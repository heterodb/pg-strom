/*
 * dpuserv.c
 *
 * A standalone command that handles XpuCommands on DPU devices
 * --------
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "dpuserv.h"

#define PEER_ADDR_LEN	80
typedef struct
{
	dlist_node			chain;	/* link to dpu_client_list */
	kern_session_info  *session;/* per-session information */
	volatile bool		in_termination; /* true, if error status */
	volatile int32_t	refcnt;	/* odd-number as long as socket is active */
	pthread_mutex_t		mutex;	/* mutex to write the socket */
	int					sockfd;	/* connection to PG-backend */
	pthread_t			worker;	/* receiver thread */
	char				peer_addr[PEER_ADDR_LEN];
} dpuClient;

static char			   *dpuserv_listen_addr = NULL;
static long				dpuserv_listen_port = -1;
static char			   *dpuserv_base_directory = NULL;
static long				dpuserv_num_workers = -1;
static char			   *dpuserv_identifier = NULL;
static const char	   *dpuserv_logfile = NULL;
static bool				verbose = false;
static pthread_mutex_t	dpu_client_mutex;
static dlist_head		dpu_client_list;
static pthread_mutex_t	dpu_command_mutex;
static pthread_cond_t	dpu_command_cond;
static dlist_head		dpu_command_list;
static volatile bool	got_sigterm = false;
static xpu_type_hash_table *dpuserv_type_htable = NULL;
static xpu_func_hash_table *dpuserv_func_htable = NULL;

/*
 * dpuClientWriteBack
 */
static void
__dpuClientWriteBack(dpuClient *dclient, struct iovec *iov, int iovcnt)
{
	pthreadMutexLock(&dclient->mutex);
	if (dclient->sockfd >= 0)
	{
		ssize_t		nbytes;

		while (iovcnt > 0)
		{
			nbytes = writev(dclient->sockfd, iov, iovcnt);
			if (nbytes > 0)
			{
				do {
					if (iov->iov_len <= nbytes)
					{
						nbytes -= iov->iov_len;
						iov++;
						iovcnt--;
					}
					else
					{
						iov->iov_base = (char *)iov->iov_base + nbytes;
						iov->iov_len -= nbytes;
						break;
					}
				} while (iovcnt > 0 && nbytes > 0);
			}
			else if (errno != EINTR)
			{
				/*
				 * Peer socket is closed? Anyway, we cannot continue to
				 * send back the message any more. So, clean up this client.
				 */
				close(dclient->sockfd);
				dclient->sockfd = -1;
				break;
			}
		}
	}
	pthreadMutexUnlock(&dclient->mutex);
}

static void
dpuClientWriteBack(dpuClient *dclient,
				   XpuCommand *resp,
				   size_t      resp_sz,
				   int         kds_nitems,
				   kern_data_store **kds_array)
{
	struct iovec   *iov_array;
	struct iovec   *iov;
	int				i, iovcnt = 0;

	iov_array = alloca(sizeof(struct iovec) * (2 * kds_nitems + 1));
	iov = &iov_array[iovcnt++];
	iov->iov_base = resp;
	iov->iov_len  = resp_sz;
	for (i=0; i < kds_nitems; i++)
	{
		kern_data_store *kds = kds_array[i];
		size_t		sz1, sz2;

		assert(kds->format == KDS_FORMAT_ROW);
		sz1 = KDS_HEAD_LENGTH(kds) + MAXALIGN(sizeof(uint32_t) * kds->nitems);
		sz2 = __kds_unpack(kds->usage);
		if (sz1 + sz2 == kds->length)
		{
			iov = &iov_array[iovcnt++];
			iov->iov_base = kds;
			iov->iov_len  = kds->length;
		}
		else
		{
			assert(sz1 + sz2 < kds->length);
			iov = &iov_array[iovcnt++];
			iov->iov_base = kds;
			iov->iov_len  = sz1;

			if (sz2 > 0)
			{
				iov = &iov_array[iovcnt++];
				iov->iov_base = (char *)kds + kds->length - sz2;
				iov->iov_len  = sz2;
			}
			kds->length = (sz1 + sz2);
		}
		resp_sz += kds->length;
	}
	resp->length = resp_sz;
	__dpuClientWriteBack(dclient, iov_array, iovcnt);
}

/*
 * dpuClientElog
 */
static void
__dpuClientElog(dpuClient *dclient,
				int errcode,
				const char *filename, int lineno,
				const char *funcname,
				const char *fmt, ...)
{
	XpuCommand		resp;
	va_list			ap;
	struct iovec	iov;
	const char	   *pos;

	for (pos = filename; *pos != '\0'; pos++)
	{
		if (pos[0] == '/' && pos[1] != '\0')
			filename = pos + 1;
	}
	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Error;
	resp.length = offsetof(XpuCommand, u.error) + sizeof(kern_errorbuf);
	resp.u.error.errcode = errcode;
	resp.u.error.lineno = lineno;
	strncpy(resp.u.error.filename, filename, KERN_ERRORBUF_FILENAME_LEN);
	strncpy(resp.u.error.funcname, funcname, KERN_ERRORBUF_FUNCNAME_LEN);

	va_start(ap, fmt);
	vsnprintf(resp.u.error.message, KERN_ERRORBUF_MESSAGE_LEN, fmt, ap);
	va_end(ap);

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__dpuClientWriteBack(dclient, &iov, 1);

	/* go to termination of this client */
	dclient->in_termination = true;
	pthread_kill(dclient->worker, SIGUSR1);
}

#define dpuClientElog(dclient,fmt,...)			\
	__dpuClientElog((dclient), ERRCODE_DEVICE_INTERNAL,	\
					__FILE__, __LINE__, __FUNCTION__,	\
					(fmt), ##__VA_ARGS__)
/*
 * xPU type/func lookup hash-table
 */
static void
__setupDevTypeLinkageTable(uint32_t xpu_type_hash_nslots)
{
	xpu_type_hash_entry *entry;
	uint32_t	i, k;

	dpuserv_type_htable = calloc(1, offsetof(xpu_type_hash_table,
											 slots[xpu_type_hash_nslots]));
	if (!dpuserv_type_htable)
		__Elog("out of memory");
	dpuserv_type_htable->nslots = xpu_type_hash_nslots;
	for (i=0; builtin_xpu_types_catalog[i].type_ops != NULL; i++)
	{
		entry = malloc(sizeof(xpu_type_hash_entry));
		if (!entry)
			__Elog("out of memory");
		memcpy(&entry->cat,
			   &builtin_xpu_types_catalog[i],
			   sizeof(xpu_type_catalog_entry));
		k = builtin_xpu_types_catalog[i].type_opcode % xpu_type_hash_nslots;
		entry->next = dpuserv_type_htable->slots[k];
		dpuserv_type_htable->slots[k] = entry;
	}
	/* add custom type */
}

static void
__setupDevFuncLinkageTable(uint32_t xpu_func_hash_nslots)
{
	xpu_func_hash_entry *entry;
	uint32_t	i, k;

	dpuserv_func_htable = calloc(1, offsetof(xpu_func_hash_table,
											 slots[xpu_func_hash_nslots]));
	if (!dpuserv_func_htable)
		__Elog("out of memory");
	dpuserv_func_htable->nslots = xpu_func_hash_nslots;
	for (i=0; builtin_xpu_functions_catalog[i].func_dptr != NULL; i++)
	{
		entry = malloc(sizeof(xpu_func_hash_entry));
		if (!entry)
			__Elog("out of memory");
		memcpy(&entry->cat,
			   &builtin_xpu_functions_catalog[i],
			   sizeof(xpu_function_catalog_entry));
		k = entry->cat.func_opcode % xpu_func_hash_nslots;
		entry->next = dpuserv_func_htable->slots[k];
		dpuserv_func_htable->slots[k] = entry;
	}
	/* add custom functions here */
}

static bool
__resolveDevicePointersWalker(kern_expression *kexp,
							  const xpu_type_hash_table *xtype_htable,
							  const xpu_func_hash_table *xfunc_htable)
{
	const xpu_type_hash_entry *dtype_hentry;
	const xpu_func_hash_entry *dfunc_hentry;
	kern_expression *karg;
	uint32_t	i, k, n;

	/* lookup device function */
	k = (uint32_t)kexp->opcode % xfunc_htable->nslots;
	for (dfunc_hentry = xfunc_htable->slots[k];
		 dfunc_hentry != NULL;
		 dfunc_hentry = dfunc_hentry->next)
	{
		if (dfunc_hentry->cat.func_opcode == kexp->opcode)
			break;
	}
	if (!dfunc_hentry)
	{
		fprintf(stderr, "device function pointer for opcode:%u not found.\n",
				(int)kexp->opcode);
		return false;
	}
	kexp->fn_dptr = dfunc_hentry->cat.func_dptr;

	/* lookup device type */
	k = kexp->exptype % xtype_htable->nslots;
	for (dtype_hentry = xtype_htable->slots[k];
		 dtype_hentry != NULL;
		 dtype_hentry = dtype_hentry->next)
	{
		if (dtype_hentry->cat.type_opcode == kexp->exptype)
			break;
	}
	if (!dtype_hentry)
	{
		fprintf(stderr, "device type pointer for opcode:%u not found.\n",
				(int)kexp->exptype);
		return false;
	}
	kexp->expr_ops = dtype_hentry->cat.type_ops;

	/* special handling for Projection */
	if (kexp->opcode == FuncOpCode__Projection)
	{
		n = kexp->u.proj.nexprs + kexp->u.proj.nattrs;
		for (i=0; i < n; i++)
		{
			kern_projection_desc *desc = &kexp->u.proj.desc[i];

			k = desc->slot_type % xtype_htable->nslots;
			for (dtype_hentry = xtype_htable->slots[k];
				 dtype_hentry != NULL;
				 dtype_hentry = dtype_hentry->next)
			{
				if (dtype_hentry->cat.type_opcode == desc->slot_type)
					break;
			}
			if (dtype_hentry)
				desc->slot_ops = dtype_hentry->cat.type_ops;
			else if (i >= kexp->u.proj.nexprs)
				desc->slot_ops = NULL;	/* PostgreSQL generic projection */
			else
			{
				fprintf(stderr, "device type pointer for opcode:%u not found.\n",
						(int)desc->slot_type);
				return false;
			}
		}
	}

	/* arguments  */
	for (i=0, karg=KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg=KEXP_NEXT_ARG(karg))
	{
		if (!__KEXP_IS_VALID(kexp,karg))
		{
			fprintf(stderr, "xPU code corruption at args[%d]\n", i);
			return false;
		}
		if (!__resolveDevicePointersWalker(karg,
										   xtype_htable,
										   xfunc_htable))
			return false;
	}
	return true;
}

static bool
xpuServResolveDevicePointers(kern_session_info *session,
							 const xpu_type_hash_table *xtype_htable,
							 const xpu_func_hash_table *xfunc_htable,
							 const xpu_encode_info *xpu_encode_catalog)
{
	xpu_encode_info *encode = SESSION_ENCODE(session);
	kern_expression *kexp;

	if (session->xpucode_scan_quals)
	{
		kexp = (kern_expression *)((char *)session + session->xpucode_scan_quals);
		if (!__resolveDevicePointersWalker(kexp,
										   xtype_htable,
										   xfunc_htable))
			return false;
	}

	if (session->xpucode_scan_projs)
	{
		kexp = (kern_expression *)((char *)session + session->xpucode_scan_projs);
		if (!__resolveDevicePointersWalker(kexp,
										   xtype_htable,
										   xfunc_htable))
			return false;
	}

	if (encode)
	{
		for (int i=0; (xpu_encode_catalog[i].enc_mblen &&
					   xpu_encode_catalog[i].enc_maxlen > 0); i++)
		{
			if (strcmp(encode->encname, xpu_encode_catalog[i].encname) == 0)
			{
				encode->enc_maxlen = xpu_encode_catalog[i].enc_maxlen;
				encode->enc_mblen  = xpu_encode_catalog[i].enc_mblen;
				goto found;
			}
		}
		return false;
	}
found:
	return true;
}

/*
 * dpuservHandleOpenSession 
 */
static bool
dpuservHandleOpenSession(dpuClient *dclient, XpuCommand *xcmd)
{
	kern_session_info *session = &xcmd->u.session;
	XpuCommand		resp;
	struct iovec	iov;

	if (dclient->session)
	{
		dpuClientElog(dclient, "OpenSession is called twice");
		return false;
	}
	if (!xpuServResolveDevicePointers(session,
									  dpuserv_type_htable,
									  dpuserv_func_htable,
									  xpu_encode_catalog))
	{
		dpuClientElog(dclient, "unable to resolve device pointers");
		return false;
	}
	dclient->session = session;

	/* success status */
	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Success;
	resp.length = offsetof(XpuCommand, u);

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__dpuClientWriteBack(dclient, &iov, 1);
	return true;
}

/*
 * dpuservLoadKdsBlock
 *
 * fill up KDS_FORMAT_BLOCK using device local filesystem
 */
static kern_data_store *
dpuservLoadKdsBlock(dpuClient *dclient,
					const kern_data_store *kds_head,
					const char *pathname,
					const strom_io_vector *kds_iovec,
					char **p_base_addr)
{
	kern_data_store *kds;
	char   *data;
	int		fdesc;

	fdesc = open(pathname, O_RDONLY | O_DIRECT | O_NOATIME);
	if (fdesc < 0)
	{
		dpuClientElog(dclient, "failed on open('%s'): %m", pathname);
		return NULL;
	}

	data = malloc(kds_head->length + PAGE_SIZE);
	if (!data)
	{
		close(fdesc);
		dpuClientElog(dclient, "out of memory: %m");
		return NULL;
	}

	/*
	 * Due to the restriction of O_DIRECT, ((char *)kds + kds->block_offset)
	 * must be aligned to PAGE_SIZE.
	 */
	assert(kds_head->block_nloaded == 0);
	kds = (kern_data_store *)
		(PAGE_ALIGN(data + kds_head->block_offset) - kds_head->block_offset);
	memcpy(kds, kds_head, kds_head->block_offset);

	if (kds_iovec)
	{
		char	   *base = (char *)kds + kds->block_offset;

		for (int i=0; i < kds_iovec->nr_chunks; i++)
		{
			const strom_io_chunk *ioc = &kds_iovec->ioc[i];
			char	   *dest = base + ioc->m_offset;
			ssize_t		offset = PAGE_SIZE * ioc->fchunk_id;
			ssize_t		length = PAGE_SIZE * ioc->nr_pages;
			ssize_t		nbytes;

			assert(dest + length <= (char *)kds + kds->length);
			while (length > 0)
			{
				nbytes = pread(fdesc, dest, length, offset);
				if (nbytes > 0)
				{
					assert(nbytes <= length);
					dest += nbytes;
					length -= nbytes;
					offset -= nbytes;
				}
				else if (nbytes == 0 || errno != EINTR)
				{
					dpuClientElog(dclient, "failed on pread('%s', %ld, %ld) = %ld: %m",
								  pathname, length, offset, nbytes);
					free(data);
					close(fdesc);
					return NULL;
				}
			}
		}
	}
	close(fdesc);
	*p_base_addr = data;

	return kds;
}

/* ----------------------------------------------------------------
 *
 * DPU Kernel Projection
 *
 * ---------------------------------------------------------------- */
int
ExecProjectionOuterRow(kern_context *kcxt,
					   kern_expression *kexp,	/* LoadVars + Projection */
					   kern_data_store *kds_dst,
					   kern_data_store *kds_outer,
					   HeapTupleHeaderData *htup_outer,
					   int num_inners,
					   kern_data_store **kds_inners,
					   HeapTupleHeaderData **htup_inners)
{
	xpu_int4_t	__tupsz;

	/*
	 * First, extract the variables from outer/inner tuples, and
	 * calculate expressions, if any.
	 */
	assert(kexp->opcode  == FuncOpCode__LoadVars &&
		   kexp->exptype == TypeOpCode__int4);
	if (!ExecLoadVarsOuterRow(kcxt,
							  kexp,
							  (xpu_datum_t *)&__tupsz,
							  kds_outer,
							  htup_outer,
							  num_inners,
							  kds_inners,
							  htup_inners))
		return -1;
	if (!__tupsz.isnull && __tupsz.value > 0)
	{
		kern_expression *kproj = KEXP_FIRST_ARG(kexp);
		int32_t		tupsz = MAXALIGN(__tupsz.value);
		uint32_t	rowid;
		size_t		offset;
		size_t		newsz;
		kern_tupitem *tupitem;

		offset = __kds_unpack(kds_dst->usage) + tupsz;
		newsz = (KDS_HEAD_LENGTH(kds_dst) +
				 MAXALIGN(sizeof(uint32_t) * (kds_dst->nitems + 1)) +
				 offset);
		if (newsz > kds_dst->length)
			return 0;	/* retry with new kds_dst */
		kds_dst->usage = __kds_packed(offset);
		rowid = kds_dst->nitems++;
		KDS_GET_ROWINDEX(kds_dst)[rowid] = kds_dst->usage;
		tupitem = (kern_tupitem *)
			((char *)kds_dst + kds_dst->length - offset);
		tupitem->rowid = rowid;
		tupitem->t_len = kern_form_heaptuple(kcxt, kproj, kds_dst, &tupitem->htup);

		return tupsz;
	}
	return -1;
}

/* ----------------------------------------------------------------
 *
 * DPU service SCAN handler
 *
 * ---------------------------------------------------------------- */
typedef struct
{
	kern_errorbuf	kerror;
	uint32_t		nitems_in;
	uint32_t		nitems_out;
	uint32_t		extra_sz;
} kern_dpuscan;

static void
__handleDpuScanExecBlock(dpuClient *dclient,
						 kern_data_store *kds_src,
						 kern_data_store *kds_dst_head)
{
	kern_session_info  *session = dclient->session;
	kern_expression	   *kexp_scan_quals = SESSION_KEXP_SCAN_QUALS(session);
	kern_expression	   *kexp_scan_projs = SESSION_KEXP_SCAN_PROJS(session);
	kern_context	   *kcxt;
	kern_data_store	   *kds_dst = NULL;
	kern_data_store	  **kds_dst_array = NULL;
	uint32_t			kds_dst_nrooms = 0;
	uint32_t			kds_dst_nitems = 0;
	uint32_t			block_index;
	uint32_t			stat_nitems_in = 0;
	uint32_t			stat_nitems_out = 0;
	kern_dpuscan		kdscan;
	XpuCommand			resp;

	assert(kds_src->format == KDS_FORMAT_BLOCK &&
		   kexp_scan_quals->opcode == FuncOpCode__LoadVars &&
		   kexp_scan_projs->opcode == FuncOpCode__LoadVars);
	memset(&kdscan, 0, sizeof(kern_dpuscan));
	INIT_KERNEL_CONTEXT(kcxt, session);

	for (block_index = 0; block_index < kds_src->nitems; block_index++)
	{
		PageHeaderData *page = KDS_BLOCK_PGPAGE(kds_src, block_index);
		uint32_t		lp_nitems = PageGetMaxOffsetNumber(page);
		uint32_t		lp_index;

		for (lp_index = 0; lp_index < lp_nitems; lp_index++)
		{
			ItemIdData	   *lpp = &page->pd_linp[lp_index];
			HeapTupleHeaderData *htup;
			xpu_bool_t		retval;
			size_t			sz;
			int				status;

			if (!ItemIdIsNormal(lpp))
				continue;
			kcxt_reset(kcxt);
			htup = (HeapTupleHeaderData *) PageGetItem(page, lpp);
			stat_nitems_in++;
			if (!ExecLoadVarsOuterRow(kcxt,
									  kexp_scan_quals,
									  (xpu_datum_t *)&retval,
									  kds_src,
									  htup,
									  0, NULL, NULL))
			{
				/* error to abort */
				__dpuClientElog(dclient,
								kcxt->errcode,
								kcxt->error_filename,
								kcxt->error_lineno,
								kcxt->error_funcname,
								kcxt->error_message);
				goto bailout;
			}
			else if (retval.isnull || !retval.value)
			{
				/* not matched */
				continue;
			}
			stat_nitems_out++;

		retry:
			if (!kds_dst)
			{
				if (kds_dst_nitems >= kds_dst_nrooms)
				{
					kern_data_store	  **__kds_dst_array;

					kds_dst_nrooms = 2 * kds_dst_nrooms + 10;
					__kds_dst_array = alloca(sizeof(kern_data_store *) * kds_dst_nrooms);
					memcpy(__kds_dst_array, kds_dst_array,
						   sizeof(kern_data_store *) * kds_dst_nitems);
					kds_dst_array = __kds_dst_array;
				}
				sz = KDS_HEAD_LENGTH(kds_dst_head) + PGSTROM_CHUNK_SIZE;
				kds_dst = malloc(sz);
				if (!kds_dst)
				{
					dpuClientElog(dclient, "out of memory");
					goto bailout;
				}
				memcpy(kds_dst, kds_dst_head, KDS_HEAD_LENGTH(kds_dst_head));
				kds_dst->length = sz;
				kds_dst_array[kds_dst_nitems++] = kds_dst;
			}
			status = ExecProjectionOuterRow(kcxt,
											kexp_scan_projs,
											kds_dst,
											kds_src,
											htup,
											0, NULL, NULL);
			if (status == 0)
			{
				kds_dst = NULL;
				goto retry;
			}
			else if (status < 0)
			{
				__dpuClientElog(dclient,
								kcxt->errcode,
								kcxt->error_filename,
								kcxt->error_lineno,
								kcxt->error_funcname,
								kcxt->error_message);
				goto bailout;
			}
		}
	}
	/* write back the results */
	memset(&resp, 0, offsetof(XpuCommand, u.results));
	resp.magic = XpuCommandMagicNumber;
	resp.tag   = XpuCommandTag__Success;
	resp.u.results.chunks_nitems = kds_dst_nitems;
	resp.u.results.chunks_offset = offsetof(XpuCommand, u.results.stats.scan.data);
	resp.u.results.stats.scan.nitems_in = stat_nitems_in;
	resp.u.results.stats.scan.nitems_out = stat_nitems_out;
	dpuClientWriteBack(dclient,
					   &resp, resp.u.results.chunks_offset,
					   kds_dst_nitems,
					   kds_dst_array);
bailout:
	for (int i=0; i < kds_dst_nitems; i++)
		free(kds_dst_array[i]);
}

static void
dpuservHandleDpuScanExec(dpuClient *dclient, XpuCommand *xcmd)
{
	const char		   *kds_src_pathname = NULL;
	strom_io_vector	   *kds_src_iovec = NULL;
	kern_data_store	   *kds_src_head = NULL;
	kern_data_store	   *kds_dst_head = NULL;
	kern_data_store	   *kds_src = NULL;

	if (xcmd->u.scan.kds_src_pathname)
		kds_src_pathname = (char *)xcmd + xcmd->u.scan.kds_src_pathname;
	if (xcmd->u.scan.kds_src_iovec)
		kds_src_iovec = (strom_io_vector *)((char *)xcmd + xcmd->u.scan.kds_src_iovec);
	if (xcmd->u.scan.kds_src_offset)
		kds_src_head = (kern_data_store *)((char *)xcmd + xcmd->u.scan.kds_src_offset);
	if (xcmd->u.scan.kds_dst_offset)
		kds_dst_head = (kern_data_store *)((char *)xcmd + xcmd->u.scan.kds_dst_offset);
	if (!kds_src_pathname || !kds_src_iovec || !kds_src_head || !kds_dst_head)
	{
		dpuClientElog(dclient, "kern_data_store is corrupted");
		return;
	}

	if (kds_src_head->format == KDS_FORMAT_BLOCK)
	{
		char   *base_addr;

		kds_src = dpuservLoadKdsBlock(dclient,
									  kds_src_head,
									  kds_src_pathname,
									  kds_src_iovec,
									  &base_addr);
		if (kds_src)
		{
			__handleDpuScanExecBlock(dclient, kds_src, kds_dst_head);
			free(base_addr);
		}
	}
#if 0
	else if (kds_src_head->format == KDS_FORMAT_ARROW)
	{
		//do arrow mode
	}
#endif
	else
	{
		dpuClientElog(dclient, "not a supported kern_data_store format");
		return;
	}
}












/*
 * getDpuClient
 */
static void
getDpuClient(dpuClient *dclient, int count)
{
	int32_t		refcnt;

	refcnt = __atomic_fetch_add(&dclient->refcnt, count, __ATOMIC_SEQ_CST);
	assert(refcnt > 0);
}

/*
 * putDpuClient
 */
static void
putDpuClient(dpuClient *dclient, int count)
{
	int32_t		refcnt;

	refcnt = __atomic_sub_fetch(&dclient->refcnt, count, __ATOMIC_SEQ_CST);
	assert(refcnt >= 0);
	if (refcnt == 0)
	{
		pthreadMutexLock(&dpu_client_mutex);
		if (dclient->chain.prev && dclient->chain.next)
		{
			dlist_delete(&dclient->chain);
			memset(&dclient->chain, 0, sizeof(dlist_node));
		}
		pthreadMutexUnlock(&dpu_client_mutex);
		
		if (dclient->session)
		{
			void   *xcmd = ((char *)dclient->session -
							offsetof(XpuCommand, u.session));
			free(xcmd);
		}
		close(dclient->sockfd);
		free(dclient);
	}
}

/*
 * dpuservDpuWorkerMain
 */
static void *
dpuservDpuWorkerMain(void *__priv)
{
	long	worker_id = (long)__priv;

	if (verbose)
		fprintf(stderr, "[worker-%lu] DPU service worker start.\n", worker_id);
	pthreadMutexLock(&dpu_command_mutex);
	while (!got_sigterm)
	{
		if (!dlist_is_empty(&dpu_command_list))
		{
			dlist_node	   *dnode = dlist_pop_head_node(&dpu_command_list);
			XpuCommand	   *xcmd = dlist_container(XpuCommand, chain, dnode);
			dpuClient	   *dclient;
			pthreadMutexUnlock(&dpu_command_mutex);

			dclient = xcmd->priv;
			/*
			 * MEMO: If the least bit of gclient->refcnt is not set,
			 * it means the gpu-client connection is no longer available.
			 * (monitor thread has already gone)
			 */
			if ((dclient->refcnt & 1) == 1)
			{
				switch (xcmd->tag)
				{
					case XpuCommandTag__OpenSession:
						if (dpuservHandleOpenSession(dclient, xcmd))
							xcmd = NULL;	/* session information shall be kept until
											 * end of the session. */
						if (verbose)
							fprintf(stderr, "[DPU-%ld@%s] OpenSession ... %s\n",
									worker_id, dclient->peer_addr,
									(xcmd != NULL ? "failed" : "ok"));
						break;
					case XpuCommandTag__XpuScanExec:
						dpuservHandleDpuScanExec(dclient, xcmd);
						if (verbose)
							fprintf(stderr, "[DPU-%ld@%s] DpuScanExec\n",
									worker_id, dclient->peer_addr);
						break;
					default:
						fprintf(stderr, "[DPU-%ld@%s] unknown xPU command (tag=%u, len=%ld)\n",
								worker_id, dclient->peer_addr,
								xcmd->tag, xcmd->length);
						break;
				}
			}
			if (xcmd)
				free(xcmd);
			putDpuClient(dclient, 2);
			pthreadMutexLock(&dpu_command_mutex);
		}
		else
		{
			pthreadCondWait(&dpu_command_cond,
							&dpu_command_mutex);
		}
	}
	pthreadMutexUnlock(&dpu_command_mutex);
	if (verbose)
		fprintf(stderr, "[worker-%lu] DPU service worker terminated.\n", worker_id);
	return NULL;
}

/*
 * dpuservMonitorClient
 */
static void *
__dpuServAllocCommand(void *__priv, size_t sz)
{
	return malloc(sz);
}

static void
__dpuServAttachCommand(void *__priv, XpuCommand *xcmd)
{
	dpuClient  *dclient = (dpuClient *)__priv;

	getDpuClient(dclient, 2);
	xcmd->priv = dclient;

	if (verbose)
		fprintf(stderr, "[%s] received xcmd (tag=%u len=%lu)\n",
				dclient->peer_addr,
				xcmd->tag, xcmd->length);

	pthreadMutexLock(&dpu_command_mutex);
	dlist_push_tail(&dpu_command_list, &xcmd->chain);
	pthreadCondSignal(&dpu_command_cond);
	pthreadMutexUnlock(&dpu_command_mutex);
}

TEMPLATE_XPU_CONNECT_RECEIVE_COMMANDS(__dpuServ)

static void *
dpuservMonitorClient(void *__priv)
{
	dpuClient  *dclient = (dpuClient *)__priv;

	if (verbose)
		fprintf(stderr, "[%s] connection start\n", dclient->peer_addr);
	while (!got_sigterm && !dclient->in_termination)
	{
		struct pollfd  pfd;
		int		rv;

		pfd.fd = dclient->sockfd;
		pfd.events = POLLIN | POLLRDHUP;
		pfd.revents = 0;
		rv = poll(&pfd, 1, -1);
		if (rv < 0)
		{
			if (errno == EINTR)
				continue;
			fprintf(stderr, "[%s] failed on poll(2): %m\n", dclient->peer_addr);
			break;
		}
		else if (rv > 0)
		{
			assert(rv == 1);
			if (pfd.revents == POLLIN)
			{
				if (__dpuServReceiveCommands(dclient->sockfd, dclient,
											 dclient->peer_addr) < 0)
					break;
			}
			else if (pfd.revents & ~POLLIN)
			{
				if (verbose)
					fprintf(stderr, "[%s] peer socket closed\n", dclient->peer_addr);
				break;
			}
		}
	}
	dclient->in_termination = true;
	putDpuClient(dclient, 1);
	if (verbose)
		fprintf(stderr, "[%s] connection terminated\n", dclient->peer_addr);
	return NULL;
}

static void
dpuserv_signal_handler(int signum)
{
	int		errno_saved = errno;

	if (signum == SIGTERM)
		got_sigterm = true;
	if (verbose)
		fprintf(stderr, "got signal (%d)\n", signum);
	errno = errno_saved;
}

static int
dpuserv_main(struct sockaddr *addr, socklen_t addr_len)
{
	pthread_t  *dpuserv_workers;
	int			serv_fd;
	int			epoll_fd;
	struct epoll_event epoll_ev;

	/* setup signal handler */
	signal(SIGTERM, dpuserv_signal_handler);
	signal(SIGUSR1, dpuserv_signal_handler);

	/* start worker threads */
	dpuserv_workers = alloca(sizeof(pthread_t) * dpuserv_num_workers);
	for (long i=0; i < dpuserv_num_workers; i++)
	{
		if ((errno = pthread_create(&dpuserv_workers[i], NULL,
									dpuservDpuWorkerMain, (void *)i)) != 0)
			__Elog("failed on pthread_create: %m");
	}
	
	/* setup server socket */
	serv_fd = socket(addr->sa_family, SOCK_STREAM, 0);
	if (serv_fd < 0)
		__Elog("failed on socket(2): %m");
	if (bind(serv_fd, addr, addr_len) != 0)
		__Elog("failed on bind(2): %m");
	if (listen(serv_fd, dpuserv_num_workers) != 0)
		__Elog("failed on listen(2): %m");

	/* setup epoll */
	epoll_fd = epoll_create(1);
	if (epoll_fd < 0)
		__Elog("failed on epoll_create: %m");
	epoll_ev.events = EPOLLIN;
	epoll_ev.data.fd = serv_fd;
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, serv_fd, &epoll_ev) != 0)
		__Elog("failed on epoll_ctl(EPOLL_CTL_ADD): %m");
	epoll_ev.events = EPOLLIN | EPOLLRDHUP;
	epoll_ev.data.fd = fileno(stdin);
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fileno(stdin), &epoll_ev) != 0)
		__Elog("failed on epoll_ctl(EPOLL_CTL_ADD): %m");
	
	while (!got_sigterm)
	{
		int		rv;

		rv = epoll_wait(epoll_fd, &epoll_ev, 1, 2000);
		if (rv > 0)
		{
			assert(rv == 1);
			if (epoll_ev.data.fd == serv_fd)
			{
				struct sockaddr		peer;
				socklen_t			peer_sz = sizeof(struct sockaddr);
				int					client_fd;

				if ((epoll_ev.events & ~EPOLLIN) != 0)
					__Elog("listen socket raised unexpected error events (%08x): %m",
						   epoll_ev.events);
				
				client_fd = accept(serv_fd, &peer, &peer_sz);
				if (client_fd < 0)
				{
					if (errno != EINTR)
						__Elog("failed on accept: %m");
				}
				else
				{
					dpuClient  *dclient;

					dclient = calloc(1, sizeof(dpuClient));
					if (!dclient)
						__Elog("out of memory: %m");
					dclient->refcnt = 1;
					pthreadMutexInit(&dclient->mutex);
					dclient->sockfd = client_fd;
					if (peer.sa_family == AF_INET)
						inet_ntop(peer.sa_family,
								  (char *)&peer +
								  offsetof(struct sockaddr_in, sin_addr),
								  dclient->peer_addr, PEER_ADDR_LEN);
					else if (peer.sa_family == AF_INET6)
						inet_ntop(peer.sa_family,
								  (char *)&peer +
								  offsetof(struct sockaddr_in6, sin6_addr),
								  dclient->peer_addr, PEER_ADDR_LEN);
					else
						snprintf(dclient->peer_addr, PEER_ADDR_LEN,
								 "Unknown DpuClient");

					pthreadMutexLock(&dpu_client_mutex);
					if ((errno = pthread_create(&dclient->worker, NULL,
												dpuservMonitorClient,
												dclient)) == 0)
					{
						dlist_push_tail(&dpu_client_list,
										&dclient->chain);
					}
					else
					{
						fprintf(stderr, "failed on pthread_create: %m\n");
						close(client_fd);
						free(dclient);
					}
					pthreadMutexUnlock(&dpu_client_mutex);
				}
			}
			else
			{
				char	buffer[1024];
				ssize_t	nbytes;

				assert(epoll_ev.data.fd == fileno(stdin));
				if ((epoll_ev.events & ~EPOLLIN) != 0)
					got_sigterm = true;
				else
				{
					/* make stdin buffer empty */
					nbytes = read(epoll_ev.data.fd, buffer, 1024);
					if (nbytes < 0)
						__Elog("failed on read(stdin): %m");
				}
			}
		}
		else if (rv < 0 && errno != EINTR)
			__Elog("failed on poll(2): %m");
	}
	close(serv_fd);

	/* wait for completion of worker threads */
	pthread_cond_broadcast(&dpu_command_cond);
	for (int i=0; i < dpuserv_num_workers; i++)
		pthread_join(dpuserv_workers[i], NULL);
	pthreadMutexLock(&dpu_client_mutex);
	while (!dlist_is_empty(&dpu_client_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&dpu_client_list);
		dpuClient  *dclient = dlist_container(dpuClient, chain, dnode);

		pthreadMutexUnlock(&dpu_client_mutex);

		pthread_kill(dclient->worker, SIGUSR1);
		pthread_join(dclient->worker, NULL);

		pthreadMutexLock(&dpu_client_mutex);
	}
	pthreadMutexUnlock(&dpu_client_mutex);
	printf("OK terminate\n");
	return 0;
}

int
main(int argc, char *argv[])
{
	static struct option command_options[] = {
		{"addr",       required_argument, 0, 'a'},
		{"port",       required_argument, 0, 'p'},
		{"directory",  required_argument, 0, 'd'},
		{"nworkers",   required_argument, 0, 'n'},
		{"identifier", required_argument, 0, 'i'},
		{"log",        required_argument, 0, 'l'},
		{"verbose",    no_argument,       0, 'v'},
		{"help",       no_argument,       0, 'h'},
		{NULL, 0, 0, 0},
	};
	struct sockaddr *addr;
	socklen_t	addr_len;

	/* init misc variables */
	pthreadMutexInit(&dpu_client_mutex);
	dlist_init(&dpu_client_list);
	pthreadMutexInit(&dpu_command_mutex);
	pthreadCondInit(&dpu_command_cond);
	dlist_init(&dpu_command_list);

	/* parse command line options */
	for (;;)
	{
		int		c = getopt_long(argc, argv, "a:p:d:n:i:l:vh",
								command_options, NULL);
		char   *end;

		if (c < 0)
			break;
		switch (c)
		{
			case 'a':
				if (dpuserv_listen_addr)
					__Elog("-a|--addr option was given twice");
				dpuserv_listen_addr = optarg;
				break;
			
			case 'p':
				if (dpuserv_listen_port > 0)
					__Elog("-p|--port option was given twice");
				dpuserv_listen_port = strtol(optarg, &end, 10);
				if (*optarg == '\0' || *end != '\0')
					__Elog("port number [%s] is not valid", optarg);
				if (dpuserv_listen_port < 1024 || dpuserv_listen_port > USHRT_MAX)
					__Elog("port number [%ld] is out of range", dpuserv_listen_port);
				break;

			case 'd':
				if (dpuserv_base_directory)
					__Elog("-d|--directory option was given twice");
				dpuserv_base_directory = optarg;
				break;
				
			case 'n':
				if (dpuserv_num_workers > 0)
					__Elog("-n|--num-workers option was given twice");
				dpuserv_num_workers = strtol(optarg, &end, 10);
				if (*optarg == '\0' || *end != '\0')
					__Elog("number of workers [%s] is not valid", optarg);
				if (dpuserv_num_workers < 1)
					__Elog("number of workers %ld is out of range",
						   dpuserv_num_workers);
				break;

			case 'i':
				if (dpuserv_identifier)
					__Elog("-i|--identifier option was given twice");
				dpuserv_identifier = optarg;
				break;

			case 'l':
				if (dpuserv_logfile)
					__Elog("-l|--log option was given twice");
				dpuserv_logfile = optarg;
				break;
				
			case 'v':
				verbose = true;
				break;
			default:	/* --help */
				fputs("usage: dpuserv [OPTIONS]\n"
					  "\n"
					  "\t-p|--port=PORT           listen port (default: 6543)\n"
					  "\t-d|--directory=DIR       tablespace base (default: .)\n"
					  "\t-n|--nworkers=N_WORKERS  number of workers (default: auto)\n"
					  "\t-i|--identifier=IDENT    security identifier\n"
					  "\t-v|--verbose             verbose output\n"
					  "\t-h|--help                shows this message\n",
					  stderr);
				return 1;
		}
	}
	/* apply default values */
	if (dpuserv_listen_port < 0)
		dpuserv_listen_port = 6543;
	if (!dpuserv_base_directory)
		dpuserv_base_directory = ".";
	if (dpuserv_num_workers < 0)
		dpuserv_num_workers = Max(4 * sysconf(_SC_NPROCESSORS_ONLN), 20);
	if (dpuserv_logfile)
	{
		FILE   *stdlog;

		stdlog = fopen(dpuserv_logfile, "ab");
		if (!stdlog)
			__Elog("failed on fopen('%s','ab'): %m", dpuserv_logfile);
		fclose(stderr);
		stderr = stdlog;
	}

	/* change the current working directory */
	if (chdir(dpuserv_base_directory) != 0)
		__Elog("failed on chdir('%s'): %m", dpuserv_base_directory);
	/* resolve host and port */
	if (!dpuserv_listen_addr)
	{
		static struct sockaddr_in __addr;

		memset(&__addr, 0, sizeof(__addr));
		__addr.sin_family = AF_INET;
		__addr.sin_port = htons(dpuserv_listen_port);
		__addr.sin_addr.s_addr = htonl(INADDR_ANY);
		addr = (struct sockaddr *)&__addr;
		addr_len = sizeof(__addr);
	}
	else
	{
		struct addrinfo hints;
		struct addrinfo *res;
		char		temp[50];

		memset(&hints, 0, sizeof(struct addrinfo));
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;
		snprintf(temp, sizeof(temp), "%ld", dpuserv_listen_port);
		if (getaddrinfo(dpuserv_listen_addr, temp, &hints, &res) != 0)
			__Elog("failed on getaddrinfo('%s',%ld): %m",
				   dpuserv_listen_addr,
				   dpuserv_listen_port);
		addr = res->ai_addr;
		addr_len = res->ai_addrlen;
	}
	/*
	 * setup device type/func lookup table
	 */
	__setupDevTypeLinkageTable(2 * TypeOpCode__BuiltInMax + 20);
	__setupDevFuncLinkageTable(2 * FuncOpCode__BuiltInMax + 100);

	return dpuserv_main(addr, addr_len);
}
