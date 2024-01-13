/*
 * dpuserv.c
 *
 * A standalone command that handles XpuCommands on DPU devices
 * --------
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "dpuserv.h"

struct groupby_final_buffer;

#define PEER_ADDR_LEN	80
typedef struct
{
	dlist_node			chain;	/* link to dpu_client_list */
	kern_session_info  *session;/* per-session information */
	kern_multirels	   *kmrels;		/* join inner buffer */
	size_t				kmrels_sz;	/* join inner buffer mmap-sz */
	struct groupby_final_buffer *gf_buf; /* group-by final buffer */
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
 * dpuTaskExecState
 */
struct dpuTaskExecState
{
	kern_errorbuf	kerror;
	kern_data_store *kds_dst_head;
	kern_data_store *kds_dst;
	kern_data_store **kds_dst_array;
	bool		   (*handleDpuTaskFinalDepth)(dpuClient *dclient,
											  struct dpuTaskExecState *dtes,
											  kern_context *kcxt);
	uint32_t		kds_dst_nrooms;
	uint32_t		kds_dst_nitems;
	uint32_t		nitems_raw;		/* nitems in the raw data chunk */
	uint32_t		nitems_in;		/* nitems after the scan_quals */
	uint32_t		nitems_out;		/* nitems of final results */
	uint32_t		num_rels;		/* >0, if JOIN */
	struct {
		uint32_t	nitems_gist;	/* nitems picked up by GiST index */
		uint32_t	nitems_out;		/* nitems after this depth */
	} stats[1];
};
typedef struct dpuTaskExecState		dpuTaskExecState;



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
				   dpuTaskExecState *dtes)
{
	XpuCommand	   *resp;
	struct iovec   *iov_array;
	struct iovec   *iov;
	int				iovcnt = 0;
	int				resp_sz;

	/* Xcmd for the response */
	resp_sz = MAXALIGN(offsetof(XpuCommand, u.results.stats[dtes->num_rels]));
	resp = alloca(resp_sz);
	memset(resp, 0, resp_sz);
	resp->magic = XpuCommandMagicNumber;
	resp->tag   = XpuCommandTag__Success;
	resp->u.results.chunks_offset = resp_sz;
	resp->u.results.chunks_nitems = dtes->kds_dst_nitems;
	resp->u.results.nitems_raw = dtes->nitems_raw;
	resp->u.results.nitems_in  = dtes->nitems_in;
	resp->u.results.nitems_out = dtes->nitems_out;
	resp->u.results.num_rels   = dtes->num_rels;
	for (int i=0; i < dtes->num_rels; i++)
	{
		resp->u.results.stats[i].nitems_gist = dtes->stats[i].nitems_gist;
		resp->u.results.stats[i].nitems_out  = dtes->stats[i].nitems_out;
	}

	/* Setup iovec */
	iov_array = alloca(sizeof(struct iovec) * (2 * dtes->kds_dst_nitems + 1));
	iov = &iov_array[iovcnt++];
	iov->iov_base = resp;
	iov->iov_len  = resp_sz;
	for (int i=0; i < dtes->kds_dst_nitems; i++)
	{
		kern_data_store *kds = dtes->kds_dst_array[i];
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
 * Get/Put Group-By Final Buffer
 */
struct groupby_final_buffer
{
	dlist_node	chain;
	int			refcnt;
	uint32_t	pgsql_port_number;
	uint32_t	pgsql_plan_node_id;
	uint32_t	pgsql_client_hash;
	pthread_rwlock_t kds_final_rwlock;
	kern_data_store *kds_final;
};
typedef struct groupby_final_buffer		groupby_final_buffer;

static pthread_mutex_t	groupby_final_buffer_lock;
#define GROUPBY_FINAL_BUFFER_HASHSZ		200
static dlist_head		groupby_final_buffer_hash[GROUPBY_FINAL_BUFFER_HASHSZ];

static bool
dpuServGetGroupByFinalBuffer(dpuClient *dclient, kern_session_info *session)
{
	kern_data_store *kds_final;
	dlist_head	   *slot;
	dlist_iter		iter;
	uint32_t		hash;
	struct {
		uint32_t	pgsql_port_number;
		uint32_t	pgsql_plan_node_id;
	} hkey;
	groupby_final_buffer *gf_buf;

	assert(session->groupby_kds_final != 0);
	hkey.pgsql_port_number  = session->pgsql_port_number;
	hkey.pgsql_plan_node_id = session->pgsql_plan_node_id;
	hash = pg_hash_any(dclient->peer_addr, strlen(dclient->peer_addr));
	hash ^= pg_hash_any(&hkey, sizeof(hkey));

	/* lookup the hash table first */
	pthreadMutexLock(&groupby_final_buffer_lock);
	slot = &groupby_final_buffer_hash[hash % GROUPBY_FINAL_BUFFER_HASHSZ];
	dlist_foreach (iter, slot)
	{
		gf_buf = dlist_container(groupby_final_buffer, chain, iter.cur);

		if (gf_buf->pgsql_port_number == session->pgsql_port_number &&
			gf_buf->pgsql_plan_node_id == session->pgsql_plan_node_id &&
			gf_buf->pgsql_client_hash == hash)
		{
			gf_buf->refcnt++;
			goto found;
		}
	}
	/* not found, so create a new one */
	kds_final = (kern_data_store *)((char *)session + session->groupby_kds_final);
	gf_buf = calloc(sizeof(groupby_final_buffer), 1);
	if (!gf_buf)
	{
		pthreadMutexUnlock(&groupby_final_buffer_lock);
		fprintf(stderr, "out of memory 1\n");
		return false;
	}
	gf_buf->kds_final = malloc(kds_final->length);
	if (!gf_buf->kds_final)
	{
		pthreadMutexUnlock(&groupby_final_buffer_lock);
		free(gf_buf);
		fprintf(stderr, "out of memory 2\n");
        return false;
	}
	gf_buf->refcnt = 1;
	gf_buf->pgsql_port_number  = session->pgsql_port_number;
	gf_buf->pgsql_plan_node_id = session->pgsql_plan_node_id;
	gf_buf->pgsql_client_hash  = hash;
	pthreadRWLockInit(&gf_buf->kds_final_rwlock);
	memcpy(gf_buf->kds_final, kds_final, KDS_HEAD_LENGTH(kds_final));

	dlist_push_tail(slot, &gf_buf->chain);
found:
	pthreadMutexUnlock(&groupby_final_buffer_lock);
	dclient->gf_buf = gf_buf;
	return true;
}

static void
dpuServPutGroupByFinalBuffer(groupby_final_buffer *gf_buf)
{
	pthreadMutexLock(&groupby_final_buffer_lock);
	Assert(gf_buf->refcnt > 0);
	if (--gf_buf->refcnt == 0)
	{
		dlist_delete(&gf_buf->chain);
		free(gf_buf->kds_final);
		free(gf_buf);
	}
	pthreadMutexUnlock(&groupby_final_buffer_lock);
}

/*
 * mmap/munmap session buffer
 */
static bool
dpuServMapSessionBuffers(dpuClient *dclient, kern_session_info *session)
{
	char		namebuf[100];
	int			fdesc;
	struct stat	stat_buf;
	void	   *mmap_addr;
	size_t		mmap_sz;

	if (session->join_inner_handle != 0)
	{
		snprintf(namebuf, sizeof(namebuf),
				 ".pgstrom_shmbuf_%u_%d",
				 session->pgsql_port_number,
				 session->join_inner_handle);
		fdesc = open(namebuf, O_RDWR);
		if (fdesc < 0)
			return false;
		if (fstat(fdesc, &stat_buf) != 0)
		{
			close(fdesc);
			return false;
		}
		mmap_sz = PAGE_ALIGN(stat_buf.st_size);
		mmap_addr = mmap(NULL, mmap_sz,
						 PROT_READ | PROT_WRITE,
						 MAP_SHARED,
						 fdesc, 0);
		if (mmap_addr == MAP_FAILED)
		{
			close(fdesc);
			return false;
		}
		dclient->kmrels = mmap_addr;
		dclient->kmrels_sz = mmap_sz;
	}

	if (session->groupby_kds_final)
	{
		if (!dpuServGetGroupByFinalBuffer(dclient, session))
			return false;
	}
	return true;
}

static void
dpuServUnmapSessionBuffers(dpuClient *dclient)
{
	if (dclient->kmrels)
	{
		if (munmap(dclient->kmrels,
				   dclient->kmrels_sz) != 0)
			fprintf(stderr, "failed on munmap(%p-%p): %m\n",
					(char *)dclient->kmrels,
					(char *)dclient->kmrels + dclient->kmrels_sz - 1);
	}
	if (dclient->gf_buf)
		dpuServPutGroupByFinalBuffer(dclient->gf_buf);
}

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

static const xpu_type_hash_entry *
__lookupDevTypeLinkageTable(TypeOpCode type_opcode)
{
	const xpu_type_hash_entry *entry;
	uint32_t	k;

	if (!dpuserv_type_htable)
		return NULL;

	k = (uint32_t)type_opcode % dpuserv_type_htable->nslots;
	for (entry = dpuserv_type_htable->slots[k];
		 entry != NULL;
		 entry = entry->next)
	{
		if (entry->cat.type_opcode == type_opcode)
			break;
	}
	return entry;
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

static const xpu_func_hash_entry *
__lookupDevFuncLinkageTable(FuncOpCode func_opcode)
{
	const xpu_func_hash_entry *entry;
	uint32_t	k;

	if (!dpuserv_func_htable)
		return NULL;
	k = (uint32_t)func_opcode % dpuserv_func_htable->nslots;
	for (entry = dpuserv_func_htable->slots[k];
		 entry != NULL;
		 entry = entry->next)
	{
		if (entry->cat.func_opcode == func_opcode)
			break;
	}
	return entry;
}

static bool
__resolveDevicePointersWalker(kern_expression *kexp)
{
	const xpu_type_hash_entry *dtype_hentry;
	const xpu_func_hash_entry *dfunc_hentry;
	kern_expression *karg;
	uint32_t	i;

	/* lookup device function */
	dfunc_hentry = __lookupDevFuncLinkageTable(kexp->opcode);
	if (!dfunc_hentry)
	{
		fprintf(stderr, "device function pointer for opcode:%u not found.\n",
				(int)kexp->opcode);
		return false;
	}
	kexp->fn_dptr = dfunc_hentry->cat.func_dptr;

	/* lookup device type */
	dtype_hentry = __lookupDevTypeLinkageTable(kexp->exptype);
	if (!dtype_hentry)
	{
		fprintf(stderr, "device type pointer for opcode:%u not found.\n",
				(int)kexp->exptype);
		return false;
	}
	kexp->expr_ops = dtype_hentry->cat.type_ops;

	/* some special cases */
	switch (kexp->opcode)
	{
		case FuncOpCode__CaseWhenExpr:
			if (kexp->u.casewhen.case_comp)
			{
				karg = (kern_expression *)
					((char *)kexp + kexp->u.casewhen.case_comp);
				if (!__KEXP_IS_VALID(kexp, karg))
				{
					fprintf(stderr, "xPU code corruption at case_comp\n");
					return false;
				}
				if (!__resolveDevicePointersWalker(karg))
					return false;
			}
			if (kexp->u.casewhen.case_else)
			{
				karg = (kern_expression *)
					((char *)kexp + kexp->u.casewhen.case_else);
				if (!__KEXP_IS_VALID(kexp, karg))
				{
					fprintf(stderr, "xPU code corruption at case_else\n");
					return false;
				}
				if (!__resolveDevicePointersWalker(karg))
					return false;
			}
			break;

		default:
			break;
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
		if (!__resolveDevicePointersWalker(karg))
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
	kern_varslot_desc *kvslot_desc = SESSION_KVARS_SLOT_DESC(session);
	xpu_encode_info *encode = SESSION_ENCODE(session);
	kern_expression *__kexp[20];
	int		nitems = 0;

	__kexp[nitems++] = SESSION_KEXP_LOAD_VARS(session, -1);
	__kexp[nitems++] = SESSION_KEXP_MOVE_VARS(session, -1);
    __kexp[nitems++] = SESSION_KEXP_SCAN_QUALS(session);
    __kexp[nitems++] = SESSION_KEXP_JOIN_QUALS(session, -1);
    __kexp[nitems++] = SESSION_KEXP_HASH_VALUE(session, -1);
    __kexp[nitems++] = SESSION_KEXP_GIST_EVALS(session, -1);
    __kexp[nitems++] = SESSION_KEXP_PROJECTION(session);
    __kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYHASH(session);
    __kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYLOAD(session);
    __kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYCOMP(session);
    __kexp[nitems++] = SESSION_KEXP_GROUPBY_ACTIONS(session);
	/* fixup device pointers */
	for (int i=0; i < nitems; i++)
	{
		if (__kexp[i] && !__resolveDevicePointersWalker(__kexp[i]))
			return false;
	}

	/* fixup kern_varslot_desc also */
	for (int i=0; i < session->kcxt_kvars_nslots; i++)
	{
		const xpu_type_hash_entry *entry;

		entry = __lookupDevTypeLinkageTable(kvslot_desc[i].vs_type_code);
		if (!entry)
		{
			fprintf(stderr, "device type for opcode:%u not found.\n",
					(int)kvslot_desc[i].vs_type_code);
			return false;
		}
		kvslot_desc[i].vs_ops = entry->cat.type_ops;
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
	if (!dpuServMapSessionBuffers(dclient, session))
	{
		dpuClientElog(dclient, "unable to map DPU-serv session buffer");
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
__dpuservLoadKdsCommon(dpuClient *dclient,
					   const kern_data_store *kds_head,
					   size_t preload_sz,
					   const char *pathname,
					   const strom_io_vector *kds_iovec,
					   char **p_base_addr)
{
	kern_data_store *kds;
	char	   *data;
	char	   *end		__attribute__((unused));
	int			fdesc;

	fdesc = open(pathname, O_RDONLY | O_DIRECT | O_NOATIME);
	if (fdesc < 0)
	{
		dpuClientElog(dclient, "failed on open('%s'): %m", pathname);
		return NULL;
	}

	data = malloc(kds_head->length + 2 * PAGE_SIZE);
	if (!data)
	{
		close(fdesc);
		dpuClientElog(dclient, "out of memory: %m");
		return NULL;
	}
	end = data + kds_head->length + 2 * PAGE_SIZE;

	/*
	 * due to the restriction of O_DIRECT, ((char *)kds + preload_sz) must
	 * be aligned to PAGE_SIZE.
	 */
	assert(kds_head->block_nloaded == 0);
	kds = (kern_data_store *)(PAGE_ALIGN(data + preload_sz) - preload_sz);
	memcpy(kds, kds_head, preload_sz);
	if (kds_iovec)
	{
		char   *base = (char *)kds + preload_sz;
		int		count = 0;

		assert(PAGE_ALIGN(base) == (uintptr_t)base);
		for (int i=0; i < kds_iovec->nr_chunks; i++)
		{
			const strom_io_chunk *ioc = &kds_iovec->ioc[i];
			char	   *dest   = base + ioc->m_offset;
			ssize_t		offset = PAGE_SIZE * (size_t)ioc->fchunk_id;
			ssize_t		length = PAGE_SIZE * (size_t)ioc->nr_pages;
			ssize_t		nbytes;

			assert(dest + length <= end);
			while (length > 0)
			{
				nbytes = pread(fdesc, dest, length, offset);
				if (nbytes > 0)
				{
					assert(nbytes <= length);
					dest   += nbytes;
					offset += nbytes;
					length -= nbytes;
					count++;
				}
				else if (nbytes == 0)
				{
					/*
					 * Due to PAGE_SIZE alignment, we may try to read the file
					 * over the tail.
					 */
					memset(dest, 0, length);
					break;
				}
				else if (errno != EINTR)
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

static kern_data_store *
dpuservLoadKdsBlock(dpuClient *dclient,
					const kern_data_store *kds_head,
					const char *pathname,
					const strom_io_vector *kds_iovec,
					char **p_base_addr)
{
	Assert(kds_head->format == KDS_FORMAT_BLOCK &&
		   kds_head->block_nloaded == 0);
	return __dpuservLoadKdsCommon(dclient,
								  kds_head,
								  kds_head->block_offset,
								  pathname,
								  kds_iovec,
								  p_base_addr);
}

static kern_data_store *
dpuservLoadKdsArrow(dpuClient *dclient,
					const kern_data_store *kds_head,
					const char *pathname,
					const strom_io_vector *kds_iovec,
					char **p_base_addr)
{
	Assert(kds_head->format == KDS_FORMAT_ARROW);
	return __dpuservLoadKdsCommon(dclient,
								  kds_head,
								  KDS_HEAD_LENGTH(kds_head),
								  pathname,
								  kds_iovec,
								  p_base_addr);
}

/* ----------------------------------------------------------------
 *
 * DPU Kernel Projection
 *
 * ----------------------------------------------------------------
 */
static bool
__handleDpuTaskExecProjection(dpuClient *dclient,
							  dpuTaskExecState *dtes,
							  kern_context *kcxt)
{
	kern_session_info  *session = dclient->session;
	kern_expression    *kexp_projection = SESSION_KEXP_PROJECTION(session);
	int32_t				tupsz;

	assert(kexp_projection != NULL &&
		   kexp_projection->opcode  == FuncOpCode__Projection);
	tupsz = kern_estimate_heaptuple(kcxt,
									kexp_projection,
									dtes->kds_dst_head);
	if (tupsz > 0)
	{
		kern_data_store *kds_dst = dtes->kds_dst;
		uint32_t	rowid;
		size_t		offset;
		size_t		newsz;
		kern_tupitem *tupitem;

	retry:
		/* allocate a new kds_dst on the demand */
		if (!kds_dst)
		{
			size_t	sz;

			if (dtes->kds_dst_nitems >= dtes->kds_dst_nrooms)
			{
				kern_data_store **kds_dst_array;
				uint32_t	kds_dst_nrooms = 2 * dtes->kds_dst_nrooms + 12;

				kds_dst_array = realloc(dtes->kds_dst_array,
										sizeof(kern_data_store *) * kds_dst_nrooms);
				if (!kds_dst_array)
				{
					dpuClientElog(dclient, "out of memory");
					return false;
				}
				dtes->kds_dst_array = kds_dst_array;
				dtes->kds_dst_nrooms = kds_dst_nrooms;
			}
			sz = KDS_HEAD_LENGTH(dtes->kds_dst_head) + PGSTROM_CHUNK_SIZE;
			kds_dst = malloc(sz);
			if (!kds_dst)
			{
				dpuClientElog(dclient, "out of memory");
				return false;
			}
			memcpy(kds_dst,
				   dtes->kds_dst_head,
				   KDS_HEAD_LENGTH(dtes->kds_dst_head));
			kds_dst->length = sz;
			dtes->kds_dst_array[dtes->kds_dst_nitems++] = kds_dst;
			dtes->kds_dst = kds_dst;
		}
		/* insert a tuple */
		offset = __kds_unpack(kds_dst->usage) + tupsz;
		newsz = (KDS_HEAD_LENGTH(kds_dst) +
				 MAXALIGN(sizeof(uint32_t) * (kds_dst->nitems + 1)) +
				 offset);
		if (newsz > kds_dst->length)
		{
			dtes->kds_dst = kds_dst = NULL;
			goto retry;
		}
		kds_dst->usage = __kds_packed(offset);
		rowid = kds_dst->nitems++;
		KDS_GET_ROWINDEX(kds_dst)[rowid] = kds_dst->usage;
		tupitem = (kern_tupitem *)
			((char *)kds_dst + kds_dst->length - offset);
		tupitem->rowid = rowid;
		tupitem->t_len = kern_form_heaptuple(kcxt,
											 kexp_projection,
											 kds_dst,
											 &tupitem->htup);
		dtes->nitems_out++;
		return true;
	}
	return false;
}

/* ----------------------------------------------------------------
 *
 * DPU Kernel PreAgg
 *
 * ----------------------------------------------------------------
 */
static int32_t
__writeOutOneTuplePreAgg(kern_context *kcxt,
						 kern_data_store *kds_final,
						 HeapTupleHeaderData *htup,
						 kern_expression *kexp_actions)
{
	int			nattrs = Min(kds_final->ncols, kexp_actions->u.pagg.nattrs);
	uint32_t	t_hoff, t_next;
	uint16_t	t_infomask = HEAP_HASNULL;
	char	   *buffer = NULL;

	t_hoff = MAXALIGN(offsetof(HeapTupleHeaderData,
							   t_bits) + BITMAPLEN(nattrs));
	if (htup)
	{
		memset(htup, 0, t_hoff);
		htup->t_choice.t_datum.datum_typmod = kds_final->tdtypmod;
		htup->t_choice.t_datum.datum_typeid = kds_final->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
		htup->t_infomask2 = (nattrs & HEAP_NATTS_MASK);
		htup->t_hoff = t_hoff;
	}
	/* walk on the columns */
	for (int j=0; j < nattrs; j++)
	{
		kern_aggregate_desc *desc = &kexp_actions->u.pagg.desc[j];
		kern_colmeta   *cmeta = &kds_final->colmeta[j];
		xpu_datum_t	   *xdatum;
		int				nbytes;

		assert((char *)cmeta > (char *)kds_final &&
			   (char *)cmeta < (char *)kds_final + kds_final->length);
		assert(cmeta->attalign > 0 && cmeta->attalign <= 8);
		t_next = TYPEALIGN(cmeta->attalign, t_hoff);
		if (htup)
		{
			if (t_next > t_hoff)
				memset((char *)htup + t_hoff, 0, t_next - t_hoff);
			buffer = (char *)htup + t_next;
		}

		switch (desc->action)
		{
			case KAGG_ACTION__VREF:
				assert(desc->arg0_slot_id >= 0 &&
					   desc->arg0_slot_id < kcxt->kvars_nslots);
				xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
				if (XPU_DATUM_ISNULL(xdatum))
					nbytes = 0;
				else
				{
					nbytes = xdatum->expr_ops->xpu_datum_write(kcxt,
															   buffer,
															   cmeta,
															   xdatum);
					if (nbytes < 0)
						return -1;
				}
				break;

			case KAGG_ACTION__NROWS_ANY:
			case KAGG_ACTION__NROWS_COND:
				assert(cmeta->attlen == sizeof(int64_t));
				nbytes = sizeof(int64_t);
				if (buffer)
					*((int64_t *)buffer) = 0;
				break;

			case KAGG_ACTION__PMIN_INT32:
			case KAGG_ACTION__PMIN_INT64:
				nbytes = sizeof(kagg_state__pminmax_int64_packed);
				if (buffer)
				{
					kagg_state__pminmax_int64_packed *r =
						(kagg_state__pminmax_int64_packed *)buffer;
					r->nitems = 0;
					r->value = LONG_MAX;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_int64_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PMAX_INT32:
			case KAGG_ACTION__PMAX_INT64:
				nbytes = sizeof(kagg_state__pminmax_int64_packed);
				if (buffer)
				{
					kagg_state__pminmax_int64_packed *r =
						(kagg_state__pminmax_int64_packed *)buffer;
					r->nitems = 0;
					r->value = LONG_MIN;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_int64_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PMIN_FP64:
				nbytes = sizeof(kagg_state__pminmax_fp64_packed);
				if (buffer)
				{
					kagg_state__pminmax_fp64_packed *r =
						(kagg_state__pminmax_fp64_packed *)buffer;
					r->nitems = 0;
					r->value = DBL_MAX;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_fp64_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PMAX_FP64:
				nbytes = sizeof(kagg_state__pminmax_fp64_packed);
				if (buffer)
				{
					kagg_state__pminmax_fp64_packed *r =
						(kagg_state__pminmax_fp64_packed *)buffer;
					r->nitems = 0;
					r->value = -DBL_MAX;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_fp64_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PSUM_INT:
			case KAGG_ACTION__PAVG_INT:
				nbytes = sizeof(kagg_state__psum_int_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__psum_int_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__psum_int_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PSUM_FP:
			case KAGG_ACTION__PAVG_FP:
				nbytes = sizeof(kagg_state__psum_fp_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__psum_fp_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__psum_fp_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__STDDEV:
				nbytes = sizeof(kagg_state__stddev_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__stddev_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__stddev_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__COVAR:
				nbytes = sizeof(kagg_state__covar_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__covar_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__covar_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			default:
				fprintf(stderr, "Bug? unknown DpuPreAgg action: %d",
						(int)desc->action);
				return -1;
		}
		if (htup && nbytes > 0)
			htup->t_bits[j>>3] |= (1<<(j&7));
		t_hoff = t_next + nbytes;
	}
	if (htup)
		htup->t_infomask = t_infomask;
	return t_hoff;
}

/*
 * __insertOneTupleNoGroups
 */
static kern_tupitem *
__insertOneTupleNoGroups(kern_context *kcxt,
						 kern_data_store *kds_final,
						 kern_expression *kexp_groupby_actions)
{
	kern_tupitem   *tupitem;
	int32_t			tupsz;
	uint32_t		required;
	uint32_t		usage;
	size_t			total_sz;

	assert(kds_final->format == KDS_FORMAT_ROW &&
		   kds_final->nitems == 0 &&
		   kds_final->hash_nslots == 0);
	/* estimate length */
	tupsz = __writeOutOneTuplePreAgg(kcxt, kds_final, NULL,
									 kexp_groupby_actions);
	assert(tupsz > 0);
	required = MAXALIGN(offsetof(kern_tupitem, htup) + tupsz);
	total_sz = (KDS_HEAD_LENGTH(kds_final) +
				MAXALIGN(sizeof(uint32_t)) +
				required + __kds_unpack(kds_final->usage));
	if (total_sz > kds_final->length)
		return NULL;	/* out of memory */
	usage = __atomic_add_uint32(&kds_final->usage, __kds_packed(required));
	tupitem = (kern_tupitem *)((char *)kds_final
							   + kds_final->length
							   - __kds_unpack(usage)
							   - required);
	__writeOutOneTuplePreAgg(kcxt, kds_final,
							 &tupitem->htup,
							 kexp_groupby_actions);
	tupitem->t_len = tupsz;
	tupitem->rowid = 0;
	KDS_GET_ROWINDEX(kds_final)[kds_final->nitems++]
		= __kds_packed((char *)kds_final
					   + kds_final->length
					   - (char *)tupitem);
	return tupitem;
}

/*
 * __insertOneTupleGroupBy
 */
static kern_hashitem *
__insertOneTupleGroupBy(kern_context *kcxt,
						kern_data_store *kds_final,
						kern_expression *kexp_groupby_actions)
{
	kern_hashitem  *hitem;
	int32_t			tupsz;
	uint32_t		required;
	union {
		uint64_t	u64;
		struct {
			uint32_t nitems;
			uint32_t usage;
		} kds;
	} oldval, curval, newval;

	assert(kds_final->format == KDS_FORMAT_HASH &&
		   kds_final->hash_nslots > 0);
	/* estimate length */
	tupsz = __writeOutOneTuplePreAgg(kcxt, kds_final, NULL,
									 kexp_groupby_actions);
	assert(tupsz > 0);
	required = MAXALIGN(offsetof(kern_hashitem, t.htup) + tupsz);

	/* expand kds_final */
	curval.kds.nitems = __volatileRead(&kds_final->nitems);
	curval.kds.usage  = __volatileRead(&kds_final->usage);
	for (;;)
	{
		size_t		total_sz;

		newval.kds.nitems = curval.kds.nitems + 1;
		newval.kds.usage  = curval.kds.usage  + __kds_packed(required);
		total_sz = (KDS_HEAD_LENGTH(kds_final) +
					MAXALIGN(sizeof(uint32_t) * (kds_final->hash_nslots +
												 newval.kds.nitems)) +
					__kds_unpack(curval.kds.usage));
		if (total_sz > kds_final->length)
			return NULL;	/* out of memory */
		oldval.u64 = __atomic_cas_uint64((uint64_t *)&kds_final->nitems,
										 curval.u64,
										 newval.u64);
		if (oldval.u64 == curval.u64)
			break;
		curval.u64 = oldval.u64;
	}
	hitem = (kern_hashitem *)((char *)kds_final
							  + kds_final->length
							  - __kds_unpack(newval.kds.usage));
	__writeOutOneTuplePreAgg(kcxt, kds_final,
							 &hitem->t.htup,
							 kexp_groupby_actions);
	hitem->t.t_len = tupsz;
	hitem->t.rowid = newval.kds.nitems - 1;
	KDS_GET_ROWINDEX(kds_final)[hitem->t.rowid]
		= __kds_packed((char *)kds_final
					   + kds_final->length
					   - (char *)&hitem->t);
	return hitem;
}

/*
 * __update_preagg__nrows_any
 */
static inline void
__update_preagg__nrows_any(kern_context *kcxt,
						   char *buffer,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc)
{
	__atomic_add_uint64((uint64_t *)buffer, 1);
}

/*
 * __update_preagg__nrows_cond
 */
static inline void
__update_preagg__nrows_cond(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc)
{
	xpu_datum_t	   *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
		__atomic_add_uint64((uint64_t *)buffer, 1);
}

/*
 * __update_preagg__pmin_intXX
 */
static inline void
__update_preagg__pmin_int32(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc)
{
	xpu_int4_t	   *xdatum = (xpu_int4_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_int4_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_min_int64(&r->value, xdatum->value);
	}
}

static inline void
__update_preagg__pmin_int64(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc)
{
	xpu_int8_t	   *xdatum = (xpu_int8_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_int8_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_min_int64(&r->value, xdatum->value);
	}
}

/*
 * __update_preagg__pmax_intXX
 */
static inline void
__update_preagg__pmax_int32(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc)
{
	xpu_int4_t	   *xdatum = (xpu_int4_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_int4_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_max_int64(&r->value, xdatum->value);
	}
}

static inline void
__update_preagg__pmax_int64(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc)
{
	xpu_int8_t	   *xdatum = (xpu_int8_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_int8_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_max_int64(&r->value, xdatum->value);
	}
}

/*
 * __update_preagg__pmin_fp64
 */
static inline void
__update_preagg__pmin_fp64(kern_context *kcxt,
						   char *buffer,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc)
{
	xpu_float8_t   *xdatum = (xpu_float8_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__pminmax_fp64_packed *r =
			(kagg_state__pminmax_fp64_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_float8_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_min_fp64(&r->value, xdatum->value);
	}
}

/*
 * __update_preagg__pmax_fp
 */
static inline void
__update_preagg__pmax_fp64(kern_context *kcxt,
						   char *buffer,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc)
{
	xpu_float8_t   *xdatum = (xpu_float8_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__pminmax_fp64_packed *r =
			(kagg_state__pminmax_fp64_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_float8_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_min_fp64(&r->value, xdatum->value);
	}
}

/*
 * __update_preagg__psum_int
 */
static inline void
__update_preagg__psum_int(kern_context *kcxt,
						  char *buffer,
						  kern_colmeta *cmeta,
						  kern_aggregate_desc *desc)
{
	xpu_int8_t	   *xdatum = (xpu_int8_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__psum_int_packed *r =
			(kagg_state__psum_int_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_int8_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_add_int64(&r->sum, xdatum->value);
	}
}

/*
 * __update_preagg__psum_fp
 */
static inline void
__update_preagg__psum_fp(kern_context *kcxt,
						 char *buffer,
						 kern_colmeta *cmeta,
						 kern_aggregate_desc *desc)
{
	xpu_float8_t   *xdatum = (xpu_float8_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__psum_fp_packed *r =
			(kagg_state__psum_fp_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_float8_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_add_fp64(&r->sum, xdatum->value);
	}
}

/*
 * __update_preagg__pstddev
 */
static inline void
__update_preagg__pstddev(kern_context *kcxt,
						 char *buffer,
						 kern_colmeta *cmeta,
						 kern_aggregate_desc *desc)
{
	xpu_float8_t   *xdatum = (xpu_float8_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
	{
		kagg_state__stddev_packed *r =
			(kagg_state__stddev_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_float8_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_add_fp64(&r->sum_x,  xdatum->value);
		__atomic_add_fp64(&r->sum_x2, xdatum->value * xdatum->value);
	}
}

/*
 * __update_preagg__pcovar
 */
static inline void
__update_preagg__pcovar(kern_context *kcxt,
						char *buffer,
						kern_colmeta *cmeta,
						kern_aggregate_desc *desc)
{
	xpu_float8_t   *xdatum = (xpu_float8_t *)
		kcxt->kvars_slot[desc->arg0_slot_id];
	xpu_float8_t   *ydatum = (xpu_float8_t *)
		kcxt->kvars_slot[desc->arg1_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum) && !XPU_DATUM_ISNULL(ydatum))
	{
		kagg_state__covar_packed *r =
			(kagg_state__covar_packed *)buffer;
		assert(xdatum->expr_ops == &xpu_float8_ops &&
			   ydatum->expr_ops == &xpu_float8_ops);
		__atomic_add_uint32(&r->nitems, 1);
		__atomic_add_fp64(&r->sum_x,  xdatum->value);
		__atomic_add_fp64(&r->sum_xx, xdatum->value * xdatum->value);
		__atomic_add_fp64(&r->sum_y,  ydatum->value);
		__atomic_add_fp64(&r->sum_yy, ydatum->value * ydatum->value);
		__atomic_add_fp64(&r->sum_xy, xdatum->value * ydatum->value);
	}
}

/*
 * __updateOneTupleDpuPreAgg (for both of NoGroups and GroupBy)
 */
static void
__updateOneTupleDpuPreAgg(kern_context *kcxt,
						  kern_data_store *kds_final,
						  HeapTupleHeaderData *htup,
						  kern_expression *kexp_groupby_actions)
{
	int			nattrs = (htup->t_infomask2 & HEAP_NATTS_MASK);
	bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	uint32_t	t_hoff;
	char	   *buffer = NULL;

	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (heap_hasnull)
		t_hoff += BITMAPLEN(nattrs);
	t_hoff = MAXALIGN(t_hoff);

	for (int j=0; j < nattrs; j++)
	{
		kern_aggregate_desc *desc = &kexp_groupby_actions->u.pagg.desc[j];
		kern_colmeta   *cmeta = &kds_final->colmeta[j];

		if (heap_hasnull && att_isnull(j, htup->t_bits))
		{
			/* only grouping-key may have NULL */
			assert(desc->action == KAGG_ACTION__VREF);
			continue;
		}

		if (cmeta->attlen > 0)
			t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
		else if (!VARATT_NOT_PAD_BYTE((char *)htup + t_hoff))
			t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
		buffer = ((char *)htup + t_hoff);
		if (cmeta->attlen > 0)
			t_hoff += cmeta->attlen;
		else
			t_hoff += VARSIZE_ANY(buffer);

		switch (desc->action)
        {
			case KAGG_ACTION__NROWS_ANY:
				__update_preagg__nrows_any(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__NROWS_COND:
				__update_preagg__nrows_cond(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMIN_INT32:
				__update_preagg__pmin_int32(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMIN_INT64:
				__update_preagg__pmin_int64(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMAX_INT32:
				__update_preagg__pmax_int32(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMAX_INT64:
				__update_preagg__pmax_int64(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMIN_FP64:
				__update_preagg__pmin_fp64(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMAX_FP64:
				__update_preagg__pmax_fp64(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PSUM_INT:
			case KAGG_ACTION__PAVG_INT:
				__update_preagg__psum_int(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PSUM_FP:
			case KAGG_ACTION__PAVG_FP:
				__update_preagg__psum_fp(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__STDDEV:
				__update_preagg__pstddev(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__COVAR:
				__update_preagg__pcovar(kcxt, buffer, cmeta, desc);
				break;
			default:
				/*
				 * No more partial aggregation exists after grouping-keys
				 */
				return;
		}
	}
}

/*
 * expandGroupByFinalBuffer
 *
 * NOTE: this function must be called under kds_final_rwlock WRITE-LOCK
 */
static bool
expandGroupByFinalBuffer(groupby_final_buffer *gf_buf)
{
	kern_data_store *kds_old = gf_buf->kds_final;
	kern_data_store *kds_new;
	size_t		sz, length;

	length = kds_old->length + Min(kds_old->length, 1UL<<30);
	kds_new = malloc(length);
	if (!kds_new)
		return false;
	/* early half */
	sz = (KDS_HEAD_LENGTH(kds_old) +
		  MAXALIGN(sizeof(uint32_t) * (kds_old->nitems +
									   kds_old->hash_nslots)));
	memcpy(kds_new, kds_old, sz);
	kds_new->length = length;

	/* later falf */
	sz = __kds_unpack(kds_old->usage);
	memcpy((char *)kds_new + kds_new->length - sz,
		   (char *)kds_old + kds_old->length - sz, sz);

	/* swap them */
	gf_buf->kds_final = kds_new;
	free(kds_old);

	return true;
}

/*
 * __handleDpuTaskExecNoGroupPreAgg
 */
static bool
__handleDpuTaskExecNoGroupPreAgg(dpuClient *dclient,
								 dpuTaskExecState *dtes,
								 kern_context *kcxt)
{
	groupby_final_buffer *gf_buf = dclient->gf_buf;
	kern_session_info  *session = dclient->session;
	kern_expression	   *kexp_groupby_actions = SESSION_KEXP_GROUPBY_ACTIONS(session);
	kern_expression	   *karg;
	kern_data_store	   *kds_final;
	kern_tupitem	   *tupitem = NULL;
	bool				has_exclusive = false;
	int					i;

	assert(kexp_groupby_actions->opcode == FuncOpCode__AggFuncs);
	/* fillup kvars_slot if it involves expression */
	for (i=0, karg = KEXP_FIRST_ARG(kexp_groupby_actions);
		 i < kexp_groupby_actions->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		assert(karg->opcode == FuncOpCode__SaveExpr);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, NULL))
			return false;
	}

	pthreadRWLockReadLock(&gf_buf->kds_final_rwlock);
	while (!tupitem)
	{
		kds_final = gf_buf->kds_final;

		assert(kds_final->format == KDS_FORMAT_ROW);
		if (kds_final->nitems == 1)
		{
			/* almost case; destination tuple already exists */
			tupitem = KDS_GET_TUPITEM(kds_final, 0);
			assert(tupitem != NULL);
		}
		else if (!has_exclusive)
		{
			pthreadRWLockUnlock(&gf_buf->kds_final_rwlock);
			pthreadRWLockWriteLock(&gf_buf->kds_final_rwlock);
			has_exclusive = true;
		}
		else
		{
			assert(kds_final->nitems == 0);
			tupitem = __insertOneTupleNoGroups(kcxt, kds_final,
											   kexp_groupby_actions);
			if (!tupitem &&
				!expandGroupByFinalBuffer(gf_buf))
			{
				/* out of memory */
				pthreadRWLockUnlock(&gf_buf->kds_final_rwlock);
				return false;
			}
		}
	}
	/* update the partial aggregation */
	__updateOneTupleDpuPreAgg(kcxt, kds_final,
							  &tupitem->htup,
							  kexp_groupby_actions);
	pthreadRWLockUnlock(&gf_buf->kds_final_rwlock);
	return true;
}

/*
 * __handleDpuTaskExecGroupByPreAgg
 */
static bool
__handleDpuTaskExecGroupByPreAgg(dpuClient *dclient,
								 dpuTaskExecState *dtes,
								 kern_context *kcxt)
{
	groupby_final_buffer *gf_buf = dclient->gf_buf;
	kern_session_info  *session = dclient->session;
	kern_expression	   *kexp_groupby_keyhash = SESSION_KEXP_GROUPBY_KEYHASH(session);
	kern_expression	   *kexp_groupby_keyload = SESSION_KEXP_GROUPBY_KEYLOAD(session);
	kern_expression	   *kexp_groupby_keycomp = SESSION_KEXP_GROUPBY_KEYCOMP(session);
	kern_expression	   *kexp_groupby_actions = SESSION_KEXP_GROUPBY_ACTIONS(session);
	kern_expression	   *karg;
	kern_data_store	   *kds_final;
	xpu_int4_t			hash;
	kern_hashitem	   *hitem;
	bool				has_exclusive = false;
	int					i;

	assert(kexp_groupby_keyhash != NULL &&
		   kexp_groupby_keyload != NULL &&
		   kexp_groupby_keycomp != NULL &&
		   kexp_groupby_actions != NULL);
	/* fillup kvars_slot if it involves expression */
	for (i=0, karg = KEXP_FIRST_ARG(kexp_groupby_actions);
		 i < kexp_groupby_actions->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		assert(karg->opcode == FuncOpCode__SaveExpr);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, NULL))
			return false;
	}
	/*
	 * calculation of GROUP BY key hash-value
	 */
	if (!EXEC_KERN_EXPRESSION(kcxt, kexp_groupby_keyhash, &hash))
		return false;
	if (XPU_DATUM_ISNULL(&hash))
		return false;

	pthreadRWLockReadLock(&gf_buf->kds_final_rwlock);
	kds_final = gf_buf->kds_final;
	assert(kds_final->format == KDS_FORMAT_HASH);
	do {
		uint32_t   *hslot = KDS_GET_HASHSLOT(kds_final, hash.value);
		uint32_t	hoffset = __volatileRead(hslot);
		uint32_t	saved = hoffset;
		xpu_bool_t	status;

		if (hoffset == UINT_MAX)
		{
			/* someone already hold the hslot-lock */
			sched_yield();
			continue;
		}

		for (hitem = KDS_HASH_NEXT_ITEM(kds_final, hoffset);
			 hitem != NULL;
			 hitem = KDS_HASH_NEXT_ITEM(kds_final, hitem->next))
		{
			bool	saved_compare_nulls = kcxt->kmode_compare_nulls;

			if (hitem->hash != hash.value)
				continue;

			kcxt->kmode_compare_nulls = true;
			ExecLoadVarsHeapTuple(kcxt,
								  kexp_groupby_keyload,
								  -2,
								  kds_final,
								  &hitem->t.htup);
			if (EXEC_KERN_EXPRESSION(kcxt, kexp_groupby_keycomp, &status))
			{
				kcxt->kmode_compare_nulls = saved_compare_nulls;
				if (!XPU_DATUM_ISNULL(&status) && status.value)
					break;
			}
			kcxt->kmode_compare_nulls = saved_compare_nulls;
		}

		if (!hitem && __atomic_cas_uint32(hslot, saved, UINT_MAX) == saved)
		{
			/* hslot-lock is now acquired */
			hitem = __insertOneTupleGroupBy(kcxt, kds_final,
											kexp_groupby_actions);
			if (hitem)
			{
				hitem->hash = hash.value;
				hitem->next = saved;
				hoffset = ((char *)kds_final
						   + kds_final->length
						   - (char *)hitem);
				/* insert and unlock */
				__atomic_write_uint32(hslot, __kds_packed(hoffset));
				break;
			}
			else
			{
				/* unlock; by out of the memory */
				__atomic_write_uint32(hslot, saved);
				if (!has_exclusive)
				{
					pthreadRWLockUnlock(&gf_buf->kds_final_rwlock);
					pthreadRWLockWriteLock(&gf_buf->kds_final_rwlock);
					has_exclusive = true;
				}
				else
				{
					/* expand the kds_final buffer */
					if (!expandGroupByFinalBuffer(gf_buf))
					{
						pthreadRWLockUnlock(&gf_buf->kds_final_rwlock);
						return false;
					}
				}
			}
		}
	} while (!hitem);
	/* update the partial aggregation */
	__updateOneTupleDpuPreAgg(kcxt, kds_final,
							  &hitem->t.htup,
							  kexp_groupby_actions);
	pthreadRWLockUnlock(&gf_buf->kds_final_rwlock);

	return true;
}

/* ----------------------------------------------------------------
 *
 * DPU service SCAN/JOIN handler
 *
 * ----------------------------------------------------------------
 */
static bool
__handleDpuTaskExecHashJoin(dpuClient *dclient,
							dpuTaskExecState *dtes,
							kern_context *kcxt,
							int depth);

static bool
__handleDpuTaskExecNestLoop(dpuClient *dclient,
							dpuTaskExecState *dtes,
							kern_context *kcxt,
							int depth)
{
	kern_session_info  *session = dclient->session;
	kern_multirels	   *kmrels = dclient->kmrels;
	kern_data_store	   *kds_heap = KERN_MULTIRELS_INNER_KDS(kmrels,depth-1);
	bool			   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels,depth-1);
	kern_expression	   *kexp_load_vars = SESSION_KEXP_LOAD_VARS(session,depth);
	kern_expression	   *kexp_join_quals = SESSION_KEXP_JOIN_QUALS(session,depth);
	bool				matched = false;

	for (uint32_t rowid=0; rowid < kds_heap->nitems; rowid++)
	{
		kern_tupitem   *tupitem;
		uint32_t		offset;
		xpu_int4_t		status;

		offset = KDS_GET_ROWINDEX(kds_heap)[rowid];
		tupitem = (kern_tupitem *)((char *)kds_heap +
								   kds_heap->length -
								   __kds_unpack(offset));
		kcxt_reset(kcxt);
		ExecLoadVarsHeapTuple(kcxt, kexp_load_vars, depth,
							  kds_heap, &tupitem->htup);
		if (!EXEC_KERN_EXPRESSION(kcxt, kexp_join_quals, &status))
			return false;
		assert(!XPU_DATUM_ISNULL(&status));
		if (status.value > 0)
		{
			if (depth >= kmrels->num_rels)
			{
				if (!dtes->handleDpuTaskFinalDepth(dclient, dtes, kcxt))
					return false;
			}
			else if (kmrels->chunks[depth].is_nestloop)
			{
				if (!__handleDpuTaskExecNestLoop(dclient, dtes, kcxt, depth+1))
					return false;
			}
			else
			{
				if (!__handleDpuTaskExecHashJoin(dclient, dtes, kcxt, depth+1))
					return false;
			}
		}
		if (status.value != 0)
		{
			matched = true;
			if (oj_map)
				oj_map[rowid] = true;
		}
	}
	/* LEFT OUTER if needed */
	if (kmrels->chunks[depth-1].left_outer && !matched)
	{
		ExecLoadVarsHeapTuple(kcxt, kexp_load_vars, depth,
							  kds_heap, NULL);
		if (depth >= kmrels->num_rels)
		{
			if (!dtes->handleDpuTaskFinalDepth(dclient, dtes, kcxt))
				return false;
		}
		else if (kmrels->chunks[depth].is_nestloop)
		{
			if (!__handleDpuTaskExecNestLoop(dclient, dtes, kcxt, depth+1))
				return false;
		}
		else
		{
			if (!__handleDpuTaskExecHashJoin(dclient, dtes, kcxt, depth+1))
				return false;
		}
	}
	return true;
}

static bool
__handleDpuTaskExecHashJoin(dpuClient *dclient,
							dpuTaskExecState *dtes,
							kern_context *kcxt,
							int depth)
{
	kern_session_info  *session = dclient->session;
	kern_multirels	   *kmrels = dclient->kmrels;
	kern_data_store	   *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth-1);
	bool			   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth-1);
	kern_expression	   *kexp_load_vars = SESSION_KEXP_LOAD_VARS(session,depth);
	kern_expression	   *kexp_join_quals = SESSION_KEXP_JOIN_QUALS(session,depth);
	kern_expression	   *kexp_hash_value = SESSION_KEXP_HASH_VALUE(session,depth);
	kern_hashitem	   *khitem;
	xpu_int4_t			hash;
	xpu_int4_t			status;
	bool				matched = false;

	if (!EXEC_KERN_EXPRESSION(kcxt, kexp_hash_value, &hash))
		return false;
	assert(!XPU_DATUM_ISNULL(&hash));
	for (khitem = KDS_HASH_FIRST_ITEM(kds_hash, hash.value);
		 khitem != NULL;
		 khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem->next))
	{
		if (khitem->hash != hash.value)
			continue;
		ExecLoadVarsHeapTuple(kcxt, kexp_load_vars, depth,
							  kds_hash, &khitem->t.htup);
		kcxt_reset(kcxt);
		if (!EXEC_KERN_EXPRESSION(kcxt, kexp_join_quals, &status))
			return false;
		assert(!XPU_DATUM_ISNULL(&status));
		if (status.value > 0)
		{
			if (depth >= kmrels->num_rels)
			{
				if (!dtes->handleDpuTaskFinalDepth(dclient, dtes, kcxt))
					return false;
			}
			else if (kmrels->chunks[depth].is_nestloop)
			{
				if (!__handleDpuTaskExecNestLoop(dclient, dtes, kcxt, depth+1))
					return false;
			}
			else
			{
				if (!__handleDpuTaskExecHashJoin(dclient, dtes, kcxt, depth+1))
					return false;
			}
		}
		if (status.value != 0)
		{
			matched = true;
			if (oj_map)
				oj_map[khitem->t.rowid] = true;
		}
	}
	/* LEFT OUTER if needed */
	if (kmrels->chunks[depth-1].left_outer && !matched)
	{
		ExecLoadVarsHeapTuple(kcxt, kexp_load_vars, depth,
							  kds_hash, NULL);
		if (depth >= kmrels->num_rels)
		{
			if (!dtes->handleDpuTaskFinalDepth(dclient, dtes, kcxt))
				return false;
		}
		else if (kmrels->chunks[depth].is_nestloop)
		{
			if (!__handleDpuTaskExecNestLoop(dclient, dtes, kcxt, depth+1))
				return false;
		}
		else
		{
			if (!__handleDpuTaskExecHashJoin(dclient, dtes, kcxt, depth+1))
				return false;
		}
	}
	return true;
}

static bool
__handleDpuScanExecBlock(dpuClient *dclient,
						 dpuTaskExecState *dtes,
						 kern_data_store *kds_src)
{
	kern_session_info  *session = dclient->session;
	kern_multirels	   *kmrels = dclient->kmrels;
	kern_expression	   *kexp_load_vars = SESSION_KEXP_LOAD_VARS(session, 0);
	kern_expression	   *kexp_scan_quals = SESSION_KEXP_SCAN_QUALS(session);
	kern_context	   *kcxt;
	uint32_t			block_index;

	assert(kds_src->format == KDS_FORMAT_BLOCK &&
		   kexp_load_vars->opcode == FuncOpCode__LoadVars);
	assert(!kmrels || kmrels->num_rels > 0);
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

			if (!ItemIdIsNormal(lpp))
				continue;
			htup = (HeapTupleHeaderData *) PageGetItem(page, lpp);
			dtes->nitems_raw++;
			kcxt_reset(kcxt);
			if (ExecLoadVarsOuterRow(kcxt,
									 kexp_load_vars,
									 kexp_scan_quals,
									 kds_src,
									 htup))
			{
				dtes->nitems_in++;
				if (!kmrels)
				{
					if (!dtes->handleDpuTaskFinalDepth(dclient, dtes, kcxt))
						return false;
				}
				else if (kmrels->chunks[0].is_nestloop)
				{
					/* NEST-LOOP */
					if (!__handleDpuTaskExecNestLoop(dclient, dtes, kcxt, 1))
						return false;
				}
				else
				{
					/* HASH-JOIN */
					if (!__handleDpuTaskExecHashJoin(dclient, dtes, kcxt, 1))
						return false;
				}
			}
			else if (kcxt->errcode != ERRCODE_STROM_SUCCESS)
			{
				__dpuClientElog(dclient,
								kcxt->errcode,
								kcxt->error_filename,
								kcxt->error_lineno,
								kcxt->error_funcname,
								kcxt->error_message);
				return false;
			}
		}
	}
	return true;
}

static bool
__handleDpuScanExecArrow(dpuClient *dclient,
						 dpuTaskExecState *dtes,
						 kern_data_store *kds_src)
{
	kern_session_info  *session = dclient->session;
	kern_multirels	   *kmrels = dclient->kmrels;
	kern_expression	   *kexp_load_vars = SESSION_KEXP_LOAD_VARS(session, 0);
	kern_expression	   *kexp_scan_quals = SESSION_KEXP_SCAN_QUALS(session);
	kern_context	   *kcxt;
	uint32_t			kds_index;

	assert(kds_src->format == KDS_FORMAT_ARROW &&
		   kexp_load_vars->opcode == FuncOpCode__LoadVars &&
		   kexp_scan_quals->exptype == TypeOpCode__bool);
	INIT_KERNEL_CONTEXT(kcxt, session);
	for (kds_index = 0; kds_index < kds_src->nitems; kds_index++)
	{
		kcxt_reset(kcxt);
		if (ExecLoadVarsOuterArrow(kcxt,
								   kexp_load_vars,
								   kexp_scan_quals,
								   kds_src,
								   kds_index))
		{
			dtes->nitems_in++;
			if (!kmrels)
			{
				if (!dtes->handleDpuTaskFinalDepth(dclient, dtes, kcxt))
					return false;
			}
			else if (kmrels->chunks[0].is_nestloop)
			{
				/* NEST-LOOP */
				if (!__handleDpuTaskExecNestLoop(dclient, dtes, kcxt, 1))
					return false;
			}
			else
			{
				/* HASH-JOIN */
				if (!__handleDpuTaskExecHashJoin(dclient, dtes, kcxt, 1))
					return false;
			}
		}
		else if (kcxt->errcode != ERRCODE_STROM_SUCCESS)
		{
			__dpuClientElog(dclient,
							kcxt->errcode,
							kcxt->error_filename,
							kcxt->error_lineno,
							kcxt->error_funcname,
							kcxt->error_message);
			return false;
		}
	}
	dtes->nitems_raw += kds_src->nitems;
	return true;
}

/*
 * dpuservHandleDpuTaskExec
 */
static void
dpuservHandleDpuTaskExec(dpuClient *dclient, XpuCommand *xcmd)
{
	kern_session_info  *session = dclient->session;
	dpuTaskExecState   *dtes;
	const char		   *kds_src_pathname = NULL;
	strom_io_vector	   *kds_src_iovec = NULL;
	kern_data_store	   *kds_src_head = NULL;
	kern_data_store	   *kds_dst_head = NULL;
	kern_data_store	   *kds_src = NULL;
	int					sz, num_rels = 0;

	if (xcmd->u.task.kds_src_pathname)
		kds_src_pathname = (char *)xcmd + xcmd->u.task.kds_src_pathname;
	if (xcmd->u.task.kds_src_iovec)
		kds_src_iovec = (strom_io_vector *)((char *)xcmd + xcmd->u.task.kds_src_iovec);
	if (xcmd->u.task.kds_src_offset)
		kds_src_head = (kern_data_store *)((char *)xcmd + xcmd->u.task.kds_src_offset);
	if (xcmd->u.task.kds_dst_offset)
		kds_dst_head = (kern_data_store *)((char *)xcmd + xcmd->u.task.kds_dst_offset);
	if (!kds_src_pathname || !kds_src_iovec || !kds_src_head || !kds_dst_head)
	{
		dpuClientElog(dclient, "kern_data_store is corrupted");
		return;
	}
	/* setup dpuTaskExecState */
	if (dclient->kmrels)
		num_rels = dclient->kmrels->num_rels;
	sz = offsetof(dpuTaskExecState, stats[num_rels]);
	dtes = alloca(sz);
	memset(dtes, 0, sz);
	dtes->kds_dst_head = kds_dst_head;
	dtes->num_rels = num_rels;
	if (session->xpucode_groupby_actions == 0)
	{
		assert(session->xpucode_projection != 0);
		dtes->handleDpuTaskFinalDepth = __handleDpuTaskExecProjection;
	}
	else if (session->xpucode_groupby_keyhash != 0 &&
			 session->xpucode_groupby_keyload != 0 &&
			 session->xpucode_groupby_keycomp != 0)
	{
		dtes->handleDpuTaskFinalDepth = __handleDpuTaskExecGroupByPreAgg;
	}
	else
	{
		dtes->handleDpuTaskFinalDepth = __handleDpuTaskExecNoGroupPreAgg;
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
			if (__handleDpuScanExecBlock(dclient, dtes, kds_src))
				dpuClientWriteBack(dclient, dtes);
			free(base_addr);
		}
	}
	else if (kds_src_head->format == KDS_FORMAT_ARROW)
	{
		char   *base_addr;

		kds_src = dpuservLoadKdsArrow(dclient,
									  kds_src_head,
									  kds_src_pathname,
									  kds_src_iovec,
									  &base_addr);
		if (kds_src)
		{
			if (__handleDpuScanExecArrow(dclient, dtes, kds_src))
				dpuClientWriteBack(dclient, dtes);
			free(base_addr);
		}
	}
	else
	{
		dpuClientElog(dclient, "not a supported kern_data_store format");
		return;
	}
	/* cleanup resources */
	if (dtes->kds_dst_array)
	{
		for (int i=0; i < dtes->kds_dst_nitems; i++)
			free(dtes->kds_dst_array[i]);
		free(dtes->kds_dst_array);
	}
}

/*
 * dpuservHandleDpuTaskFinal
 */
static void
dpuservHandleDpuTaskFinal(dpuClient *dclient, XpuCommand *xcmd)
{
	groupby_final_buffer *gf_buf = dclient->gf_buf;
	kern_multirels *kmrels = dclient->kmrels;
	XpuCommand		resp;
	struct iovec   *iovec_array;
	struct iovec   *iov;
	int				iovcnt = 0;
	bool			gf_buf_locked = false;
	size_t			resp_sz;

	/* iovec allocation */
	iovec_array = alloca(sizeof(struct iovec) *
						 ((kmrels ? kmrels->num_rels : 0) + 8));
	/* Xcmd for the response */
	memset(&resp, 0, sizeof(XpuCommand));
	resp_sz = MAXALIGN(offsetof(XpuCommand, u.results.stats));
	resp.magic = XpuCommandMagicNumber;
	resp.tag   = XpuCommandTag__Success;

	iov = &iovec_array[iovcnt++];
    iov->iov_base = &resp;
    iov->iov_len  = resp_sz;

	/*
	 * Pack outer join map if this is the final call on this device
	 */
	if (xcmd->u.fin.final_this_device && kmrels)
	{
		uint32_t	ojmap_length = 0;

		for (int i=0; i < kmrels->num_rels; i++)
		{
			kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, i);
			bool	   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, i);

			if (oj_map)
			{
				uint32_t	map_sz = MAXALIGN(sizeof(bool) * kds_in->nitems);

				iov = &iovec_array[iovcnt++];
				iov->iov_base = oj_map;
				iov->iov_len = map_sz;

				ojmap_length += map_sz;
			}
		}
		if (ojmap_length > 0)
		{
			resp.u.results.final_this_device = true;
			resp.u.results.ojmap_offset = resp_sz;
			resp.u.results.ojmap_length = ojmap_length;
			resp_sz += ojmap_length;
		}
	}

	/*
	 * KDS-Final buffer if DpuPreAgg
	 */
	if (xcmd->u.fin.final_plan_node && gf_buf)
	{
		kern_data_store *kds_final;
		size_t		sz1, sz2, sz3;

		pthreadRWLockReadLock(&gf_buf->kds_final_rwlock);
		gf_buf_locked = true;

		kds_final = gf_buf->kds_final;
		if (kds_final->format == KDS_FORMAT_HASH)
		{
			assert(kds_final->hash_nslots > 0);
			sz1 = KDS_HEAD_LENGTH(kds_final);
			iov = &iovec_array[iovcnt++];
			iov->iov_base = kds_final;
			iov->iov_len  = sz1;

			sz2 = MAXALIGN(sizeof(uint32_t) * kds_final->nitems);
			if (sz2 > 0)
			{
				iov = &iovec_array[iovcnt++];
				iov->iov_base = KDS_GET_ROWINDEX(kds_final);
				iov->iov_len  = sz2;
			}

			sz3 = __kds_unpack(kds_final->usage);
			if (sz3 > 0)
			{
				iov = &iovec_array[iovcnt++];
				iov->iov_base = (char *)kds_final + kds_final->length - sz3;
				iov->iov_len  = sz3;
			}
			/* fixup kds */
			kds_final->format = KDS_FORMAT_ROW;
			kds_final->hash_nslots = 0;
			kds_final->length = (sz1 + sz2 + sz3);
		}
		else
		{
			assert(kds_final->format == KDS_FORMAT_ROW &&
				   kds_final->hash_nslots == 0);
			sz1 = (KDS_HEAD_LENGTH(kds_final) +
				   MAXALIGN(sizeof(uint32_t) * kds_final->nitems));
			if (sz1 > 0)
			{
				iov = &iovec_array[iovcnt++];
				iov->iov_base = kds_final;
				iov->iov_len  = sz1;
			}
			sz2 = __kds_unpack(kds_final->usage);
			if (sz2 > 0)
			{
				iov = &iovec_array[iovcnt++];
				iov->iov_base = (char *)kds_final + kds_final->length - sz2;
				iov->iov_len  = sz2;
			}
			/* fixup kds */
			kds_final->length = sz1 + sz2;
		}
		resp.u.results.final_plan_node = true;
		resp.u.results.chunks_nitems = 1;
		resp.u.results.chunks_offset = resp_sz;
		resp_sz += kds_final->length;
	}
	resp.length = resp_sz;
	__dpuClientWriteBack(dclient, iovec_array, iovcnt);

	if (gf_buf_locked)
		pthreadRWLockUnlock(&gf_buf->kds_final_rwlock);
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
		dpuServUnmapSessionBuffers(dclient);
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
					case XpuCommandTag__XpuTaskExec:
						dpuservHandleDpuTaskExec(dclient, xcmd);
						if (verbose)
							fprintf(stderr, "[DPU-%ld@%s] CMD=XpuTaskExec\n",
									worker_id, dclient->peer_addr);
						break;
					case XpuCommandTag__XpuTaskFinal:
						dpuservHandleDpuTaskFinal(dclient, xcmd);
						if (verbose)
							fprintf(stderr, "[DPU-%ld@%s] CMD=XpuTaskFinal\n",
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
	signal(SIGPIPE, SIG_IGN);

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
				union {
					struct sockaddr		addr;
					struct sockaddr_in	in;
					struct sockaddr_in6	in6;
				} peer;
				socklen_t	peer_sz = sizeof(peer);
				int			client_fd;

				if ((epoll_ev.events & ~EPOLLIN) != 0)
					__Elog("listen socket raised unexpected error events (%08x): %m",
						   epoll_ev.events);

				client_fd = accept(serv_fd, &peer.addr, &peer_sz);
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
					if (peer.addr.sa_family == AF_INET)
					{
						inet_ntop(peer.in.sin_family,
								  &peer.in.sin_addr,
								  dclient->peer_addr,
								  PEER_ADDR_LEN);
					}
					else if (peer.addr.sa_family == AF_INET6)
					{
						inet_ntop(peer.in6.sin6_family,
								  &peer.in6.sin6_addr,
								  dclient->peer_addr,
								  PEER_ADDR_LEN);
					}
					else
					{
						snprintf(dclient->peer_addr, PEER_ADDR_LEN,
								 "Unknown DpuClient");
					}
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

	pthreadMutexInit(&groupby_final_buffer_lock);
	for (int i=0; i < GROUPBY_FINAL_BUFFER_HASHSZ; i++)
		dlist_init(&groupby_final_buffer_hash[i]);

	return dpuserv_main(addr, addr_len);
}
