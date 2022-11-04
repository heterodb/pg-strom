/*
 * gpu_service.c
 *
 * A background worker process that handles any interactions with GPU
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/*
 * gpuContext / gpuModule / gpuMemory
 */
typedef struct
{
	CUmodule		cuda_module;
	HTAB		   *cuda_type_htab;
	HTAB		   *cuda_func_htab;
	xpu_encode_info *cuda_encode_catalog;
} gpuModule;

typedef struct
{
	pthread_mutex_t	lock;
	size_t			total_sz;	/* total pool size */
	size_t			hard_limit;
	size_t			keep_limit;
	dlist_head		segment_list;
} gpuMemoryPool;

struct gpuContext
{
	dlist_node		chain;
	int				serv_fd;		/* for accept(2) */
	int				cuda_dindex;
	CUdevice		cuda_device;
	CUcontext		cuda_context;
	gpuModule		normal;			/* optimized kernel */
	gpuModule		debug;			/* non-optimized debug kernel */
	gpuMemoryPool	pool;
	/* GPU client */
	pthread_mutex_t	client_lock;
	dlist_head		client_list;
	/* XPU commands */
	pthread_cond_t	cond;
	pthread_mutex_t	lock;
	dlist_head		command_list;
	/* GPU task workers */
	int				n_workers;
	pthread_t		workers[1];
};
typedef struct gpuContext	gpuContext;
#define SizeOfGpuContext	\
	offsetof(gpuContext, workers[pgstrom_max_async_gpu_tasks])

/*
 * variables
 */
int		pgstrom_max_async_gpu_tasks;	/* GUC */
bool	pgstrom_load_gpu_debug_module;	/* GUC */
__thread int		CU_DINDEX_PER_THREAD = -1;
__thread CUdevice	CU_DEVICE_PER_THREAD = -1;
__thread CUcontext	CU_CONTEXT_PER_THREAD = NULL;
__thread CUevent	CU_EVENT_PER_THREAD = NULL;
static __thread gpuContext *GpuWorkerCurrentContext = NULL;
static volatile int		gpuserv_bgworker_got_signal = 0;
static dlist_head		gpuserv_gpucontext_list;
static int				gpuserv_epoll_fdesc = -1;

void	gpuservBgWorkerMain(Datum arg);

/*
 * cuStrError
 */
const char *
cuStrError(CUresult rc)
{
	static __thread char buffer[300];
	const char *err_name = NULL;
	const char *err_string = NULL;

	if (cuGetErrorName(rc, &err_name) != CUDA_SUCCESS)
	{
		snprintf(buffer, sizeof(buffer), "Unknown CUDA Error (%d)", (int)rc);
	}
	else if (cuGetErrorString(rc, &err_string) != CUDA_SUCCESS)
	{
		return err_name;
	}
	else
	{
		snprintf(buffer, sizeof(buffer), "%s - %s", err_name, err_string);
	}
	return buffer;
}

/* ----------------------------------------------------------------
 *
 * GPU Memory Allocator
 *
 * ----------------------------------------------------------------
 */
static int		pgstrom_gpu_mempool_segment_sz_kb;	/* GUC */
static double	pgstrom_gpu_mempool_max_ratio;		/* GUC */
static double	pgstrom_gpu_mempool_min_ratio;		/* GUC */
static int		pgstrom_gpu_mempool_release_delay;	/* GUC */
typedef struct
{
	dlist_node		chain;
	size_t			segment_sz;
	size_t			active_sz;		/* == 0 can be released */
	CUdeviceptr		devptr;
	dlist_head		free_chunks;	/* list of free chunks */
	dlist_head		addr_chunks;	/* list of ordered chunks */
	struct timeval	tval;
	unsigned long	iomap_handle;	/* for old nvme_strom kmod */
} gpuMemorySegment;

typedef struct
{
	dlist_node		free_chain;
	dlist_node		addr_chain;
	gpuMemorySegment *mseg;
	gpuMemChunk		c;
} gpuMemoryChunkInternal;

static const gpuMemoryChunkInternal *
__gpuMemAllocFromSegment(gpuMemoryPool *pool,
						 gpuMemorySegment *mseg,
						 size_t bytesize)
{
	gpuMemoryChunkInternal *chunk;
	gpuMemoryChunkInternal *buddy;
	dlist_iter		iter;

	dlist_foreach(iter, &mseg->free_chunks)
	{
		chunk = dlist_container(gpuMemoryChunkInternal,
								free_chain, iter.cur);
		if (bytesize <= chunk->c.length)
		{
			size_t	surplus = chunk->c.length - bytesize;

			/* try to split, if free chunk is enough large (>4MB) */
			if (surplus > (4UL << 20))
			{
				buddy = calloc(1, sizeof(gpuMemoryChunkInternal));
				if (!buddy)
					return NULL;	/* out of memory */
				chunk->c.length -= surplus;

				buddy->mseg   = mseg;
				buddy->c.base = mseg->devptr;
				buddy->c.offset = chunk->c.offset + chunk->c.length;
				buddy->c.length = surplus;
				buddy->c.iomap_handle = mseg->iomap_handle;
				dlist_insert_after(&chunk->free_chain, &buddy->free_chain);
				dlist_insert_after(&chunk->addr_chain, &buddy->addr_chain);
			}
			/* mark it as an active chunk */
			dlist_delete(&chunk->free_chain);
			memset(&chunk->free_chain, 0, sizeof(dlist_node));
			mseg->active_sz += chunk->c.length;

			/* update the LRU ordered segment list and timestamp */
			gettimeofday(&mseg->tval, NULL);
			dlist_move_head(&pool->segment_list, &mseg->chain);

			return chunk;
		}
	}
	return NULL;
}

static gpuMemorySegment *
__gpuMemAllocNewSegment(gpuMemoryPool *pool, size_t segment_sz)
{
	gpuMemorySegment *mseg = calloc(1, sizeof(gpuMemorySegment));
	gpuMemoryChunkInternal *chunk = calloc(1, sizeof(gpuMemoryChunkInternal));
	CUresult	rc;

	if (!mseg || !chunk)
		goto error;
	mseg->segment_sz = segment_sz;
	mseg->active_sz = 0;
	dlist_init(&mseg->free_chunks);
	dlist_init(&mseg->addr_chunks);

	rc = cuMemAlloc(&mseg->devptr, mseg->segment_sz);
	if (rc != CUDA_SUCCESS)
		goto error;
	rc = gpuDirectMapGpuMemory(mseg->devptr,
							   mseg->segment_sz,
							   &mseg->iomap_handle);
	if (rc != CUDA_SUCCESS)
		goto error;

	chunk->mseg   = mseg;
	chunk->c.base = mseg->devptr;
	chunk->c.offset = 0;
	chunk->c.length = segment_sz;
	chunk->c.iomap_handle = mseg->iomap_handle;
	dlist_push_head(&mseg->free_chunks, &chunk->free_chain);
	dlist_push_head(&mseg->addr_chunks, &chunk->addr_chain);

	dlist_push_head(&pool->segment_list, &mseg->chain);
	pool->total_sz += segment_sz;

	return mseg;
error:
	if (mseg->devptr)
		cuMemFree(mseg->devptr);
	if (mseg)
		free(mseg);
	if (chunk)
		free(chunk);
	return NULL;
}

const gpuMemChunk *
gpuMemAlloc(size_t bytesize)
{
	gpuMemoryPool  *pool = &GpuWorkerCurrentContext->pool;
	dlist_iter		iter;
	size_t			segment_sz;
	const gpuMemoryChunkInternal *chunk = NULL;

	bytesize = MAXALIGN(bytesize);
	pthreadMutexLock(&pool->lock);
	dlist_foreach(iter, &pool->segment_list)
	{
		gpuMemorySegment *mseg = dlist_container(gpuMemorySegment,
												 chain, iter.cur);
		if (mseg->active_sz + bytesize <= mseg->segment_sz)
		{
			chunk = __gpuMemAllocFromSegment(pool, mseg, bytesize);
			if (chunk)
				goto out_unlock;
		}
	}
	segment_sz = ((size_t)pgstrom_gpu_mempool_segment_sz_kb << 10);
	if (segment_sz < bytesize)
		segment_sz = bytesize;
	if (pool->total_sz + segment_sz <= pool->hard_limit)
	{
		gpuMemorySegment *mseg = __gpuMemAllocNewSegment(pool, segment_sz);

		if (mseg)
			chunk = __gpuMemAllocFromSegment(pool, mseg, bytesize);
	}
out_unlock:	
	pthreadMutexUnlock(&pool->lock);

	return (chunk ? &chunk->c : NULL);
}

void
gpuMemFree(const gpuMemChunk *__chunk)
{
	gpuMemoryPool  *pool = &GpuWorkerCurrentContext->pool;
	gpuMemorySegment *mseg;
	gpuMemoryChunkInternal *chunk;
	gpuMemoryChunkInternal *buddy;
	dlist_node	   *dnode;

	chunk = (gpuMemoryChunkInternal *)
		((char *)__chunk - offsetof(gpuMemoryChunkInternal, c));
	Assert(!chunk->free_chain.prev && !chunk->free_chain.next);

	pthreadMutexLock(&pool->lock);
	/* revert this chunk state to 'free' */
	mseg = chunk->mseg;
	mseg->active_sz -= chunk->c.length;
	dlist_push_head(&mseg->free_chunks,
					&chunk->free_chain);

	/* try merge if next chunk is also free */
	if (dlist_has_next(&mseg->addr_chunks,
					   &chunk->addr_chain))
	{
		dnode = dlist_next_node(&mseg->addr_chunks,
								&chunk->addr_chain);
		buddy = dlist_container(gpuMemoryChunkInternal,
								addr_chain, dnode);
		if (buddy->free_chain.prev && buddy->addr_chain.next)
		{
			Assert(chunk->c.offset + chunk->c.length == buddy->c.offset);
			dlist_delete(&buddy->free_chain);
			dlist_delete(&buddy->addr_chain);
			chunk->c.length += buddy->c.length;
			free(buddy);
		}
	}
	/* try merge if prev chunk is also free */
	if (dlist_has_prev(&mseg->addr_chunks,
					   &chunk->addr_chain))
	{
		dnode = dlist_prev_node(&mseg->addr_chunks,
								&chunk->addr_chain);
		buddy = dlist_container(gpuMemoryChunkInternal,
								addr_chain, dnode);
		/* merge if prev chunk is also free */
		if (buddy->free_chain.prev && buddy->addr_chain.next)
		{
			Assert(buddy->c.offset + buddy->c.length == chunk->c.offset);
			dlist_delete(&chunk->free_chain);
			dlist_delete(&chunk->addr_chain);
			buddy->c.length += chunk->c.length;
			free(chunk);
		}
	}
	/* update the LRU ordered segment list and timestamp */
	gettimeofday(&mseg->tval, NULL);
	dlist_move_head(&pool->segment_list, &mseg->chain);
	pthreadMutexUnlock(&pool->lock);
}

static void
gpuMemoryPoolMaintenance(gpuContext *gcontext)
{
	gpuMemoryPool  *pool = &gcontext->pool;
	dlist_iter		iter;
	struct timeval	tval;
	int64			tdiff;
	CUresult		rc;

	if (!pthreadMutexTryLock(&pool->lock))
		return;
	if (pool->total_sz > pool->keep_limit)
	{
		gettimeofday(&tval, NULL);
		dlist_reverse_foreach(iter, &pool->segment_list)
		{
			gpuMemorySegment *mseg = dlist_container(gpuMemorySegment,
													 chain, iter.cur);
			/* still in active? */
			if (mseg->active_sz != 0)
				continue;

			/* enough time to release is elapsed? */
			tdiff = ((tval.tv_sec  - mseg->tval.tv_sec)  * 1000 +
					 (tval.tv_usec - mseg->tval.tv_usec) / 1000);
			if (tdiff < pgstrom_gpu_mempool_release_delay)
				continue;

			/* ok, this segment should be released */
			rc = gpuDirectUnmapGpuMemory(mseg->devptr,
										 mseg->iomap_handle);
			if (rc != CUDA_SUCCESS)
				__FATAL("failed on gpuDirectUnmapGpuMemory: %s", cuStrError(rc));
			rc = cuMemFree(mseg->devptr);
			if (rc != CUDA_SUCCESS)
				__FATAL("failed on cuMemFree: %s", cuStrError(rc));
			/* detach segment */
			dlist_delete(&mseg->chain);
			while (!dlist_is_empty(&mseg->addr_chunks))
			{
				dlist_node	   *dnode = dlist_pop_head_node(&mseg->addr_chunks);
				gpuMemoryChunkInternal *chunk;

				chunk = dlist_container(gpuMemoryChunkInternal,
										addr_chain, dnode);
				Assert(chunk->free_chain.prev &&
					   chunk->free_chain.next);
				free(chunk);
			}
			fprintf(stderr, "GPU-%d: i/o mapped device memory %lu bytes released",
					gcontext->cuda_dindex, mseg->segment_sz);
			free(mseg);
			break;
		}
	}
	pthreadMutexUnlock(&pool->lock);
}

static void
gpuMemoryPoolInit(gpuMemoryPool *pool, size_t dev_total_memsz)
{
	pthreadMutexInit(&pool->lock);
	pool->total_sz = 0;
	pool->hard_limit = pgstrom_gpu_mempool_max_ratio * (double)dev_total_memsz;
	pool->keep_limit = pgstrom_gpu_mempool_min_ratio * (double)dev_total_memsz;
	dlist_init(&pool->segment_list);
}

/*
 * gpuservLoadKdsBlock
 *
 * fill up KDS_FORMAT_BLOCK using GPU-Direct
 */
const gpuMemChunk *
gpuservLoadKdsBlock(gpuClient *gclient,
					kern_data_store *kds,
					const char *pathname,
					strom_io_vector *kds_iovec)
{
	const gpuMemChunk *chunk;
	GPUDirectFileDesc gdfdesc;
	CUresult	rc;
	size_t		len;

	if (!gpuDirectFileDescOpenByPath(&gdfdesc, pathname))
	{
		gpuClientELog(gclient, "failed on gpuDirectFileDescOpenByPath('%s')", pathname);
		return NULL;
	}

	chunk = gpuMemAlloc(kds->length);
	if (!chunk)
	{
		gpuClientELog(gclient, "failed on gpuMemAlloc(%zu)", kds->length);
		goto error_1;
	}

	len = kds->block_offset + kds->block_nloaded * BLCKSZ;
	rc = cuMemcpyHtoD(chunk->base + chunk->offset, kds, len);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientELog(gclient, "failed on cuMemcpyHtoD: %s", cuStrError(rc));
		goto error_2;
	}

	if (!gpuDirectFileReadIOV(&gdfdesc,
							  chunk->base,
							  chunk->iomap_handle,
							  chunk->offset + len,
							  kds_iovec))
	{
		gpuClientELog(gclient, "failed on gpuDirectFileReadIOV");
		goto error_2;
	}
	gpuDirectFileDescClose(&gdfdesc);
	return chunk;

error_2:
	gpuMemFree(chunk);
error_1:
	gpuDirectFileDescClose(&gdfdesc);
	return NULL;
}

/*
 * gpuServiceGoingTerminate
 */
bool
gpuServiceGoingTerminate(void)
{
	return (gpuserv_bgworker_got_signal != 0);
}

/*
 * gpuClientPut
 */
static void
gpuClientPut(gpuClient *gclient)
{
	if (pg_atomic_sub_fetch_u32(&gclient->refcnt, 2) == 0)
	{
		gpuContext *gcontext = gclient->gcontext;
		CUresult	rc;

		pthreadMutexLock(&gcontext->client_lock);
		dlist_delete(&gclient->chain);
		pthreadMutexUnlock(&gcontext->client_lock);

		if (gclient->sockfd >= 0)
			close(gclient->sockfd);
		if (gclient->session)
		{
			CUdeviceptr		dptr = ((CUdeviceptr)gclient->session -
									offsetof(XpuCommand, u.session));
			rc = cuMemFree(dptr);
			if (rc != CUDA_SUCCESS)
				fprintf(stderr, "failed on cuMemFree: %s\n", cuStrError(rc));
		}
		free(gclient);
	}
}

/*
 * gpuClientWriteBack
 */
static void
__gpuClientWriteBack(gpuClient *gclient, struct iovec *iov, int iovcnt)
{
	pthreadMutexLock(&gclient->mutex);
	if (gclient->sockfd >= 0)
	{
		ssize_t		nbytes;

		while (iovcnt > 0)
		{
			nbytes = writev(gclient->sockfd, iov, iovcnt);
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
				 * Peer socket is closed? Anyway, it looks we cannot continue
				 * to send back the message any more. So, clean up this gpuClient.
				 */
				pg_atomic_fetch_and_u32(&gclient->refcnt, ~1U);
				close(gclient->sockfd);
				gclient->sockfd = -1;
				break;
			}
		}
	}
	pthreadMutexUnlock(&gclient->mutex);
}

void
gpuClientWriteBack(gpuClient  *gclient,
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
	__gpuClientWriteBack(gclient, iov_array, iovcnt);
}

/*
 * gpuClientELog
 */
void
__gpuClientELogRaw(gpuClient *gclient, kern_errorbuf *errorbuf)
{
	XpuCommand		resp;
	struct iovec	iov;

	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Error;
	resp.length = offsetof(XpuCommand, u.error) + sizeof(kern_errorbuf);
	memcpy(&resp.u.error, errorbuf, sizeof(kern_errorbuf));

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);
}

void
__gpuClientELog(gpuClient *gclient,
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
	resp.u.error.errcode = errcode,
	resp.u.error.lineno = lineno;
	strncpy(resp.u.error.filename, filename, KERN_ERRORBUF_FILENAME_LEN);
	strncpy(resp.u.error.funcname, funcname, KERN_ERRORBUF_FUNCNAME_LEN);

	va_start(ap, fmt);
	vsnprintf(resp.u.error.message, KERN_ERRORBUF_MESSAGE_LEN, fmt, ap);
	va_end(ap);

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);

	/* unable to continue GPU service, so try to restart */
	if (errcode == ERRCODE_DEVICE_FATAL)
	{
		fprintf(stderr, "(%s:%d, %s) GPU faal - %s\n",
				resp.u.error.filename,
				resp.u.error.lineno,
				resp.u.error.funcname,
				resp.u.error.message);
		gpuserv_bgworker_got_signal |= (1 << SIGHUP);
		pg_memory_barrier();
		SetLatch(MyLatch);
	}
}

/*
 * gpuservHandleOpenSession
 */
static bool
__resolveDevicePointersWalker(gpuModule *gmodule, kern_expression *kexp,
							  char *emsg, size_t emsg_sz)
{
	xpu_function_catalog_entry *xpu_func;
	xpu_type_catalog_entry *xpu_type;
	kern_expression *karg;
	int		i, n;

	/* lookup device function */
	xpu_func = hash_search(gmodule->cuda_func_htab,
						   &kexp->opcode,
						   HASH_FIND, NULL);
	if (!xpu_func)
	{
		snprintf(emsg, emsg_sz,
				 "device function pointer for opcode:%u not found.",
				 (int)kexp->opcode);
		return false;
	}
	kexp->fn_dptr = xpu_func->func_dptr;

	/* lookup device type operator */
	xpu_type = hash_search(gmodule->cuda_type_htab,
						   &kexp->exptype,
						   HASH_FIND, NULL);
	if (!xpu_type)
	{
		snprintf(emsg, emsg_sz,
				 "device type pointer for opcode:%u not found.",
				 (int)kexp->exptype);
		return false;
	}
	kexp->expr_ops = xpu_type->type_ops;

	if (kexp->opcode == FuncOpCode__Projection)
	{
		n = kexp->u.proj.nexprs + kexp->u.proj.nattrs;
		for (i=0; i < n; i++)
		{
			kern_projection_desc *desc = &kexp->u.proj.desc[i];

			xpu_type = hash_search(gmodule->cuda_type_htab,
								   &desc->slot_type,
								   HASH_FIND, NULL);
			if (xpu_type)
				desc->slot_ops = xpu_type->type_ops;
			else if (i >= kexp->u.proj.nexprs)
				desc->slot_ops = NULL;	/* PostgreSQL generic projection */
			else
			{
				snprintf(emsg, emsg_sz,
						 "device type pointer for opcode:%u not found.",
						 (int)desc->slot_type);
				return false;
			}
			fprintf(stderr, "desc[%d] slot_id=%u slot_type=%d\n", i, desc->slot_id, desc->slot_type);
		}
	}

	for (i=0, karg=KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg=KEXP_NEXT_ARG(karg))
	{
		if (!__KEXP_IS_VALID(kexp,karg))
		{
			snprintf(emsg, emsg_sz, "XPU code corruption at args[%d]", i);
			return false;
		}
		if (!__resolveDevicePointersWalker(gmodule, karg, emsg, emsg_sz))
			return false;
	}
	return true;
}

static bool
__resolveDevicePointers(gpuModule *gmodule,
						kern_session_info *session,
						char *emsg, size_t emsg_sz)
{
	xpu_encode_info	*encode = SESSION_ENCODE(session);
	kern_expression *kexp;

	if (session->xpucode_scan_quals)
	{
		kexp = (kern_expression *)((char *)session + session->xpucode_scan_quals);
		if (!__resolveDevicePointersWalker(gmodule, kexp, emsg, emsg_sz))
			return false;
	}

	if (session->xpucode_scan_projs)
	{
		kexp = (kern_expression *)((char *)session + session->xpucode_scan_projs);
		if (!__resolveDevicePointersWalker(gmodule, kexp, emsg, emsg_sz))
			return false;
	}

	if (encode)
	{
		xpu_encode_info *catalog = gmodule->cuda_encode_catalog;
		int		i;

		for (i=0; ; i++)
		{
			if (!catalog[i].enc_mblen || catalog[i].enc_maxlen < 1)
			{
				snprintf(emsg, emsg_sz,
						 "encode [%s] was not found.", encode->encname);
				return false;
			}
			if (strcmp(encode->encname, catalog[i].encname) == 0)
			{
				encode->enc_maxlen = catalog[i].enc_maxlen;
				encode->enc_mblen  = catalog[i].enc_mblen;
				break;
			}
		}
	}
	return true;
}

static bool
gpuservHandleOpenSession(gpuClient *gclient, XpuCommand *xcmd)
{
	gpuContext	   *gcontext = gclient->gcontext;
	gpuModule	   *gmodule;
	kern_session_info *session = &xcmd->u.session;
	XpuCommand		resp;
	char			emsg[512];
	struct iovec	iov;

	if (gclient->session)
	{
		gpuClientELog(gclient, "OpenSession is called twice");
		return false;
	}

	/* resolve device pointers */
	if (session->xpucode_use_debug_code && pgstrom_load_gpu_debug_module)
		gmodule = &gcontext->debug;
	else
		gmodule = &gcontext->normal;
	if (!__resolveDevicePointers(gmodule, session, emsg, sizeof(emsg)))
	{
		gpuClientELog(gclient, "%s", emsg);
		return false;
	}

	fprintf(stderr, "session kcxt_kvars_nslots=%u kcxt_extra_bufsz=%u\n",
			session->kcxt_kvars_nslots,
			session->kcxt_extra_bufsz);

	gclient->session = session;
	gclient->cuda_module = gmodule->cuda_module;

	/* success status */
	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Success;
	resp.length = offsetof(XpuCommand, u);

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);

	return true;
}

/*
 * gpuservGpuWorkerMain -- actual worker
 */
static void *
gpuservGpuWorkerMain(void *__arg)
{
	gpuContext *gcontext = (gpuContext *)__arg;
	gpuClient  *gclient;
	CUevent		cuda_event;
	CUresult	rc;

	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuCtxSetCurrent: %s", cuStrError(rc));
	rc = cuEventCreate(&cuda_event, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuEventCreate: %s", cuStrError(rc));

	GpuWorkerCurrentContext = gcontext;
	CU_DINDEX_PER_THREAD  = gcontext->cuda_dindex;
	CU_DEVICE_PER_THREAD  = gcontext->cuda_device;
	CU_CONTEXT_PER_THREAD = gcontext->cuda_context;
	CU_EVENT_PER_THREAD = cuda_event;
	pg_memory_barrier();

	pthreadMutexLock(&gcontext->lock);
	while (!gpuServiceGoingTerminate())
	{
		XpuCommand *xcmd;
		dlist_node *dnode;

		if (!dlist_is_empty(&gcontext->command_list))
		{
			dnode = dlist_pop_head_node(&gcontext->command_list);
			xcmd = dlist_container(XpuCommand, chain, dnode);
			pthreadMutexUnlock(&gcontext->lock);

			gclient = xcmd->priv;
			/*
			 * MEMO: If the least bit of gclient->refcnt is not set,
			 * it means the gpu-client connection is no longer available.
			 * (already closed, or error detected.)
			 */
			if ((pg_atomic_read_u32(&gclient->refcnt) & 1) == 1)
			{
				switch (xcmd->tag)
				{
					case XpuCommandTag__OpenSession:
						if (gpuservHandleOpenSession(gclient, xcmd))
							xcmd = NULL;	/* session information shall be kept until
											 * end of the session. */
						break;
					case XpuCommandTag__XpuScanExec:
						gpuservHandleGpuScanExec(gclient, xcmd);
						break;
					default:
						gpuClientELog(gclient, "unknown XPU command (%d)",
									  (int)xcmd->tag);
						break;
				}
			}
			if (xcmd)
				cuMemFree((CUdeviceptr)xcmd);
			gpuClientPut(gclient);
			pthreadMutexLock(&gcontext->lock);
		}
		else if (!pthreadCondWaitTimeout(&gcontext->cond,
										 &gcontext->lock,
										 5000))
		{
			pthreadMutexUnlock(&gcontext->lock);
			/* maintenance works */
			gpuMemoryPoolMaintenance(gcontext);
			pthreadMutexLock(&gcontext->lock);
		}
	}
	pthreadMutexUnlock(&gcontext->lock);
	return NULL;
}

/*
 * gpuservMonitorClient
 */
static void *
__gpuServiceAllocCommand(void *__priv, size_t sz)
{
	CUdeviceptr	devptr;
	CUresult	rc;

	rc = cuMemAllocManaged(&devptr, sz, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		return NULL;
	return (void *)devptr;
}

static void
__gpuServiceAttachCommand(void *__priv, XpuCommand *xcmd)
{
	gpuClient  *gclient = (gpuClient *)__priv;
	gpuContext *gcontext = gclient->gcontext;

	pg_atomic_fetch_add_u32(&gclient->refcnt, 2);
	xcmd->priv = gclient;

	pthreadMutexLock(&gcontext->lock);
	dlist_push_tail(&gcontext->command_list, &xcmd->chain);
	pthreadMutexUnlock(&gcontext->lock);
	pthreadCondSignal(&gcontext->cond);
}
TEMPLATE_XPU_CONNECT_RECEIVE_COMMANDS(__gpuService)

static void *
gpuservMonitorClient(void *__priv)
{
	gpuClient  *gclient = __priv;
	gpuContext *gcontext = gclient->gcontext;
	pgsocket	sockfd = gclient->sockfd;
	char		elabel[32];
	CUresult	rc;

	snprintf(elabel, sizeof(elabel), "GPU-%d", gcontext->cuda_dindex);

	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		fprintf(stderr, "[%s; %s:%d] failed on cuCtxSetCurrent: %s\n",
				elabel, __FILE_NAME__,__LINE__, cuStrError(rc));
		goto out;
	}
	GpuWorkerCurrentContext = gcontext;
	pg_memory_barrier();

	for (;;)
	{
		struct pollfd  pfd;
		int		nevents;

		pfd.fd = sockfd;
		pfd.events = POLLIN;
		pfd.revents = 0;
		nevents = poll(&pfd, 1, -1);
		if (nevents < 0)
		{
			if (errno == EINTR)
				continue;
			fprintf(stderr, "[%s; %s:%d] failed on poll(2): %m\n",
					elabel, __FILE_NAME__,__LINE__);
			break;
		}
		if (nevents == 0)
			continue;
		Assert(nevents == 1);
		if (pfd.revents == POLLIN)
		{
			if (__gpuServiceReceiveCommands(sockfd, gclient, elabel) < 0)
				break;
		}
		else if (pfd.revents & ~POLLIN)
		{
			fprintf(stderr, "[%s; %s:%d] peer socket closed.\n",
					elabel, __FILE_NAME__,__LINE__);
			break;
		}
	}
out:
	gpuClientPut(gclient);
	return NULL;
}

/*
 * gpuservAcceptClient
 */
static void
gpuservAcceptClient(gpuContext *gcontext)
{
	gpuClient  *gclient;
	pgsocket	sockfd;
	int			errcode;

	sockfd = accept(gcontext->serv_fd, NULL, NULL);
	if (sockfd < 0)
	{
		elog(LOG, "GPU%d: could not accept new connection: %m",
			 gcontext->cuda_dindex);
		pg_usleep(10000L);		/* wait 10ms */
		return;
	}

	gclient = calloc(1, sizeof(gpuClient));
	if (!gclient)
	{
		elog(LOG, "GPU%d: out of memory: %m",
			 gcontext->cuda_dindex);
		close(sockfd);
		return;
	}
	gclient->gcontext = gcontext;
	pg_atomic_init_u32(&gclient->refcnt, 3);
	pthreadMutexInit(&gclient->mutex);
	gclient->sockfd = sockfd;

	if ((errcode = pthread_create(&gclient->worker, NULL,
								  gpuservMonitorClient,
								  gclient)) != 0)
	{
		elog(LOG, "failed on pthread_create: %s", strerror(errcode));
		close(sockfd);
		free(gclient);
		return;
	}
	pthreadMutexLock(&gcontext->client_lock);
	dlist_push_tail(&gcontext->client_list, &gclient->chain);
	pthreadMutexUnlock(&gcontext->client_lock);
}

/*
 * __setupDevTypeLinkageTable
 */
static HTAB *
__setupDevTypeLinkageTable(gpuModule *gmodule)
{
	xpu_type_catalog_entry *xpu_types_catalog;
	HASHCTL		hctl;
	HTAB	   *htab = NULL;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;
	int			i;
	
	/* build device type table */
	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = sizeof(TypeOpCode);
	hctl.entrysize = sizeof(xpu_type_catalog_entry);
	hctl.hcxt = TopMemoryContext;
	htab = hash_create("CUDA device type hash table",
					   512,
					   &hctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	rc = cuModuleGetGlobal(&dptr, &nbytes, gmodule->cuda_module,
						   "builtin_xpu_types_catalog");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal: %s", cuStrError(rc));

	xpu_types_catalog = alloca(nbytes);
	rc = cuMemcpyDtoH(xpu_types_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));
	for (i=0; xpu_types_catalog[i].type_opcode != TypeOpCode__Invalid; i++)
	{
		TypeOpCode	type_opcode = xpu_types_catalog[i].type_opcode;
		xpu_type_catalog_entry *entry;
		bool		found;

		entry = hash_search(htab, &type_opcode, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? duplicated TypeOpCode: %u", (uint32_t)type_opcode);
		Assert(entry->type_opcode == type_opcode);
		entry->type_ops = xpu_types_catalog[i].type_ops;
	}
	return htab;
}

/*
 * __setupDevFuncLinkageTable
 */
static HTAB *
__setupDevFuncLinkageTable(gpuModule *gmodule)
{
	xpu_function_catalog_entry *xpu_funcs_catalog;
	HASHCTL		hctl;
	HTAB	   *htab;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;
	int			i;

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = sizeof(FuncOpCode);
	hctl.entrysize = sizeof(xpu_function_catalog_entry);
	hctl.hcxt = TopMemoryContext;
	htab = hash_create("CUDA device function hash table",
					   1024,
					   &hctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	rc = cuModuleGetGlobal(&dptr, &nbytes, gmodule->cuda_module,
						   "builtin_xpu_functions_catalog");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal: %s", cuStrError(rc));
	xpu_funcs_catalog = alloca(nbytes);
	rc = cuMemcpyDtoH(xpu_funcs_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));
	for (i=0; xpu_funcs_catalog[i].func_opcode != FuncOpCode__Invalid; i++)
	{
		FuncOpCode	func_opcode = xpu_funcs_catalog[i].func_opcode;
		xpu_function_catalog_entry *entry;
		bool		found;

		entry = hash_search(htab, &func_opcode, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? duplicated FuncOpCode: %u", (uint32_t)func_opcode);
		Assert(entry->func_opcode == func_opcode);
		entry->func_dptr = xpu_funcs_catalog[i].func_dptr;
	}
	return htab;
}

/*
 * __setupDevEncodeLinkageCatalog
 */
static xpu_encode_info *
__setupDevEncodeLinkageCatalog(gpuModule *gmodule)
{
	xpu_encode_info *xpu_encode_catalog;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;

	rc = cuModuleGetGlobal(&dptr, &nbytes, gmodule->cuda_module,
                           "xpu_encode_catalog");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal: %s", cuStrError(rc));
	xpu_encode_catalog = MemoryContextAlloc(TopMemoryContext, nbytes);
	rc = cuMemcpyDtoH(xpu_encode_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));

	return xpu_encode_catalog;
}

/*
 * gpuservSetupGpuModule
 */
static void
gpuservSetupGpuModule(gpuModule *gmodule, bool debug_module)
{
	CUlinkState	lstate;
	CUjit_option jit_options[16];
	void	   *jit_option_values[16];
	int			jit_index = 0;
	void	   *bin_image;
	size_t		bin_length;
	char		log_buffer[16384];
	char	   *cuda_builtin_objs;
	char	   *tok, *saveptr;
	CUresult	rc;

	/* Limit max number of registers per threads for ABI compatibility */
	jit_options[jit_index] = CU_JIT_MAX_REGISTERS;
	jit_option_values[jit_index] = (void *)CUDA_MAXREGCOUNT;
	jit_index++;

	/* Get optimal binary to the current context */
	jit_options[jit_index] = CU_JIT_TARGET_FROM_CUCONTEXT;
	jit_option_values[jit_index] = NULL;
	jit_index++;

	/* Compile with L1 cache enabled */
	jit_options[jit_index] = CU_JIT_CACHE_MODE;
	jit_option_values[jit_index] = (void *)CU_JIT_CACHE_OPTION_CA;
	jit_index++;

	if (debug_module)
	{
		jit_options[jit_index] = CU_JIT_GENERATE_DEBUG_INFO;
		jit_option_values[jit_index] = (void *)1UL;
		jit_index++;

		jit_options[jit_index] = CU_JIT_GENERATE_LINE_INFO;
		jit_option_values[jit_index] = (void *)1UL;
		jit_index++;

		jit_options[jit_index] = CU_JIT_OPTIMIZATION_LEVEL;
		jit_option_values[jit_index] = (void *)0UL;
		jit_index++;
	}
	/* Link log buffer */
	jit_options[jit_index] = CU_JIT_ERROR_LOG_BUFFER;
	jit_option_values[jit_index] = (void *)log_buffer;
	jit_index++;

	jit_options[jit_index] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
	jit_option_values[jit_index] = (void *)sizeof(log_buffer);
	jit_index++;

	rc = cuLinkCreate(jit_index, jit_options, jit_option_values, &lstate);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkCreate: %s", cuStrError(rc));

	/* Add builtin fatbin files */
	cuda_builtin_objs = alloca(sizeof(CUDA_BUILTIN_OBJS) + 1);
	strcpy(cuda_builtin_objs, CUDA_BUILTIN_OBJS);
	for (tok = strtok_r(cuda_builtin_objs, " ", &saveptr);
		 tok != NULL;
		 tok = strtok_r(NULL, " ", &saveptr))
	{
		char	pathname[MAXPGPATH];

		snprintf(pathname, MAXPGPATH,
				 PGSHAREDIR "/pg_strom/%s%s.fatbin",
				 __trim(tok), 
				 debug_module ? ".debug" : "");
		rc = cuLinkAddFile(lstate, CU_JIT_INPUT_FATBINARY,
						   pathname, 0, NULL, NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLinkAddFile('%s'): %s",
				 pathname, cuStrError(rc));
	}
	//TODO: Load the extra CUDA module

	/* do the linkage */
	rc = cuLinkComplete(lstate, &bin_image, &bin_length);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkComplete: %s\n%s",
			 cuStrError(rc), log_buffer);

	rc = cuModuleLoadData(&gmodule->cuda_module, bin_image);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleLoadData: %s", cuStrError(rc));

	rc = cuLinkDestroy(lstate);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkDestroy: %s", cuStrError(rc));

	/* setup XPU linkage hash tables */
	gmodule->cuda_type_htab = __setupDevTypeLinkageTable(gmodule);
	gmodule->cuda_func_htab = __setupDevFuncLinkageTable(gmodule);
	gmodule->cuda_encode_catalog = __setupDevEncodeLinkageCatalog(gmodule);
}

/*
 * __gpuContextTerminateWorkers
 *
 * Wake up all the worker threads, and terminate them.
 * The event_fd is polled with edge-trigger mode for individual wake-up,
 * thus we have to clear them to wake up all.
 */
static void
__gpuContextTerminateWorkers(gpuContext *gcontext)
{
	int		i;

	gpuserv_bgworker_got_signal |= (1 << SIGHUP);
	pg_memory_barrier();

	pthreadCondBroadcast(&gcontext->cond);
	for (i=0; i < gcontext->n_workers; i++)
		pthread_join(gcontext->workers[i], NULL);
}

/*
 * gpuservSetupGpuContext
 */
static gpuContext *
gpuservSetupGpuContext(int cuda_dindex)
{
	GpuDevAttributes *dattrs = &gpuDevAttrs[cuda_dindex];
	gpuContext *gcontext = NULL;
	CUresult	rc;
	size_t		stack_sz;
	int			i, status;
	struct sockaddr_un addr;
	struct epoll_event ev;

	/* gpuContext allocation */
	gcontext = calloc(1, SizeOfGpuContext);
	if (!gcontext)
		elog(ERROR, "out of memory");
	gcontext->serv_fd = -1;
	gcontext->cuda_dindex = cuda_dindex;
	gpuMemoryPoolInit(&gcontext->pool, dattrs->DEV_TOTAL_MEMSZ);
	pthreadMutexInit(&gcontext->client_lock);
	dlist_init(&gcontext->client_list);

	pthreadCondInit(&gcontext->cond);
	pthreadMutexInit(&gcontext->lock);
	dlist_init(&gcontext->command_list);

	PG_TRY();
	{
		/* Open the listen socket for this GPU */
		gcontext->serv_fd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (gcontext->serv_fd < 0)
			elog(ERROR, "failed on socket(2): %m");
		snprintf(addr.sun_path, sizeof(addr.sun_path),
				 ".pg_strom.%u.gpu%u.sock",
				 PostmasterPid, gcontext->cuda_dindex);
		addr.sun_family = AF_UNIX;
		if (bind(gcontext->serv_fd, (struct sockaddr *) &addr, sizeof(addr)) != 0)
			elog(ERROR, "failed on bind('%s'): %m", addr.sun_path);
		if (listen(gcontext->serv_fd, 32) != 0)
			elog(ERROR, "failed on listen(2): %m");
		ev.events = EPOLLIN;
		ev.data.ptr = gcontext;
		if (epoll_ctl(gpuserv_epoll_fdesc,
					  EPOLL_CTL_ADD,
					  gcontext->serv_fd, &ev) != 0)
			elog(ERROR, "failed on epoll_ctl(2): %m");

		/* Setup raw CUDA context */
		rc = cuDeviceGet(&gcontext->cuda_device, dattrs->DEV_ID);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", cuStrError(rc));
		rc = cuCtxCreate(&gcontext->cuda_context,
						 CU_CTX_SCHED_AUTO,
						 gcontext->cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s", cuStrError(rc));

		rc = cuCtxGetLimit(&stack_sz, CU_LIMIT_STACK_SIZE);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxGetLimit: %s", cuStrError(rc));
		//stack_sz += 6144;	// 6kB extra stack
		stack_sz += 2048;
		rc = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, stack_sz);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxSetLimit: %s", cuStrError(rc));

		gpuservSetupGpuModule(&gcontext->normal, false);
		if (pgstrom_load_gpu_debug_module)
			gpuservSetupGpuModule(&gcontext->debug, true);

		/* launch worker threads */
		for (i=0; i < pgstrom_max_async_gpu_tasks; i++)
		{
			if ((status = pthread_create(&gcontext->workers[i], NULL,
										 gpuservGpuWorkerMain, gcontext)) != 0)
			{
				__gpuContextTerminateWorkers(gcontext);
				elog(ERROR, "failed on pthread_create: %s", strerror(status));
			}
			gcontext->n_workers++;
		}
	}
	PG_CATCH();
	{
		if (gcontext->cuda_context)
			cuCtxDestroy(gcontext->cuda_context);
		if (gcontext->serv_fd >= 0)
			close(gcontext->serv_fd);
		free(gcontext);
		PG_RE_THROW();
	}
	PG_END_TRY();
	return gcontext;
}

/*
 * gpuservCleanupGpuContext
 */
static void
gpuservCleanupGpuContext(gpuContext *gcontext)
{
	CUresult	rc;

	__gpuContextTerminateWorkers(gcontext);

	while (!dlist_is_empty(&gcontext->client_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gcontext->client_list);
		gpuClient  *gclient = dlist_container(gpuClient, chain, dnode);

		if (close(gclient->sockfd) != 0)
			elog(LOG, "failed on close(sockfd): %m");
	}
	if (close(gcontext->serv_fd) != 0)
		elog(LOG, "failed on close(serv_fd): %m");
	rc = cuCtxDestroy(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(LOG, "failed on cuCtxDestroy: %s", cuStrError(rc));
}

/*
 * SIGTERM/SIGHUP handler
 */
static void
gpuservBgWorkerSignal(SIGNAL_ARGS)
{
	int		saved_errno = errno;

	gpuserv_bgworker_got_signal |= (1 << postgres_signal_arg);

	pg_memory_barrier();

	SetLatch(MyLatch);

	errno = saved_errno;
}

/*
 * gpuservClenupListenSocket
 */
static void
gpuservCleanupOnProcExit(int code, Datum arg)
{
	int		i;

	/* cleanup UNIX domain socket */
	for (i=0; i < numGpuDevAttrs; i++)
	{
		struct stat	stat_buf;
		char		path[MAXPGPATH];

		snprintf(path, sizeof(path),
				 ".pg_strom.%u.gpu%u.sock", PostmasterPid, i);
		if (stat(path, &stat_buf) == 0 &&
			(stat_buf.st_mode & S_IFMT) == S_IFSOCK)
		{
			if (unlink(path) < 0)
				elog(LOG, "failed on unlink('%s'): %m", path);
		}
	}
}

/*
 * gpuservBgWorkerMain
 */
void
gpuservBgWorkerMain(Datum arg)
{
	CUresult	rc;
	int			dindex;

	pqsignal(SIGTERM, gpuservBgWorkerSignal);
	pqsignal(SIGHUP,  gpuservBgWorkerSignal);
	BackgroundWorkerUnblockSignals();

	/* Disable MPS */
	if (setenv("CUDA_MPS_PIPE_DIRECTORY", "/dev/null", 1) != 0)
		elog(ERROR, "failed on setenv: %m");

	/* Registration of resource cleanup handler */
	dlist_init(&gpuserv_gpucontext_list);
	before_shmem_exit(gpuservCleanupOnProcExit, 0);

	/* Open epoll descriptor */
	gpuserv_epoll_fdesc = epoll_create(30);
	if (gpuserv_epoll_fdesc < 0)
		elog(ERROR, "failed on epoll_create: %m");

	/* Init GPU Context for each devices */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", cuStrError(rc));
	PG_TRY();
	{
		for (dindex=0; dindex < numGpuDevAttrs; dindex++)
		{
			gpuContext *gcontext = gpuservSetupGpuContext(dindex);
			dlist_push_tail(&gpuserv_gpucontext_list, &gcontext->chain);
		}

		while (!gpuServiceGoingTerminate())
		{
			struct epoll_event	ep_ev;
			int		status;

			if (!PostmasterIsAlive())
				elog(FATAL, "unexpected postmaster dead");
			CHECK_FOR_INTERRUPTS();

			status = epoll_wait(gpuserv_epoll_fdesc, &ep_ev, 1, 4000);
			if (status < 0)
			{
				if (errno != EINTR)
				{
					elog(LOG, "failed on epoll_wait: %m");
					break;
				}
			}
			else if (status > 0)
			{
				/* errors on server socker? */
				if ((ep_ev.events & ~EPOLLIN) != 0)
					break;
				/* any connection pending? */
				if ((ep_ev.events & EPOLLIN) != 0)
					gpuservAcceptClient((gpuContext *)ep_ev.data.ptr);
			}
		}
	}
	PG_CATCH();
	{
		while (!dlist_is_empty(&gpuserv_gpucontext_list))
		{
			dlist_node *dnode = dlist_pop_head_node(&gpuserv_gpucontext_list);
			gpuContext *gcontext = dlist_container(gpuContext, chain, dnode);
			gpuservCleanupGpuContext(gcontext);
		}
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* cleanup */
	while (!dlist_is_empty(&gpuserv_gpucontext_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gpuserv_gpucontext_list);
		gpuContext *gcontext = dlist_container(gpuContext, chain, dnode);
		gpuservCleanupGpuContext(gcontext);
	}

	/*
	 * If it received only SIGHUP (no SIGTERM), try to restart rather than
	 * shutdown.
	 */
	if (gpuserv_bgworker_got_signal == (1 << SIGHUP))
		proc_exit(1);
}

/*
 * pgstrom_init_gpu_service
 */
void
pgstrom_init_gpu_service(void)
{
	BackgroundWorker worker;

	Assert(numGpuDevAttrs > 0);
	DefineCustomIntVariable("pg_strom.max_async_gpu_tasks",
							"Max number of asynchronous GPU tasks",
							NULL,
							&pgstrom_max_async_gpu_tasks,
							8,
							1,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.load_gpu_debug_module",
							 "Loads GPU debug module",
							 NULL,
							 &pgstrom_load_gpu_debug_module,
							 true,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpu_mempool_segment_sz",
							"Segment size of GPU memory pool",
							NULL,
							&pgstrom_gpu_mempool_segment_sz_kb,
							1048576,	/* 1GB */
							262144,		/* 256MB */
							16777216,	/* 16GB */
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.gpu_mempool_max_ratio",
							 "GPU memory pool: maximum usable ratio for memory pool",
							 NULL,
							 &pgstrom_gpu_mempool_max_ratio,
							 0.50,		/* 50% */
							 0.20,		/* 20% */
							 0.80,		/* 80% */
							 PGC_SIGHUP,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.gpu_mempool_min_ratio",
							 "GPU memory pool: minimum preserved ratio memory pool",
							 NULL,
							 &pgstrom_gpu_mempool_min_ratio,
							 0.05,		/*  8% */
							 0.0,		/*  0% */
							 pgstrom_gpu_mempool_max_ratio,
							 PGC_SIGHUP,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpu_mempool_release_delay",
							"GPU memory pool: time to release device memory segment after the last chunk is released",
							NULL,
							&pgstrom_gpu_mempool_release_delay,
							5000,		/* 5sec */
							1,
							INT_MAX,
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_MS | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);
	
	memset(&worker, 0, sizeof(BackgroundWorker));
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = 5;
	snprintf(worker.bgw_name, BGW_MAXLEN, "PG-Strom GPU Service");
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_strom");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "gpuservBgWorkerMain");
	worker.bgw_main_arg = 0;
	RegisterBackgroundWorker(&worker);
}
