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
} gpuModule;

typedef struct
{
	pthread_mutex_t	lock;
	size_t			total_sz;	/* total pool size */
	size_t			hard_limit;
	size_t			keep_limit;
	dlist_head		segment_list;
} gpuMemoryPool;

typedef struct
{
	dlist_node		chain;
	int				serv_fd;		/* for accept(2) */
	int				epoll_fd;		/* for epoll(2) */
	int				event_fd;		/* to wakeup threads */
	int				cuda_dindex;
	CUdevice		cuda_device;
	CUcontext		cuda_context;
	gpuModule		normal;			/* optimized kernel */
	gpuModule		debug;			/* non-optimized debug kernel */
	gpuMemoryPool	pool;
	/* GPU client management */
	pthread_mutex_t	lock;			/* protect the list below */
	dlist_head		client_list;	/* list of gpuClient */
	dlist_head		task_list;		/* list of gpuTask */
	volatile bool	terminate_workers; /* true, to terminate workers */
	pthread_t		workers[1];
} gpuContext;
#define SizeOfGpuContext		\
	offsetof(gpuContext, workers[pgstrom_max_async_gpu_tasks])
#define EPOLL_FLAGS__SERVER_FD		(EPOLLIN|EPOLLET|EPOLLONESHOT)
#define EPOLL_FLAGS__EVENT_FD_ONE	(EPOLLIN|EPOLLET)
#define EPOLL_FLAGS__EVENT_FD_ANY	(EPOLLIN)
#define EPOLL_FLAGS__CLIENT_FD		(EPOLLIN|EPOLLRDHUP|EPOLLET|EPOLLONESHOT)
#define EPOLL_DPTR__SERVER_FD		(NULL)
#define EPOLL_DPTR__EVENT_FD		((void *)(~0UL))

typedef struct
{
	dlist_node		chain;
	gpuContext	   *gcontext;
	pg_atomic_uint32 refcnt;
	//memo: state information like kern_gpuscan with kparams
	//      will be saved here.
	pthread_mutex_t	mutex;		/* mutex to write the socket */
	int				sockfd;		/* connection to PG backend */
} gpuClient;

/*
 * variables
 */
int		pgstrom_max_async_gpu_tasks;	/* GUC */
bool	pgstrom_load_gpu_debug_module;	/* GUC */
static __thread gpuContext *GpuWorkerCurrentContext = NULL;
#define CU_CONTEXT_PER_THREAD	(GpuWorkerCurrentContext->cuda_context)
#define CU_DEVICE_PER_THREAD	(GpuWorkerCurrentContext->cuda_device)
static volatile int		gpuserv_bgworker_got_signal = 0;
static dlist_head		gpuserv_gpucontext_list;

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
} gpuMemorySegment;

typedef struct
{
	dlist_node		free_chain;
	dlist_node		addr_chain;
	size_t			offset;
	size_t			length;
} gpuMemoryChunk;

static CUresult
__gpuMemAllocFromSegment(gpuMemoryPool *pool,
						 gpuMemorySegment *mseg,
						 CUdeviceptr *dptr, size_t bytesize)
{
	gpuMemoryChunk *chunk;
	dlist_iter		iter;

	dlist_foreach(iter, &mseg->free_chunks)
	{
		chunk = dlist_container(gpuMemoryChunk, free_chain, iter.cur);
		if (bytesize <= chunk->length)
		{
			size_t	surplus = chunk->length - bytesize;

			/* try to split, if free chunk is enough large (>1MB) */
			if (surplus > (1UL << 20))
			{
				gpuMemoryChunk *buddy = calloc(1, sizeof(gpuMemoryChunk));

				if (!buddy)
					return CUDA_ERROR_OUT_OF_MEMORY;
				chunk->length -= surplus;
				buddy->offset = chunk->offset + chunk->length;
				buddy->length = surplus;
				dlist_insert_after(&chunk->free_chain, &buddy->free_chain);
				dlist_insert_after(&chunk->addr_chain, &buddy->addr_chain);
			}
			/* mark it as an active chunk */
			dlist_delete(&chunk->free_chain);
			memset(&chunk->free_chain, 0, sizeof(dlist_node));
			mseg->active_sz += chunk->length;

			*dptr = (mseg->devptr + chunk->offset);

			/* update the LRU ordered segment list and timestamp */
			gettimeofday(&mseg->tval, NULL);
			dlist_move_head(&pool->segment_list, &mseg->chain);
			return CUDA_SUCCESS;
		}
	}
	return CUDA_ERROR_OUT_OF_MEMORY;
}

static gpuMemorySegment *
__gpuMemAllocNewSegment(gpuMemoryPool *pool, size_t segment_sz)
{
	gpuMemorySegment *mseg = calloc(1, sizeof(gpuMemorySegment));
	gpuMemoryChunk *chunk = calloc(1, sizeof(gpuMemoryChunk));
	CUresult		rc;

	if (!mseg || !chunk)
		goto error;
	mseg->segment_sz = segment_sz;
	mseg->active_sz = 0;
	dlist_init(&mseg->free_chunks);
	dlist_init(&mseg->addr_chunks);

	rc = cuMemAlloc(&mseg->devptr, mseg->segment_sz);
	if (rc != CUDA_SUCCESS)
		goto error;

	chunk->offset = 0;
	chunk->length = segment_sz;
	dlist_push_head(&mseg->free_chunks, &chunk->free_chain);
	dlist_push_head(&mseg->addr_chunks, &chunk->addr_chain);

	pool->total_sz += segment_sz;

	return mseg;
error:
	if (mseg)
		free(mseg);
	if (chunk)
		free(chunk);
	return NULL;
}

CUresult
gpuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
	gpuMemoryPool  *pool = &GpuWorkerCurrentContext->pool;
	dlist_iter		iter;
	CUresult		rc;
	size_t			segment_sz;

	bytesize = MAXALIGN(bytesize);
	pthreadMutexLock(&pool->lock);
	dlist_foreach(iter, &pool->segment_list)
	{
		gpuMemorySegment *mseg = dlist_container(gpuMemorySegment,
												 chain, iter.cur);
		if (mseg->active_sz + bytesize <= mseg->segment_sz)
		{
			rc = __gpuMemAllocFromSegment(pool, mseg, dptr, bytesize);
			if (rc == CUDA_SUCCESS)
				goto out_unlock;
		}
	}
	segment_sz = ((size_t)pgstrom_gpu_mempool_segment_sz_kb << 10);
	if (segment_sz < bytesize)
		segment_sz = bytesize;
	rc = CUDA_ERROR_OUT_OF_MEMORY;
	if (pool->total_sz + segment_sz <= pool->hard_limit)
	{
		gpuMemorySegment *mseg = __gpuMemAllocNewSegment(pool, segment_sz);

		if (mseg)
			rc = __gpuMemAllocFromSegment(pool, mseg, dptr, bytesize);
	}
out_unlock:	
	pthreadMutexUnlock(&pool->lock);

	return rc;
}

static CUresult
__gpuMemFree(gpuMemoryPool *pool,
			 gpuMemorySegment *mseg,
			 CUdeviceptr devptr)
{
	dlist_iter		iter;
	gpuMemoryChunk *chunk;
	gpuMemoryChunk *buddy;
	dlist_node	   *dnode;

	dlist_foreach(iter, &mseg->addr_chunks)
	{
		chunk = dlist_container(gpuMemoryChunk, addr_chain, iter.cur);
		if (devptr == mseg->devptr + chunk->offset)
		{
			Assert(mseg->active_sz >= chunk->length);
			mseg->active_sz -= chunk->length;
			Assert(!chunk->free_chain.prev && !chunk->free_chain.next);
			dlist_push_head(&mseg->free_chunks, &chunk->free_chain);

			/* merge if next chunk is also free */
			if (dlist_has_next(&mseg->addr_chunks, &chunk->addr_chain))
			{
				dnode = dlist_next_node(&mseg->addr_chunks,
										&chunk->addr_chain);
				buddy = dlist_container(gpuMemoryChunk, addr_chain, dnode);
				if (buddy->free_chain.prev && buddy->addr_chain.next)
				{
					Assert(buddy->offset == chunk->offset + chunk->length);
					dlist_delete(&buddy->free_chain);
					dlist_delete(&buddy->addr_chain);
					chunk->length += buddy->length;
					free(buddy);
				}
			}
			/* merge if prev chunk is also free */
			if (dlist_has_prev(&mseg->addr_chunks, &chunk->addr_chain))
			{
				dnode = dlist_prev_node(&mseg->addr_chunks,
										&chunk->addr_chain);
				buddy = dlist_container(gpuMemoryChunk, addr_chain, dnode);
				if (buddy->free_chain.prev && buddy->addr_chain.next)
				{
					Assert(chunk->offset == buddy->offset + buddy->length);
					dlist_delete(&chunk->free_chain);
					dlist_delete(&chunk->addr_chain);
					buddy->length += chunk->length;
					free(chunk);
				}
			}
			/* update the LRU ordered segment list and timestamp */
			gettimeofday(&mseg->tval, NULL);
			dlist_move_head(&pool->segment_list, &mseg->chain);
			return CUDA_SUCCESS;
		}
	}
	/* not found */
	return CUDA_ERROR_INVALID_VALUE;
}

CUresult
gpuMemFree(CUdeviceptr devptr)
{
	gpuMemoryPool  *pool = &GpuWorkerCurrentContext->pool;
	dlist_iter		iter;
	CUresult		rc = CUDA_ERROR_INVALID_VALUE;

	pthreadMutexLock(&pool->lock);
	dlist_foreach (iter, &pool->segment_list)
	{
		gpuMemorySegment *mseg = dlist_container(gpuMemorySegment,
												 chain, iter.cur);
		if (devptr >= mseg->devptr &&
			devptr <  mseg->devptr + mseg->segment_sz)
		{
			rc = __gpuMemFree(pool, mseg, devptr);
			break;
		}
	}
	pthreadMutexUnlock(&pool->lock);

	return rc;
}

static void
gpuMemoryPoolMaintenance(void)
{
	gpuMemoryPool  *pool = &GpuWorkerCurrentContext->pool;
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
			rc = cuMemFree(mseg->devptr);
			if (rc != CUDA_SUCCESS)
				__FATAL("failed on cuMemFree: %s", cuStrError(rc));
			/* detach segment */
			dlist_delete(&mseg->chain);
			while (!dlist_is_empty(&mseg->addr_chunks))
			{
				dlist_node	   *dnode = dlist_pop_head_node(&mseg->addr_chunks);
				gpuMemoryChunk *chunk = dlist_container(gpuMemoryChunk,
														addr_chain, dnode);
				Assert(chunk->free_chain.prev &&
					   chunk->free_chain.next);
				free(chunk);
			}
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
 * gpuClientGet
 */
static inline void
gpuClientGet(gpuClient *gclient)
{
	pg_atomic_add_fetch_u32(&gclient->refcnt, 1);
}

/*
 * gpuClientPut
 */
static void
gpuClientPut(gpuClient *gclient)
{
	if (pg_atomic_sub_fetch_u32(&gclient->refcnt, 1) == 0)
	{
		gpuContext *gcontext = gclient->gcontext;

		pthreadMutexLock(&gcontext->lock);
		dlist_delete(&gclient->chain);
		pthreadMutexUnlock(&gcontext->lock);

		if (epoll_ctl(gcontext->epoll_fd,
					  EPOLL_CTL_DEL,
					  gclient->sockfd, NULL) != 0)
			__FATAL("failed on epoll_ctl(EPOLL_CTL_DEL): %m\n");
		fprintf(stderr, "close the client socket!\n");
		close(gclient->sockfd);
		free(gclient);
	}
}

/*
 * gpuservAcceptConnection
 */
static void
gpuservAcceptConnection(gpuContext *gcontext)
{
	gpuClient  *gclient = NULL;
	int			sockfd;
	struct epoll_event ep_event;

	sockfd = accept(gcontext->serv_fd, NULL, NULL);
	/* enables serv_fd again */
	ep_event.events = EPOLL_FLAGS__SERVER_FD;
	ep_event.data.ptr = EPOLL_DPTR__SERVER_FD;
	if (epoll_ctl(gcontext->epoll_fd,
				  EPOLL_CTL_MOD,
				  gcontext->serv_fd,
				  &ep_event) != 0)
		__FATAL("failed on epoll_ctl(serv_fd): %m");

	if (sockfd < 0)
	{
		fprintf(stderr, "GPU%d: cound not accept new connection: %m\n",
				gcontext->cuda_dindex);
		pg_usleep(100000L);		/* wait 0.1 sec */
		return;
	}

	gclient = calloc(1, sizeof(gpuClient));
	if (!gclient)
	{
		fprintf(stderr, "out of memory: %m\n");
		close(sockfd);
		return;
	}
	gclient->gcontext = gcontext;
	pg_atomic_init_u32(&gclient->refcnt, 1);
	pthreadMutexInit(&gclient->mutex);
	gclient->sockfd = sockfd;
	pthreadMutexLock(&gcontext->lock);
	dlist_push_tail(&gcontext->client_list, &gclient->chain);
	pthreadMutexUnlock(&gcontext->lock);

	ep_event.events = EPOLL_FLAGS__CLIENT_FD;
	ep_event.data.ptr = gclient;
	if (epoll_ctl(gcontext->epoll_fd,
				  EPOLL_CTL_ADD,
				  gclient->sockfd,
				  &ep_event) != 0)
		__FATAL("failed on epoll_ctl(sockfd): %m");
}

/*
 * gpuservHandleMessage
 */
static void
gpuservHandleMessage(gpuClient *gclient)
{
	char		buf[1024];
	ssize_t		nbytes;

	nbytes = read(gclient->sockfd, buf, 1024);
	fprintf(stderr, "read %zd bytes from client socket\n", nbytes);
}

/*
 * gpuservGpuWorkerMain
 */
static void *
gpuservGpuWorkerMain(void *__arg)
{
	gpuContext *gcontext = (gpuContext *)__arg;

	GpuWorkerCurrentContext = gcontext;
	pg_memory_barrier();
	
	while (!gcontext->terminate_workers)
	{
		struct epoll_event event;
		int			nevents;
#if 0
		pthreadMutexLock(&gcontext->lock);
		//check gcontext->task_list, and fetch one if any
		pthreadMutexUnlock(&gcontext->lock);

		if (gtask)
		{
			//handle one gpuTask
			continue;
		}
#endif
		nevents = epoll_wait(gcontext->epoll_fd, &event, 1, 15000);
		if (nevents < 0)
		{
			if (errno == EINTR)
			{
				fprintf(stderr, "epoll_wait = %d (errno %m)\n", nevents);
				continue;
			}
			fprintf(stderr, "epoll_wait = %d (errno %m), exit\n", nevents);
			break;
		}
		else if (nevents > 0)
		{
			if (event.data.ptr == EPOLL_DPTR__SERVER_FD)
			{
				 fprintf(stderr, "epoll wakeup by server socket\n");

				 gpuservAcceptConnection(gcontext);
			}
			else if (event.data.ptr == EPOLL_DPTR__EVENT_FD)
			{
				fprintf(stderr, "epoll wakeup by eventfd\n");
			}
			else if (event.events & EPOLLRDHUP)
			{
				fprintf(stderr, "epoll shutdown\n");
				gpuClientPut((gpuClient *)event.data.ptr);
			}
			else
			{
				gpuClient  *gclient = event.data.ptr;

				if (event.events & EPOLLIN)
					gpuservHandleMessage(gclient);
				else
					fprintf(stderr, "unexpected epoll event: %08x\n", event.events);

				event.events = EPOLL_FLAGS__CLIENT_FD;
				if (epoll_ctl(gcontext->epoll_fd,
							  EPOLL_CTL_MOD,
							  gclient->sockfd, &event) != 0)
					__FATAL("failed on epoll_ctl(sockfd): %m");
			}
		}
		else
		{
			/* maintenance works */
			gpuMemoryPoolMaintenance();
		}
	}
	return NULL;
}

/*
 * gpuservSetupGpuLinkage
 */
static void
gpuservSetupGpuLinkage(gpuModule *gmodule)
{
	HASHCTL		hctl;
	HTAB	   *cuda_type_htab = NULL;
	HTAB	   *cuda_func_htab = NULL;
	xpu_type_catalog_entry *xpu_types_catalog;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;
	int			i;
	
	/* build device type table */
	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = sizeof(TypeOpCode);
	hctl.entrysize = sizeof(xpu_type_catalog_entry);
	hctl.hcxt = TopMemoryContext;
	cuda_type_htab = hash_create("CUDA device type hash table",
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
		xpu_datum_operators *type_ops = xpu_types_catalog[i].type_ops;
		xpu_type_catalog_entry *entry;
		bool		found;

		entry = hash_search(cuda_type_htab, &type_opcode, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? duplicated TypeOpCode: %u", (uint32_t)type_opcode);
		entry->type_opcode = type_opcode;
		entry->type_ops = type_ops;
	}
	//TODO: Extra device types


	
	gmodule->cuda_type_htab = cuda_type_htab;
	gmodule->cuda_func_htab = cuda_func_htab;
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

	gpuservSetupGpuLinkage(gmodule);
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
	struct epoll_event ep_event;
	int64_t		one = 1;

	gcontext->terminate_workers = true;
	pg_memory_barrier();

	ep_event.events = EPOLL_FLAGS__EVENT_FD_ANY;
	ep_event.data.ptr = EPOLL_DPTR__EVENT_FD;
	if (epoll_ctl(gcontext->epoll_fd,
				  EPOLL_CTL_MOD,
				  gcontext->event_fd, &ep_event) != 0)
		__FATAL("failed on epoll_ctl(event_fd): %m\n");
	write(gcontext->event_fd, &one, sizeof(int64_t));
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
	int			i, status;
	struct sockaddr_un addr;
	struct epoll_event ep_ev;

	/* gpuContext allocation */
	gcontext = calloc(1, SizeOfGpuContext);
	if (!gcontext)
		elog(ERROR, "out of memory");
	gcontext->cuda_dindex = cuda_dindex;
	gcontext->serv_fd = -1;
	gcontext->epoll_fd = -1;
	gcontext->event_fd = -1;
	gpuMemoryPoolInit(&gcontext->pool, dattrs->DEV_TOTAL_MEMSZ);
	pthreadMutexInit(&gcontext->lock);
	dlist_init(&gcontext->client_list);
	dlist_init(&gcontext->task_list);

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

		/* Setup raw CUDA context */
		rc = cuDeviceGet(&gcontext->cuda_device, dattrs->DEV_ID);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", cuStrError(rc));
		rc = cuCtxCreate(&gcontext->cuda_context,
						 CU_CTX_SCHED_AUTO,
						 gcontext->cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s", cuStrError(rc));
		gpuservSetupGpuModule(&gcontext->normal, false);
		if (pgstrom_load_gpu_debug_module)
			gpuservSetupGpuModule(&gcontext->debug, true);

		/* creation of epoll */
		gcontext->epoll_fd = epoll_create(32);
		if (gcontext->epoll_fd < 0)
			elog(ERROR, "failed on epoll_create: %m");
		/* register the server socket */
		ep_ev.events = EPOLL_FLAGS__SERVER_FD;
		ep_ev.data.ptr = EPOLL_DPTR__SERVER_FD;
		if (epoll_ctl(gcontext->epoll_fd,
					  EPOLL_CTL_ADD,
					  gcontext->serv_fd, &ep_ev) != 0)
			elog(ERROR, "failed on epoll_ctl: %m");
		/* creation of eventfd */
		gcontext->event_fd = eventfd(0, 0);
		if (gcontext->event_fd < 0)
			elog(ERROR, "failed on eventfd: %m");

		ep_ev.events = EPOLL_FLAGS__EVENT_FD_ONE;
		ep_ev.data.ptr = EPOLL_DPTR__EVENT_FD;
		if (epoll_ctl(gcontext->epoll_fd,
					  EPOLL_CTL_ADD,
					  gcontext->event_fd, &ep_ev) != 0)
			elog(ERROR, "failed on epoll_ctl: %m");

		/* launch the worker threads */
		for (i=0; i < pgstrom_max_async_gpu_tasks; i++)
		{
			if ((status = pthread_create(&gcontext->workers[i], NULL,
										 gpuservGpuWorkerMain, gcontext)) != 0)
			{
				__gpuContextTerminateWorkers(gcontext);
				while (i > 0)
					pthread_join(gcontext->workers[--i], NULL);
				elog(ERROR, "failed on pthread_create: %s", strerror(status));
			}
		}
	}
	PG_CATCH();
	{
		if (gcontext->cuda_context)
			cuCtxDestroy(gcontext->cuda_context);
		if (gcontext->event_fd >= 0)
			close(gcontext->event_fd);
		if (gcontext->epoll_fd >= 0)
			close(gcontext->epoll_fd);
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
	int			i;

	__gpuContextTerminateWorkers(gcontext);
	for (i=0; i < pgstrom_max_async_gpu_tasks; i++)
	{
		if (pthread_join(gcontext->workers[i], NULL) != 0)
			elog(LOG, "failed on pthread_join: %m");
	}

	while (!dlist_is_empty(&gcontext->client_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gcontext->client_list);
		gpuClient  *gclient = dlist_container(gpuClient, chain, dnode);

		if (close(gclient->sockfd) != 0)
			elog(LOG, "failed on close(sockfd): %m");
	}
	if (close(gcontext->serv_fd) != 0)
		elog(LOG, "failed on close(serv_fd): %m");
	if (close(gcontext->epoll_fd) != 0)
		elog(LOG, "failed on close(epoll_fd): %m");
	if (close(gcontext->event_fd) != 0)
		elog(LOG, "failed on close(event_fd): %m");
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

		while (!gpuserv_bgworker_got_signal)
		{
			int		ev = WaitLatch(MyLatch,
								   WL_LATCH_SET |
								   WL_TIMEOUT |
								   WL_POSTMASTER_DEATH,
								   1000L,
								   PG_WAIT_EXTENSION);
			ResetLatch(MyLatch);
			if (ev & WL_POSTMASTER_DEATH)
				elog(FATAL, "unexpected postmaster dead");
			CHECK_FOR_INTERRUPTS();
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
 * Backend side functions
 * ------------------------------------------------
 */
#define GPUSERV_CONNECTION_NSLOTS		24
static dlist_head	gpuserv_connection_slots[GPUSERV_CONNECTION_NSLOTS];
static int			gpuserv_num_connections = 0;

typedef struct
{
	dlist_node	chain;
	int			cuda_dindex;
	pgsocket	sockfd;
	ResourceOwner resowner;
} gpuservConnTrack;

/*
 * gpuservOpenConnection
 */
pgsocket
gpuservOpenConnection(int cuda_dindex)
{
	struct sockaddr_un addr;
	pgsocket		sockfd;
	dlist_head	   *slot;
	gpuservConnTrack *track;

	if (cuda_dindex < 0 || cuda_dindex >= numGpuDevAttrs)
		elog(ERROR, "GPU%d is not installed", cuda_dindex);

	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");

	addr.sun_family = AF_UNIX;
	snprintf(addr.sun_path, sizeof(addr.sun_path),
			 ".pg_strom.%u.gpu%u.sock",
			 PostmasterPid, cuda_dindex);
	if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
	{
		int		__errno = errno;

		close(sockfd);
		elog(ERROR, "failed on connect('%s'): %s",
			 addr.sun_path, strerror(__errno));
	}
	/* remember the connection */
	track = calloc(1, sizeof(gpuservConnTrack));
	if (!track)
	{
		close(sockfd);
		elog(ERROR, "out of memory");
	}
	track->cuda_dindex = cuda_dindex;
	track->sockfd = sockfd;
	track->resowner = CurrentResourceOwner;
	slot = &gpuserv_connection_slots[sockfd % GPUSERV_CONNECTION_NSLOTS];
	dlist_push_tail(slot, &track->chain);
	gpuserv_num_connections++;

	/* TODO: initial negatiation here? */
	return sockfd;
}

/*
 * gpuservCloseConnection
 */
void
gpuservCloseConnection(pgsocket sockfd)
{
	dlist_iter		iter;
	dlist_head	   *slot
		= &gpuserv_connection_slots[sockfd % GPUSERV_CONNECTION_NSLOTS];
	dlist_foreach (iter, slot)
	{
		gpuservConnTrack *track = dlist_container(gpuservConnTrack,
												  chain, iter.cur);
		if (track->sockfd)
		{
			gpuserv_num_connections--;
			dlist_delete(&track->chain);
			if (close(track->sockfd) != 0)
				elog(LOG, "failed on close(sockfd): %m");
			free(track);
			return;
		}
	}
	elog(ERROR, "socket %d is not tracked as a client of GPU service", sockfd);
}

/*
 * gpuservCleanupConnections
 */
static void
gpuservCleanupConnections(ResourceReleasePhase phase,
						  bool isCommit,
						  bool isTopLevel,
						  void *arg)
{
	int		i;

	if (phase != RESOURCE_RELEASE_BEFORE_LOCKS || gpuserv_num_connections == 0)
		return;
	for (i=0; i < GPUSERV_CONNECTION_NSLOTS; i++)
	{
		dlist_head	   *slot = &gpuserv_connection_slots[i];
		dlist_mutable_iter iter;

		dlist_foreach_modify(iter, slot)
		{
			gpuservConnTrack *track = dlist_container(gpuservConnTrack,
													  chain, iter.cur);
			if (track->resowner == CurrentResourceOwner)
			{
				gpuserv_num_connections--;
				dlist_delete(&track->chain);
				if (isCommit)
					elog(LOG, "Bug? socket %d is not closed on ExecEnd",
						 track->sockfd);
				if (close(track->sockfd) != 0)
					elog(LOG, "failed on close(sockfd): %m");
				free(track);
			}
		}
	}
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
							 0.10,		/* 50% */
							 0.0,		/* 20% */
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

	RegisterResourceReleaseCallback(gpuservCleanupConnections, NULL);
}
