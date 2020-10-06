/*
 * cufile.c
 *
 * A thin wrapper to call cuFile library functions.
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#include "pg_strom.h"
#include <dlfcn.h>

#ifdef WITH_CUFILE
/*
 * cuFileError - note that it is not a cuFile API
 */
const char *
cuFileError(CUfileError_t rv)
{
	return cufileop_status_error(rv.err);
}

/*
 * cuFileDriverOpen
 */
static CUfileError_t (*p_cuFileDriverOpen)(void) = NULL;

CUfileError_t
cuFileDriverOpen(void)
{
	return p_cuFileDriverOpen();
}

/*
 * cuFileDriverClose
 */
static CUfileError_t (*p_cuFileDriverClose)(void) = NULL;

CUfileError_t
cuFileDriverClose(void)
{
	return p_cuFileDriverClose();
}

/*
 * cuFileDriverGetProperties
 */
static CUfileError_t (*p_cuFileDriverGetProperties)(
	CUfileDrvProps_t *props) = NULL;

CUfileError_t
cuFileDriverGetProperties(CUfileDrvProps_t *props)
{
	return p_cuFileDriverGetProperties(props);
}

/*
 * cuFileDriverSetPollMode
 */
static CUfileError_t (*p_cuFileDriverSetPollMode)(
	bool poll,
	size_t poll_threshold_size) = NULL;

CUfileError_t
cuFileDriverSetPollMode(bool poll, size_t poll_threshold_size)
{
	return p_cuFileDriverSetPollMode(poll, poll_threshold_size);
}

/*
 * cuFileDriverSetMaxDirectIOSize
 */
static CUfileError_t (*p_cuFileDriverSetMaxDirectIOSize)(
	size_t max_direct_io_size) = NULL;

CUfileError_t
cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)
{
	return p_cuFileDriverSetMaxDirectIOSize(max_direct_io_size);
}

/*
 * cuFileDriverSetMaxCacheSize
 */
static CUfileError_t (*p_cuFileDriverSetMaxCacheSize)(
	size_t max_cache_size) = NULL;

CUfileError_t
cuFileDriverSetMaxCacheSize(size_t max_cache_size)
{
	return p_cuFileDriverSetMaxCacheSize(max_cache_size);
}

/*
 * cuFileDriverSetMaxPinnedMemSize
 */
static CUfileError_t (*p_cuFileDriverSetMaxPinnedMemSize)(
	size_t max_pinned_size) = NULL;

CUfileError_t
cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)
{
	return p_cuFileDriverSetMaxPinnedMemSize(max_pinned_size);
}

/*
 * cuFileHandleRegister
 */
static CUfileError_t (*p_cuFileHandleRegister)(
	CUfileHandle_t *fh,
	CUfileDescr_t *descr) = NULL;

CUfileError_t
cuFileHandleRegister(CUfileHandle_t *fh, CUfileDescr_t *descr)
{
	return p_cuFileHandleRegister(fh, descr);
}

/*
 * cuFileHandleDeregister
 */
static void (*p_cuFileHandleDeregister)(
	CUfileHandle_t fh) = NULL;

void
cuFileHandleDeregister(CUfileHandle_t fh)
{
	return p_cuFileHandleDeregister(fh);
}

/*
 * cuFileBufRegister
 */
static CUfileError_t (*p_cuFileBufRegister)(
	const void *devPtr_base,
	size_t length,
	int flags) = NULL;

CUfileError_t
cuFileBufRegister(const void *devPtr_base, size_t length, int flags)
{
	return p_cuFileBufRegister(devPtr_base, length, flags);
}

/*
 * cuFileBufDeregister
 */
static CUfileError_t (*p_cuFileBufDeregister)(
	const void *devPtr_base) = NULL;

CUfileError_t cuFileBufDeregister(const void *devPtr_base)
{
	return p_cuFileBufDeregister(devPtr_base);
}

/*
 * cuFileRead
 */
static ssize_t (*p_cuFileRead)(
	CUfileHandle_t fh,
	void *devPtr_base,
	size_t size,
	off_t file_offset,
	off_t devPtr_offset) = NULL;

ssize_t
cuFileRead(CUfileHandle_t fh,
		   void *devPtr_base,
		   size_t size,
		   off_t file_offset,
		   off_t devPtr_offset)
{
	return p_cuFileRead(fh, devPtr_base, size, file_offset, devPtr_offset);
}

/*
 * cuFileWrite
 */
static ssize_t (*p_cuFileWrite)(
	CUfileHandle_t fh,
	const void *devPtr_base,
	size_t size,
	off_t file_offset,
	off_t devPtr_offset) = NULL;

ssize_t cuFileWrite(CUfileHandle_t fh,
					const void *devPtr_base,
					size_t size,
					off_t file_offset,
					off_t devPtr_offset)
{
	return p_cuFileWrite(fh,devPtr_base,size,file_offset,devPtr_offset);
}

/*
 * __cuFileReadAsync - an alternative implementation unless the official
 * distribution does not provide cuFileReadAsync API
 */
#define CUFILE_NUM_ASYNC_IO_THREADS		12
#define CUFILE_ASYNC_IO_THREADS_MASK	((1UL << CUFILE_NUM_ASYNC_IO_THREADS) - 1)
static pthread_mutex_t	cufile_async_io_mutex;
static pthread_cond_t	cufile_async_io_cond;
static Datum			cufile_async_io_threads;
static dlist_head		cufile_async_io_queue;
static int				cufile_async_io_unitsz;		/* I/O size in kB */

/* GUC checker */
static bool
cufile_async_io_unitsz_checker(int *p_newval, void **extra, GucSource source)
{
	int		newval = *p_newval;

	if ((newval & (newval - 1)) != 0)
		elog(ERROR, "pg_strom.cufile_io_unitsz must be power of 2");
	return true;
}

typedef struct
{
	CUcontext		cuda_context;
	pthread_mutex_t	lock;
	pthread_cond_t	cond;
	int				refcnt;
	int				errcode;
	char			errmsg[160];
} cufile_async_io_state;

typedef struct
{
	dlist_node		chain;
	CUfileHandle_t	fhandle;
	off_t			file_offset;
	size_t			bytesize;
	CUdeviceptr		devptr_base;
	off_t			devptr_offset;
	cufile_async_io_state *async_io_state;
} cufile_async_io_request;

static void *
__cuFileAsyncIOThread(void *__arg)
{
	cl_int		thread_id = PointerGetDatum(__arg);

	pthreadMutexLock(&cufile_async_io_mutex);
	for (;;)
	{
		cufile_async_io_state *async_io_state;
		cufile_async_io_request *req;
		dlist_node *dnode;
		ssize_t		nbytes;
		CUresult	rc;

		if (!dlist_is_empty(&cufile_async_io_queue))
		{
			dnode = dlist_pop_head_node(&cufile_async_io_queue);
			req = dlist_container(cufile_async_io_request, chain, dnode);
			pthreadMutexUnlock(&cufile_async_io_mutex);

			async_io_state = req->async_io_state;
			rc = cuCtxSetCurrent(async_io_state->cuda_context);

			if (rc != CUDA_SUCCESS)
			{
				pthreadMutexLock(&async_io_state->lock);
				if (async_io_state->errcode == 0)
				{
					async_io_state->errcode = EINVAL;
					snprintf(async_io_state->errmsg, 160,
							 "failed on cuCtxSetCurrent: %s", errorText(rc));
				}
			}
			else
			{
				nbytes = cuFileRead(req->fhandle,
									(void *)req->devptr_base,
									req->bytesize,
									req->file_offset,
									req->devptr_offset);
				pthreadMutexLock(&async_io_state->lock);
				if (nbytes != req->bytesize)
				{
					if (async_io_state->errcode == 0)
					{
						async_io_state->errcode = EIO;
						if (nbytes < 0)
						{
							snprintf(async_io_state->errmsg, 160,
									 "failed on cuFileRead: %s",
									 cufileop_status_error(-nbytes));
						}
						else
						{
							snprintf(async_io_state->errmsg, 160,
									 "cuFileRead returned %zu for %zu bytes",
									 nbytes, req->bytesize);
						}
					}
				}
			}
			Assert(async_io_state->refcnt > 0);
			async_io_state->refcnt--;
			if (async_io_state->refcnt == 0)
				pthreadCondBroadcast(&async_io_state->cond);
			pthreadMutexUnlock(&async_io_state->lock);
			/* unbind the CUDA context */
			cuCtxSetCurrent(NULL);

			free(req);

			pthreadMutexLock(&cufile_async_io_mutex);
		}
		else if (!pthreadCondWaitTimeout(&cufile_async_io_cond,
										 &cufile_async_io_mutex,
										 10000L))
		{
			if (dlist_is_empty(&cufile_async_io_queue))
			{
				cufile_async_io_threads &= ~(1UL << thread_id);
				break;
			}
		}
	}
	pthreadMutexUnlock(&cufile_async_io_mutex);
	return NULL;
}

void *
__cuFileReadAsync(CUfileHandle_t fhandle,
				  CUdeviceptr devptr_base,
				  off_t devptr_offset,
				  strom_io_chunk *io_chunk,
				  void *__async_io_state)
{
	cufile_async_io_state   *async_io_state = __async_io_state;
	cufile_async_io_request *req;
	pthread_t	thread;
	Datum		mask;
	int			units = ((Datum)cufile_async_io_unitsz << 10) / PAGE_SIZE;
	int			i, j;
	size_t		sz;
	off_t		file_offset;
	int			errcode = 0;
	int			nr_reqs = 0;
	dlist_head	req_list;

	/* setup a state object at the first call */
	if (!async_io_state)
	{
		async_io_state = calloc(1, sizeof(cufile_async_io_state));
		if (!async_io_state)
			return NULL;	/* out of memory */
		async_io_state->cuda_context = CU_CONTEXT_PER_THREAD;
		pthreadMutexInit(&async_io_state->lock, 0);
		pthreadCondInit(&async_io_state->cond, 0);
		async_io_state->refcnt = 0;
	}
	else
	{
		Assert(async_io_state->cuda_context == CU_CONTEXT_PER_THREAD);
	}

	/* setup individual i/o request */
	file_offset = PAGE_SIZE * io_chunk->fchunk_id;
	devptr_offset += io_chunk->m_offset;
	dlist_init(&req_list);
	for (i=0; i < io_chunk->nr_pages; i += units)
	{
		req = calloc(1, sizeof(cufile_async_io_request));
		if (!req)
		{
			errcode = errno;
			goto error;
		}
		sz = Min(io_chunk->nr_pages - i, units) * PAGE_SIZE;
		req->fhandle = fhandle;
		req->file_offset = file_offset;
		req->bytesize = sz;
		req->devptr_base = devptr_base;
		req->devptr_offset = devptr_offset;
		req->async_io_state = async_io_state;

		file_offset += sz;
		devptr_offset += sz;
		nr_reqs++;

		dlist_push_tail(&req_list, &req->chain);
	}
	/* enqueue the request message */
	pthreadMutexLock(&cufile_async_io_mutex);
	dlist_append_tail(&cufile_async_io_queue, &req_list);
	pthreadMutexLock(&async_io_state->lock);
	async_io_state->refcnt += nr_reqs;
	pthreadMutexUnlock(&async_io_state->lock);

	/* launch worker threads if not active */
	if (cufile_async_io_threads != CUFILE_ASYNC_IO_THREADS_MASK)
	{
		for (j=0, mask=1; j < CUFILE_NUM_ASYNC_IO_THREADS; j++, mask *= 2)
		{
			if ((cufile_async_io_threads & mask) == 0)
			{
				errcode = pthread_create(&thread, NULL,
										 __cuFileAsyncIOThread,
										 (void *)Int32GetDatum(j));
				if (errcode != 0)
					goto error_locked;
				cufile_async_io_threads |= mask;
			}
		}
	}
	/* wake up a worker thread to invoke cuFileRead */
	if (nr_reqs > 1)
		pthreadCondBroadcast(&cufile_async_io_cond);
	else
		pthreadCondSignal(&cufile_async_io_cond);
	pthreadMutexUnlock(&cufile_async_io_mutex);

	return async_io_state;

error:
	pthreadMutexLock(&cufile_async_io_mutex);
error_locked:
	/* Remove pending request and set error status */
	while (!dlist_is_empty(&cufile_async_io_queue))
	{
		cufile_async_io_state  *io_state;
		dlist_node	   *dnode = dlist_pop_head_node(&cufile_async_io_queue);

		req = dlist_container(cufile_async_io_request, chain, dnode);
		io_state = req->async_io_state;
		pthreadMutexLock(&io_state->lock);
		if (io_state->errcode == 0)
		{
			io_state->errcode = errcode;
			snprintf(io_state->errmsg, 160,
					 "failed on __cuFileReadAsync: %s", strerror(errcode));
		}
		Assert(io_state->refcnt > 0);
		io_state->refcnt--;
		if (io_state->refcnt == 0)
			pthreadCondBroadcast(&io_state->cond);
		pthreadMutexUnlock(&io_state->lock);

		free(req);
	}
	pthreadMutexUnlock(&cufile_async_io_mutex);
	/* Is a first call? */
	if (!__async_io_state)
		free(async_io_state);
	return NULL;
}

void
__cuFileReadWait(void *__async_io_state)
{
	cufile_async_io_state  *async_io_state = __async_io_state;
	cufile_async_io_state	temp;

	if (!async_io_state)
		return;

	pthreadMutexLock(&async_io_state->lock);
	while (async_io_state->refcnt > 0)
	{
		pthreadCondWait(&async_io_state->cond,
						&async_io_state->lock);
	}
	pthreadMutexUnlock(&async_io_state->lock);

	if (async_io_state->errcode != 0)
	{
		memcpy(&temp, async_io_state, sizeof(cufile_async_io_state));
		free(async_io_state);
		werror("failed on __cuFileReadAsync[%s]", temp.errmsg);
	}
	free(async_io_state);
}

/*
 * lookup_cufile_function
 */
static void *
lookup_cufile_function(void *handle, const char *func_name)
{
	void   *func_addr = dlsym(handle, func_name);

	if (!func_addr)
		elog(ERROR, "could not find cuFile symbol \"%s\" - %s",
			 func_name, dlerror());
	return func_addr;
}
#endif

#define LOOKUP_CUFILE_FUNCTION(func_name)		\
	p_##func_name = lookup_cufile_function(handle, #func_name)

/*
 * pgstrom_init_cufile
 */
void
pgstrom_init_cufile(void)
{
#ifdef WITH_CUFILE
	char		namebuf[MAXPGPATH];
	void	   *handle;

	/* version attached on the production release? */
	snprintf(namebuf, sizeof(namebuf),
			 "/usr/local/gds/lib/libcufile.so");
	handle = dlopen(namebuf, RTLD_NOW | RTLD_LOCAL);
	if (!handle)
	{
		elog(LOG, "unable to open '%s', cuFile is disabled: %m", namebuf);
		return;
	}

	PG_TRY();
	{
		LOOKUP_CUFILE_FUNCTION(cuFileDriverOpen);
		LOOKUP_CUFILE_FUNCTION(cuFileDriverClose);
		LOOKUP_CUFILE_FUNCTION(cuFileDriverGetProperties);
		LOOKUP_CUFILE_FUNCTION(cuFileDriverSetPollMode);
		LOOKUP_CUFILE_FUNCTION(cuFileDriverSetMaxDirectIOSize);
		LOOKUP_CUFILE_FUNCTION(cuFileDriverSetMaxCacheSize);
		LOOKUP_CUFILE_FUNCTION(cuFileDriverSetMaxPinnedMemSize);

		LOOKUP_CUFILE_FUNCTION(cuFileHandleRegister);
		LOOKUP_CUFILE_FUNCTION(cuFileHandleDeregister);
		LOOKUP_CUFILE_FUNCTION(cuFileBufRegister);
		LOOKUP_CUFILE_FUNCTION(cuFileBufDeregister);
		LOOKUP_CUFILE_FUNCTION(cuFileRead);
		LOOKUP_CUFILE_FUNCTION(cuFileWrite);
	}
	PG_CATCH();
	{
		dlclose(handle);

		p_cuFileDriverOpen = NULL;
		p_cuFileDriverClose = NULL;
		p_cuFileDriverGetProperties = NULL;
		p_cuFileDriverSetPollMode = NULL;
		p_cuFileDriverSetMaxDirectIOSize = NULL;
		p_cuFileDriverSetMaxCacheSize = NULL;
		p_cuFileDriverSetMaxPinnedMemSize = NULL;
		p_cuFileHandleRegister = NULL;
		p_cuFileHandleDeregister = NULL;
		p_cuFileBufRegister = NULL;
		p_cuFileBufDeregister = NULL;
		p_cuFileRead = NULL;
		p_cuFileWrite = NULL;

		elog(LOG, "failed on lookup cuFile symbols, cuFile is disabled.");
		FlushErrorState();
	}
	PG_END_TRY();

	DefineCustomIntVariable("pg_strom.cufile_io_unitsz",
							"I/O size on cuFileRead invocations",
							"Note that this parameter may be removed in the future version without notifications",
							&cufile_async_io_unitsz,
							2048,		/* 2MB */
							128,		/* 128kB */
							16384,		/* 16MB */
							PGC_SUSET,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							cufile_async_io_unitsz_checker, NULL, NULL);
	
	/* init for __cuFileReadAsync */
	pthreadMutexInit(&cufile_async_io_mutex, 0);
	pthreadCondInit(&cufile_async_io_cond, 0);
	cufile_async_io_threads = 0;
	dlist_init(&cufile_async_io_queue);
#endif /* WITH_CUFILE */
}

