/*
 * perfmon.h
 *
 * Definition of Performance monitor structure
 * --
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#ifndef PERFMON_H
#define PERFMON_H

/*
 * Performance monitor structure
 */
typedef struct {
	cl_bool			enabled;
	cl_bool			prime_in_gpucontext;
	GpuTaskKind		task_kind;
	/*-- memory allocation counter --*/
	cl_uint			num_dmabuf_alloc;
	cl_uint			num_dmabuf_free;
	cl_uint			num_gpumem_alloc;
	cl_uint			num_gpumem_free;
	cl_uint			num_iomapped_alloc;
	cl_uint			num_iomapped_free;
	cl_double		tv_dmabuf_alloc;
	cl_double		tv_dmabuf_free;
	cl_double		tv_gpumem_alloc;
	cl_double		tv_gpumem_free;
	cl_double		tv_iomapped_alloc;
	cl_double		tv_iomapped_free;
	size_t			size_dmabuf_total;
	size_t			size_gpumem_total;
	size_t			size_iomapped_total;
	/*-- message send/recv counter --*/
	cl_double		tv_sendmsg;
	cl_double		tv_recvmsg;
	/*-- build cuda program --*/
	struct timeval	tv_build_start;
	struct timeval	tv_build_end;
	/*-- time for task pending --*/

	/*-- time for I/O stuff --*/
	cl_double		time_inner_load;	/* time to load the inner relation */
	cl_double		time_outer_load;	/* time to load the outer relation */
	cl_double		time_materialize;	/* time to materialize the result */
	/*-- DMA data transfer --*/
	cl_uint			num_dma_send;	/* number of DMA send request */
	cl_uint			num_dma_recv;	/* number of DMA receive request */
	cl_ulong		bytes_dma_send;	/* bytes of DMA send */
	cl_ulong		bytes_dma_recv;	/* bytes of DMA receive */
	cl_double		time_dma_send;	/* time to send host=>device data */
	cl_double		time_dma_recv;	/* time to receive device=>host data */
	/*-- specific items for each GPU logic --*/
	cl_uint			num_tasks;			/* number of tasks completed */
	cl_double		time_launch_cuda;	/* time to kick CUDA commands */
	cl_double		time_sync_tasks;	/* time to synchronize tasks */
	/*-- for each GPU logic --*/
	struct {
		cl_uint		num_kern_main;
		cl_double	tv_kern_main;
		cl_double	tv_kern_exec_quals;
		cl_double	tv_kern_projection;
	} gscan;
	struct {
		cl_uint		num_kern_main;
		cl_uint		num_kern_outer_scan;
		cl_uint		num_kern_exec_nestloop;
		cl_uint		num_kern_exec_hashjoin;
		cl_uint		num_kern_outer_nestloop;
		cl_uint		num_kern_outer_hashjoin;
		cl_uint		num_kern_projection;
		cl_uint		num_kern_rows_dist;
		cl_uint		num_global_retry;
		cl_uint		num_major_retry;
		cl_uint		num_minor_retry;
		cl_double	tv_kern_main;
		cl_double	tv_kern_outer_scan;
		cl_double	tv_kern_exec_nestloop;
		cl_double	tv_kern_exec_hashjoin;
		cl_double	tv_kern_outer_nestloop;
		cl_double	tv_kern_outer_hashjoin;
		cl_double	tv_kern_projection;
		cl_double	tv_kern_rows_dist;
		/* DMA of inner multi relations */
		cl_uint		num_inner_dma_send;
		cl_ulong	bytes_inner_dma_send;
		cl_double	tv_inner_dma_send;
	} gjoin;
	struct {
		cl_uint		num_kern_main;
		cl_uint		num_kern_prep;
		cl_uint		num_kern_nogrp;
		cl_uint		num_kern_lagg;
		cl_uint		num_kern_gagg;
		cl_uint		num_kern_fagg;
		cl_uint		num_kern_fixvar;
		cl_double	tv_kern_main;
		cl_double	tv_kern_prep;
		cl_double	tv_kern_nogrp;
		cl_double	tv_kern_lagg;
		cl_double	tv_kern_gagg;
		cl_double	tv_kern_fagg;
		cl_double	tv_kern_fixvar;
	} gpreagg;
	struct {
		cl_uint		num_kern_proj;
		cl_uint		num_kern_main;
		cl_uint		num_kern_lsort;	/* gpusort_bitonic_local */
		cl_uint		num_kern_ssort;	/* gpusort_bitonic_step */
		cl_uint		num_kern_msort;	/* gpusort_bitonic_merge */
		cl_uint		num_kern_fixvar;/* gpusort_fixup_pointers */
		cl_double	tv_kern_proj;
		cl_double	tv_kern_main;
		cl_double	tv_kern_lsort;
		cl_double	tv_kern_ssort;
		cl_double	tv_kern_msort;
		cl_double	tv_kern_fixvar;
		cl_double	tv_cpu_sort;
	} gsort;
} pgstrom_perfmon;

/*
 * pgstromWorkerStatistics
 *
 * PostgreSQL v9.6 does not allow to assign run-time statistics counter
 * on the DSM area because of oversight during the development cycle.
 * So, we need to allocate independent shared memory are to write back
 * performance counter to the master backend.
 * Likely, this restriction shall be removed at PostgreSQL v10.
 */
typedef struct
{
	slock_t			lock;
	Instrumentation	worker_instrument;
	pgstrom_perfmon	worker_pfm;
	/* for GpuJoin */
	struct {
		size_t		inner_nitems;
		size_t		right_nitems;
	} gpujoin[FLEXIBLE_ARRAY_MEMBER];
} pgstromWorkerStatistics;

/*
 * macro definitions for performance counter
 */
#define PFMON_BEGIN(pfm,tv1)					\
	do {										\
		if ((pfm)->enabled)						\
			gettimeofday((tv1), NULL);			\
	} while(0)

#define PFMON_END(pfm,field,tv1,tv2)			\
	do {										\
		if ((pfm)->enabled)						\
		{										\
			gettimeofday((tv2), NULL);			\
			(pfm)->field +=												\
				((double)(((tv2)->tv_sec - (tv1)->tv_sec) * 1000000L +	\
						  ((tv2)->tv_usec - (tv1)->tv_usec)) / 1000.0);	\
		}										\
	} while(0)

#define PFMON_EVENT_RECORD(node,ev_field,cuda_stream)			\
	do {														\
		if (((GpuTask_v2 *)(node))->perfmon)					\
		{														\
			CUresult __rc = cuEventRecord((node)->ev_field,		\
										  cuda_stream);			\
			if (__rc != CUDA_SUCCESS)							\
				elog(ERROR, "failed on cuEventRecord: %s",		\
					 errorText(__rc));							\
		}														\
	} while(0)

#define PFMON_EVENT_CREATE(node,ev_field)						\
	do {														\
		if (((GpuTask_v2 *)(node))->perfmon)					\
		{														\
			CUresult	__rc = cuEventCreate(&(node)->ev_field,	\
											 CU_EVENT_DEFAULT);	\
			if (__rc != CUDA_SUCCESS)							\
				elog(ERROR, "failed on cuEventCreate: %s",		\
					 errorText(__rc));							\
		}														\
	} while(0)

#define PFMON_EVENT_DESTROY(node,ev_field)						\
	do {														\
		if ((node)->ev_field)									\
		{														\
			CUresult __rc = cuEventDestroy((node)->ev_field);	\
			if (__rc != CUDA_SUCCESS)                           \
				elog(WARNING, "failed on cuEventDestroy: %s",   \
					 errorText(__rc));                          \
			(node)->ev_field = NULL;                            \
		}														\
	} while(0)

#define PFMON_EVENT_ELAPSED(node,tv_field,ev_start,ev_stop)		\
	do {														\
		CUresult	__rc;										\
		float		__elapsed;									\
																\
		if (((GpuTask_v2 *)(node))->perfmon &&					\
			(ev_start) && (ev_stop))							\
		{														\
			__rc = cuEventElapsedTime(&__elapsed,				\
									  (ev_start),				\
									  (ev_stop));				\
			if (__rc == CUDA_SUCCESS)							\
				(node)->tv_field += __elapsed;					\
			else												\
			{													\
				elog(WARNING, "failed on cuEventElapsedTime: %s",	\
					 errorText(__rc));								\
				((GpuTask_v2 *)(node))->perfmon = false;		\
				goto skip_perfmon;								\
			}													\
		}														\
	} while(0)

#define PERFMON_TIMEVAL_DIFF(TV1,TV2)							\
		((cl_double)((TV2.tv_sec * 1000000 + TV2.tv_usec) -		\
					 (TV1.tv_sec * 1000000 + TV1.tv_usec)) / 1000000.0)


#define PFMON_ADD_TIMEVAL(tv_sum,tv1,tv2)						\
	do {														\
		(tv_sum)->tv_sec += (tv2)->tv_sec - (tv1)->tv_sec;		\
		if ((tv2)->tv_usec > (tv1)->tv_usec)					\
			(tv_sum)->tv_usec += (tv2)->tv_usec - (tv1)->tv_usec;	\
		else													\
		{														\
			(tv_sum)->tv_sec--;									\
			(tv_sum)->tv_usec += 1000000 + (tv2)->tv_usec - (tv1)->tv_usec; \
		}														\
	} while(0)

#define PFMON_TIMEVAL_AS_FLOAT(tval)			\
	(((cl_double)(tval)->tv_sec) +				\
	 ((cl_double)(tval)->tv_usec / 1000000.0))

#endif	/* PERFMON_H */
