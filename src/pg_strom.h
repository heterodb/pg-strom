/*
 * pg_strom.h
 *
 * Header file of pg_strom module
 *
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef PG_STROM_H
#define PG_STROM_H
#include "commands/explain.h"
#include "fmgr.h"
#include "lib/ilist.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "nodes/relation.h"
#include "storage/fd.h"
#include "storage/latch.h"
#include "storage/lock.h"
#include "storage/proc.h"
#include "storage/spin.h"
#include "utils/resowner.h"
#include <cuda.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <sys/time.h>
#include "cuda_common.h"

/*
 * --------------------------------------------------------------------
 *
 * Configuration sections
 *
 * NOTE: We uses configuration of the host PostgreSQL system, instead of
 * own configure script, not to mismatch prerequisites for module build.
 * However, some (possible) configuration will lead unexpected behavior.
 * So, we put some checks to prevent unexpected host configurations.
 *
 * --------------------------------------------------------------------
 */
#if PG_VERSION_NUM < 90500
#error Not supported PostgreSQL version
#else
/* check only for v9.5 series */
#if PG_VERSION_NUM < 90600
#ifndef PG_USE_INLINE
/* inline function became minimum requirement at v9.6 */
#error PG-Strom expects inline function is supported by compiler
#endif	/* PG_USE_INLINE */
#endif
#endif
#if SIZEOF_DATUM != 8
#error PG-Strom expects 64bit platform
#endif
#ifndef USE_FLOAT4_BYVAL
#error PG-Strom expects float32 is referenced by value, not reference
#endif
#ifndef USE_FLOAT8_BYVAL
#error PG-Strom expexts float64 is referenced by value, not reference
#endif
#ifndef HAVE_INT64_TIMESTAMP
#error PG-Strom expects timestamp has 64bit integer format
#endif

/*
 * --------------------------------------------------------------------
 *
 * Constant Definitions
 *
 * --------------------------------------------------------------------
 */


/*
 * --------------------------------------------------------------------
 *
 * Type Definitions
 *
 * --------------------------------------------------------------------
 */

/*
 * Performance monitor structure
 */
typedef struct {
	cl_bool		enabled;
	cl_uint		num_samples;
	/*-- perfmon to load and materialize --*/
	cl_double	time_inner_load;	/* time to load the inner relation */
	cl_double	time_outer_load;	/* time to load the outer relation */
	cl_double	time_materialize;	/* time to materialize the result */
	/*-- perfmon to launch CUDA kernel --*/
	cl_double	time_launch_cuda;	/* time to kick CUDA commands */
	cl_double	time_sync_tasks;	/* time to synchronize tasks */
	/*-- perfmon for DMA send/recv --*/
	cl_uint		num_dma_send;	/* number of DMA send request */
	cl_uint		num_dma_recv;	/* number of DMA receive request */
	cl_ulong	bytes_dma_send;	/* bytes of DMA send */
	cl_ulong	bytes_dma_recv;	/* bytes of DMA receive */
	cl_double	time_dma_send;	/* time to send host=>device data */
	cl_double	time_dma_recv;	/* time to receive device=>host data */
	/*-- (special perfmon for gpuscan) --*/
	cl_uint		num_kern_qual;	/* number of qual eval kernel execution */
	cl_double	time_kern_qual;	/* time to execute qual eval kernel */
	/*-- (special perfmon for gpuhashjoin) --*/
	cl_uint		num_kern_join;	/* number of hash-join kernel execution */
	cl_uint		num_kern_proj;	/* number of projection kernel execution */
	cl_double	time_kern_join;	/* time to execute hash-join kernel */
	cl_double	time_kern_proj;	/* time to execute projection kernel */
	/*-- (special perfmon for gpupreagg) --*/
	cl_uint		num_kern_prep;	/* number of preparation kernel execution */
	cl_uint		num_kern_lagg;	/* number of local reduction kernel exec */
	cl_uint		num_kern_gagg;	/* number of global reduction kernel exec */
	cl_uint		num_kern_fagg;	/* number of final reduction kernel exec */
	cl_uint		num_kern_nogrp;	/* number of nogroup reduction kernel exec */
	cl_uint		num_kern_fixvar;/* number of varlena fixup kernel exec */
	cl_double	time_kern_prep;	/* time to execute preparation kernel */
	cl_double	time_kern_lagg;	/* time to execute local reduction kernel */
	cl_double	time_kern_gagg;	/* time to execute global reduction kernel */
	cl_double	time_kern_fagg;	/* time to execute final reduction kernel */
	cl_double	time_kern_fixvar; /* time to execute varlena fixup kernel */
	cl_double	time_kern_nogrp;/* time to execute nogroup reduction kernel */
	/*-- (special perfmon for gpusort) --*/
	cl_uint		num_prep_sort;	/* number of GPU sort preparation kernel */
	cl_uint		num_gpu_sort;	/* number of GPU bitonic sort execution */
	cl_uint		num_cpu_sort;	/* number of GPU merge sort execution */
	cl_double	time_prep_sort;	/* time to execute GPU sort prep kernel */
	cl_double	time_gpu_sort;	/* time to execute GPU bitonic sort */
	cl_double	time_cpu_sort;	/* time to execute CPU merge sort */
	cl_double	time_cpu_sort_real;	/* real time to execute CPU merge sort */
	cl_double	time_cpu_sort_min;	/* min time to execute CPU merge sort */
	cl_double	time_cpu_sort_max;	/* max time to execute CPU merge sort */
	cl_double	time_bgw_sync;	/* time to synchronize bgworkers*/

	/*-- for debugging usage --*/
	cl_double	time_debug1;	/* time for debugging purpose.1 */
	cl_double	time_debug2;	/* time for debugging purpose.2 */
	cl_double	time_debug3;	/* time for debugging purpose.3 */
	cl_double	time_debug4;	/* time for debugging purpose.4 */

	struct timeval	tv;	/* result of gettimeofday(2) when enqueued */
} pgstrom_perfmon;

/* time interval in milliseconds */
#define PERFMON_BEGIN(pfm_accum,tv1)			\
	do {										\
		if ((pfm_accum)->enabled)				\
			gettimeofday((tv1), NULL);			\
	} while(0)

#define PERFMON_END(pfm_accum,field,tv1,tv2)					\
	do {														\
		if ((pfm_accum)->enabled)								\
		{														\
			gettimeofday((tv2), NULL);							\
			(pfm_accum)->field +=								\
				((double)(((tv2)->tv_sec - (tv1)->tv_sec) * 1000000L +	\
						  ((tv2)->tv_usec - (tv1)->tv_usec)) / 1000.0);	\
		}														\
	} while(0)

#define CUDA_EVENT_RECORD(node,ev_field)						\
	do {														\
		if (((GpuTask *)(node))->pfm.enabled)					\
		{														\
			CUresult __rc =	cuEventRecord((node)->ev_field,		\
										  ((GpuTask *)(node))->cuda_stream); \
			if (__rc != CUDA_SUCCESS)							\
				elog(ERROR, "failed on cuEventRecord: %s",		\
					 errorText(__rc));							\
		}														\
	} while(0)

#define CUDA_EVENT_CREATE(node,ev_field)					\
	do {													\
		CUresult __rc = cuEventCreate(&(node)->ev_field,	\
									  CU_EVENT_DEFAULT);	\
		if (__rc != CUDA_SUCCESS)							\
			elog(ERROR, "failed on cuEventCreate: %s",		\
				 errorText(__rc));							\
	} while(0)

#define CUDA_EVENT_DESTROY(node,ev_field)						\
	do {														\
		if ((node)->ev_field)									\
		{														\
			CUresult __rc = cuEventDestroy((node)->ev_field);	\
			if (__rc != CUDA_SUCCESS)							\
				elog(WARNING, "failed on cuEventDestroy: %s",	\
					 errorText(__rc));							\
			(node)->ev_field = NULL;							\
		}														\
	} while(0)

#define CUDA_EVENT_ELAPSED(node,pfm_field,ev_start,ev_stop)		\
	do {														\
		CUresult	__rc;										\
		float		__elapsed;									\
																\
		if ((node)->ev_start != NULL && (node)->ev_stop != NULL)\
		{														\
			__rc = cuEventElapsedTime(&__elapsed,				\
									  (node)->ev_start,			\
									  (node)->ev_stop);			\
			if (__rc != CUDA_SUCCESS)							\
				elog(ERROR, "failed on cuEventElapsedTime: %s",	\
					 errorText(__rc));							\
			((GpuTask *)(node))->pfm.pfm_field += __elapsed;	\
		}														\
	} while(0)

/*
 *
 *
 *
 *
 */
struct GpuMemBlock;
typedef struct
{
	struct GpuMemBlock *empty_block;
	dlist_head		active_blocks;
	dlist_head		unused_chunks;	/* cache for GpuMemChunk entries */
	dlist_head		unused_blocks;	/* cache for GpuMemBlock entries */
	dlist_head		hash_slots[59];	/* hash to find out GpuMemChunk */
} GpuMemHead;

typedef struct
{
	dlist_node		chain;			/* dual link to the global list */
	int				refcnt;			/* reference counter */
	ResourceOwner	resowner;		/* ResourceOwner owns this GpuContext */
	MemoryContext	memcxt;			/* Memory context for host pinned mem */
	dlist_head		pds_list;		/* list of pgstrom_data_store */
	cl_int			num_context;	/* number of CUDA context */
	cl_int			next_context;
	struct {
		CUdevice	cuda_device;
		CUcontext	cuda_context;
		GpuMemHead	cuda_memory;	/* wrapper of device memory allocation */
		size_t		gmem_used;		/* device memory allocated */
	} gpu[FLEXIBLE_ARRAY_MEMBER];
} GpuContext;

typedef struct GpuTask		GpuTask;
typedef struct GpuTaskState	GpuTaskState;

struct GpuTaskState
{
	CustomScanState	css;
	GpuContext	   *gcontext;
	kern_parambuf  *kern_params;	/* Const/Param buffer */
	const char	   *kern_define;	/* per session definition */
	const char	   *kern_source;	/* GPU kernel source on the fly */
	cl_uint			extra_flags;	/* flags for static inclusion */
	const char	   *source_pathname;
	CUmodule	   *cuda_modules;	/* CUmodules for each CUDA context */
	bool			scan_done;		/* no rows to read, if true */
	bool			outer_bulk_exec;/* true, if it bulk-exec on outer-node */
	bool			be_row_format;	/* true, if KDS_FORMAT_ROW is required */
	BlockNumber		curr_blknum;	/* current block number to scan table */
	BlockNumber		last_blknum;	/* last block number to scan table */
	TupleTableSlot *scan_overflow;	/* temp buffer, if unable to load */
	cl_long			curr_index;		/* current position on the curr_task */
	struct GpuTask *curr_task;		/* a task currently processed */
	slock_t			lock;			/* protection of the fields below */
	dlist_head		tracked_tasks;	/* for resource tracking */
	dlist_head		running_tasks;	/* list for running tasks */
	dlist_head		pending_tasks;	/* list for pending tasks */
	dlist_head		completed_tasks;/* list for completed tasks */
	dlist_head		ready_tasks;	/* list for ready tasks */
	cl_uint			num_running_tasks;
	cl_uint			num_pending_tasks;
	cl_uint			num_completed_tasks;
	cl_uint			num_ready_tasks;
	/* callbacks */
	bool		  (*cb_task_process)(GpuTask *gtask);
	bool		  (*cb_task_complete)(GpuTask *gtask);
	void		  (*cb_task_release)(GpuTask *gtask);
	void		  (*cb_task_polling)(GpuTaskState *gts);
	GpuTask		 *(*cb_next_chunk)(GpuTaskState *gts);
	TupleTableSlot *(*cb_next_tuple)(GpuTaskState *gts);
	/* extended executor */
	struct pgstrom_data_store *(*cb_bulk_exec)(GpuTaskState *gts,
											   size_t chunk_size);
	/* performance counter  */
	pgstrom_perfmon	pfm_accum;
};
#define GTS_GET_SCAN_TUPDESC(gts)				\
	(((GpuTaskState *)(gts))->css.ss.ss_ScanTupleSlot->tts_tupleDescriptor)
#define GTS_GET_RESULT_TUPDESC(gts)				\
  (((GpuTaskState *)(gts))->css.ss.ps.ps_ResultTupleSlot->tts_tupleDescriptor)

struct GpuTask
{
	dlist_node		chain;		/* link to task state list */
	dlist_node		tracker;	/* link to task tracker list */
	GpuTaskState   *gts;
	bool			no_cuda_setup;	/* true, if no need to set up stream */
	bool			cpu_fallback;	/* true, if task needs CPU fallback */
	cl_uint			cuda_index;		/* index of the cuda_context */
	CUcontext		cuda_context;	/* just reference, no cleanup needed */
	CUdevice		cuda_device;	/* just reference, no cleanup needed */
	CUstream		cuda_stream;	/* owned for each GpuTask */
	CUmodule		cuda_module;	/* just reference, no cleanup needed */
	kern_errorbuf	kerror;		/* error status on CUDA kernel execution */
	pgstrom_perfmon	pfm;
};

/*
 * Type declarations for code generator
 */
//#define DEVINFO_IS_NEGATIVE				0x00000001
#define DEVTYPE_IS_VARLENA				0x00000002
#define DEVTYPE_HAS_INTERNAL_FORMAT		0x00000004
#define DEVFUNC_NEEDS_TIMELIB			0x00000008
#define DEVFUNC_NEEDS_TEXTLIB			0x00000010
#define DEVFUNC_NEEDS_NUMERIC			0x00000020
#define DEVFUNC_NEEDS_MATHLIB			0x00000040
#define DEVFUNC_NEEDS_MONEY				0x00000080
#define DEVFUNC_INCL_FLAGS				0x000000f8
#define DEVKERNEL_NEEDS_GPUSCAN			0x00010000
#define DEVKERNEL_NEEDS_GPUJOIN			0x00020000
#define DEVKERNEL_NEEDS_GPUPREAGG		0x00040000
#define DEVKERNEL_NEEDS_GPUSORT			0x00080000
#define DEVKERNEL_NEEDS_LIBCUDART		0x80000000

struct devtype_info;
struct devfunc_info;

typedef struct devtype_info {
	Oid			type_oid;
	uint32		type_flags;
	int16		type_length;
	int16		type_align;
	bool		type_byval;
	bool		type_is_negative;
	char	   *type_name;	/* name of device type; same of SQL's type */
	char	   *type_base;	/* base name of this type (like varlena) */
	/* oid of type related functions */
	Oid			type_eqfunc;	/* function to check equality */
	Oid			type_cmpfunc;	/* function to compare two values */
} devtype_info;

typedef struct devfunc_info {
	Oid			func_oid;		/* OID of the SQL function */
	Oid			func_collid;	/* OID of collation, if collation aware */
	int32		func_flags;		/* Extra flags of this function */
	bool		func_is_negative;	/* True, if not supported by GPU */
	bool		func_is_strict;		/* True, if NULL strict function */
	List	   *func_args;		/* argument types by devtype_info */
	devtype_info *func_rettype;	/* result type by devtype_info */
	const char *func_sqlname;	/* name of the function in SQL side */
	const char *func_devname;	/* name of the function in device side */
	const char *func_decl;	/* declaration of device function, if any */
} devfunc_info;

/*
 * pgstrom_data_store - a data structure with row- or column- format
 * to exchange a data chunk between the host and opencl server.
 */
typedef struct pgstrom_data_store
{
	dlist_node	pds_chain;	/* link to GpuContext->pds_list */
	cl_int		refcnt;		/* reference counter */
	FileName	kds_fname;	/* filename, if file-mapped */
	Size		kds_offset;	/* offset of mapped file */
	Size		kds_length;	/* length of the kernel data store */
	kern_data_store *kds;
	struct pgstrom_data_store *ptoast;
} pgstrom_data_store;

/*
 * --------------------------------------------------------------------
 *
 * Function Declarations
 *
 * --------------------------------------------------------------------
 */

/*
 * cuda_mmgr.c
 */
extern void cudaHostMemAssert(void *pointer);

extern MemoryContext
HostPinMemContextCreate(MemoryContext parent,
                        const char *name,
						CUcontext cuda_context,
                        Size block_size_init,
                        Size block_size_max);
/*
 * cuda_control.c
 */
extern Size gpuMemMaxAllocSize(void);
extern CUdeviceptr __gpuMemAlloc(GpuContext *gcontext,
								 int cuda_index,
								 size_t bytesize);
extern void __gpuMemFree(GpuContext *gcontext,
						 int cuda_index,
						 CUdeviceptr dptr);
extern CUdeviceptr gpuMemAlloc(GpuTask *gtask, size_t bytesize);
extern void gpuMemFree(GpuTask *gtask, CUdeviceptr dptr);
extern GpuContext *pgstrom_get_gpucontext(void);
extern void pgstrom_put_gpucontext(GpuContext *gcontext);

extern void pgstrom_cleanup_gputaskstate(GpuTaskState *gts);
extern void pgstrom_release_gputaskstate(GpuTaskState *gts);
extern void pgstrom_init_gputaskstate(GpuContext *gcontext, GpuTaskState *gts);
extern void pgstrom_init_gputask(GpuTaskState *gts, GpuTask *gtask);
extern void pgstrom_release_gputask(GpuTask *gtask);
extern GpuTask *pgstrom_fetch_gputask(GpuTaskState *gts);
extern pgstrom_data_store *pgstrom_exec_chunk_gputask(GpuTaskState *gts,
													  size_t chunk_size);
extern TupleTableSlot *pgstrom_exec_gputask(GpuTaskState *gts);
extern bool pgstrom_recheck_gputask(GpuTaskState *gts, TupleTableSlot *slot);
extern void pgstrom_cleanup_gputask_cuda_resources(GpuTask *gtask);
extern size_t gpuLocalMemSize(void);
extern cl_uint gpuMaxThreadsPerBlock(void);
extern void pgstrom_compute_workgroup_size(size_t *p_grid_size,
										   size_t *p_block_size,
										   CUfunction function,
										   CUdevice device,
										   bool maximize_blocksize,
										   size_t nitems,
										   size_t dynamic_shmem_per_thread);
extern void pgstrom_compute_workgroup_size_2d(size_t *p_grid_xsize,
											  size_t *p_grid_ysize,
											  size_t *p_block_xsize,
											  size_t *p_block_ysize,
											  CUfunction function,
											  CUdevice device,
											  size_t x_nitems,
											  size_t y_nitems,
											  size_t dynamic_shmem_per_xitems,
											  size_t dynamic_shmem_per_yitems,
											  size_t dynamic_shmem_per_thread);
extern void pgstrom_init_cuda_control(void);
extern cl_ulong pgstrom_baseline_cuda_capability(void);
extern const char *errorText(int errcode);
extern const char *errorTextKernel(kern_errorbuf *kerror);
extern Datum pgstrom_scoreboard_info(PG_FUNCTION_ARGS);
extern Datum pgstrom_device_info(PG_FUNCTION_ARGS);

/*
 * cuda_program.c
 */
extern const char *pgstrom_cuda_source_file(GpuTaskState *gts);
extern bool pgstrom_load_cuda_program(GpuTaskState *gts);
extern void pgstrom_preload_cuda_program(GpuTaskState *gts);
extern void pgstrom_assign_cuda_program(GpuTaskState *gts,
										List *used_params,
										const char *kern_source,
										int extra_flags);
extern void pgstrom_init_cuda_program(void);
extern Datum pgstrom_program_info(PG_FUNCTION_ARGS);

/*
 * codegen.c
 */
typedef struct {
	StringInfoData	str;
	List	   *type_defs;	/* list of devtype_info in use */
	List	   *func_defs;	/* list of devfunc_info in use */
	List	   *used_params;/* list of Const/Param in use */
	List	   *used_vars;	/* list of Var in use */
	Bitmapset  *param_refs;	/* referenced parameters */
	const char *var_label;	/* prefix of var reference, if exist */
	const char *kds_label;	/* label to reference kds, if exist */
	const char *kds_index_label; /* label to reference kds_index, if exist */
	List	   *pseudo_tlist;/* pseudo tlist expression, if any */
	int			extra_flags;/* external libraries to be included */
} codegen_context;

extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);
extern devfunc_info *pgstrom_devfunc_lookup(Oid func_oid, Oid func_collid);
extern devtype_info *pgstrom_devtype_lookup_and_track(Oid type_oid,
											  codegen_context *context);
extern devfunc_info *pgstrom_devfunc_lookup_and_track(Oid func_oid,
													  Oid func_collid,
											  codegen_context *context);

extern char *pgstrom_codegen_expression(Node *expr, codegen_context *context);
extern void pgstrom_codegen_func_declarations(StringInfo buf,
											  codegen_context *context);
extern void pgstrom_codegen_param_declarations(StringInfo buf,
											   codegen_context *context);
extern void pgstrom_codegen_var_declarations(StringInfo buf,
											 codegen_context *context);
extern bool pgstrom_device_expression(Expr *expr);
extern void pgstrom_init_codegen_context(codegen_context *context);
extern void pgstrom_init_codegen(void);

/*
 * datastore.c
 */
extern Size pgstrom_chunk_size(void);
extern Size pgstrom_chunk_size_limit(void);
extern bool pgstrom_bulk_exec_supported(const PlanState *planstate);
extern cl_uint estimate_num_chunks(Path *pathnode);
extern pgstrom_data_store *ChunkExecProcNode(GpuTaskState *gts,
											 size_t chunk_size);
extern Datum pgstrom_fixup_kernel_numeric(Datum numeric_datum);
extern bool pgstrom_fetch_data_store(TupleTableSlot *slot,
									 pgstrom_data_store *pds,
									 size_t row_index,
									 HeapTuple tuple);
extern bool kern_fetch_data_store(TupleTableSlot *slot,
								  kern_data_store *kds,
								  size_t row_index,
								  HeapTuple tuple);
extern pgstrom_data_store *pgstrom_acquire_data_store(pgstrom_data_store *pds);
extern void pgstrom_release_data_store(pgstrom_data_store *pds);
extern void pgstrom_expand_data_store(GpuContext *gcontext,
									  pgstrom_data_store *pds,
									  Size kds_length_new,
									  cl_uint nslots_new);
extern void pgstrom_shrink_data_store(pgstrom_data_store *pds);
extern void init_kernel_data_store(kern_data_store *kds,
								   TupleDesc tupdesc,
								   Size length,
								   int format,
								   uint nrooms,
								   bool internal_format);
extern pgstrom_data_store *
pgstrom_create_data_store_row(GpuContext *gcontext,
							  TupleDesc tupdesc,
							  Size length,
							  bool file_mapped);
extern pgstrom_data_store *
pgstrom_create_data_store_slot(GpuContext *gcontext,
							   TupleDesc tupdesc,
							   cl_uint nrooms,
							   bool internal_format,
							   Size extra_length,
							   pgstrom_data_store *ptoast);
extern pgstrom_data_store *
pgstrom_create_data_store_hash(GpuContext *gcontext,
							   TupleDesc tupdesc,
							   Size length,
							   cl_uint nslots,
							   bool file_mapped);
extern void
pgstrom_file_mmap_data_store(FileName kds_fname,
							 Size kds_offset,
							 Size kds_length,
							 kern_data_store **p_kds,
							 kern_data_store **p_ktoast);

extern int pgstrom_data_store_insert_block(pgstrom_data_store *pds,
										   Relation rel,
										   BlockNumber blknum,
										   Snapshot snapshot,
										   bool page_prune);
extern bool pgstrom_data_store_insert_tuple(pgstrom_data_store *pds,
											TupleTableSlot *slot);
extern bool pgstrom_data_store_insert_hashitem(pgstrom_data_store *pds,
											   TupleTableSlot *slot,
											   cl_uint hash_value);
extern void pgstrom_dump_data_store(pgstrom_data_store *pds);
extern void pgstrom_init_datastore(void);

/*
 * gpuscan.c
 */
extern bool pgstrom_pullup_outer_scan(Plan *plannode,
									  bool allow_expression,
									  List **p_outer_qual);
extern bool pgstrom_path_is_gpuscan(const Path *path);
extern bool pgstrom_plan_is_gpuscan(const Plan *plan);
extern Node *replace_varnode_with_tlist_dev(Node *node, List *tlist_dev);
extern AttrNumber add_unique_expression(Expr *expr, List **p_targetlist,
										bool resjunk);
extern pgstrom_data_store *pgstrom_exec_scan_chunk(GpuTaskState *gts,
												   Size chunk_length);
extern void pgstrom_rewind_scan_chunk(GpuTaskState *gts);
extern void pgstrom_post_planner_gpuscan(PlannedStmt *pstmt, Plan **p_plan);
extern void assign_gpuscan_session_info(StringInfo buf, GpuTaskState *gts);
extern void pgstrom_init_gpuscan(void);

/*
 * gpujoin.c
 */
extern bool pgstrom_path_is_gpujoin(Path *pathnode);
extern bool pgstrom_plan_is_gpujoin(const Plan *plannode);
extern void pgstrom_post_planner_gpujoin(PlannedStmt *pstmt, Plan **p_plan);
extern void assign_gpujoin_session_info(StringInfo buf, GpuTaskState *gts);
extern void	pgstrom_init_gpujoin(void);

/*
 * gpupreagg.c
 */
extern void pgstrom_try_insert_gpupreagg(PlannedStmt *pstmt, Agg *agg);
extern bool pgstrom_plan_is_gpupreagg(const Plan *plan);
extern void pgstrom_init_gpupreagg(void);

/*
 * gpusort.c
 */
extern void pgstrom_try_insert_gpusort(PlannedStmt *pstmt, Plan **p_plan);
extern bool pgstrom_plan_is_gpusort(const Plan *plan);
extern void pgstrom_init_gpusort(void);

/*
 * main.c
 */
extern bool		pgstrom_enabled;
extern bool		pgstrom_perfmon_enabled;
extern bool		pgstrom_bulkexec_enabled;
extern int		pgstrom_max_async_tasks;
extern double	pgstrom_gpu_setup_cost;
extern double	pgstrom_gpu_dma_cost;
extern double	pgstrom_gpu_operator_cost;
extern double	pgstrom_gpu_tuple_cost;
extern double	pgstrom_nrows_growth_ratio_limit;
extern double	pgstrom_nrows_growth_margin;
extern double	pgstrom_num_threads_margin;
extern double	pgstrom_chunk_size_margin;

extern void _PG_init(void);
extern const char *pgstrom_strerror(cl_int errcode);

extern void pgstrom_explain_expression(List *expr_list, const char *qlabel,
									   PlanState *planstate,
									   List *deparse_context,
									   List *ancestors, ExplainState *es,
									   bool force_prefix,
									   bool convert_to_and);

extern void show_scan_qual(List *qual, const char *qlabel,
						   PlanState *planstate, List *ancestors,
						   ExplainState *es);
extern void show_instrumentation_count(const char *qlabel, int which,
									   PlanState *planstate, ExplainState *es);
extern void pgstrom_accum_perfmon(pgstrom_perfmon *accum,
								  const pgstrom_perfmon *pfm);
extern void
pgstrom_explain_gputaskstate(GpuTaskState *gts, ExplainState *es);

/*
 * grafter.c
 */
extern void pgstrom_init_grafter(void);

/*
 * opencl_*.h
 */
extern const char *pgstrom_cuda_common_code;
extern const char *pgstrom_cuda_gpuscan_code;
extern const char *pgstrom_cuda_gpujoin_code;
extern const char *pgstrom_cuda_gpupreagg_code;
extern const char *pgstrom_cuda_gpusort_code;
extern const char *pgstrom_cuda_mathlib_code;
extern const char *pgstrom_cuda_textlib_code;
extern const char *pgstrom_cuda_timelib_code;
extern const char *pgstrom_cuda_numeric_code;
extern const char *pgstrom_cuda_money_code;
extern const char *pgstrom_cuda_terminal_code;

/*
 * createplan.c
 */
extern Plan *create_plan_recurse(PlannerInfo *root, Path *best_path);

/* ----------------------------------------------------------------
 *
 * Miscellaneous static inline functions
 *
 * ---------------------------------------------------------------- */

/*
 * int/float reinterpret functions
 */
static inline cl_double
long_as_double(cl_long ival)
{
	union {
		cl_long		ival;
		cl_double	fval;
	} datum;
	datum.ival = ival;
	return datum.fval;
}

static inline cl_long
double_as_long(cl_double fval)
{
	union {
		cl_long		ival;
		cl_double	fval;
	} datum;
	datum.fval = fval;
	return datum.ival;
}

static inline cl_float
int_as_float(cl_int ival)
{
	union {
		cl_int		ival;
		cl_float	fval;
	} datum;
	datum.ival = ival;
	return datum.fval;
}

static inline cl_int
float_as_int(cl_float fval)
{
	union {
		cl_int		ival;
		cl_float	fval;
	} datum;
	datum.fval = fval;
	return datum.ival;
}

/*
 * get_next_log2
 *
 * It returns N of the least 2^N value that is larger than or equal to
 * the supplied value.
 */
static inline int
get_next_log2(Size size)
{
	int		shift = 0;

	if (size == 0 || size == 1)
		return 0;
	size--;
#ifdef __GNUC__
	shift = sizeof(Size) * BITS_PER_BYTE - __builtin_clzl(size);
#else
#if SIZEOF_VOID_P == 8
	if ((size & 0xffffffff00000000UL) != 0)
	{
		size >>= 32;
		shift += 32;
	}
#endif
	if ((size & 0xffff0000UL) != 0)
	{
		size >>= 16;
		shift += 16;
	}
	if ((size & 0x0000ff00UL) != 0)
	{
		size >>= 8;
		shift += 8;
	}
	if ((size & 0x000000f0UL) != 0)
	{
		size >>= 4;
		shift += 4;
	}
	if ((size & 0x0000000cUL) != 0)
	{
		size >>= 2;
		shift += 2;
	}
	if ((size & 0x00000002UL) != 0)
	{
		size >>= 1;
		shift += 1;
	}
	if ((size & 0x00000001UL) != 0)
		shift += 1;
#endif	/* !__GNUC__ */
	return shift;
}

/*
 * It translate an alignment character into width
 */
static inline int
typealign_get_width(char type_align)
{
	if (type_align == 'c')
		return sizeof(cl_char);
	else if (type_align == 's')
		return sizeof(cl_short);
	else if (type_align == 'i')
		return sizeof(cl_int);
	else if (type_align == 'd')
		return sizeof(cl_long);
	Assert(false);
	elog(ERROR, "unexpected type alignment: %c", type_align);
	return -1;	/* be compiler quiet */
}

#ifndef forfour
#define forfour(cell1, list1, cell2, list2, cell3, list3, cell4, list4)	\
	for ((cell1) = list_head(list1), (cell2) = list_head(list2),	\
		 (cell3) = list_head(list3), (cell4) = list_head(list4);	\
		 (cell1) != NULL && (cell2) != NULL &&						\
		 (cell3) != NULL && (cell4) != NULL;						\
		 (cell1) = lnext(cell1), (cell2) = lnext(cell2),			\
		 (cell3) = lnext(cell3), (cell4) = lnext(cell4))
#endif

static inline char *
bytesz_unitary_format(Size nbytes)
{
	if (nbytes > (Size)(1UL << 43))
		return psprintf("%.2fTB", (double)nbytes / (double)(1UL << 40));
	else if (nbytes > (double)(1UL << 33))
		return psprintf("%.2fGB", (double)nbytes / (double)(1UL << 30));
	else if (nbytes > (double)(1UL << 23))
		return psprintf("%.2fMB", (double)nbytes / (double)(1UL << 20));
	else if (nbytes > (double)(1UL << 13))
		return psprintf("%.2fKB", (double)nbytes / (double)(1UL << 10));
	return psprintf("%uB", (unsigned int)nbytes);
}

static inline char *
milliseconds_unitary_format(double milliseconds)
{
	if (milliseconds > 300000.0)    /* more then 5min */
		return psprintf("%.2fmin", milliseconds / 60000.0);
	else if (milliseconds > 8000.0) /* more than 8sec */
		return psprintf("%.2fsec", milliseconds / 1000.0);
	return psprintf("%.2fms", milliseconds);
}

#endif	/* PG_STROM_H */
