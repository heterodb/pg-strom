/*
 * pg_strom.h
 *
 * Header file of pg_strom module
 *
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#ifndef PG_USE_INLINE
#error PG-Strom expects inline function is supported by compiler
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
	cl_uint		num_kern_nogrp;	/* number of nogroup reduction kernel exec */
	cl_double	time_kern_prep;	/* time to execute preparation kernel */
	cl_double	time_kern_lagg;	/* time to execute local reduction kernel */
	cl_double	time_kern_gagg;	/* time to execute global reduction kernel */
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
	dlist_node		chain;
	int				refcnt;
	ResourceOwner	resowner;
	MemoryContext	memcxt;
	dlist_head		state_list;		/* list of GpuTaskState */
	dlist_head		pds_list;		/* list of pgstrom_data_store */
	cl_int			num_context;
	cl_int			next_context;
	struct {
		CUdevice	cuda_device;
		CUcontext	cuda_context;
		GpuMemHead	cuda_memory;	/* wrapper of device memory allocation */
	} gpu[FLEXIBLE_ARRAY_MEMBER];
} GpuContext;

typedef struct GpuTask		GpuTask;
typedef struct GpuTaskState	GpuTaskState;

struct GpuTaskState
{
	CustomScanState	css;
	dlist_node		chain;
	GpuContext	   *gcontext;
	const char	   *kern_source;
	cl_uint			extra_flags;
	const char	   *source_pathname;
	CUmodule	   *cuda_modules;	/* CUmodules for each CUDA context */
	bool			scan_done;		/* no rows to read, if true */
	bool			scan_bulk;		/* bulk outer load, if true */
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
	void		  (*cb_cleanup)(GpuTaskState *gts);
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
	cl_uint			cuda_index;		/* index of the cuda_context */
	CUcontext		cuda_context;	/* just reference, no cleanup needed */
	CUdevice		cuda_device;	/* just reference, no cleanup needed */
	CUstream		cuda_stream;	/* owned for each GpuTask */
	CUmodule		cuda_module;	/* just reference, no cleanup needed */
	cl_int			errcode;
	pgstrom_perfmon	pfm;
};

/*
 * Type declarations for code generator
 */
#define DEVINFO_IS_NEGATIVE				0x00000001
#define DEVTYPE_IS_VARLENA				0x00000002
#define DEVTYPE_HAS_INTERNAL_FORMAT		0x00000004
#define DEVFUNC_NEEDS_TIMELIB			0x00000008
#define DEVFUNC_NEEDS_TEXTLIB			0x00000010
#define DEVFUNC_NEEDS_NUMERIC			0x00000020
#define DEVFUNC_NEEDS_MATHLIB			0x00000040
#define DEVFUNC_INCL_FLAGS				0x00000078
#define DEVKERNEL_NEEDS_GPUSCAN			0x00010000
#define DEVKERNEL_NEEDS_HASHJOIN		0x00020000
#define DEVKERNEL_NEEDS_GPUPREAGG		0x00040000
#define DEVKERNEL_NEEDS_GPUSORT			0x00080000
#define DEVKERNRL_NEEDS_GPUJOIN			0x00100000

struct devtype_info;
struct devfunc_info;

typedef struct devtype_info {
	Oid			type_oid;
	uint32		type_flags;
	int16		type_length;
	int16		type_align;
	char	   *type_name;	/* name of device type; same of SQL's type */
	char	   *type_base;	/* base name of this type (like varlena) */
	/* oid of type related functions */
	Oid			type_eqfunc;	/* function to check equality */
	Oid			type_cmpfunc;	/* function to compare two values */
} devtype_info;

typedef struct devfunc_info {
	int32		func_flags;
	Oid			func_namespace;
	Oid		   *func_argtypes;
	const char *func_name;	/* name of SQL function */
	List	   *func_args;	/* list of devtype_info */
	devtype_info *func_rettype;
	Oid			func_collid;/* OID of collation, if collation aware */
	const char *func_alias;	/* name of declared device function */
	const char *func_decl;	/* declaration of device function */
} devfunc_info;

/*
 * pgstrom_data_store - a data structure with row- or column- format
 * to exchange a data chunk between the host and opencl server.
 */
typedef struct pgstrom_data_store
{
	dlist_node	pds_chain;		/* link to GpuContext->pds_list */
	FileName	kds_fname;	/* filename, if file-mapped */
	Size		kds_offset;	/* offset of mapped file */
	Size		kds_length;	/* length of the kernel data store */
	kern_data_store *kds;
	struct pgstrom_data_store *ptoast;
} pgstrom_data_store;

/* --------------------------------------------------------------------
 *
 * Private enhancement of CustomScan Interface
 *
 * --------------------------------------------------------------------
 */
typedef void *(*pgstromExecBulkScan_type)(CustomScanState *node);

typedef struct
{
	CustomExecMethods	c;
	pgstromExecBulkScan_type ExecCustomBulk;
} PGStromExecMethods;

/*
 * Extra flags of CustomPath/Scan node.
 *
 * CUSTOMPATH_SUPPORT_BULKLOAD is set, if this CustomScan node support
 * bulkload mode and parent node can cann BulkExecProcNode() instead of
 * the usual ExecProcNode().
 *
 * CUSTOMPATH_PREFERE_ROW_FORMAT is set by parent node to inform
 * preferable format of tuple delivered row-by-row mode. Usually, we put
 * a record with format of either of tts_tuple or tts_values/tts_isnull.
 * It informs child node a preferable output if parent can collaborate.
 * Note that this flag never enforce the child node format. It's just
 * a hint.
 */
#define CUSTOMPATH_SUPPORT_BULKLOAD			0x10000000
#define CUSTOMPATH_PREFERE_ROW_FORMAT		0x20000000

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
extern MemoryContext
HostPinMemContextCreate(MemoryContext parent,
                        const char *name,
						CUcontext cuda_context,
                        Size block_size_init,
                        Size block_size_max);
/*
 * cuda_control.c
 */
typedef enum {
	ResourceContextNormal,
	ResourceContextCommit,
	ResourceContextAbort,
} PGStromResourceContext;

extern PGStromResourceContext pgstrom_resource_context(void);

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
extern void pgstrom_sync_gpucontext(GpuContext *gcontext);
extern void pgstrom_put_gpucontext(GpuContext *gcontext);

extern void pgstrom_cleanup_gputaskstate(GpuTaskState *gts);
extern void pgstrom_release_gputaskstate(GpuTaskState *gts);
extern void pgstrom_init_gputaskstate(GpuContext *gcontext, GpuTaskState *gts);
extern void pgstrom_init_gputask(GpuTaskState *gts, GpuTask *gtask);
extern GpuTask *pgstrom_fetch_gputask(GpuTaskState *gts);
extern TupleTableSlot *pgstrom_exec_gputask(GpuTaskState *gts);
extern bool pgstrom_recheck_gputask(GpuTaskState *gts, TupleTableSlot *slot);
extern void pgstrom_cleanup_gputask_cuda_resources(GpuTask *gtask);
extern void pgstrom_compute_workgroup_size(size_t *p_grid_size,
										   size_t *p_block_size,
										   CUfunction function,
										   CUdevice device,
										   bool maximum_blocksize,
										   size_t nitems,
										   size_t dynamic_shmem_per_thread);
extern void pgstrom_init_cuda_control(void);
extern int pgstrom_baseline_cuda_capability(void);
extern const char *errorText(int errcode);
extern Datum pgstrom_device_info(PG_FUNCTION_ARGS);

/*
 * cuda_program.c
 */
extern const char *pgstrom_cuda_source_file(const char *kern_source,
											uint32 extra_flags);
extern bool pgstrom_load_cuda_program(GpuTaskState *gts);
extern void pgstrom_preload_cuda_program(GpuTaskState *gts);
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
extern char *pgstrom_codegen_func_declarations(codegen_context *context);
extern char *pgstrom_codegen_param_declarations(codegen_context *context);
extern char *pgstrom_codegen_var_declarations(codegen_context *context);
extern char *pgstrom_codegen_bulk_var_declarations(codegen_context *context,
												   Plan *outer_plan,
												   Bitmapset *attr_refs);
extern bool pgstrom_codegen_available_expression(Expr *expr);
extern void pgstrom_init_codegen_context(codegen_context *context);
extern void pgstrom_init_codegen(void);

/*
 * datastore.c
 */
extern Size pgstrom_chunk_size(void);
extern kern_parambuf *
pgstrom_create_kern_parambuf(List *used_params,
                             ExprContext *econtext);
extern Plan *pgstrom_try_replace_plannode(Plan *child_plan,
										  List *range_tables,
										  List **pullup_quals);
extern pgstrom_data_store *BulkExecProcNode(PlanState *node);
extern Datum pgstrom_fixup_kernel_numeric(Datum numeric_datum);
extern bool pgstrom_fetch_data_store(TupleTableSlot *slot,
									 pgstrom_data_store *pds,
									 size_t row_index,
									 HeapTuple tuple);
extern bool kern_fetch_data_store(TupleTableSlot *slot,
								  kern_data_store *kds,
								  size_t row_index,
								  HeapTuple tuple);
extern void pgstrom_release_data_store(pgstrom_data_store *pds);
extern void pgstrom_expand_data_store(GpuContext *gcontext,
									  pgstrom_data_store *pds,
									  Size kds_length_new);
extern void pgstrom_shrink_data_store(pgstrom_data_store *pds);
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
							   pgstrom_data_store *ptoast);
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
extern void pgstrom_dump_data_store(pgstrom_data_store *pds);
extern void pgstrom_init_datastore(void);

/*
 * gpuscan.c
 */
extern Plan *gpuscan_pullup_devquals(Plan *plannode, List **pullup_quals);
extern Plan *gpuscan_try_replace_seqscan(SeqScan *seqscan,
										 List *range_tables,
										 List **pullup_quals);
extern bool pgstrom_path_is_gpuscan(const Path *path);
extern bool pgstrom_plan_is_gpuscan(const Plan *plan);
extern void pgstrom_gpuscan_setup_bulkslot(PlanState *outer_ps,
										   ProjectionInfo **p_bulk_proj,
										   TupleTableSlot **p_bulk_slot);
extern void pgstrom_init_gpuscan(void);

/*
 * multirels.c
 */
struct pgstrom_multirels;

extern bool	pgstrom_plan_is_multirels(const Plan *plan);
extern CustomScan *
pgstrom_create_multirels_plan(PlannerInfo *root,
							  int depth, Path *outer_path,
							  int nbatches, Size buffer_size, double threshold,
							  int nslots, List *hash_keys);
extern struct pgstrom_multirels *
pgstrom_multirels_exec_bulk(PlanState *plannode);
extern Size pgstrom_multirels_get_length(struct pgstrom_multirels *pmrels);
extern void pgstrom_multirels_dma_send(struct pgstrom_multirels *pmrels);
exterb void pgstrom_multirels_dma_recv(struct pgstrom_multirels *pmrels);
extern void	pgstrom_init_multirels(void);

/*
 * gpujoin.c
 */





/*
 * gpuhashjoin.c
 */
extern bool pgstrom_plan_is_gpuhashjoin(const Plan *plan);
extern bool pgstrom_plan_is_multihash(const Plan *plan);
extern void pgstrom_gpuhashjoin_setup_bulkslot(PlanState *outer_ps,
											   ProjectionInfo **p_bulk_proj,
											   TupleTableSlot **p_bulk_slot);
extern void pgstrom_init_gpuhashjoin(void);

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
extern void pgstrom_init_gpusort(void);

/*
 * main.c
 */
extern bool	pgstrom_enabled(void);
extern bool pgstrom_perfmon_enabled;
extern bool pgstrom_debug_bulkload_enabled;
extern int	pgstrom_max_async_chunks;
extern int	pgstrom_min_async_chunks;
extern double pgstrom_gpu_setup_cost;
extern double pgstrom_gpu_operator_cost;
extern double pgstrom_gpu_tuple_cost;
extern double pgstrom_row_population_max;
extern double pgstrom_row_population_margin;
extern void _PG_init(void);
extern const char *pgstrom_strerror(cl_int errcode);
extern void show_scan_qual(List *qual, const char *qlabel,
						   PlanState *planstate, List *ancestors,
						   ExplainState *es);
extern void show_instrumentation_count(const char *qlabel, int which,
									   PlanState *planstate, ExplainState *es);
extern void pgstrom_accum_perfmon(pgstrom_perfmon *accum,
								  const pgstrom_perfmon *pfm);
extern void
pgstrom_explain_gputaskstate(GpuTaskState *gts, ExplainState *es);
extern void _outToken(StringInfo str, const char *s);
extern Value *formBitmapset(const Bitmapset *bms);
extern Bitmapset *deformBitmapset(const Value *value);

/*
 * grafter.c
 */
extern void pgstrom_init_grafter(void);

/*
 * opencl_*.h
 */
extern const char *pgstrom_cuda_common_code;
extern const char *pgstrom_cuda_gpuscan_code;
extern const char *pgstrom_cuda_gpupreagg_code;
extern const char *pgstrom_cuda_hashjoin_code;
extern const char *pgstrom_cuda_gpusort_code;
extern const char *pgstrom_cuda_mathlib_code;
extern const char *pgstrom_cuda_textlib_code;
extern const char *pgstrom_cuda_timelib_code;
extern const char *pgstrom_cuda_numeric_code;

/* ----------------------------------------------------------------
 *
 * Miscellaneous static inline functions
 *
 * ---------------------------------------------------------------- */

/* binary available pstrcpy() */
static inline void *
pmemcpy(void *from, size_t sz)
{
	/*
	 * Note that usual palloc() has 1GB limitation because of historical
	 * reason, so we have to use MemoryContextAllocHuge instead in case
	 * when we expect sz > 1GB.
	 * Also, *_huge has identical implementation expect for size checks,
	 * we don't need to check the cases.
	 */
	void   *dest = MemoryContextAllocHuge(CurrentMemoryContext, sz);

	return memcpy(dest, from, sz);
}

/* additional dlist stuff */
static inline int
dlist_length(dlist_head *head)
{
	dlist_iter	iter;
	int			count = 0;

	dlist_foreach(iter, head)
		count++;
	return count;
}

static inline void
dlist_move_tail(dlist_head *head, dlist_node *node)
{
	/* fast path if it's already at the head */
	if (head->head.next == node)
		return;
	dlist_delete(node);
    dlist_push_tail(head, node);

    dlist_check(head);
}

static inline void
dlist_move_all(dlist_head *dest, dlist_head *src)
{
	Assert(dlist_is_empty(dest));

	dest->head.next = dlist_head_node(src);
	dest->head.prev = dlist_tail_node(src);
	dlist_head_node(src)->prev = &dest->head;
	dlist_tail_node(src)->next = &dest->head;

	dlist_init(src);
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

#endif	/* PG_STROM_H */
