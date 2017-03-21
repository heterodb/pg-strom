/*
 * pg_strom.h
 *
 * Header file of pg_strom module
 *
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#include "storage/buf.h"
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
#ifdef PG_MIN_VERSION_NUM
#if PG_VERSION_NUM < PG_MIN_VERSION_NUM
#error Base PostgreSQL version is too OLD for this PG-Strom code
#endif
#endif	/* PG_MIN_VERSION_NUM */

#ifdef PG_MAX_VERSION_NUM
#if PG_VERSION_NUM >= PG_MAX_VERSION_NUM
#error Base PostgreSQL version is too NEW for this PG-Strom code
#endif
#endif	/* PG_MAX_VERSION_NUM */

/* inline function is minimum requirement. fortunately, it also
 * become prerequisite of PostgreSQL at v9.6.
 */
#if PG_VERSION_NUM < 90600
#ifndef PG_USE_INLINE
#error PG-Strom expects inline function is supported by compiler
#endif	/* PG_USE_INLINE */
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
	cl_bool		prime_in_gpucontext;
	cl_uint		extra_flags;	/* one of DEVKERNEL_NEEDS_GPUXXXX */
	/*-- memory allocation counter --*/
	cl_uint		num_host_malloc;
	cl_uint		num_host_mfree;
	cl_uint		num_dev_malloc;
	cl_uint		num_dev_mfree;
	cl_double	tv_host_malloc;
	cl_double	tv_host_mfree;
	cl_double	tv_dev_malloc;
	cl_double	tv_dev_mfree;
	/*-- build cuda program --*/
	cl_double	tv_build_kernel;
	struct timeval	tv_build_start;
	struct timeval	tv_build_end;
	/*-- time for task pending --*/

	/*-- time for I/O stuff --*/
	cl_double	time_inner_load;	/* time to load the inner relation */
	cl_double	time_outer_load;	/* time to load the outer relation */
	cl_double	time_materialize;	/* time to materialize the result */
	/*-- DMA data transfer --*/
	cl_uint		num_dma_send;	/* number of DMA send request */
	cl_uint		num_dma_recv;	/* number of DMA receive request */
	cl_ulong	bytes_dma_send;	/* bytes of DMA send */
	cl_ulong	bytes_dma_recv;	/* bytes of DMA receive */
	cl_double	time_dma_send;	/* time to send host=>device data */
	cl_double	time_dma_recv;	/* time to receive device=>host data */
	/*-- specific items for each GPU logic --*/
	cl_uint		num_tasks;			/* number of tasks completed */
	cl_double	time_launch_cuda;	/* time to kick CUDA commands */
	cl_double	time_sync_tasks;	/* time to synchronize tasks */
	/*-- for each GPU logic --*/
	struct {
		cl_uint		num_kern_main;
		cl_uint		num_kern_exec_quals;
		cl_uint		num_kern_projection;
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
	cl_int		   *p_keep_freemem;	/* flag to control memory cache policy */
	/*
	 * Performance statistics
	 */
	cl_int		   *p_num_host_malloc;	/* # of cuMemAllocHost calls */
	cl_int		   *p_num_host_mfree;	/* # of cuMemFreeHost calls */
	struct timeval *p_tv_host_malloc;	/* total time for cuMemAllocHost */
	struct timeval *p_tv_host_mfree;	/* total time for cuMemFreeHost */
	cl_int			num_dev_malloc;
	cl_int			num_dev_mfree;
	struct timeval	tv_dev_malloc;
	struct timeval	tv_dev_mfree;


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
	bool			be_row_format;	/* true, if KDS_FORMAT_ROW is required */
	bool			outer_bulk_exec;/* true, if it bulk-exec on outer-node */
	Instrumentation	outer_instrument; /* run time statistics */
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
	GpuTask		 *(*cb_next_chunk)(GpuTaskState *gts);
	void		  (*cb_switch_task)(GpuTaskState *gts, GpuTask *gtask);
	TupleTableSlot *(*cb_next_tuple)(GpuTaskState *gts);
	/* extended executor */
	struct pgstrom_data_store *(*cb_bulk_exec)(GpuTaskState *gts,
											   size_t chunk_size);
	/* performance counter  */
	pgstrom_perfmon	pfm;
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
};

/*
 * Type declarations for code generator
 */
#define DEVKERNEL_NEEDS_GPUSCAN			0x00000001	/* GpuScan logic */
#define DEVKERNEL_NEEDS_GPUJOIN			0x00000002	/* GpuJoin logic */
#define DEVKERNEL_NEEDS_GPUPREAGG		0x00000004	/* GpuPreAgg logic */
#define DEVKERNEL_NEEDS_GPUSORT			0x00000008	/* GpuSort logic */
#define DEVKERNEL_NEEDS_PLCUDA			0x00000080	/* PL/CUDA related */

#define DEVKERNEL_NEEDS_DYNPARA			0x00000100
#define DEVKERNEL_NEEDS_MATRIX		   (0x00000200 | DEVKERNEL_NEEDS_DYNPARA)
#define DEVKERNEL_NEEDS_TIMELIB			0x00000400
#define DEVKERNEL_NEEDS_TEXTLIB			0x00000800
#define DEVKERNEL_NEEDS_NUMERIC			0x00001000
#define DEVKERNEL_NEEDS_MATHLIB			0x00002000
#define DEVKERNEL_NEEDS_MONEY			0x00004000

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
	const struct devtype_info *type_array;	/* array type of itself, if any */
	const struct devtype_info *type_element;/* element type of array, if any */
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

typedef struct devexpr_info {
	NodeTag		expr_tag;		/* tag of the expression */
	Oid			expr_collid;	/* OID of collation, if collation aware */
	List	   *expr_args;		/* argument types by devtype_info */
	devtype_info *expr_rettype;	/* result type by devtype_info */
	Datum		expr_extra1;	/* 1st extra information per node type */
	Datum		expr_extra2;	/* 2nd extra information per node type */
	const char *expr_name;	/* name of the function in device side */
	const char *expr_decl;	/* declaration of device function, if any */
} devexpr_info;

/*
 * pgstrom_data_store - a data structure with row- or column- format
 * to exchange a data chunk between the host and opencl server.
 */
typedef struct pgstrom_data_store
{
	dlist_node	pds_chain;	/* link to GpuContext->pds_list */
	cl_int		refcnt;		/* reference counter */
	Size		kds_length;	/* length of the kernel data store */
	kern_data_store *kds;
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
                        Size block_size_max,
						cl_int **pp_keep_freemem,
						cl_int **pp_num_host_malloc,
						cl_int **pp_num_host_mfree,
						struct timeval **pp_tv_host_malloc,
						struct timeval **pp_tv_host_mfree);
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
extern void pgstrom_init_gputaskstate(GpuContext *gcontext,
									  GpuTaskState *gts,
									  EState *estate);
extern void pgstrom_activate_gputaskstate(GpuTaskState *gts);
extern void pgstrom_deactivate_gputaskstate(GpuTaskState *gts);
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
extern void optimal_workgroup_size(size_t *p_grid_size,
								   size_t *p_block_size,
								   CUfunction function,
								   CUdevice device,
								   size_t nitems,
								   size_t dynamic_shmem_per_block,
								   size_t dynamic_shmem_per_thread);
extern void largest_workgroup_size(size_t *p_grid_size,
								   size_t *p_block_size,
								   CUfunction function,
								   CUdevice device,
								   size_t nitems,
								   size_t dynamic_shmem_per_block,
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
extern bool pgstrom_load_cuda_program(GpuTaskState *gts, bool is_preload);
extern CUmodule *plcuda_load_cuda_program(GpuContext *gcontext,
										  const char *kern_source,
										  cl_uint extra_flags);
extern char *pgstrom_build_session_info(GpuTaskState *gts,
										const char *kern_source,
										cl_uint extra_flags);
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
	List	   *expr_defs;	/* list of devexpr_info in use */
	List	   *used_params;/* list of Const/Param in use */
	List	   *used_vars;	/* list of Var in use */
	Bitmapset  *param_refs;	/* referenced parameters */
	const char *var_label;	/* prefix of var reference, if exist */
	const char *kds_label;	/* label to reference kds, if exist */
	const char *kds_index_label; /* label to reference kds_index, if exist */
	List	   *pseudo_tlist;/* pseudo tlist expression, if any */
	int			extra_flags;/* external libraries to be included */
} codegen_context;

extern void pgstrom_codegen_typeoid_declarations(StringInfo buf);
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
extern void pgstrom_codegen_expr_declarations(StringInfo buf,
											  codegen_context *context);
extern void pgstrom_codegen_param_declarations(StringInfo buf,
											   codegen_context *context);
extern void pgstrom_codegen_var_declarations(StringInfo buf,
											 codegen_context *context);
extern void codegen_tempvar_declaration(StringInfo buf, const char *varname);
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
extern pgstrom_data_store *BulkExecProcNode(GpuTaskState *gts,
											 size_t chunk_size);
extern bool pgstrom_fetch_data_store(TupleTableSlot *slot,
									 pgstrom_data_store *pds,
									 size_t row_index,
									 HeapTuple tuple);
extern bool kern_fetch_data_store(TupleTableSlot *slot,
								  kern_data_store *kds,
								  size_t row_index,
								  HeapTuple tuple);
extern pgstrom_data_store *PDS_retain(pgstrom_data_store *pds);
extern void PDS_release(pgstrom_data_store *pds);
extern void PDS_expand_size(GpuContext *gcontext,
							pgstrom_data_store *pds,
							Size kds_length_new);
extern void PDS_shrink_size(pgstrom_data_store *pds);
extern void init_kernel_data_store(kern_data_store *kds,
								   TupleDesc tupdesc,
								   Size length,
								   int format,
								   uint nrooms,
								   bool use_internal);

extern pgstrom_data_store *PDS_create_row(GpuContext *gcontext,
										  TupleDesc tupdesc,
										  Size length);
extern pgstrom_data_store *PDS_create_slot(GpuContext *gcontext,
										   TupleDesc tupdesc,
										   cl_uint nrooms,
										   Size extra_length,
										   bool use_internal);
extern pgstrom_data_store *PDS_create_hash(GpuContext *gcontext,
										   TupleDesc tupdesc,
										   Size length);
extern int PDS_insert_block(pgstrom_data_store *pds,
							Relation rel,
							BlockNumber blknum,
							Snapshot snapshot,
							BufferAccessStrategy strategy);
extern bool PDS_insert_tuple(pgstrom_data_store *pds,
							 TupleTableSlot *slot);
extern bool PDS_insert_hashitem(pgstrom_data_store *pds,
								TupleTableSlot *slot,
								cl_uint hash_value);
extern void PDS_build_hashtable(pgstrom_data_store *pds);
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
extern void assign_gpusort_session_info(StringInfo buf, GpuTaskState *gts);
extern void pgstrom_init_gpusort(void);

/*
 * pl_cuda.c
 */
extern Datum plcuda_function_validator(PG_FUNCTION_ARGS);
extern Datum plcuda_function_handler(PG_FUNCTION_ARGS);
extern Datum plcuda_function_source(PG_FUNCTION_ARGS);
extern void pgstrom_init_plcuda(void);

/*
 * main.c
 */
extern bool		pgstrom_enabled;
extern bool		pgstrom_perfmon_enabled;
extern bool		pgstrom_bulkexec_enabled;
extern bool		pgstrom_cpu_fallback_enabled;
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
extern void pgstrom_explain_outer_bulkexec(GpuTaskState *gts,
										   List *deparse_context,
										   List *ancestors,
										   ExplainState *es);
extern void show_scan_qual(List *qual, const char *qlabel,
						   PlanState *planstate, List *ancestors,
						   ExplainState *es);
extern void show_instrumentation_count(const char *qlabel, int which,
									   PlanState *planstate, ExplainState *es);
extern void pgstrom_init_perfmon(GpuTaskState *gts);
extern void pgstrom_explain_gputaskstate(GpuTaskState *gts, ExplainState *es);

/*
 * macro definitions for performance counter
 */
#define PERFMON_BEGIN(pfm,tv1)					\
	do {										\
		if ((pfm)->enabled)						\
			gettimeofday((tv1), NULL);			\
	} while(0)

#define PERFMON_END(pfm,field,tv1,tv2)			\
	do {										\
		if ((pfm)->enabled)						\
		{										\
			gettimeofday((tv2), NULL);			\
			(pfm)->field +=												\
				((double)(((tv2)->tv_sec - (tv1)->tv_sec) * 1000000L +	\
						  ((tv2)->tv_usec - (tv1)->tv_usec)) / 1000.0);	\
		}										\
	} while(0)

#define CUDA_EVENT_RECORD(node,ev_field)						\
	do {														\
		if (((GpuTask *)(node))->gts->pfm.enabled)				\
		{														\
			CUresult __rc = cuEventRecord((node)->ev_field,		\
							((GpuTask *)(node))->cuda_stream);	\
			if (__rc != CUDA_SUCCESS)							\
				elog(ERROR, "failed on cuEventRecord: %s",		\
					 errorText(__rc));							\
		}														\
	} while(0)

#define CUDA_EVENT_CREATE(node,ev_field)						\
	do {														\
		if (((GpuTask *)(node))->gts->pfm.enabled)				\
		{														\
			CUresult __rc = cuEventCreate(&(node)->ev_field,	\
										  CU_EVENT_DEFAULT);	\
			if (__rc != CUDA_SUCCESS)							\
				elog(ERROR, "failed on cuEventCreate: %s",		\
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

#define CUDA_EVENT_ELAPSED(node,pfm_field,ev_start,ev_stop,bailout) \
	do {														\
		CUresult	__rc;										\
		float		__elapsed;									\
																\
		if ((ev_start) != NULL && (ev_stop) != NULL)			\
		{														\
			__rc = cuEventElapsedTime(&__elapsed,				\
									  (ev_start),				\
									  (ev_stop));				\
			if (__rc != CUDA_SUCCESS)							\
			{													\
				elog(WARNING, "failed on cuEventElapsedTime: %s",	\
					 errorText(__rc));							\
				goto bailout;									\
			}													\
			((GpuTask *)(node))->gts->pfm.pfm_field += __elapsed;	\
		}														\
	} while(0)

#define PFMON_TIMEVAL_DIFF(tv1,tv2)								\
	((cl_double)(((tv2)->tv_sec * 1000000 + (tv2)->tv_usec) -	\
				 ((tv1)->tv_sec * 1000000 + (tv1)->tv_usec)) / 1000000.0)

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

/*
 * Device Code generated from cuda_*.h
 */
extern const char *pgstrom_cuda_common_code;
extern const char *pgstrom_cuda_dynpara_code;
extern const char *pgstrom_cuda_matrix_code;
extern const char *pgstrom_cuda_gpuscan_code;
extern const char *pgstrom_cuda_gpujoin_code;
extern const char *pgstrom_cuda_gpupreagg_code;
extern const char *pgstrom_cuda_gpusort_code;
extern const char *pgstrom_cuda_mathlib_code;
extern const char *pgstrom_cuda_textlib_code;
extern const char *pgstrom_cuda_timelib_code;
extern const char *pgstrom_cuda_numeric_code;
extern const char *pgstrom_cuda_money_code;
extern const char *pgstrom_cuda_plcuda_code;
extern const char *pgstrom_cuda_terminal_code;

/* ----------------------------------------------------------------
 *
 * Miscellaneous static inline functions
 *
 * ---------------------------------------------------------------- */

/* Max/Min macros that takes 3 or more arguments */
#define Max3(a,b,c)		((a) > (b) ? Max((a),(c)) : Max((b),(c)))
#define Max4(a,b,c,d)	Max(Max((a),(b)), Max((c),(d)))

#define Min3(a,b,c)		((a) > (b) ? Min((a),(c)) : Min((b),(c)))
#define Min4(a,b,c,d)	Min(Min((a),(b)), Min((c),(d)))

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
#define forfour(lc1, list1, lc2, list2, lc3, list3, lc4, list4)		\
	for ((lc1) = list_head(list1), (lc2) = list_head(list2),		\
		 (lc3) = list_head(list3), (lc4) = list_head(list4);		\
		 (lc1) != NULL && (lc2) != NULL && (lc3) != NULL &&			\
		 (lc4) != NULL;												\
		 (lc1) = lnext(lc1), (lc2) = lnext(lc2), (lc3) = lnext(lc3),\
		 (lc4) = lnext(lc4))
#endif

static inline char *
format_bytesz(Size nbytes)
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
format_millisec(double milliseconds)
{
	if (milliseconds > 300000.0)    /* more then 5min */
		return psprintf("%.2fmin", milliseconds / 60000.0);
	else if (milliseconds > 8000.0) /* more than 8sec */
		return psprintf("%.2fsec", milliseconds / 1000.0);
	return psprintf("%.2fms", milliseconds);
}

/* old style */
static inline char *
milliseconds_unitary_format(double milliseconds)
{
	return format_millisec(milliseconds);
}

#endif	/* PG_STROM_H */
