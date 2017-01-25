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
#include "nodes/extensible.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "nodes/readfuncs.h"
#include "nodes/relation.h"
#include "port/atomics.h"
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
#include <float.h>
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

#define PGSTROM_SCHEMA_NAME		"pgstrom"

/*
 * GpuContext_v2
 *
 * 
 *
 *
 */
typedef struct SharedGpuContext
{
	dlist_node	chain;
	cl_uint		context_id;		/* a unique ID of the GpuContext */

	slock_t		lock;			/* lock of the field below */
	cl_int		refcnt;			/* refcount by backend/gpu-server */
	cl_int		device_id;		/* a unique index of the GPU device */
	PGPROC	   *server;			/* PGPROC of GPU/CUDA Server */
	PGPROC	   *backend;		/* PGPROC of Backend Process */
	/* state of asynchronous tasks */
	dlist_head	dma_buffer_list;/* tracker of DMA buffers */
	cl_int		num_async_tasks;/* num of tasks passed to GPU server */
	/* performance monitor */
	struct {
		bool		enabled;
		cl_uint		num_dmabuf_alloc;
		cl_uint		num_dmabuf_free;
		cl_uint		num_gpumem_alloc;
		cl_uint		num_gpumem_free;
		cl_uint		num_iomapped_alloc;
		cl_uint		num_iomapped_free;
		cl_float	tv_dmabuf_alloc;
		cl_float	tv_dmabuf_free;
		cl_float	tv_gpumem_alloc;
		cl_float	tv_gpumem_free;
		cl_float	tv_iomapped_alloc;
		cl_float	tv_iomapped_free;
		size_t		size_dmabuf_total;
		size_t		size_gpumem_total;
		size_t		size_iomapped_total;
	} pfm;
} SharedGpuContext;

#define INVALID_GPU_CONTEXT_ID		(-1)

typedef struct GpuContext_v2
{
	dlist_node		chain;
	cl_int			refcnt;		/* refcount by local GpuTaskState */
	pgsocket		sockfd;		/* connection between backend <-> server */
	ResourceOwner	resowner;
	SharedGpuContext *shgcon;
	dlist_head		restrack[FLEXIBLE_ARRAY_MEMBER];
} GpuContext_v2;

/* Identifier of the Gpu Programs */
typedef cl_long					ProgramId;
#define INVALID_PROGRAM_ID		(-1L)

/*
 * GpuTask and related
 */
typedef enum {
	GpuTaskKind_GpuScan,
	GpuTaskKind_GpuJoin,
	GpuTaskKind_GpuPreAgg,
	GpuTaskKind_GpuSort,
	GpuTaskKind_PL_CUDA,
} GpuTaskKind;

/* definition of performance-monitor structure */
#include "perfmon.h"

typedef struct GpuTask_v2		GpuTask_v2;
typedef struct GpuTaskState_v2	GpuTaskState_v2;

/*
 * GpuTaskState
 *
 * A common structure of the state machine of GPU related tasks.
 */
struct NVMEScanState;

struct GpuTaskState_v2
{
	CustomScanState	css;
	GpuContext_v2  *gcontext;
	GpuTaskKind		task_kind;		/* one of GpuTaskKind_* */
	ProgramId		program_id;		/* CUDA Program (to be acquired) */
	cl_uint			revision;		/* incremented for each ExecRescan */
	kern_parambuf  *kern_params;	/* Const/Param buffer */
	bool			scan_done;		/* True, if no more rows to read */
	bool			row_format;		/* True, if KDS_FORMAT_ROW is required */

	/* fields for outer scan */
	bool			outer_bulk_exec;/* True, if bulk-exec mode is supported */
	Cost			outer_startup_cost;	/* copy from the outer path node */
	Cost			outer_total_cost;	/* copy from the outer path node */
	double			outer_plan_rows;	/* copy from the outer path node */
	int				outer_plan_width;	/* copy from the outer path node */
	cl_uint			outer_nrows_per_block;
	Instrumentation	outer_instrument; /* runtime statistics, if any */
	TupleTableSlot *scan_overflow;	/* temporary buffer, if no space on PDS */
	struct NVMEScanState *nvme_sstate;/* A state object for NVMe-Strom.
									   * If not NULL, GTS prefers BLOCK format
									   * as source data store. Then, SSD2GPU
									   * Direct DMA will be kicked.
									   */
	/*
	 * fields to fetch rows from the current task
	 *
	 * NOTE: @curr_index is sufficient to point a particular row of KDS,
	 * if format is ROW, HASH and SLOT. However, BLOCK format has no direct
	 * pointer for each rows. It contains @nitems blocks and individual block
	 * contains uncertain number of rows. So, at BLOCK format, @curr_index
	 * is index of the current block, and @curr_lp_index is also index of
	 * the current line pointer.
	 * For all format, @curr_index == @nitems means no rows any more.
	 */
	cl_long			curr_index;		/* current position on the curr_task */
	cl_long			curr_lp_index;	/* index of LinePointer in a block */
	HeapTupleData	curr_tuple;		/* internal use of PDS_fetch() */
	struct GpuTask_v2 *curr_task;	/* a GpuTask currently processed */

	/* callbacks used by gputasks.c */
	GpuTask_v2	 *(*cb_next_task)(GpuTaskState_v2 *gts);
	void		  (*cb_ready_task)(GpuTaskState_v2 *gts, GpuTask_v2 *gtask);
	void		  (*cb_switch_task)(GpuTaskState_v2 *gts, GpuTask_v2 *gtask);
	TupleTableSlot *(*cb_next_tuple)(GpuTaskState_v2 *gts);
	struct pgstrom_data_store *(*cb_bulk_exec)(GpuTaskState_v2 *gts,
											   size_t chunk_size);
	/* list to manage GpuTasks */
	dlist_head		ready_tasks;	/* list of tasks already processed */
	cl_uint			num_ready_tasks;/* length of the list above */

	/* performance monitor */
	pgstrom_perfmon	pfm;			/* local counter */
	pgstromWorkerStatistics *worker_stat;

//	pgstrom_perfmon *pfm_master;	/* shared state, if under Gather */
};
#define GTS_GET_SCAN_TUPDESC(gts)				\
	(((GpuTaskState_v2 *)(gts))->css.ss.ss_ScanTupleSlot->tts_tupleDescriptor)
#define GTS_GET_RESULT_TUPDESC(gts)				\
	(((GpuTaskState_v2 *)(gts))->css.ss.ps.ps_ResultTupleSlot->tts_tupleDescriptor)

/*
 * GpuTask
 *
 * It is a unit of task to be sent GPU server. Thus, this object must be
 * allocated on the DMA buffer area.
 */
struct GpuTask_v2
{
	kern_errorbuf	kerror;			/* error status of the task */
	dlist_node		chain;			/* link to the task state list */
	GpuTaskKind		task_kind;		/* same with GTS's one */
	ProgramId		program_id;		/* same with GTS's one */
	GpuTaskState_v2 *gts;			/* GTS reference in the backend */
	cl_uint			revision;		/* same with GTS's one when kicked */
	bool			row_format;		/* true, if row-format is preferred */
	bool			cpu_fallback;	/* true, if task needs CPU fallback */
	bool			perfmon;		/* true, if perfmon is required */
	int				file_desc;		/* file-descriptor on backend side */
	/* fields below are valid only server */
	GpuContext_v2  *gcontext;		/* session info of GPU server */
	CUstream		cuda_stream;	/* stream object assigned on the task */
	int				peer_fdesc;		/* FD moved via SCM_RIGHTS */
	unsigned long	dma_task_id;	/* Task-ID of Async SSD2GPU DMA */
};
typedef struct GpuTask_v2 GpuTask_v2;

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
	const char *max_const;		/* static initializer, if any */
	const char *min_const;		/* static initializer, if any */
	const char *zero_const;		/* static initializer, if any */
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
 * pgstrom_data_store - a data structure with various format to exchange
 * a data chunk between the host and CUDA server.
 */
typedef struct pgstrom_data_store
{
	pg_atomic_uint32	refcnt;		/* reference counter */

	/*
	 * NOTE: Extra information for KDS_FORMAT_BLOCK.
	 * @nblocks_uncached is number of PostgreSQL blocks, to be processed
	 * by NVMe-Strom. If @nblocks_uncached > 0, the tail of PDS shall be
	 * filled up by an array of strom_dma_chunk.
	 */
	cl_uint				nblocks_uncached;

	/*
	 * NOTE: @ntasks_running is an independent counter regardless of the
	 * @refcnt. It represents number of concurrent tasks which reference
	 * the PDS. So, once @ntasks_running gets back to zero when no new
	 * tasks will be never attached any more, we can determine it is the
	 * last task that references this PDS.
	 * GpuPreAgg uses this mechanism to terminate its final reduction
	 * buffer.
	 * datastore.c does not care about this counter, so individual logics
	 * have to manage the counter with proper locking mechanism by itself.
	 */
	cl_uint				ntasks_running;

	/* data chunk in kernel portion */
	kern_data_store kds	__attribute__ ((aligned (sizeof(cl_ulong))));
} pgstrom_data_store;

/*
 * State structure of NVMe-Strom per GpuTaskState
 */
typedef struct NVMEScanState
{
	cl_uint			nrows_per_block;
	cl_uint			nblocks_per_chunk;
	BlockNumber		curr_segno;
	Buffer			curr_vmbuffer;
	BlockNumber		nr_segs;
	struct {
		File		vfd;
		BlockNumber	segno;
	} mdfd[FLEXIBLE_ARRAY_MEMBER];
} NVMEScanState;

/*
 * --------------------------------------------------------------------
 *
 * Function Declarations
 *
 * --------------------------------------------------------------------
 */

/*
 * gpu_device.c
 */
typedef struct DevAttributes
{
	cl_int		DEV_ID;
	char		DEV_NAME[256];
	size_t		DEV_TOTAL_MEMSZ;
	cl_int		CORES_PER_MPU;
#define DEV_ATTR(LABEL,a,b,c)					\
	cl_int		LABEL;
#include "device_attrs.h"
#undef DEV_ATTR
} DevAttributes;

extern DevAttributes   *devAttrs;
extern cl_int			numDevAttrs;
extern cl_ulong			devComputeCapability;
extern cl_ulong			devBaselineMemorySize;
extern cl_uint			devBaselineMaxThreadsPerBlock;

extern bool	gpu_scoreboard_mem_alloc(size_t nbytes);
extern void	gpu_scoreboard_mem_free(size_t nbytes);
extern void gpu_scoreboard_dump(void);

extern void pgstrom_init_gpu_device(void);
extern Datum pgstrom_device_info(PG_FUNCTION_ARGS);

/*
 * dma_buffer.c
 */
extern void *dmaBufferAlloc(GpuContext_v2 *gcontext, Size required);
extern void *dmaBufferRealloc(void *pointer, Size required);
extern bool dmaBufferValidatePtr(void *pointer);
extern Size dmaBufferSize(void *pointer);
extern Size dmaBufferChunkSize(void *pointer);
extern void dmaBufferFree(void *pointer);
extern void dmaBufferFreeAll(SharedGpuContext *shgcon);
extern Size dmaBufferMaxAllocSize(void);
extern Datum pgstrom_dma_buffer_alloc(PG_FUNCTION_ARGS);
extern Datum pgstrom_dma_buffer_free(PG_FUNCTION_ARGS);
extern Datum pgstrom_dma_buffer_info(PG_FUNCTION_ARGS);
extern void pgstrom_init_dma_buffer(void);

/*
 * gpu_context.c
 */
extern GpuContext_v2 *MasterGpuContext(void);
extern GpuContext_v2 *AllocGpuContext(bool with_connection);
extern GpuContext_v2 *AttachGpuContext(pgsocket sockfd,
									   cl_int context_id,
									   BackendId backend_id,
									   cl_int device_id);
extern GpuContext_v2 *GetGpuContext(GpuContext_v2 *gcontext);
extern bool PutGpuContext(GpuContext_v2 *gcontext);
extern bool ForcePutGpuContext(GpuContext_v2 *gcontext);
extern bool GpuContextIsEstablished(GpuContext_v2 *gcontext);

extern Size gpuMemMaxAllocSize(void);
extern CUresult	gpuMemAlloc_v2(GpuContext_v2 *gcontext,
							   CUdeviceptr *p_devptr, size_t bytesize);
extern CUresult	gpuMemFree_v2(GpuContext_v2 *gcontext, CUdeviceptr devptr);
extern void trackCudaProgram(GpuContext_v2 *gcontext, ProgramId program_id);
extern void untrackCudaProgram(GpuContext_v2 *gcontext, ProgramId program_id);
extern void trackIOMapMem(GpuContext_v2 *gcontext, CUdeviceptr devptr);
extern void untrackIOMapMem(GpuContext_v2 *gcontext, CUdeviceptr devptr);
extern void trackSSD2GPUDMA(GpuContext_v2 *gcontext,
							unsigned long dma_task_id);
extern void untrackSSD2GPUDMA(GpuContext_v2 *gcontext,
							  unsigned long dma_task_id);
extern void pgstrom_init_gpu_context(void);

/*
 * gpu_server.c
 */
extern int				gpuserv_cuda_dindex;
extern CUdevice			gpuserv_cuda_device;
extern CUcontext		gpuserv_cuda_context;

extern bool IsGpuServerProcess(void);
extern int	numGpuServerProcesses(void);
extern void gpuservTryToWakeUp(void);
extern void gpuservOpenConnection(GpuContext_v2 *gcontext);
extern bool gpuservSendGpuTask(GpuContext_v2 *gcontext, GpuTask_v2 *gtask);
extern bool gpuservRecvGpuTasks(GpuContext_v2 *gcontext, long timeout);
extern void gpuservPushGpuTask(GpuContext_v2 *gcontext, GpuTask_v2 *gtask);
extern void gpuservCompleteGpuTask(GpuTask_v2 *gtask, bool is_urgent);

extern void pgstrom_init_gpu_server(void);

/* service routines */
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

/*
 * gpu_tasks.c
 */
extern void pgstromInitGpuTaskState(GpuTaskState_v2 *gts,
									GpuContext_v2 *gcontext,
									GpuTaskKind task_kind,
									List *used_params,
									EState *estate);
extern TupleTableSlot *pgstromExecGpuTaskState(GpuTaskState_v2 *gts);
extern pgstrom_data_store *pgstromBulkExecGpuTaskState(GpuTaskState_v2 *gts,
													   size_t chunk_size);
extern void pgstromRescanGpuTaskState(GpuTaskState_v2 *gts);
extern void pgstromReleaseGpuTaskState(GpuTaskState_v2 *gts);
extern void pgstromExplainGpuTaskState(GpuTaskState_v2 *gts,
									   ExplainState *es);
extern void pgstrom_merge_worker_statistics(GpuTaskState_v2 *gts);
extern void pgstromExplainOuterScan(GpuTaskState_v2 *gts,
									List *deparse_context,
									List *ancestors,
									ExplainState *es,
									List *outer_quals,
									Cost outer_startup_cost,
									Cost outer_total_cost,
									double outer_plan_rows,
									int outer_plan_width);

extern void pgstromInitGpuTask(GpuTaskState_v2 *gts, GpuTask_v2 *gtask);
extern int	pgstromProcessGpuTask(GpuTask_v2 *gtask,
								  CUmodule cuda_module,
								  CUstream cuda_stream);
extern int	pgstromCompleteGpuTask(GpuTask_v2 *gtask);
extern void pgstromReleaseGpuTask(GpuTask_v2 *gtask);

extern const char *__errorText(int errcode, const char *filename, int lineno);
#define errorText(errcode)		__errorText((errcode),__FILE__,__LINE__)
extern const char *errorTextKernel(kern_errorbuf *kerror);
extern void pgstrom_init_gputasks(void);

/*
 * cuda_program.c
 */
extern void pgstrom_get_cuda_program(GpuContext_v2 *gcontext,
									 ProgramId program_id);
extern void pgstrom_put_cuda_program(GpuContext_v2 *gcontext,
									 ProgramId program_id);
extern ProgramId pgstrom_create_cuda_program(GpuContext_v2 *gcontext,
											 cl_uint extra_flags,
											 const char *kern_source,
											 const char *kern_define,
											 bool try_async_build);
extern char *pgstrom_build_session_info(cl_uint extra_flags,
										GpuTaskState_v2 *gts);
extern CUmodule pgstrom_load_cuda_program(ProgramId program_id, long timeout);
extern bool pgstrom_wait_cuda_program(ProgramId program_id, long timeout);

extern ProgramId pgstrom_try_build_cuda_program(void);

extern const char *pgstrom_cuda_source_file(ProgramId program_id);
#if 0
extern bool pgstrom_load_cuda_program_legacy(GpuTaskState *gts,
											 bool is_preload);
extern CUmodule *plcuda_load_cuda_program_legacy(GpuContext *gcontext,
												 const char *kern_source,
												 cl_uint extra_flags);
extern void pgstrom_assign_cuda_program(GpuTaskState *gts,
										List *used_params,
										const char *kern_source,
										int extra_flags);
#endif
extern void pgstrom_init_cuda_program(void);
//extern Datum pgstrom_program_info(PG_FUNCTION_ARGS);

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
extern bool pgstrom_device_expression(Expr *expr);
extern void pgstrom_init_codegen_context(codegen_context *context);
extern void pgstrom_init_codegen(void);

/*
 * datastore.c
 */
extern bool pgstrom_bulk_exec_supported(const PlanState *planstate);
extern cl_uint estimate_num_chunks(Path *pathnode);
extern bool pgstrom_fetch_data_store(TupleTableSlot *slot,
									 pgstrom_data_store *pds,
									 size_t row_index,
									 HeapTuple tuple);
/*
extern bool kern_fetch_data_store(TupleTableSlot *slot,
								  kern_data_store *kds,
								  size_t row_index,
								  HeapTuple tuple);
*/
extern bool PDS_fetch_tuple(TupleTableSlot *slot,
							pgstrom_data_store *pds,
							GpuTaskState_v2 *gts);
extern pgstrom_data_store *PDS_retain(pgstrom_data_store *pds);
extern void PDS_release(pgstrom_data_store *pds);
extern pgstrom_data_store *PDS_expand_size(GpuContext_v2 *gcontext,
										   pgstrom_data_store *pds,
										   Size kds_length_new);
extern void PDS_shrink_size(pgstrom_data_store *pds);
extern void init_kernel_data_store(kern_data_store *kds,
								   TupleDesc tupdesc,
								   Size length,
								   int format,
								   uint nrooms,
								   bool use_internal);

extern pgstrom_data_store *PDS_create_row(GpuContext_v2 *gcontext,
										  TupleDesc tupdesc,
										  Size length);
extern pgstrom_data_store *PDS_create_slot(GpuContext_v2 *gcontext,
										   TupleDesc tupdesc,
										   cl_uint nrooms,
										   Size extra_length,
										   bool use_internal);
extern pgstrom_data_store *PDS_duplicate_slot(GpuContext_v2 *gcontext,
											  kern_data_store *kds_head,
											  cl_uint nrooms,
											  cl_uint extra_unitsz);
extern pgstrom_data_store *PDS_create_hash(GpuContext_v2 *gcontext,
										   TupleDesc tupdesc,
										   Size length);
extern pgstrom_data_store *PDS_create_block(GpuContext_v2 *gcontext,
											TupleDesc tupdesc,
											struct NVMEScanState *nvme_sstate);
extern void PDS_init_heapscan_state(GpuTaskState_v2 *gts,
									cl_uint nrows_per_block);
extern void PDS_end_heapscan_state(GpuTaskState_v2 *gts);
extern bool PDS_exec_heapscan(GpuTaskState_v2 *gts,
							  pgstrom_data_store *pds,
							  int *p_filedesc);
extern cl_uint NVMESS_NBlocksPerChunk(struct NVMEScanState *nvme_sstate);

#define PGSTROM_DATA_STORE_BLOCK_FILEPOS(pds)							\
	((loff_t *)((char *)KERN_DATA_STORE_BLOCK_PGPAGE(&(pds)->kds,		\
													 (pds)->kds.nrooms) - \
				(sizeof(loff_t) * (pds)->nblocks_uncached)))
extern void PDS_fillup_blocks(pgstrom_data_store *pds, int file_desc);

extern bool PDS_insert_tuple(pgstrom_data_store *pds,
							 TupleTableSlot *slot);
extern bool PDS_insert_hashitem(pgstrom_data_store *pds,
								TupleTableSlot *slot,
								cl_uint hash_value);
extern void PDS_build_hashtable(pgstrom_data_store *pds);
extern void pgstrom_init_datastore(void);

/*
 * nvme_strom.c
 */
#include "nvme_strom.h"

extern Size gpuMemSizeIOMap(void);
extern CUresult	gpuMemAllocIOMap(GpuContext_v2 *gcontext,
								 CUdeviceptr *p_devptr, size_t bytesize);
extern CUresult	gpuMemFreeIOMap(GpuContext_v2 *gcontext,
								CUdeviceptr devptr);
extern void gpuMemCopyFromSSDAsync(GpuTask_v2 *gtask,
								   CUdeviceptr m_kds,
								   pgstrom_data_store *pds,
								   CUstream cuda_stream);
extern void gpuMemCopyFromSSDWait(GpuTask_v2 *gtask,
								  CUstream cuda_stream);

extern void dump_iomap_buffer_info(void);
extern Datum pgstrom_iomap_buffer_info(PG_FUNCTION_ARGS);
extern void pgstrom_init_nvme_strom(void);

extern bool ScanPathWillUseNvmeStrom(PlannerInfo *root, RelOptInfo *baserel);
extern bool RelationCanUseNvmeStrom(Relation relation);
extern bool RelationWillUseNvmeStrom(Relation relation,
									 BlockNumber *p_nr_blocks);

/*
 * gpuscan.c
 */
extern bool cost_discount_gpu_projection(PlannerInfo *root, RelOptInfo *rel,
										 Cost *p_discount_per_tuple);
extern void cost_gpuscan_common(PlannerInfo *root,
								RelOptInfo *scan_rel,
								List *scan_quals,
								int parallel_workers,
								double *p_parallel_divisor,
								double *p_scan_ntuples,
								double *p_scan_nchunks,
								cl_uint *p_nrows_per_block,
								Cost *p_startup_cost,
								Cost *p_run_cost);
extern void codegen_gpuscan_quals(StringInfo kern,
								  codegen_context *context,
								  Index scanrelid,
								  List *dev_quals);
extern bool add_unique_expression(Expr *expr, List **p_tlist, bool resjunk);
extern bool pgstrom_pullup_outer_scan(const Path *outer_path,
									  Index *p_outer_relid,
									  List **p_outer_quals,
									  bool *parallel_aware);
extern bool pgstrom_path_is_gpuscan(const Path *path);
extern bool pgstrom_plan_is_gpuscan(const Plan *plan);

extern void gpuscan_rewind_position(GpuTaskState_v2 *gts);

extern pgstrom_data_store *gpuscanExecScanChunk(GpuTaskState_v2 *gts,
												int *p_filedesc);
extern void gpuscanRewindScanChunk(GpuTaskState_v2 *gts);

extern Size ExecGpuScanEstimateDSM(CustomScanState *node,
								   ParallelContext *pcxt);
extern void ExecGpuScanInitDSM(CustomScanState *node,
							   ParallelContext *pcxt,
							   void *coordinate);
extern void ExecGpuScanInitWorker(CustomScanState *node,
								  shm_toc *toc,
								  void *coordinate);
extern int	gpuscan_process_task(GpuTask_v2 *gtask,
								 CUmodule cuda_module,
								 CUstream cuda_stream);
extern int	gpuscan_complete_task(GpuTask_v2 *gtask);
extern void gpuscan_release_task(GpuTask_v2 *gtask);
extern void assign_gpuscan_session_info(StringInfo buf,
										GpuTaskState_v2 *gts);
extern void pgstrom_init_gpuscan(void);

/*
 * gpujoin.c
 */
extern bool pgstrom_path_is_gpujoin(Path *pathnode);
extern bool pgstrom_plan_is_gpujoin(const Plan *plannode);
extern int	gpujoin_process_task(GpuTask_v2 *gtask,
								 CUmodule cuda_module,
								 CUstream cuda_stream);
extern int	gpujoin_complete_task(GpuTask_v2 *gtask);
extern void	gpujoin_release_task(GpuTask_v2 *gtask);
extern void assign_gpujoin_session_info(StringInfo buf,
										GpuTaskState_v2 *gts);
extern void gpujoin_merge_worker_statistics(GpuTaskState_v2 *gts);
extern void gpujoin_accum_worker_statistics(GpuTaskState_v2 *gts);
extern void	pgstrom_init_gpujoin(void);

/*
 * gpupreagg.c
 */
extern bool pgstrom_plan_is_gpupreagg(const Plan *plan);
extern void gpupreagg_post_planner(PlannedStmt *pstmt, CustomScan *cscan);
extern int	gpupreagg_process_task(GpuTask_v2 *gtask,
								   CUmodule cuda_module,
								   CUstream cuda_stream);
extern int	gpupreagg_complete_task(GpuTask_v2 *gtask);
extern void	gpupreagg_release_task(GpuTask_v2 *gtask);
extern void assign_gpupreagg_session_info(StringInfo buf,
										  GpuTaskState_v2 *gts);
extern void pgstrom_init_gpupreagg(void);

/*
 * gpusort.c
 */
extern void pgstrom_try_insert_gpusort(PlannedStmt *pstmt, Plan **p_plan);
extern bool pgstrom_plan_is_gpusort(const Plan *plan);
extern void assign_gpusort_session_info(StringInfo buf, GpuTaskState_v2 *gts);
extern void pgstrom_init_gpusort(void);

/*
 * pl_cuda.c
 */
extern Datum plcuda_function_validator(PG_FUNCTION_ARGS);
extern Datum plcuda_function_handler(PG_FUNCTION_ARGS);
extern Datum plcuda_function_source(PG_FUNCTION_ARGS);
extern void pgstrom_init_plcuda(void);

/*
 * matrix.c
 */
extern Datum array_matrix_accum(PG_FUNCTION_ARGS);
extern Datum array_matrix_accum_varbit(PG_FUNCTION_ARGS);
extern Datum varbit_to_int4_array(PG_FUNCTION_ARGS);
extern Datum int4_array_to_varbit(PG_FUNCTION_ARGS);
extern Datum array_matrix_final_int2(PG_FUNCTION_ARGS);
extern Datum array_matrix_final_int4(PG_FUNCTION_ARGS);
extern Datum array_matrix_final_int8(PG_FUNCTION_ARGS);
extern Datum array_matrix_final_float4(PG_FUNCTION_ARGS);
extern Datum array_matrix_final_float8(PG_FUNCTION_ARGS);
extern Datum array_matrix_unnest(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_int2(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_int4(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_int8(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_float4(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_float8(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_int2(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_int4(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_int8(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_float4(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_float8(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_accum(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_final_int2(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_final_int4(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_final_int8(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_final_float4(PG_FUNCTION_ARGS);
extern Datum array_matrix_rbind_final_float8(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_accum(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_final_int2(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_final_int4(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_final_int8(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_final_float4(PG_FUNCTION_ARGS);
extern Datum array_matrix_cbind_final_float8(PG_FUNCTION_ARGS);
extern Datum array_matrix_transpose_int2(PG_FUNCTION_ARGS);
extern Datum array_matrix_transpose_int4(PG_FUNCTION_ARGS);
extern Datum array_matrix_transpose_int8(PG_FUNCTION_ARGS);
extern Datum array_matrix_transpose_float4(PG_FUNCTION_ARGS);
extern Datum array_matrix_transpose_float8(PG_FUNCTION_ARGS);
extern Datum float4_as_int4(PG_FUNCTION_ARGS);
extern Datum int4_as_float4(PG_FUNCTION_ARGS);
extern Datum float8_as_int8(PG_FUNCTION_ARGS);
extern Datum int8_as_float8(PG_FUNCTION_ARGS);
extern Datum array_matrix_validation(PG_FUNCTION_ARGS);
extern Datum array_matrix_height(PG_FUNCTION_ARGS);
extern Datum array_matrix_width(PG_FUNCTION_ARGS);
extern Datum array_matrix_rawsize(PG_FUNCTION_ARGS);

/*
 * main.c
 */
extern bool		pgstrom_enabled;
extern bool		pgstrom_perfmon_enabled;
extern bool		pgstrom_debug_kernel_source;
extern bool		pgstrom_bulkexec_enabled;
extern bool		pgstrom_cpu_fallback_enabled;
extern int		pgstrom_max_async_tasks;
extern int		pgstrom_min_async_tasks;
extern double	pgstrom_gpu_setup_cost;
extern double	pgstrom_gpu_dma_cost;
extern double	pgstrom_gpu_operator_cost;
extern double	pgstrom_nrows_growth_ratio_limit;
extern double	pgstrom_nrows_growth_margin;
extern double	pgstrom_chunk_size_margin;
extern Size		pgstrom_chunk_size(void);
extern Size		pgstrom_chunk_size_limit(void);

extern Path *pgstrom_create_dummy_path(PlannerInfo *root,
									   Path *subpath,
									   PathTarget *target);
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
 * pmakeFloat - for convenient; makeFloat + psprintf
 */
static inline Value *
pmakeFloat(cl_double float_value)
{
	return makeFloat(psprintf("%.*e", DBL_DIG+3, float_value));
}

/*
 * get_prev_log2
 *
 * It returns N of the largest 2^N value that is smaller than or equal to
 * the supplied value.
 */
static inline int
get_prev_log2(Size size)
{
	int		shift = 0;

	if (size == 0 || size == 1)
		return 0;
	size >>= 1;
#if __GNUC__
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
