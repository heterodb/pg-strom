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
#include "storage/lock.h"
#include "storage/fd.h"
#include "storage/spin.h"
#include "utils/resowner.h"
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <sys/time.h>
#include "cuda.h"
#include "device_common.h"

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

/*
 * --------------------------------------------------------------------
 *
 * Constant Definitions
 *
 * --------------------------------------------------------------------
 */
#define PGSTROM_TEMP_DIR	"pg_strom_temp"





/*
 * --------------------------------------------------------------------
 *
 * Type Definitions
 *
 * --------------------------------------------------------------------
 */

/*
 * Tag of shared memory object classes
 */
typedef enum {
	StromTag_DevProgram = 1001,
	StromTag_MsgQueue,
	StromTag_ParamBuf,
	StromTag_DataStore,
	StromTag_GpuScan,
	StromTag_GpuHashJoin,
	StromTag_HashJoinTable,
	StromTag_GpuPreAgg,
	StromTag_GpuSort,
} StromTag;

typedef struct {
	StromTag	stag;			/* StromTag_* */
} StromObject;

#define StromTagIs(PTR,IDENT) \
	(((StromObject *)(PTR))->stag == StromTag_##IDENT)

static inline const char *
StromTagGetLabel(StromObject *sobject)
{
	static char msgbuf[80];
#define StromTagGetLabelEntry(IDENT)		\
	case StromTag_##IDENT: return #IDENT

	switch (sobject->stag)
	{
		StromTagGetLabelEntry(DevProgram);
		StromTagGetLabelEntry(MsgQueue);
		StromTagGetLabelEntry(ParamBuf);
		StromTagGetLabelEntry(DataStore);
		StromTagGetLabelEntry(GpuScan);
		StromTagGetLabelEntry(GpuPreAgg);
		StromTagGetLabelEntry(GpuHashJoin);
		StromTagGetLabelEntry(HashJoinTable);
		StromTagGetLabelEntry(GpuSort);
		default:
			snprintf(msgbuf, sizeof(msgbuf),
					 "unknown tag (%u)", sobject->stag);
			break;
	}
#undef StromTagGetLabelEntry
	return msgbuf;
}

/*
 * Performance monitor structure
 */
typedef struct {
	cl_bool		enabled;
	cl_uint		num_samples;
	/*-- perfmon to load and materialize --*/
	cl_ulong	time_inner_load;	/* time to load the inner relation */
	cl_ulong	time_outer_load;	/* time to load the outer relation */
	cl_ulong	time_materialize;	/* time to materialize the result */
	/*-- perfmon for message exchanging --*/
	cl_ulong	time_in_sendq;		/* waiting time in the server mqueue */
	cl_ulong	time_in_recvq;		/* waiting time in the response mqueue */
	cl_ulong	time_kern_build;	/* max time to build opencl kernel */
	/*-- perfmon for DMA send/recv --*/
	cl_uint		num_dma_send;	/* number of DMA send request */
	cl_uint		num_dma_recv;	/* number of DMA receive request */
	cl_ulong	bytes_dma_send;	/* bytes of DMA send */
	cl_ulong	bytes_dma_recv;	/* bytes of DMA receive */
	cl_ulong	time_dma_send;	/* time to send host=>device data */
	cl_ulong	time_dma_recv;	/* time to receive device=>host data */
	/*-- perfmon for kernel execution --*/
	cl_uint		num_kern_exec;	/* number of main kernel execution */
	cl_ulong	time_kern_exec;	/* time to execute main kernel */
	/*-- (special perfmon for gpuhashjoin) --*/
	cl_uint		num_kern_proj;	/* number of projection kernel execution */
	cl_ulong	time_kern_proj;	/* time to execute projection kernel */
	/*-- (special perfmon for gpupreagg) --*/
	cl_uint		num_kern_prep;	/* number of preparation kernel execution */
	cl_uint		num_kern_lagg;	/* number of local reduction kernel exec */
	cl_uint		num_kern_gagg;	/* number of global reduction kernel exec */
	cl_ulong	time_kern_prep;	/* time to execute preparation kernel */
	cl_ulong	time_kern_lagg;	/* time to execute local reduction kernel */
	cl_ulong	time_kern_gagg;	/* time to execute global reduction kernel */
	/*-- (special perfmon for gpusort) --*/
	cl_uint		num_gpu_sort;	/* number of GPU bitonic sort execution */
	cl_uint		num_cpu_sort;	/* number of GPU merge sort execution */
	cl_ulong	time_gpu_sort;	/* time to execute GPU bitonic sort */
	cl_ulong	time_cpu_sort;	/* time to execute CPU merge sort */
	cl_ulong	time_cpu_sort_real;	/* real time to execute CPU merge sort */
	cl_ulong	time_bgw_sync;	/* time to synchronich BGWorkers */

	/*-- for debugging usage --*/
	cl_ulong	time_debug1;	/* time for debugging purpose.1 */
	cl_ulong	time_debug2;	/* time for debugging purpose.2 */
	cl_ulong	time_debug3;	/* time for debugging purpose.3 */
	cl_ulong	time_debug4;	/* time for debugging purpose.4 */

	struct timeval	tv;	/* result of gettimeofday(2) when enqueued */
} pgstrom_perfmon;

#define timeval_diff(tv1,tv2)						\
	(((tv2)->tv_sec * 1000000L + (tv2)->tv_usec) -	\
	 ((tv1)->tv_sec * 1000000L + (tv1)->tv_usec))

/*
 *
 *
 *
 *
 */
typedef struct
{
	dlist_node		chain;
	int				refcnt;
	ResourceOwner	resowner;
	MemoryContext	memcxt;
	dlist_head		state_list;		/* list of GpuTaskState */
	dlist_head		pds_list;		/* list of pgstrom_data_store */
	cl_int			num_context;
	cl_int			cur_context;
	CUcontext		dev_context[FLEXIBLE_ARRAY_MEMBER];
} GpuContext;

typedef struct GpuTaskState
{
	dlist_node		chain;
	GpuContext	   *gcontext;
	const char	   *kern_source;
	cl_uint			extra_flags;
	CUmodule		cuda_module;	/* module object built from cuda_binary */
	slock_t			lock;			/* protection of the list below */
	dlist_head		running_tasks;
	dlist_head		pending_tasks;
	dlist_head		completed_tasks;
	cl_uint			num_running_tasks;
	cl_uint			num_pending_tasks;
	cl_uint			num_completed_tasks;
	void		  (*cb_cleanup)(struct GpuTaskState *gtstate);
	pgstrom_perfmon	pfm_sum;
} GpuTaskState;

typedef struct GpuTask
{
	dlist_node		chain;
	GpuTaskState   *gts;
	CUstream		cuda_stream;
	CUdevice		cuda_device;	/* just reference, no cleanup needed */
	CUcontext		cuda_context;	/* just reference, no cleanup needed */
	cl_int			errcode;
	void		  (*cb_process)(struct GpuTask *gtask);
	void		  (*cb_release)(struct GpuTask *gtask);
	pgstrom_perfmon	pfm;
} GpuTask;

/*
 * Kernel Param/Const buffer
 */
typedef struct {
	StromObject		sobj;
	slock_t			lock;
	int				refcnt;
	kern_parambuf	kern;
} pgstrom_parambuf;

/*
 * Type declarations for code generator
 */
#define DEVINFO_IS_NEGATIVE			0x0001
#define DEVTYPE_IS_VARLENA			0x0002
#define DEVTYPE_IS_BUILTIN			0x0004
#define DEVFUNC_NEEDS_TIMELIB		0x0008
#define DEVFUNC_NEEDS_TEXTLIB		0x0010
#define DEVFUNC_NEEDS_NUMERIC		0x0020
#define DEVFUNC_NEEDS_MATHLIB		0x0040
#define DEVFUNC_INCL_FLAGS			0x0078
#define DEVKERNEL_DISABLE_OPTIMIZE	0x0100
#define DEVKERNEL_NEEDS_GPUSCAN		0x0200
#define DEVKERNEL_NEEDS_HASHJOIN	0x0400
#define DEVKERNEL_NEEDS_GPUPREAGG	0x0800
#define DEVKERNEL_NEEDS_GPUSORT		0x1000

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
	dlist_node	chain;
	FileName	kds_fname;	/* filename, if file-mapped */
	int			kds_fdesc;	/* file descriptor, if file-mapped */
	Size		kds_length;	/* copy of kds->length */
	kern_data_store *kds;
	struct pgstrom_data_store *ktoast;
} pgstrom_data_store;

/*
 * pgstrom_bulk_slot
 *
 * A data structure to move a chunk of data with keeping pgstrom_data_store
 * data format on shared memory segment. It reduces cost for data copy.
 */
typedef struct
{
	NodeTag			type;
	pgstrom_data_store *pds;
	cl_int			nvalids;	/* length of rindex. -1 means all valid */
	cl_uint			rindex[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_bulkslot;

typedef pgstrom_data_store *(*pgstromExecBulkScan)(CustomScanState *node);

/* --------------------------------------------------------------------
 *
 * Private enhancement of CustomScan Interface
 *
 * --------------------------------------------------------------------
 */
typedef struct
{
	CustomExecMethods	c;
	void   *(*ExecCustomBulk)(CustomScanState *node);
} PGStromExecMethods;

static inline void *
BulkExecProcNode(PlanState *node)
{
	CHECK_FOR_INTERRUPTS();

	if (node->chgParam != NULL)		/* something changed */
		ExecReScan(node);			/* let ReScan handle this */

	/* rough check, not sufficient... */
	if (IsA(node, CustomScanState))
	{
		CustomScanState *css = (CustomScanState *) node;
		PGStromExecMethods *methods = (PGStromExecMethods *) css->methods;
		Assert(methods->ExecCustomBulk != NULL);
		return methods->ExecCustomBulk(css);
	}
	elog(ERROR, "unrecognized node type: %d", (int) nodeTag(node));
}

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
                        Size minContextSize,
                        Size initBlockSize,
                        Size maxBlockSize);
/*
 * cuda_control.c
 */
extern void pgstrom_init_cuda_control(void);
extern GpuContext *pgstrom_get_gpucontext(void);
extern void pgstrom_put_gpucontext(GpuContext *gcontext);
extern void pgstrom_assign_cuda_stream(GpuContext *gcontext, GpuTask *task);
extern const char *cuda_strerror(CUresult errcode);
extern Datum pgstrom_device_info(PG_FUNCTION_ARGS);

/*
 * cuda_program.c
 */
extern bool pgstrom_get_cuda_program(GpuTaskState *gts,
									 const char *source,
									 int32 extra_flags);
extern void pgstrom_put_cuda_program(GpuTaskState *gts);
extern void pgstrom_init_cuda_program(void);

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
	const char *ktoast_label;/* label to reference ktoast, if exist */
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
extern pgstrom_data_store *
pgstrom_create_data_store_row(GpuContext *gcontext,
							  TupleDesc tupdesc,
							  Size length,
							  char *file_name);
extern pgstrom_data_store *
pgstrom_create_data_store_slot(GpuContext *gcontext,
							   TupleDesc tupdesc,
							   cl_uint nrooms,
							   bool internal_format);
extern pgstrom_data_store *
pgstrom_file_map_data_store(const char *kds_fname, Size kds_length);
extern void
pgstrom_file_unmap_data_store(pgstrom_data_store *pds);

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
extern Plan *gpuscan_try_replace_relscan(Plan *plan,
										 List *range_table,
										 Bitmapset *attr_refs,
										 List **p_upper_quals);
extern bool pgstrom_path_is_gpuscan(const Path *path);
extern bool pgstrom_plan_is_gpuscan(const Plan *plan);
extern void pgstrom_gpuscan_setup_bulkslot(PlanState *outer_ps,
										   ProjectionInfo **p_bulk_proj,
										   TupleTableSlot **p_bulk_slot);
extern void pgstrom_init_gpuscan(void);

/*
 * gpuhashjoin.c
 */
struct pgstrom_multihash_tables;/* to avoid include opencl_hashjoin.h here */
extern struct pgstrom_multihash_tables *
multihash_get_tables(struct pgstrom_multihash_tables *mhtables);
extern void
multihash_put_tables(struct pgstrom_multihash_tables *mhtables);

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

extern Datum gpupreagg_partial_nrows(PG_FUNCTION_ARGS);
extern Datum gpupreagg_pseudo_expr(PG_FUNCTION_ARGS);
extern Datum gpupreagg_psum_int(PG_FUNCTION_ARGS);
extern Datum gpupreagg_psum_float4(PG_FUNCTION_ARGS);
extern Datum gpupreagg_psum_float8(PG_FUNCTION_ARGS);
extern Datum gpupreagg_psum_x2_float(PG_FUNCTION_ARGS);
extern Datum gpupreagg_psum_numeric(PG_FUNCTION_ARGS);
extern Datum gpupreagg_psum_x2_numeric(PG_FUNCTION_ARGS);
extern Datum gpupreagg_corr_psum_x(PG_FUNCTION_ARGS);
extern Datum gpupreagg_corr_psum_y(PG_FUNCTION_ARGS);
extern Datum gpupreagg_corr_psum_x2(PG_FUNCTION_ARGS);
extern Datum gpupreagg_corr_psum_y2(PG_FUNCTION_ARGS);
extern Datum gpupreagg_corr_psum_xy(PG_FUNCTION_ARGS);

extern Datum pgstrom_avg_int8_accum(PG_FUNCTION_ARGS);	/* name confusing? */
extern Datum pgstrom_sum_int8_accum(PG_FUNCTION_ARGS);	/* name confusing? */
extern Datum pgstrom_sum_int8_final(PG_FUNCTION_ARGS);	/* name confusing? */
extern Datum pgstrom_sum_float8_accum(PG_FUNCTION_ARGS);
extern Datum pgstrom_variance_float8_accum(PG_FUNCTION_ARGS);
extern Datum pgstrom_covariance_float8_accum(PG_FUNCTION_ARGS);

extern Datum pgstrom_int8_avg_accum(PG_FUNCTION_ARGS);
extern Datum pgstrom_numeric_avg_accum(PG_FUNCTION_ARGS);
extern Datum pgstrom_numeric_avg_final(PG_FUNCTION_ARGS);
extern Datum pgstrom_numeric_var_accum(PG_FUNCTION_ARGS);
extern Datum pgstrom_numeric_var_samp(PG_FUNCTION_ARGS);
extern Datum pgstrom_numeric_var_pop(PG_FUNCTION_ARGS);
extern Datum pgstrom_numeric_stddev_samp(PG_FUNCTION_ARGS);
extern Datum pgstrom_numeric_stddev_pop(PG_FUNCTION_ARGS);

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
extern void _PG_init(void);
extern const char *pgstrom_strerror(cl_int errcode);
extern void show_scan_qual(List *qual, const char *qlabel,
						   PlanState *planstate, List *ancestors,
						   ExplainState *es);
extern void show_instrumentation_count(const char *qlabel, int which,
									   PlanState *planstate, ExplainState *es);
extern void print_device_kernel(const char *kern_source,
								int32 extra_flags,
								ExplainState *es);
extern void pgstrom_perfmon_add(pgstrom_perfmon *pfm_sum,
								pgstrom_perfmon *pfm_item);
extern void pgstrom_perfmon_explain(pgstrom_perfmon *pfm,
									ExplainState *es);
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
	elog(ERROR, "unexpected type alignment: %c", type_align);
	return -1;	/* be compiler quiet */
}

#endif	/* PG_STROM_H */
