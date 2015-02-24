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
#include "storage/spin.h"
#include "utils/resowner.h"
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "opencl_common.h"

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
 * Type Definitions
 *
 * --------------------------------------------------------------------
 */

/*
 * pgstrom_platform_info
 *
 * Properties of OpenCL platform being choosen. Usually, a particular
 * platform shall be choosen on starting up time according to the GUC
 * configuration (including automatic policy).
 * Note that the properties below are supported on the OpenCL 1.1 only,
 * because older drivers cannot understand newer parameter names appeared
 * in v1.2.
 */
typedef struct {
	cl_uint		pl_index;
	char	   *pl_profile;
	char	   *pl_version;
	char	   *pl_name;
	char	   *pl_vendor;
	char	   *pl_extensions;
	Size		buflen;
	char		buffer[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_platform_info;

/*
 * pgstrom_device_info
 *
 * A set of OpenCL properties of a particular device. See above comments.
 */
typedef struct {
	pgstrom_platform_info *pl_info;
	cl_uint		dev_index;
	cl_uint		dev_address_bits;
	cl_bool		dev_available;
	cl_bool		dev_compiler_available;
	cl_device_fp_config	dev_double_fp_config;
	cl_bool		dev_endian_little;
	cl_bool		dev_error_correction_support;
	cl_device_exec_capabilities dev_execution_capabilities;
	char	   *dev_device_extensions;
	cl_ulong	dev_global_mem_cache_size;
	cl_device_mem_cache_type	dev_global_mem_cache_type;
	cl_uint		dev_global_mem_cacheline_size;
	cl_ulong	dev_global_mem_size;
	cl_bool		dev_host_unified_memory;
	cl_ulong	dev_local_mem_size;
	cl_device_local_mem_type	dev_local_mem_type;
	cl_uint		dev_max_clock_frequency;
	cl_uint		dev_max_compute_units;
	cl_uint		dev_max_constant_args;
	cl_ulong	dev_max_constant_buffer_size;
	cl_ulong	dev_max_mem_alloc_size;
	size_t		dev_max_parameter_size;
	cl_uint		dev_max_samplers;
	size_t		dev_max_work_group_size;
	cl_uint		dev_max_work_item_dimensions;
	size_t		dev_max_work_item_sizes[3];
	cl_uint		dev_mem_base_addr_align;
	char	   *dev_name;
	cl_uint		dev_native_vector_width_char;
	cl_uint		dev_native_vector_width_short;
	cl_uint		dev_native_vector_width_int;
	cl_uint		dev_native_vector_width_long;
	cl_uint		dev_native_vector_width_float;
	cl_uint		dev_native_vector_width_double;
	char	   *dev_opencl_c_version;
	cl_uint		dev_preferred_vector_width_char;
	cl_uint		dev_preferred_vector_width_short;
	cl_uint		dev_preferred_vector_width_int;
	cl_uint		dev_preferred_vector_width_long;
	cl_uint		dev_preferred_vector_width_float;
	cl_uint		dev_preferred_vector_width_double;
	char	   *dev_profile;
	size_t		dev_profiling_timer_resolution;
	cl_command_queue_properties	dev_queue_properties;
	cl_device_fp_config	dev_single_fp_config;
	cl_device_type	dev_type;
	char	   *dev_vendor;
	cl_uint		dev_vendor_id;
	char	   *dev_version;
	char	   *driver_version;
	Size		buflen;
	char		buffer[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_device_info;

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
 * pgstrom_queue
 *
 * A message queue allocated on shared memory, to send messages to/from
 * OpenCL background server. A message queue is constructed with refcnt=1,
 * then its reference counter shall be incremented for each message enqueue
 * to be returned
 */
typedef struct {
	StromObject		sobj;
	dlist_node		chain;	/* link to free queues list in mqueue.c */
	PGPROC		   *owner;
	int				refcnt;
	pthread_mutex_t	lock;
	pthread_cond_t	cond;
	dlist_head		qhead;
	bool			closed;
} pgstrom_queue;

typedef struct pgstrom_message {
	StromObject		sobj;
	slock_t			lock;	/* protection for reference counter */
	cl_int			refcnt;
	cl_int			errcode;
	dlist_node		chain;
	pgstrom_queue  *respq;	/* mqueue for response message */
	void	(*cb_process)(struct pgstrom_message *message);
	void	(*cb_release)(struct pgstrom_message *message);
	pgstrom_perfmon	pfm;
} pgstrom_message;

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
#define DEVTYPE_HAS_INTERNAL_FORMAT	0x0004
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
typedef struct pgstrom_data_store {
	StromObject			sobj;
	slock_t				lock;
	volatile int		refcnt;
	kern_data_store	   *kds;		/* reference to kern_data_store */
	size_t				kds_length;	/* length of kds file */
	size_t				kds_offset;	/* offset of kds file */
	char			   *kds_fname;	/* if KDS_FORMAT_ROW_FMAP */
	int					kds_fdesc;	/* !!NOTE: valid only the backend */
	struct pgstrom_data_store *ktoast;
	ResourceOwner		resowner;	/* !!NOTE: private address!!*/
	char			   *local_pages;/* duplication of local pages */
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

typedef pgstrom_bulkslot *(*pgstromExecBulkScan)(CustomScanState *node);

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
 * shmem.c
 */
#define SHMEM_BLOCKSZ_BITS_MAX	34			/* 16GB */
#define SHMEM_BLOCKSZ_BITS		13			/*  8KB */
#define SHMEM_BLOCKSZ_BITS_RANGE	\
	(SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS)
#define SHMEM_BLOCKSZ			(1UL << SHMEM_BLOCKSZ_BITS)
#define SHMEM_ALLOC_COST		48

extern void *__pgstrom_shmem_alloc(const char *filename, int lineno,
								   Size size);
extern void *__pgstrom_shmem_alloc_alap(const char *filename, int lineno,
										Size required, Size *allocated);
extern void *__pgstrom_shmem_realloc(const char *filename, int lineno,
									 void *oldaddr, Size newsize);
#define pgstrom_shmem_alloc(size)					\
	__pgstrom_shmem_alloc(__FILE__,__LINE__,(size))
#define pgstrom_shmem_alloc_alap(size,allocated)	\
	__pgstrom_shmem_alloc_alap(__FILE__,__LINE__,(size),(allocated))
#define pgstrom_shmem_realloc(addr,size)		\
	__pgstrom_shmem_realloc(__FILE__,__LINE__,(addr),(size))
extern void pgstrom_shmem_free(void *address);
extern Size pgstrom_shmem_getsize(void *address);
extern Size pgstrom_shmem_zone_length(void);
extern Size pgstrom_shmem_maxalloc(void);
extern bool pgstrom_shmem_sanitycheck(const void *address);
extern void pgstrom_shmem_dump(void);
extern void pgstrom_setup_shmem(Size zone_length,
								bool (*callback)(void *address, Size length,
												 const char *label,
												 bool abort_on_error));
extern void pgstrom_init_shmem(void);

extern Datum pgstrom_shmem_info(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_active_info(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_slab_info(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_free_func(PG_FUNCTION_ARGS);

/*
 * mqueue.c
 */
extern pgstrom_queue *pgstrom_create_queue(void);
extern bool pgstrom_enqueue_message(pgstrom_message *message);
extern void pgstrom_reply_message(pgstrom_message *message);
extern pgstrom_message *pgstrom_dequeue_message(pgstrom_queue *queue);
extern pgstrom_message *pgstrom_try_dequeue_message(pgstrom_queue *queue);
extern pgstrom_message *pgstrom_dequeue_server_message(void);
extern void pgstrom_close_server_queue(void);
extern void pgstrom_cancel_server_loop(void);
extern void pgstrom_close_queue(pgstrom_queue *queue);
extern pgstrom_queue *pgstrom_get_queue(pgstrom_queue *mqueue);
extern void pgstrom_put_queue(pgstrom_queue *mqueue);
extern void pgstrom_put_message(pgstrom_message *msg);
extern void pgstrom_init_message(pgstrom_message *msg,
								 StromTag stag,
								 pgstrom_queue *respq,
								 void (*cb_process)(pgstrom_message *msg),
								 void (*cb_release)(pgstrom_message *msg),
								 bool perfmon_enabled);
extern void pgstrom_init_mqueue(void);
extern Datum pgstrom_mqueue_info(PG_FUNCTION_ARGS);

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
__pgstrom_create_data_store_row(const char *filename, int lineno,
								TupleDesc tupdesc,
								Size pds_length,
								Size tup_width);
#define pgstrom_create_data_store_row(tupdesc,pds_length,tup_width) \
	__pgstrom_create_data_store_row(__FILE__, __LINE__,			\
									(tupdesc),(pds_length),(tup_width))
extern pgstrom_data_store *
__pgstrom_create_data_store_row_flat(const char *filename, int lineno,
									 TupleDesc tupdesc, Size length);
#define pgstrom_create_data_store_row_flat(tupdesc,length)		\
	__pgstrom_create_data_store_row_flat(__FILE__,__LINE__,		\
										 (tupdesc),(length))

extern pgstrom_data_store *
__pgstrom_create_data_store_row_fmap(const char *filename, int lineno,
									 TupleDesc tupdesc, Size length);
#define pgstrom_create_data_store_row_fmap(tupdesc,length)		\
	__pgstrom_create_data_store_row_fmap(__FILE__,__LINE__,		\
										 (tupdesc),(length))

extern pgstrom_data_store *
__pgstrom_extend_data_store_tupslot(const char *filename, int lineno,
                                    pgstrom_data_store *pds_toast,
                                    TupleDesc tupdesc, cl_uint nrooms);
#define pgstrom_extend_data_store_tupslot(pds_toast,tupdesc,nrooms)	\
	__pgstrom_extend_data_store_tupslot(__FILE__,__LINE__,			\
										(pds_toast),(tupdesc),(nrooms))

extern kern_data_store *
filemap_kern_data_store(const char *kds_fname, size_t kds_length, int *fdesc);
extern void
fileunmap_kern_data_store(kern_data_store *kds, int fdesc);

extern pgstrom_data_store *
__pgstrom_create_data_store_tupslot(const char *filename, int lineno,
									TupleDesc tupdesc, cl_uint nrooms,
									bool internal_format);
#define pgstrom_create_data_store_tupslot(tupdesc,nrooms,internal_format) \
	__pgstrom_create_data_store_tupslot(__FILE__,__LINE__,		\
										(tupdesc),(nrooms),		\
										(internal_format))
extern pgstrom_data_store *pgstrom_get_data_store(pgstrom_data_store *pds);
extern void pgstrom_put_data_store(pgstrom_data_store *pds);
extern int pgstrom_data_store_insert_block(pgstrom_data_store *pds,
										   Relation rel,
										   BlockNumber blknum,
										   Snapshot snapshot,
										   bool page_prune);
extern bool pgstrom_data_store_insert_tuple(pgstrom_data_store *pds,
											TupleTableSlot *slot);
extern cl_int clserv_dmasend_data_store(pgstrom_data_store *pds,
										cl_command_queue kcmdq,
										cl_mem kds_buffer,
										cl_mem ktoast_buffer,
										cl_uint num_blockers,
										const cl_event *blockers,
										cl_uint *ev_index,
										cl_event *events,
										pgstrom_perfmon *pfm);
extern void pgstrom_dump_data_store(pgstrom_data_store *pds);
extern void pgstrom_init_datastore(void);

/*
 * restrack.c
 */
extern bool pgstrom_restrack_cleanup_context(void);
extern void __pgstrom_track_object(const char *filename, int lineno,
								   StromObject *sobject, Datum private);
#define pgstrom_track_object(sobject, private)			\
	__pgstrom_track_object(__FILE__,__LINE__,(sobject),(private))
extern Datum pgstrom_untrack_object(StromObject *sobject);
extern bool pgstrom_object_is_tracked(StromObject *sobject);
extern void pgstrom_init_restrack(void);

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
 * opencl_devinfo.c
 */
extern int	pgstrom_get_device_nums(void);
extern const pgstrom_device_info *pgstrom_get_device_info(unsigned int index);
extern void construct_opencl_device_info(void);
extern void pgstrom_init_opencl_devinfo(void);
extern Datum pgstrom_opencl_device_info(PG_FUNCTION_ARGS);

extern bool clserv_compute_workgroup_size(size_t *gwork_sz,
										  size_t *lwork_sz,
										  cl_kernel kernel,
										  int dev_index,
										  bool larger_is_better,
										  size_t num_threads,
										  size_t local_memsz_per_thread);
/*
 * opencl_devprog.c
 */
#define BAD_OPENCL_PROGRAM		((void *) ~0UL)
extern bool		devprog_enable_optimize;
extern cl_program clserv_lookup_device_program(Datum dprog_key,
											   pgstrom_message *msg);
extern Datum pgstrom_get_devprog_key(const char *source, int32 extra_libs);
extern void pgstrom_put_devprog_key(Datum dprog_key);
extern Datum pgstrom_retain_devprog_key(Datum dprog_key);
extern const char *pgstrom_get_devprog_errmsg(Datum dprog_key);
extern int32 pgstrom_get_devprog_extra_flags(Datum dprog_key);
extern const char *pgstrom_get_devprog_kernel_source(Datum dprog_key);
extern void pgstrom_init_opencl_devprog(void);
extern Datum pgstrom_opencl_program_info(PG_FUNCTION_ARGS);

/*
 * opencl_entry.c
 */
extern void pgstrom_init_opencl_entry(void);
extern const char *opencl_strerror(cl_int errcode);

/*
 * opencl_serv.c
 */
extern cl_platform_id		opencl_platform_id;
extern cl_context			opencl_context;
extern cl_uint				opencl_num_devices;
extern cl_device_id			opencl_devices[];
extern cl_command_queue		opencl_cmdq[];
extern volatile bool		pgstrom_clserv_exit_pending;
extern volatile bool		pgstrom_i_am_clserv;

extern int pgstrom_opencl_device_schedule(pgstrom_message *message);
extern void pgstrom_init_opencl_server(void);

extern void __clserv_log(const char *funcname,
						 const char *filename, int lineno,
						 const char *fmt, ...)
	__attribute__((format(PG_PRINTF_ATTRIBUTE, 4, 5)));
#define clserv_log(...)						\
	__clserv_log(__FUNCTION__,__FILE__,__LINE__,__VA_ARGS__)

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
extern void show_custom_flags(CustomScanState *css, ExplainState *es);
extern void show_device_kernel(Datum dprog_key, ExplainState *es);
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
extern const char *pgstrom_opencl_common_code;
extern const char *pgstrom_opencl_gpuscan_code;
extern const char *pgstrom_opencl_gpupreagg_code;
extern const char *pgstrom_opencl_hashjoin_code;
extern const char *pgstrom_opencl_gpusort_code;
extern const char *pgstrom_opencl_mathlib_code;
extern const char *pgstrom_opencl_textlib_code;
extern const char *pgstrom_opencl_timelib_code;
extern const char *pgstrom_opencl_numeric_code;

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
