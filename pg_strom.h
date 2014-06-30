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
#include "nodes/execnodes.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "storage/lock.h"
#include "storage/spin.h"
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
	StromTag_TCacheHead,
	StromTag_TCacheRowStore,
	StromTag_TCacheColumnStore,
	StromTag_TCacheToastBuf,
	StromTag_GpuScan,
	StromTag_GpuSort,
	StromTag_GpuHashJoin,
	StromTag_HashJoinTable,
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
		StromTagGetLabelEntry(TCacheHead);
		StromTagGetLabelEntry(TCacheRowStore);
		StromTagGetLabelEntry(TCacheColumnStore);
		StromTagGetLabelEntry(TCacheToastBuf);
		StromTagGetLabelEntry(GpuScan);
		StromTagGetLabelEntry(GpuHashJoin);
		StromTagGetLabelEntry(HashJoinTable);
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
	cl_ulong	time_to_load;	/* time to load data from heap/cache/subplan */
	cl_ulong	time_to_load_inner;	/* time to load inner scan */
	cl_ulong	time_tcache_build;	/* time to build tcache */
	cl_ulong	time_in_sendq;	/* waiting time in the server mqueue */
	cl_ulong	time_kern_build;/* max time to build opencl kernel */
	cl_ulong	bytes_dma_send;	/* bytes of DMA send */
	cl_ulong	bytes_dma_recv;	/* bytes of DMA receive */
	cl_uint		num_dma_send;	/* number of DMA send request */
	cl_uint		num_dma_recv;	/* number of DMA receive request */
	cl_ulong	time_dma_send;	/* time to send host=>device data */
	cl_uint		num_prep_exec;	/* number of preprocess kernel execution */
	cl_uint		num_kern_exec;	/* number of main kernel execution */
	cl_ulong	time_prep_exec;	/* time to execute preprocess kernel */
	cl_ulong	time_kern_exec;	/* time to execute main kernel */
	cl_ulong	time_dma_recv;	/* time to receive device=>host data */
	cl_ulong	time_in_recvq;	/* waiting time in the response mqueue */
	cl_ulong	time_post_exec;	/* time to execute post GPU processing
								 * like matelialization of hashjoin */
	cl_ulong	time_move_slot;	/* time to move rows to slot from rcstore */
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
	pid_t			owner;
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
#define DEVTYPE_IS_BUILTIN			0x0004
#define DEVFUNC_NEEDS_TIMELIB		0x0008
#define DEVFUNC_NEEDS_TEXTLIB		0x0010
#define DEVFUNC_NEEDS_NUMERICLIB	0x0020
#define DEVFUNC_INCL_FLAGS			0x0038
#define DEVKERNEL_NEEDS_GPUSCAN		0x0200
#define DEVKERNEL_NEEDS_GPUSORT		0x0400
#define DEVKERNEL_NEEDS_HASHJOIN	0x0800

struct devtype_info;
struct devfunc_info;

typedef struct devtype_info {
	Oid			type_oid;
	uint32		type_flags;
	int16		type_length;
	int16		type_align;
	char	   *type_name;	/* name of device type; same of SQL's type */
	char	   *type_base;	/* base name of this type (like varlena) */
} devtype_info;

typedef struct devfunc_info {
	int32		func_flags;
	Oid			func_namespace;
	Oid		   *func_argtypes;
	const char *func_name;	/* name of device function; same of SQL's func */
	List	   *func_args;	/* list of devtype_info */
	devtype_info *func_rettype;
	const char *func_decl;	/* declaration of function */
} devfunc_info;

/*
 * T-Tree Columner Cache
 */
struct tcache_head;

/*
 * tcache_row_store - uncolumnized data buffer in row-format
 */
typedef struct {
	StromObject		sobj;	/* =StromTag_TCacheRowStore */
	slock_t			refcnt_lock;
	int				refcnt;
	dlist_node		chain;
	cl_uint			usage;
	BlockNumber		blkno_max;
	BlockNumber		blkno_min;
	kern_row_store	kern;
} tcache_row_store;

/*
 * NOTE: shmem.c put a magic number to detect shared memory usage overrun.
 * So, we have a little adjustment for this padding.
 */
#define ROWSTORE_DEFAULT_SIZE	(8 * 1024 * 1024 - SHMEM_ALLOC_COST)

/*
 * tcache_toastbuf - toast buffer for each varlena column
 */
typedef struct {
	slock_t			refcnt_lock;
	int				refcnt;
	cl_uint			tbuf_length;
	cl_uint			tbuf_usage;
	cl_uint			tbuf_junk;
	char			data[FLEXIBLE_ARRAY_MEMBER];
} tcache_toastbuf;

#define TCACHE_TOASTBUF_INITSIZE	((32 << 20) - SHMEM_ALLOC_COST)	/* 32MB */

/*
 * tcache_column_store - a node or leaf entity of t-tree columnar cache
 */
typedef struct {
	StromObject		sobj;	/* StromTag_TCacheColumnStore */
	slock_t			refcnt_lock;
	int				refcnt;
	uint32			ncols;	/* length of cdata[] (.incl uncached columns) */
	uint32			nrows;	/* number of rows being cached */
	uint32			njunks;	/* number of junk rows to be removed later */
	bool			is_sorted;
	BlockNumber		blkno_max;
	BlockNumber		blkno_min;
	ItemPointerData		*ctids;
	HeapTupleHeaderData	*theads;
	struct {
		uint8	   *isnull;		/* nullmap, if NOT NULL is not set */
		char	   *values;		/* array of values in columnar format */
		tcache_toastbuf *toast;	/* toast buffer, if varlena variable */
	} cdata[FLEXIBLE_ARRAY_MEMBER];
} tcache_column_store;

#define NUM_ROWS_PER_COLSTORE	(1 << 18)	/* 256K records */

/*
 * tcache_node - leaf or 
 */
struct tcache_node {
	StromObject		sobj;	/* = StromTag_TCacheNode */
	dlist_node		chain;
	struct timeval	tv;		/* time being enqueued */
	struct tcache_node *right;	/* node with greater ctids */
	struct tcache_node *left;	/* node with less ctids */
	int				r_depth;
	int				l_depth;
	/* above fields are protected by lwlock of tc_head */

	slock_t			lock;
	tcache_column_store	*tcs;
};
typedef struct tcache_node tcache_node;

/*
 * tcache_head - a cache entry of individual relations
 */
typedef struct {
	StromObject		sobj;			/* StromTag_TCacheHead */
	dlist_node		chain;			/* link to the hash or free list */
	dlist_node		lru_chain;		/* link to the LRU list */
	dlist_node		pending_chain;	/* link to the pending list */
	int				refcnt;
	/* above fields are protected by tc_common->lock */

	/*
	 * NOTE: regarding to locking
	 *
	 * tcache_head contains two types of stores; row and column.
	 * Usually, newly written tuples are put on the row-store, then
	 * it shall be moved to column-store by columnizer background
	 * worker process.
	 * To simplifies the implementation, we right now adopt a giant-
	 * lock approach; regular backend code takes shared-lock during
	 * its execution, but columnizer process takes exclusive-lock
	 * during its translation. Usually, row -> column translation
	 * shall be done chunk-by-chunk, so this shared lock does not
	 * take so long time.
	 * Also note that an writer operation within a particular row-
	 * or column- store is protected by its individual spinlock,
	 * but it never takes giant locks.
	 */
	LOCKTAG			locktag;	/* locktag of per execution (long) lock */
	tcache_node	   *tcs_root;	/* root node of this cache */
	

	slock_t			lock;		/* short term locking for fields below */
	bool			is_ready;	/* true, if tcache is already built */
	dlist_head		free_list;	/* list of free tcache_node */
	dlist_head		block_list;	/* list of blocks of tcache_node */
	dlist_head		pending_list; /* list of pending tcahe_node */
	dlist_head		trs_list; /* list of pending tcache_row_store */
	tcache_row_store *trs_curr;	/* current available row-store */

	/* fields below are read-only once constructed (no lock needed) */
	Oid			datoid;		/* database oid of this cache */
	Oid			reloid;		/* relation oid of this cache */
	Bitmapset  *cached_attrs;	/* cached attributes in bitmap form */
	TupleDesc	tupdesc;	/* duplication of TupleDesc of underlying table.
							 * all the values, except for constr, are on
							 * the shared memory region, so its visible to
							 * all processes including columnizer.
							 */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} tcache_head;

#define TCACHE_NODE_PER_BLOCK(row_natts, col_natts)						\
	((SHMEM_BLOCKSZ - SHMEM_ALLOC_COST -								\
	  MAXALIGN(offsetof(tcache_head, attrs[(row_natts)])) -				\
	  MAXALIGN(sizeof(AttrNumber) * (col_natts))) / sizeof(tcache_node))
#define TCACHE_NODE_PER_BLOCK_BARE					\
	((SHMEM_BLOCKSZ - SHMEM_ALLOC_COST				\
	  - sizeof(dlist_node)) / sizeof(tcache_node))

/*
 * tcache_scandesc - 
 */
typedef struct {
	StromObject		sobj;		/* =StromTag_TCacheScanDesc */
	Relation		rel;
	HeapScanDesc	heapscan;	/* valid, if state == TC_STATE_NOW_BUILD */
	bool			has_exlock;	/* true, if this scan hold exclusive lock */
	cl_ulong		time_tcache_build;
	tcache_head	   *tc_head;
	BlockNumber		tcs_blkno_min;
	BlockNumber		tcs_blkno_max;
	tcache_row_store *trs_curr;
} tcache_scandesc;

/*
 * pgstrom_bulk_slot
 *
 * A data structure to move scanned/joinned data in column-oriented data
 * format, to reduce row to/from column translation during query execution.
 */
typedef struct
{
	Node			node;			/* dummy header portion */
	StromObject	   *rc_store;		/* row/column-store to be moved */
	cl_uint			nitems;			/* num of rows on this bulk-slot */
	List		   *attmap;			/* attribute numbers */
	cl_uint			rindex[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_bulkslot;

/*
 * pgstrom_materialize
 *
 * This data structure provides a definition of destination relation on
 * server side materialization. Unlike host side, this materialization
 * cannot contain expression in the target-list. What it can do is
 * re-ordering the columns of (multiple) source relations.
 */
typedef struct
{
	/* true, if column never has NULL (thus, no nullmap required) */
	cl_char		attnotnull;
	/* alignment; 1,2,4 or 8, not characters in pg_attribute */
	cl_char		attalign;
	/* length of attribute */
	cl_short	attlen;
	/* source relation index */
	cl_short	relsrc;
	/* source attribute index */
	cl_short	attsrc;
	/* destination attribute index */
	cl_short	attdst;
	/* average width of this attribute */
	cl_short	attwidth;

} materialize_colmeta;

typedef struct
{
	cl_uint			nrels;		/* number of relations being expected */
	cl_uint			ncols;		/* number of columns in destination store */
	materialize_colmeta colmeta[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_materialize;

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
extern bool pgstrom_shmem_sanitycheck(const void *address);
extern void pgstrom_shmem_dump(void);
extern void pgstrom_setup_shmem(Size zone_length,
								void *(*callback)(void *address,
												  Size length));
extern void pgstrom_init_shmem(void);

extern Datum pgstrom_shmem_info(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_active_info(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_slab_info(PG_FUNCTION_ARGS);

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
 * datastore.c
 */
extern int pgstrom_try_varlena_inline(Form_pg_attribute attr);
#if 0
extern pgstrom_vrelation *pgstrom_get_vrelation(pgstrom_vrelation *vrel);
extern void pgstrom_put_vrelation(pgstrom_vrelation *vrel);
extern pgstrom_vrelation *
pgstrom_create_vrelation_head(TupleDesc tupdesc,
                              List *vtlist_relidx,
                              List *vtlist_attidx);
extern pgstrom_vrelation *
pgstrom_populate_vrelation(pgstrom_vrelation *vrel_head,
						   StromObject *rcstore,
                           cl_uint nrels, cl_uint nitems, cl_uint nrooms);
extern List *
pgstrom_can_vrelation_projection(List *targetlist);
extern pgstrom_vrelation *
pgstrom_apply_vrelation_projection(pgstrom_vrelation *vrel, List *vrel_proj);
#endif

extern kern_parambuf *
pgstrom_create_kern_parambuf(List *used_params,
                             ExprContext *econtext);
extern bytea *kparam_make_kds_head(TupleDesc tupdesc,
								   Bitmapset *attrefs,
								   cl_uint nsyscols);
extern void kparam_refresh_kds_head(kern_parambuf *kparams,
									StromObject *rcstore,
									cl_uint nitems);
extern bytea *kparam_make_ktoast_head(TupleDesc tupdesc,
									  cl_uint nsyscols);
extern void kparam_refresh_ktoast_head(kern_parambuf *kparams,
									   StromObject *rcstore);
extern bytea *kparam_make_materialization(List *varnode_list,
										  List *source_relids);
extern List *pgstrom_make_bulk_attmap(List *targetlist, Index varno);
extern pgstrom_bulkslot *pgstrom_create_bulkslot(StromObject *rc_store,
												 List *bulk_attmap,
												 cl_uint nitems,
												 cl_uint nrooms);
extern void pgstrom_release_bulkslot(pgstrom_bulkslot *bulk);
extern bool pgstrom_plan_can_multi_exec(const PlanState *ps);

extern tcache_row_store *pgstrom_create_row_store(TupleDesc tupdesc);
extern tcache_row_store *pgstrom_get_row_store(tcache_row_store *trs);
extern void pgstrom_put_row_store(tcache_row_store *trs);

extern tcache_toastbuf *pgstrom_create_toast_buffer(Size required);
extern tcache_toastbuf *pgstrom_expand_toast_buffer(tcache_toastbuf *tbuf);
extern tcache_toastbuf *pgstrom_get_toast_buffer(tcache_toastbuf *tbuf);
extern void pgstrom_put_toast_buffer(tcache_toastbuf *tbuf);
#if 0
extern tcache_column_store *
pgstrom_create_column_store_with_projection(pgstrom_projection *pproj,
                                            cl_uint nitems,
                                            bool with_syscols);
#endif
extern tcache_column_store *pgstrom_get_column_store(tcache_column_store *pcs);
extern void pgstrom_put_column_store(tcache_column_store *pcs);

/*
 * restrack.c
 */
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
extern bool gpuscan_support_multi_exec(const CustomPlanState *cps);
extern void pgstrom_init_gpuscan(void);

/*
 * gpusort.c
 */
extern CustomPlan *pgstrom_create_gpusort_plan(Sort *original, List *rtable);
extern bool gpusort_support_multi_exec(const CustomPlanState *cps);
extern void pgstrom_init_gpusort(void);

/*
 * gpuhashjoin.c
 */
struct pgstrom_hashjoin_table;	/* to avoid including opencl_hashjoin.h here */
extern struct pgstrom_hashjoin_table *
gpuhashjoin_get_hash_table(struct pgstrom_hashjoin_table *ghash_table);
extern void
gpuhashjoin_put_hash_table(struct pgstrom_hashjoin_table *ghash_table);

extern bool gpuhashjoin_support_multi_exec(const CustomPlanState *cps);
extern void pgstrom_init_gpuhashjoin(void);

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
										  size_t num_threads,
										  size_t local_memsz_per_thread);
/*
 * opencl_devprog.c
 */
#define BAD_OPENCL_PROGRAM		((void *) ~0UL)
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
 * codegen.c
 */
typedef struct {
	StringInfoData	str;
	List	   *type_defs;	/* list of devtype_info in use */
	List	   *func_defs;	/* list of devfunc_info in use */
	List	   *used_params;/* list of Const/Param in use */
	List	   *used_vars;	/* list of Var in use */
	const char *row_index;	/* label to reference row-index, if exist */
	int			extra_flags;/* external libraries to be included */
} codegen_context;

extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);
extern devfunc_info *pgstrom_devfunc_lookup(Oid func_oid);
extern devtype_info *pgstrom_devtype_lookup_and_track(Oid type_oid,
											  codegen_context *context);
extern devfunc_info *pgstrom_devfunc_lookup_and_track(Oid func_oid,
											  codegen_context *context);
extern char *pgstrom_codegen_expression(Node *expr, codegen_context *context);
extern char *pgstrom_codegen_type_declarations(codegen_context *context);
extern char *pgstrom_codegen_func_declarations(codegen_context *context);
extern char *pgstrom_codegen_param_declarations(codegen_context *context,
												int num_skips);
extern char *pgstrom_codegen_var_declarations(codegen_context *context);
extern bool pgstrom_codegen_available_expression(Expr *expr);
extern void pgstrom_init_codegen(void);

/*
 * tcache.c
 */
extern tcache_scandesc *tcache_begin_scan(tcache_head *tc_head,
										  Relation heap_rel);
extern StromObject *tcache_scan_next(tcache_scandesc *tc_scan);
extern StromObject *tcache_scan_prev(tcache_scandesc *tc_scan);
extern void tcache_end_scan(tcache_scandesc *tc_scan);
extern void tcache_abort_scan(tcache_scandesc *tc_scan);
extern void tcache_rescan(tcache_scandesc *tc_scan);

extern tcache_head *tcache_try_create_tchead(Oid reloid, Bitmapset *required);
extern tcache_head *tcache_get_tchead(Oid reloid, Bitmapset *required);
extern void tcache_put_tchead(tcache_head *tc_head);
extern void tcache_abort_tchead(tcache_head *tc_head, Datum private);
extern bool tcache_state_is_ready(tcache_head *tc_head);


extern tcache_row_store *tcache_create_row_store(TupleDesc tupdesc);
extern tcache_row_store *tcache_get_row_store(tcache_row_store *trs);
extern void tcache_put_row_store(tcache_row_store *trs);
extern bool tcache_row_store_insert_tuple(tcache_row_store *trs,
										  HeapTuple tuple);
extern void tcache_row_store_fixup_tcs_head(tcache_row_store *trs);

extern tcache_column_store *tcache_get_column_store(tcache_column_store *tcs);
extern void tcache_put_column_store(tcache_column_store *tcs);

extern bool pgstrom_relation_has_synchronizer(Relation rel);
extern Datum pgstrom_tcache_synchronizer(PG_FUNCTION_ARGS);


extern Datum pgstrom_tcache_info(PG_FUNCTION_ARGS);
extern Datum pgstrom_tcache_node_info(PG_FUNCTION_ARGS);

extern void pgstrom_init_tcache(void);


/*
 * main.c
 */
extern bool	pgstrom_enabled;
extern bool pgstrom_perfmon_enabled;
extern int	pgstrom_max_async_chunks;
extern int	pgstrom_min_async_chunks;
extern int  pgstrom_max_inline_varlena;
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
extern void show_device_kernel(Datum dprog_key, ExplainState *es);
extern void pgstrom_perfmon_add(pgstrom_perfmon *pfm_sum,
								pgstrom_perfmon *pfm_item);
extern void pgstrom_perfmon_explain(pgstrom_perfmon *pfm,
									ExplainState *es);
extern void _outToken(StringInfo str, const char *s);
extern void _outBitmapset(StringInfo str, const Bitmapset *bms);

/*
 * grafter.c
 */
extern void pgstrom_init_grafter(void);

/*
 * debug.c
 */
extern Datum pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_free_func(PG_FUNCTION_ARGS);

/*
 * opencl_*.h
 */
extern const char *pgstrom_opencl_common_code;
extern const char *pgstrom_opencl_gpuscan_code;
extern const char *pgstrom_opencl_gpusort_code;
extern const char *pgstrom_opencl_hashjoin_code;
extern const char *pgstrom_opencl_textlib_code;
extern const char *pgstrom_opencl_timelib_code;

/* ----------------------------------------------------------------
 *
 * Miscellaneous static inline functions
 *
 * ---------------------------------------------------------------- */

/*
 * increment or decrement reference ounter or row- or column-store
 */
static inline StromObject *
pgstrom_get_rcstore(StromObject *sobject)
{
	StromObject	   *result;

	if (StromTagIs(sobject, TCacheRowStore))
	{
		tcache_row_store *trs = (tcache_row_store *) sobject;
		result = (StromObject *) pgstrom_get_row_store(trs);
	}
	else if (StromTagIs(sobject, TCacheColumnStore))
	{
		tcache_column_store *tcs = (tcache_column_store *) sobject;
		result = (StromObject *) tcache_get_column_store(tcs);
	}
	else
		elog(ERROR, "Bug? it's neither row nor column store");

	return result;
}

static inline void
pgstrom_put_rcstore(StromObject *sobject)
{
	if (StromTagIs(sobject, TCacheRowStore))
		pgstrom_put_row_store((tcache_row_store *)sobject);
	else if (StromTagIs(sobject, TCacheColumnStore))
		tcache_put_column_store((tcache_column_store *)sobject);
	else
		elog(ERROR, "Bug? it's neither row nor column store");
}

static inline cl_uint
pgstrom_nitems_rcstore(StromObject *sobject)
{
	cl_uint		nitems;

	if (StromTagIs(sobject, TCacheRowStore))
		nitems = ((tcache_row_store *) sobject)->kern.nrows;
	else if (StromTagIs(sobject, TCacheColumnStore))
		nitems = ((tcache_column_store *) sobject)->nrows;
	else if (pgstrom_i_am_clserv)
		nitems = 0;	/* tells caller rcstore is not valid */
	else
		elog(ERROR, "bug? it's neither row nor column store");
	return nitems;
}

/* binary available pstrcpy() */
static inline void *
pmemcpy(void *from, size_t sz)
{
	void   *dest = palloc(sz);

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

/*
 * utility function to access system kparams
 */

static inline kern_data_store *
KPARAM_GET_KDS_HEAD(kern_parambuf *kparams)
{
	bytea  *vl_datum = kparam_get_value(kparams, 0);

	if (!vl_datum)
		return NULL;
	return (kern_data_store *)VARDATA_ANY(vl_datum);
}

static inline kern_toastbuf *
KPARAM_GET_KTOAST_HEAD(kern_parambuf *kparams)
{
	bytea  *vl_datum = kparam_get_value(kparams, 1);

	if (!vl_datum)
		return NULL;
	return (kern_toastbuf *)VARDATA_ANY(vl_datum);
}

#endif	/* PG_STROM_H */
