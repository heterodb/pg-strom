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
#include "nodes/execnodes.h"
#include "nodes/primnodes.h"
#include "storage/spin.h"
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <CL/cl.h>
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
	StromTag_RowStore,
	StromTag_ColumnStore,
	StromTag_ToastBuf,
	StromTag_GpuScan,
	StromTag_GpuSort,
	StromTag_HashJoin,
	StromTag_TestMessage,
} StromTag;

/*
 * Performance monitor structure
 */
typedef struct {
	cl_bool		enabled;
	cl_ulong	time_to_load;	/* time to load data from heap/cache/subplan */
	cl_ulong	time_in_sendq;	/* waiting time in the server mqueue */
	cl_ulong	time_kern_build;/* time to build opencl kernel */
	cl_ulong	time_dma_send;	/* time to send host=>device data */
	cl_ulong	time_kern_exec;	/* time to execute kernel */
	cl_ulong	time_dma_recv;	/* time to receive device=>host data */
	cl_ulong	time_in_recvq;	/* waiting time in the response mqueue */
	struct timeval	tv;	/* result of gettimeofday(2) when enqueued */
} pgstrom_perfmon;

#define timeval_diff(tv1,tv2)						\
	(((tv2)->tv_sec * 1000000L + (tv2)->tv_usec) -	\
	 ((tv1)->tv_sec) * 1000000L + (tv1)->tv_usec)

#define PERFMON_ADD(pfm_sum,pfm_item)								\
	do {															\
		(pfm_sum)->time_to_load += (pfm_item)->time_to_load;		\
		(pfm_sum)->time_in_sendq += (pfm_item)->time_in_sendq;		\
		(pfm_sum)->time_kern_build += (pfm_item)->time_kern_build;	\
		(pfm_sum)->time_dma_send += (pfm_item)->time_dma_send;		\
		(pfm_sum)->time_kern_exec += (pfm_item)->time_kern_exec;	\
		(pfm_sum)->time_dma_recv += (pfm_item)->time_dma_recv;		\
		(pfm_sum)->time_in_recvq += (pfm_item)->time_in_recvq;		\
	} while(0)
#define PERFMON_EXPLAIN(pfm,es)					\
	do {										\
		ExplainPropertyFloat("time to load",							\
					 (double)(pfm)->time_to_load / 1000.0, 3, (es));	\
		ExplainPropertyFloat("time in send-mq",							\
					 (double)(pfm)->time_in_sendq / 1000.0, 3, (es));	\
		ExplainPropertyFloat("time to build kernel",					\
					 (double)(pfm)->time_kern_build / 1000.0, 3, (es)); \
		ExplainPropertyFloat("time of DMA send",						\
					 (double)(pfm)->time_dma_send / 1000.0, 3, (es)); 	\
		ExplainPropertyFloat("time of kernel exec",						\
					 (double)(pfm)->time_kern_exec / 1000.0, 3, (es));	\
		ExplainPropertyFloat("time of DMA recv",						\
					 (double)(pfm)->time_dma_recv / 1000.0, 3, (es));	\
		ExplainPropertyFloat("time in recv-mq",							\
					 (double)(pfm)->time_in_recvq / 1000.0, 3, (es));	\
	} while(0)

/*
 * pgstrom_queue
 *
 * A message queue allocated on shared memory, to send messages to/from
 * OpenCL background server. A message queue is constructed with refcnt=1,
 * then its reference counter shall be incremented for each message enqueue
 * to be returned
 */
typedef struct {
	StromTag		stag;
	dlist_node		chain;	/* link to free queues list in mqueue.c */
	pid_t			owner;
	int				refcnt;
	pthread_mutex_t	lock;
	pthread_cond_t	cond;
	dlist_head		qhead;
	bool			closed;
} pgstrom_queue;

typedef struct pgstrom_message {
	StromTag		stag;
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
	StromTag		stag;
	slock_t			lock;
	int				refcnt;
	kern_parambuf	kern;
} pgstrom_parambuf;

/*
 * Row-format data store
 */
typedef struct {
	StromTag		stag;
	dlist_node		chain;
	kern_column_store *kcs_head;	/* template of in-kernel column store */
	kern_row_store	kern;
} pgstrom_row_store;

/*
 * NOTE: shmem.c put a magic number to detect shared memory usage overrun.
 * So, we have a little adjustment for this padding.
 */
#define ROWSTORE_DEFAULT_SIZE	(8 * 1024 * 1024 - sizeof(cl_uint))

/*
 * Column-format data store
 */
typedef struct {
	StromTag		stag;
	dlist_node		chain;
	dlist_head		toast;	/* list of toast buffers */
	kern_column_store kern;
} pgstrom_column_store;

typedef struct {
	StromTag		stag;
	dlist_node		chain;
	cl_uint			usage;
	kern_toastbuf	kern;
} pgstrom_toastbuf;

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
#define DEVKERNEL_NEEDS_DEBUG		0x0100
#define DEVKERNEL_NEEDS_GPUSCAN		0x0200
#define DEVKERNEL_NEEDS_GPUSORT		0x0400
#define DEVKERNEL_NEEDS_HASHJOIN	0x0800

struct devtype_info;
struct devfunc_info;

typedef struct devtype_info {
	Oid			type_oid;
	uint32		type_flags;
	char	   *type_name;	/* name of device type; same of SQL's type */
	char	   *type_base;	/* base name of this type (like varlena) */
	char	   *type_decl;
	struct devfunc_info *type_is_null_fn;
	struct devfunc_info	*type_is_not_null_fn;
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

extern void *pgstrom_shmem_alloc(Size size);
extern void pgstrom_shmem_free(void *address);
extern bool pgstrom_shmem_sanitycheck(const void *address);
extern void pgstrom_setup_shmem(Size zone_length,
								void *(*callback)(void *address,
												  Size length));
extern void pgstrom_init_shmem(void);

extern Datum pgstrom_shmem_info(PG_FUNCTION_ARGS);

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
extern void pgstrom_init_mqueue(void);
extern Datum pgstrom_mqueue_info(PG_FUNCTION_ARGS);

/*
 * datastore.c
 */
extern kern_parambuf *
pgstrom_create_kern_parambuf(List *used_params,
                             ExprContext *econtext);
extern pgstrom_row_store *
pgstrom_load_row_store_heap(HeapScanDesc scan,
                            ScanDirection direction,
                            kern_colmeta *rs_colmeta,
                            kern_colmeta *cs_colmeta,
                            int cs_colnums,
                            bool *scan_done);
#ifdef USE_ASSERT_CHECKING
extern void SanityCheck_kern_column_store(kern_row_store *krs,
										  kern_column_store *kcs);
#else
#define SanityCheck_kern_column_store(X,Y)
#endif /* USE_ASSERT_CHECKING */

/*
 * restrack.c
 */
extern void pgstrom_track_object(StromTag *stag);
extern void pgstrom_untrack_object(StromTag *stag);
extern bool pgstrom_object_is_tracked(StromTag *stag);
extern void pgstrom_init_restrack(void);

/*
 * gpuscan.c
 */
extern void pgstrom_init_gpuscan(void);

/*
 * opencl_devinfo.c
 */
extern pgstrom_platform_info *
collect_opencl_platform_info(cl_platform_id platform);
extern pgstrom_device_info *
collect_opencl_device_info(cl_device_id device);

extern int	pgstrom_get_device_nums(void);
extern const pgstrom_device_info *pgstrom_get_device_info(unsigned int index);
extern void pgstrom_setup_opencl_devinfo(List *dev_list);
extern void pgstrom_init_opencl_devinfo(void);

extern size_t clserv_compute_workgroup_size(cl_kernel kernel,
											int dev_index,
											size_t num_threads,
											size_t local_memsz_per_thread);

/*
 * opencl_devprog.c
 */
#define BAD_OPENCL_PROGRAM		((void *) ~0UL)
extern bool pgstrom_kernel_debug;
extern cl_program clserv_lookup_device_program(Datum dprog_key,
											   pgstrom_message *msg);
extern Datum pgstrom_get_devprog_key(const char *source, int32 extra_libs);
extern void pgstrom_put_devprog_key(Datum dprog_key);
extern Datum pgstrom_retain_devprog_key(Datum dprog_key);
extern const char *pgstrom_get_devprog_errmsg(Datum dprog_key);
extern int32 pgstrom_get_devprog_extra_flags(Datum dprog_key);
extern const char *pgstrom_get_devprog_kernel_source(Datum dprog_key);
extern void pgstrom_init_opencl_devprog(void);
extern Datum pgstrom_opencl_device_info(PG_FUNCTION_ARGS);

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

/*
 * codegen.c
 */
typedef struct {
	List	   *type_defs;	/* list of devtype_info in use */
	List	   *func_defs;	/* list of devfunc_info in use */
	List	   *used_params;/* list of Const/Param in use */
	List	   *used_vars;	/* list of Var in use */
	int			extra_flags;/* external libraries to be included */
} codegen_context;

extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);
extern devfunc_info *pgstrom_devfunc_lookup(Oid func_oid);
extern char *pgstrom_codegen_expression(Node *expr, codegen_context *context);
extern char *pgstrom_codegen_declarations(codegen_context *context);
extern bool pgstrom_codegen_available_expression(Expr *expr);
extern void pgstrom_codegen_init(void);

/*
 * gpuscan.c
 */
extern void pgstrom_init_gpuscan(void);

/*
 * main.c
 */
extern bool	pgstrom_enabled;
extern bool pgstrom_perfmon_enabled;
extern int	pgstrom_max_async_chunks;
extern int	pgstrom_min_async_chunks;
extern void _PG_init(void);
extern const char *pgstrom_strerror(cl_int errcode);
extern void show_scan_qual(List *qual, const char *qlabel,
						   PlanState *planstate, List *ancestors,
						   ExplainState *es);
extern void show_instrumentation_count(const char *qlabel, int which,
									   PlanState *planstate, ExplainState *es);
extern void show_device_kernel(Datum dprog_key, ExplainState *es);

/*
 * debug.c
 */
extern void pgstrom_dump_kernel_debug(int elevel, kern_resultbuf *kresult);
extern Datum pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_free_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_create_queue_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_close_queue_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_create_testmsg_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_enqueue_testmsg_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_dequeue_testmsg_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_release_testmsg_func(PG_FUNCTION_ARGS);
extern void pgstrom_init_debug(void);

/*
 * opencl_*.h
 */
extern const char *pgstrom_opencl_common_code;
extern const char *pgstrom_opencl_gpuscan_code;

#endif	/* PG_STROM_H */
