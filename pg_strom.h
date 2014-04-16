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
#include <pthread.h>
#include <unistd.h>
#include <CL/cl.h>
#include "opencl_common.h"
#include "strom_types.h"

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
                            List *dev_attnums,
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
extern int	pgstrom_max_async_chunks;
extern int	pgstrom_min_async_chunks;
extern void _PG_init(void);
extern const char *pgstrom_strerror(cl_int errcode);
extern void show_scan_qual(List *qual, const char *qlabel,
						   PlanState *planstate, List *ancestors,
						   ExplainState *es);
extern void show_instrumentation_count(const char *qlabel, int which,
									   PlanState *planstate, ExplainState *es);
extern void show_device_kernel(const char *device_kernel, int32 extra_flags,
							   ExplainState *es);

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
