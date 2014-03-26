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
#include "fmgr.h"
#include "lib/ilist.h"
#include "nodes/pg_list.h"
#include <pthread.h>
#include <CL/cl.h>
#include "opencl_common.h"
#include "strom_types.h"

#ifndef PG_USE_INLINE
#define IF_INLINE	inline
#else
#define IF_INLINE
#endif

/*
 * mqueue.c
 */
extern pgstrom_queue *pgstrom_create_queue(bool is_server);
extern bool pgstrom_enqueue_message(pgstrom_message *message);
extern void pgstrom_reply_message(pgstrom_message *message);
extern pgstrom_message *pgstrom_dequeue_message(pgstrom_queue *queue);
extern pgstrom_message *pgstrom_try_dequeue_message(pgstrom_queue *queue);
extern pgstrom_message *pgstrom_dequeue_server_message(void);
extern void pgstrom_close_queue(pgstrom_queue *queue);
extern void pgstrom_get_queue(pgstrom_queue *queue);
extern void pgstrom_put_queue(pgstrom_queue *queue);
extern void pgstrom_setup_mqueue(void);
extern void pgstrom_init_mqueue(void);

/*
 * gpuscan.c
 */








/*
 * opencl_devinfo.c
 */
extern List *pgstrom_collect_device_info(int platform_index);
extern int	pgstrom_get_device_nums(void);
extern const pgstrom_device_info *pgstrom_get_device_info(unsigned int index);
extern void pgstrom_setup_opencl_devinfo(List *dev_list);
extern void pgstrom_init_opencl_devinfo(void);

/*
 * opencl_devprog.c
 */
extern cl_program pgstrom_lookup_opencl_devprog(Datum dprog_key);
extern Datum pgstrom_create_opencl_devprog(const char *source,
										   int32 extra_libs);
extern void pgstrom_get_opencl_devprog(Datum dprog_key);
extern void pgstrom_put_opencl_devprog(Datum dprog_key);
extern void pgstrom_setup_opencl_devprog(void);
extern void pgstrom_init_opencl_devprog(void);

/*
 * opencl_entry.c
 */
extern void pgstrom_init_opencl_entry(void);
extern const char *opencl_strerror(cl_int errcode);

/*
 * opencl_serv.c
 */
Datum pgstrom_opencl_device_info(PG_FUNCTION_ARGS);
extern void pgstrom_init_opencl_server(void);


/*
 * main.c
 */
extern void _PG_init(void);

/*
 * shmem.c
 */
typedef struct shmem_context shmem_context;

extern shmem_context *TopShmemContext;

extern shmem_context *pgstrom_shmem_context_create(const char *name);
extern void pgstrom_shmem_context_reset(shmem_context *context);
extern void pgstrom_shmem_context_delete(shmem_context *context);
extern void *pgstrom_shmem_alloc(shmem_context *contetx, Size size);
extern void pgstrom_shmem_free(void *address);

extern void pgstrom_setup_shmem(Size zone_length,
								void *(*callback)(void *address,
												  Size length,
												  void *cb_private),
								void *cb_private);
extern void pgstrom_init_shmem(void);

extern Datum pgstrom_shmem_block_info(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_context_info(PG_FUNCTION_ARGS);

/*
 * codegen_expr.c
 */
#define DEVINFO_IS_NEGATIVE			0x0001
#define DEVTYPE_IS_VARLENA			0x0002
#define DEVTYPE_IS_BUILTIN			0x0004
#define DEVFUNC_NEEDS_TIMELIB		0x0008
#define DEVFUNC_NEEDS_TEXTLIB		0x0010
#define DEVFUNC_NEEDS_NUMERICLIB	0x0020
#define DEVFUNC_INCL_FLAGS			\
	(DEVFUNC_NEEDS_TIMELIB | DEVFUNC_NEEDS_TEXTLIB | DEVFUNC_NEEDS_NUMERICLIB)

struct devtype_info;
struct devfunc_info;

typedef struct devtype_info {
	Oid			type_oid;
	uint32		type_flags;
	char	   *type_ident;
	char	   *type_base;
	char	   *type_decl;
	struct devfunc_info *type_is_null_fn;
	struct devfunc_info	*type_is_not_null_fn;
} devtype_info;

typedef struct devfunc_info {
	const char *func_name;
	Oid			func_namespace;
	Oid		   *func_argtypes;
	int32		func_flags;
	const char *func_ident;	/* identifier of device function */
	List	   *func_args;	/* list of devtype_info */
	devtype_info *func_rettype;
	const char *func_decl;	/* declaration of function */
} devfunc_info;

typedef struct {
	List	   *type_defs;	/* list of devtype_info in use */
	List	   *func_defs;	/* list of devfunc_info in use */
	List	   *used_params;/* list of Const/Param in use */
	List	   *used_vars;	/* list of Var in use */
	int			incl_flags;	/* external libraries to be included */
} codegen_context;

extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);
extern devfunc_info *pgstrom_devfunc_lookup(Oid func_oid);
extern char *pgstrom_codegen_expression(Node *expr, codegen_context *context);
extern char *pgstrom_codegen_declarations(codegen_context *context);
extern void pgstrom_codegen_expr_init(void);


/*
 * debug.c
 */
extern Datum pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_free_func(PG_FUNCTION_ARGS);

#endif	/* PG_STROM_H */
