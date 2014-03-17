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

#ifndef PG_USE_INLINE
#define IF_INLINE	inline
#else
#define IF_INLINE
#endif

/*
 * pgstrom_device_info
 *
 * Properties of opencl devices. Usually, it shall be collected on the
 * starting up time once, then kept on the shared memory segment.
 * Note that the properties below are supported on opencl 1.1, because
 * older driver (even if front one support 1.1) cannot understand newer
 * parameter name appeared in 1.2.
 */
typedef struct {
	cl_platform_id platform_id;		/* valid only OpenCL server */
	cl_context	context;			/* valid only OpenCL server */
	cl_uint		pl_index;
	char	   *pl_profile;
	char	   *pl_version;
	char	   *pl_name;
	char	   *pl_vendor;
	char	   *pl_extensions;
	Size		buflen;
	char		buffer[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_platform_info;

typedef struct {
	pgstrom_platform_info *pl_info;
	/* Device properties */
	cl_device_id	device_id;		/* valid only OpenCL server */
	cl_command_queue	cmdq;		/* valid only OpenCL server */
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
 * mqueue.c
 */
typedef struct {
	pthread_mutex_t	lock;
	pthread_cond_t	cond;
	dlist_head		qhead;
	int				refcnt;
	bool			closed;
} pgstrom_queue;

typedef struct {
	int				type;
	dlist_node		chain;
	pgstrom_queue  *respq;	/* queue for response message */
} pgstrom_message;

extern pgstrom_queue *pgstrom_create_queue(bool persistent);
extern bool pgstrom_enqueue_message(pgstrom_queue *queue,
									pgstrom_message *message);
extern pgstrom_message *pgstrom_dequeue_message(pgstrom_queue *queue);
extern pgstrom_message *pgstrom_try_dequeue_message(pgstrom_queue *queue);
extern pgstrom_message *pgstrom_dequeue_message_timeout(pgstrom_queue *queue,
														long wait_usec);
extern void pgstrom_close_queue(pgstrom_queue *queue);

extern void pgstrom_setup_mqueue(void);
extern void pgstrom_init_mqueue(void);

/*
 * opencl_devinfo.c
 */
extern pgstrom_device_info *pgstrom_get_opencl_device_info(int index);
extern int	pgstrom_get_opencl_device_num(void);
extern List *pgstrom_collect_opencl_device_info(int platform_index);

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

extern int	pgstrom_get_device_nums(void);
extern pgstrom_device_info *pgstrom_get_device_info(int index);
extern void pgstrom_register_device_info(List *dev_list);

extern shmem_context *pgstrom_get_mqueue_context(void);
extern void pgstrom_register_mqueue_context(shmem_context *context);

/*
 * debug.c
 */
extern Datum pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS);
extern Datum pgstrom_shmem_free_func(PG_FUNCTION_ARGS);

#endif	/* PG_STROM_H */
