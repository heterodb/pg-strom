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
#include <CL/cl.h>

typedef struct {
	dlist_node	chain;
	/* Platform properties */
	cl_uint		pl_index;
	char		pl_profile[32];
	char		pl_version[64];
	char		pl_name[64];
	char		pl_vendor[64];
	char		pl_extension[512];
	/* Device properties */
	cl_uint		dev_address_bits;
	cl_bool		dev_available;
	char		dev_built_in_kernels[1024];
	cl_bool		dev_compiler_available;
	cl_device_fp_config	dev_double_fp_config;
	cl_bool		dev_endian_little;
	cl_bool		dev_error_correction_support;
	cl_device_exec_capabilities dev_execution_capabilities;
	char		dev_device_extensions[512];
	cl_ulong	dev_global_mem_cache_size;
	cl_device_mem_cache_type	dev_global_mem_cache_type;
	cl_uint		dev_global_mem_cacheline_size;
	cl_ulong	dev_global_mem_size;
	cl_bool		dev_host_unified_memory;
	cl_bool		dev_linker_available;
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
	char		dev_name[128];
	cl_uint		dev_native_vector_width_char;
	cl_uint		dev_native_vector_width_short;
	cl_uint		dev_native_vector_width_int;
	cl_uint		dev_native_vector_width_long;
	cl_uint		dev_native_vector_width_float;
	cl_uint		dev_native_vector_width_double;
	char		dev_opencl_c_version[128];
	cl_uint		dev_preferred_vector_width_char;
	cl_uint		dev_preferred_vector_width_short;
	cl_uint		dev_preferred_vector_width_int;
	cl_uint		dev_preferred_vector_width_long;
	cl_uint		dev_preferred_vector_width_float;
	cl_uint		dev_preferred_vector_width_double;
	size_t		dev_printf_buffer_size;
	cl_bool		dev_preferred_interop_user_sync;
	char		dev_profile[512];
	size_t		dev_profiling_timer_resolution;
	cl_command_queue_properties	dev_queue_properties;
	cl_device_fp_config	dev_single_fp_config;
	cl_device_type	dev_type;
	char		dev_vendor[80];
	cl_uint		dev_vendor_id;
	char		dev_version[80];
	char		driver_version[80];
} pgstrom_device_info;










/*
 * opencl_entry.c
 */
extern void pgstrom_init_opencl_entry(void);
extern const char *opencl_strerror(cl_int errcode);

/*
 * opencl_serv.c
 */
extern void pgstrom_init_opencl_server(void);


/*
 * main.c
 */
extern void _PG_init(void);

/*
 * shmem.c
 */
typedef struct shmem_context shmem_context;

extern shmem_context *pgstrom_shmem_context_create(const char *name);
extern void pgstrom_shmem_context_reset(shmem_context *context);
extern void pgstrom_shmem_context_delete(shmem_context *context);
extern void *pgstrom_shmem_alloc(shmem_context *contetx, Size size);
extern void pgstrom_shmem_free(void *address);

extern void pgstrom_setup_shmem(void);
extern void pgstrom_init_shmem(void);

#endif	/* PG_STROM_H */
