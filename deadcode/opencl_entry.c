/*
 * opencl_entry.c
 *
 * Entrypoint of the OpenCL runtime
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#include "postgres.h"
#include "pg_strom.h"
#include <dlfcn.h>

/*
 * Query Platform Info
 */
static cl_int (*p_clGetPlatformIDs)(
	cl_uint	num_entries,
	cl_platform_id *platforms,
	cl_uint *num_platforms) = NULL;
static cl_int (*p_clGetPlatformInfo)(
	cl_platform_id platform,
	cl_platform_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret) = NULL;

cl_int
clGetPlatformIDs(cl_uint num_entries,
				 cl_platform_id *platforms,
				 cl_uint *num_platforms)
{
	return (*p_clGetPlatformIDs)(num_entries,
								 platforms,
								 num_platforms);
}

cl_int
clGetPlatformInfo(cl_platform_id platform,
				  cl_platform_info param_name,
				  size_t param_value_size,
				  void *param_value,
				  size_t *param_value_size_ret)
{
	return (*p_clGetPlatformInfo)(platform,
								  param_name,
								  param_value_size,
								  param_value,
								  param_value_size_ret);
}

/*
 * Query Devices
 */
static cl_int (*p_clGetDeviceIDs)(
	cl_platform_id platform,
	cl_device_type device_type,
	cl_uint num_entries,
	cl_device_id *devices,
	cl_uint *num_devices) = NULL;
static cl_int (*p_clGetDeviceInfo)(
	cl_device_id device,
	cl_device_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret) = NULL;

cl_int
clGetDeviceIDs(cl_platform_id platform,
               cl_device_type device_type,
               cl_uint num_entries,
               cl_device_id *devices,
               cl_uint *num_devices)
{
	return (*p_clGetDeviceIDs)(platform,
							   device_type,
							   num_entries,
							   devices,
							   num_devices);
}

cl_int
clGetDeviceInfo(cl_device_id device,
                cl_device_info param_name,
                size_t param_value_size,
                void *param_value,
                size_t *param_value_size_ret)
{
	return (*p_clGetDeviceInfo)(device,
								param_name,
								param_value_size,
								param_value,
								param_value_size_ret);
}

/*
 * Contexts
 */
static cl_context (*p_clCreateContext)(
	const cl_context_properties *properties,
	cl_uint num_devices,
	const cl_device_id *devices,
	void (CL_CALLBACK *pfn_notify)(
		const char *errinfo,
		const void *private_info,
		size_t cb,
		void *user_data),
	void *user_data,
	cl_int *errcode_ret) = NULL;
static cl_context (*p_clCreateContextFromType)(
	const cl_context_properties  *properties,
	cl_device_type  device_type,
	void  (CL_CALLBACK *pfn_notify) (
		const char *errinfo,
		const void  *private_info,
		size_t  cb,
		void  *user_data),
	void  *user_data,
	cl_int  *errcode_ret) = NULL;
static cl_int (*p_clRetainContext)(cl_context context) = NULL;
static cl_int (*p_clReleaseContext)(cl_context context) = NULL;
static cl_int (*p_clGetContextInfo)(
	cl_context  context,
	cl_context_info  param_name,
	size_t  param_value_size,
	void  *param_value,
	size_t  *param_value_size_ret) = NULL;

cl_context
clCreateContext(const cl_context_properties *properties,
                cl_uint num_devices,
                const cl_device_id *devices,
                void (CL_CALLBACK *pfn_notify)(
					const char *errinfo,
					const void *private_info,
					size_t cb,
					void *user_data),
                void *user_data,
                cl_int *errcode_ret)
{
	return (*p_clCreateContext)(properties,
								num_devices,
								devices,
								pfn_notify,
								user_data,
								errcode_ret);
}

cl_context
clCreateContextFromType(const cl_context_properties  *properties,
						cl_device_type  device_type,
						void  (CL_CALLBACK *pfn_notify) (
							const char *errinfo,
							const void  *private_info,
							size_t  cb,
							void  *user_data),
						void  *user_data,
						cl_int  *errcode_ret)
{
	return (*p_clCreateContextFromType)(properties,
										device_type,
										pfn_notify,
										user_data,
										errcode_ret);
}

cl_int
clRetainContext(cl_context context)
{
	return (*p_clRetainContext)(context);
}

cl_int
clReleaseContext(cl_context context)
{
	return (*p_clReleaseContext)(context);
}


cl_int
clGetContextInfo(cl_context context,
				 cl_context_info param_name,
				 size_t param_value_size,
				 void *param_value,
				 size_t *param_value_size_ret)
{
	return (*p_clGetContextInfo)(context,
								 param_name,
								 param_value_size,
								 param_value,
								 param_value_size_ret);
}

/*
 * Command Queues
 */
static cl_command_queue (*p_clCreateCommandQueue)(
	cl_context context,
	cl_device_id device,
	cl_command_queue_properties properties,
	cl_int *errcode_ret) = NULL;
static cl_int (*p_clRetainCommandQueue)(cl_command_queue cmdq) = NULL;
static cl_int (*p_clReleaseCommandQueue)(cl_command_queue cmdq) = NULL;
static cl_int (*p_clGetCommandQueueInfo)(
	cl_command_queue command_queue,
	cl_command_queue_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret) = NULL;

cl_command_queue
clCreateCommandQueue(cl_context context,
					 cl_device_id device,
					 cl_command_queue_properties properties,
					 cl_int *errcode_ret)
{
	return (*p_clCreateCommandQueue)(context,
									 device,
									 properties,
									 errcode_ret);
}

cl_int
clRetainCommandQueue(cl_command_queue command_queue)
{
	return (*p_clRetainCommandQueue)(command_queue);
}

cl_int
clReleaseCommandQueue(cl_command_queue command_queue)
{
	return (*p_clReleaseCommandQueue)(command_queue);
}


cl_int
clGetCommandQueueInfo(cl_command_queue  command_queue ,
					  cl_command_queue_info  param_name ,
					  size_t  param_value_size ,
					  void  *param_value ,
					  size_t  *param_value_size_ret )
{
	return (*p_clGetCommandQueueInfo)(command_queue,
									  param_name,
									  param_value_size,
									  param_value,
									  param_value_size_ret);
}

/*
 * Buffer Objects
 */
static cl_mem (*p_clCreateBuffer)(
	cl_context context,
	cl_mem_flags flags,
	size_t size,
	void *host_ptr,
	cl_int *errcode_ret) = NULL;
static cl_mem (*p_clCreateSubBuffer)(
	cl_mem buffer,
	cl_mem_flags flags,
	cl_buffer_create_type buffer_create_type,
	const void *buffer_create_info,
	cl_int *errcode_ret) = NULL;
static cl_int (*p_clEnqueueReadBuffer)(
	cl_command_queue command_queue,
	cl_mem buffer,
	cl_bool blocking_read,
	size_t offset,
	size_t size,
	void *ptr,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event) = NULL;
static cl_int (*p_clEnqueueWriteBuffer)(
	cl_command_queue command_queue,
	cl_mem buffer,
	cl_bool blocking_write,
	size_t offset,
	size_t size,
	const void *ptr,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event) = NULL;
#if 0
/* we're waiting for nvidia's opencl 1.2 support... */
static cl_int (*p_clEnqueueFillBuffer)(
	cl_command_queue command_queue,
	cl_mem buffer,
	const void *pattern,
	size_t pattern_size,
	size_t offset,
	size_t size,
	cl_uint num_events_in_wait_list,
	const cl_event  *event_wait_list,
	cl_event  *event) = NULL;
#endif
static cl_int (*p_clEnqueueCopyBuffer)(
	cl_command_queue command_queue,
	cl_mem src_buffer,
	cl_mem dst_buffer,
	size_t src_offset,
	size_t dst_offset,
	size_t size,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event) = NULL;
static void *(*p_clEnqueueMapBuffer)(
	cl_command_queue command_queue,
	cl_mem buffer,
	cl_bool blocking_map,
	cl_map_flags map_flags,
	size_t offset,
	size_t size,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event,
	cl_int *errcode_ret) = NULL;
static cl_int (*p_clEnqueueUnmapMemObject)(
	cl_command_queue command_queue,
	cl_mem  memobj,
	void  *mapped_ptr,
	cl_uint  num_events_in_wait_list,
	const cl_event  *event_wait_list,
	cl_event  *event) = NULL;
static cl_int (*p_clGetMemObjectInfo)(
	cl_mem memobj ,
	cl_mem_info param_name ,
	size_t param_value_size ,
	void *param_value,
	size_t *param_value_size_ret) = NULL;
static cl_int (*p_clRetainMemObject)(cl_mem memobj) = NULL;
static cl_int (*p_clReleaseMemObject)(cl_mem memobj) = NULL;
static cl_int (*p_clSetMemObjectDestructorCallback)(
	cl_mem memobj,
	void (CL_CALLBACK  *pfn_notify)(
		cl_mem memobj,
		void *user_data),
	void *user_data) = NULL;

cl_mem
clCreateBuffer(cl_context context,
			   cl_mem_flags flags,
			   size_t size,
			   void *host_ptr,
			   cl_int *errcode_ret)
{
	return (*p_clCreateBuffer)(context,
							   flags,
							   size,
							   host_ptr,
							   errcode_ret);
}

cl_mem
clCreateSubBuffer(cl_mem buffer,
				  cl_mem_flags flags,
				  cl_buffer_create_type buffer_create_type,
				  const void *buffer_create_info,
				  cl_int *errcode_ret)
{
	return (*p_clCreateSubBuffer)(buffer,
								  flags,
								  buffer_create_type,
								  buffer_create_info,
								  errcode_ret);
}

cl_int
clEnqueueReadBuffer(cl_command_queue command_queue,
					cl_mem buffer,
					cl_bool blocking_read,
					size_t offset,
					size_t size,
					void *ptr,
					cl_uint num_events_in_wait_list,
					const cl_event *event_wait_list,
					cl_event *event)
{
	return (*p_clEnqueueReadBuffer)(command_queue,
									buffer,
									blocking_read,
									offset,
									size,
									ptr,
									num_events_in_wait_list,
									event_wait_list,
									event);
}

cl_int
clEnqueueWriteBuffer(cl_command_queue command_queue,
					 cl_mem buffer,
					 cl_bool blocking_write,
					 size_t offset,
					 size_t size,
					 const void *ptr,
					 cl_uint num_events_in_wait_list,
					 const cl_event *event_wait_list,
					 cl_event *event)
{
	return (*p_clEnqueueWriteBuffer)(command_queue,
									 buffer,
									 blocking_write,
									 offset,
									 size,
									 ptr,
									 num_events_in_wait_list,
									 event_wait_list,
									 event);
}

#if 0
/*
 * NOTE: clEnqueueFillBuffer is a new feature being supported
 * from OpenCL 1.2; that beyonds the requirement of PG-Strom.
 * So, we put an alternative way to work with the driver that
 * supports OpenCL 1.1 (typically, nvidia's one)
 */
static void
pgstromEnqueueFillBufferCleanup(cl_event event,
								cl_int event_command_exec_status,
								void *user_data)
{
	clReleaseKernel((cl_kernel)user_data);
}

cl_int
clEnqueueFillBuffer(cl_command_queue command_queue,
					cl_mem buffer,
					const void *pattern,
					size_t pattern_size,
					size_t offset,
					size_t size,
					cl_uint num_events_in_wait_list,
					const cl_event *event_wait_list,
					cl_event *event)
{
	static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
	static cl_program	last_program = NULL;
	static cl_context	last_context = NULL;
	cl_kernel			kernel;
	cl_program			program;
	cl_context			context;
	cl_device_id		device;
	size_t				gworksz;
	size_t				lworksz;
	char				kernel_name[80];
	cl_int				rc;
	union {
		cl_char			v_char;
		cl_short		v_short;
		cl_int			v_int;
		cl_long			v_long;
	} pattern_value;
	cl_uint				pattern_nums;

	switch (pattern_size)
	{
		case sizeof(cl_char):
			pattern_value.v_char = *((cl_char *)pattern);
			break;
		case sizeof(cl_short):
			pattern_value.v_short = *((cl_short *)pattern);
			break;
		case sizeof(cl_int):
			pattern_value.v_int = *((cl_int *)pattern);
			break;
		case sizeof(cl_long):
			pattern_value.v_long = *((cl_long *)pattern);
			break;
		default:
			/*
			 * pattern_size was not support one, even though OpenCL 1.2
			 * spec says 16, 32, 64 or 128 bytes patterns are supported.
			 */
			return CL_INVALID_VALUE;
	}

	/* ensure alignment */
	if (offset % pattern_size != 0)
		return CL_INVALID_VALUE;
	if (size % pattern_size != 0)
		return CL_INVALID_VALUE;

	/* fetch context and device_id associated with this command queue */
	rc = clGetCommandQueueInfo(command_queue,
							   CL_QUEUE_CONTEXT,
							   sizeof(cl_context),
							   &context,
							   NULL);
	if (rc != CL_SUCCESS)
		return rc;

	pthread_mutex_lock(&lock);
	if (last_program && last_context == context)
	{
		rc = clRetainProgram(last_program);
		if (rc != CL_SUCCESS)
			goto out_unlock;
		program = last_program;
	}
	else
	{
		char			source[10240];
		const char	   *prog_source[1];
		size_t			prog_length[1];
		cl_uint			num_devices;
		cl_device_id   *device_ids;
		static struct {
			const char *type_name;
			size_t		type_size;
		} pattern_types[] = {
			{ "char",  sizeof(cl_char) },
			{ "short", sizeof(cl_short) },
			{ "int",   sizeof(cl_int) },
			{ "long",  sizeof(cl_long) },
		};
		size_t		i, ofs;

		/* fetch properties of cl_context */
		rc = clGetContextInfo(context,
							  CL_CONTEXT_NUM_DEVICES,
							  sizeof(cl_uint),
							  &num_devices,
							  NULL);
		if (rc != CL_SUCCESS)
			goto out_unlock;
		Assert(num_devices > 0);

		device_ids = calloc(num_devices, sizeof(cl_device_id));
		if (!device_ids)
		{
			rc = CL_OUT_OF_HOST_MEMORY;
			goto out_unlock;
		}
		rc = clGetContextInfo(context,
							  CL_CONTEXT_DEVICES,
							  sizeof(cl_device_id) * num_devices,
							  device_ids,
							  NULL);
		if (rc != CL_SUCCESS)
		{
			free(device_ids);
			goto out_unlock;
		}

		/* release the previous program */
		if (last_program)
		{
			rc = clReleaseProgram(last_program);
			Assert(rc == CL_SUCCESS);
			last_program = NULL;
			last_context = NULL;
		}

		/* create a program object */
		for (i=0, ofs=0; i < lengthof(pattern_types); i++)
		{
			ofs += snprintf(
				source + ofs, sizeof(source) - ofs,
				"__kernel void\n"
				"pgstromEnqueueFillBuffer_%zu(__global %s *buffer,\n"
				"                             %s value, uint nums)\n"
				"{\n"
				"  if (get_global_id(0) >= nums)\n"
				"    return;\n"
				"  buffer[get_global_id(0)] = value;\n"
				"}\n",
				pattern_types[i].type_size,
				pattern_types[i].type_name,
				pattern_types[i].type_name);
		}
		prog_source[0] = source;
		prog_length[0] = ofs;
		program = clCreateProgramWithSource(context,
											1,
											prog_source,
											prog_length,
											&rc);
		if (rc != CL_SUCCESS)
		{
			free(device_ids);
			goto out_unlock;
		}

		/* build this program object */
		rc = clBuildProgram(program,
							num_devices,
							device_ids,
							NULL,
							NULL,
							NULL);
		free(device_ids);
		if (rc != CL_SUCCESS)
		{
			clReleaseProgram(program);
			goto out_unlock;
		}

		/* acquire the program object */
		rc = clRetainProgram(program);
		if (rc != CL_SUCCESS)
		{
			clReleaseProgram(program);
			goto out_unlock;
		}
		last_program = program;
		last_context = context;
	}
	pthread_mutex_unlock(&lock);
	Assert(program != NULL);

	/* fetch a device id of this command queue */
	rc = clGetCommandQueueInfo(command_queue,
							   CL_QUEUE_DEVICE,
							   sizeof(cl_device_id),
							   &device,
							   NULL);
	if (rc != CL_SUCCESS)
		goto out_release_program;

	/* fetch a kernel object to be called */
	snprintf(kernel_name, sizeof(kernel_name),
			 "pgstromEnqueueFillBuffer_%zu", pattern_size);
	kernel = clCreateKernel(program,
							kernel_name,
							&rc);
	if (rc != CL_SUCCESS)
		goto out_release_program;

	/* 1st arg: __global <typename> *buffer */
	rc = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
	if (rc != CL_SUCCESS)
		goto out_release_kernel;

	/* 2nd arg: <typename> value */
	rc = clSetKernelArg(kernel, 1, pattern_size, &pattern_value);
	if (rc != CL_SUCCESS)
		goto out_release_kernel;

	/* 3rd arg: size_t nums */
	pattern_nums = (offset + size) / pattern_size;
	rc = clSetKernelArg(kernel, 2, sizeof(cl_uint), &pattern_nums);
	if (rc != CL_SUCCESS)
		goto out_release_kernel;

	/* calculate optimal workgroup size */
	rc = clGetKernelWorkGroupInfo(kernel,
								  device,
								  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
								  sizeof(size_t),
								  &lworksz,
								  NULL);
	Assert((lworksz & (lworksz - 1)) == 0);
	gworksz = ((size / pattern_size + lworksz - 1) / lworksz) * lworksz;

	/* enqueue a kernel, instead of clEnqueueFillBuffer */
	offset /= pattern_size;
	rc = clEnqueueNDRangeKernel(command_queue,
								kernel,
								1,
								&offset,
								&gworksz,
								&lworksz,
								num_events_in_wait_list,
								event_wait_list,
								event);
	if (rc != CL_SUCCESS)
		goto out_release_kernel;

	rc = clSetEventCallback(*event,
							CL_COMPLETE,
							pgstromEnqueueFillBufferCleanup,
							kernel);
	if (rc != CL_SUCCESS)
	{
		clWaitForEvents(1, event);
		goto out_release_kernel;
	}
	return CL_SUCCESS;

out_unlock:
	pthread_mutex_unlock(&lock);
	return rc;

out_release_kernel:
	clReleaseKernel(kernel);
out_release_program:
	clReleaseProgram(program);
	return rc;
}
#endif

cl_int
clEnqueueCopyBuffer(cl_command_queue command_queue,
					cl_mem src_buffer,
					cl_mem dst_buffer,
					size_t src_offset,
					size_t dst_offset,
					size_t size,
					cl_uint num_events_in_wait_list,
					const cl_event *event_wait_list,
					cl_event *event)
{
	return (*p_clEnqueueCopyBuffer)(command_queue,
									src_buffer,
									dst_buffer,
									src_offset,
									dst_offset,
									size,
									num_events_in_wait_list,
									event_wait_list,
									event);
}

void *
clEnqueueMapBuffer(cl_command_queue command_queue,
				   cl_mem buffer,
				   cl_bool blocking_map,
				   cl_map_flags map_flags,
				   size_t offset,
				   size_t size,
				   cl_uint num_events_in_wait_list,
				   const cl_event *event_wait_list,
				   cl_event *event,
				   cl_int *errcode_ret)
{
	return (*p_clEnqueueMapBuffer)(command_queue,
								   buffer,
								   blocking_map,
								   map_flags,
								   offset,
								   size,
								   num_events_in_wait_list,
								   event_wait_list,
								   event,
								   errcode_ret);
}

cl_int
clEnqueueUnmapMemObject(cl_command_queue command_queue,
						cl_mem memobj,
						void  *mapped_ptr,
						cl_uint num_events_in_wait_list,
						const cl_event *event_wait_list,
						cl_event  *event)
{
	return (*p_clEnqueueUnmapMemObject)(command_queue,
										memobj,
										mapped_ptr,
										num_events_in_wait_list,
										event_wait_list,
										event);
}


cl_int
clGetMemObjectInfo(cl_mem memobj,
				   cl_mem_info param_name,
				   size_t param_value_size,
				   void *param_value,
				   size_t *param_value_size_ret)
{
	return (*p_clGetMemObjectInfo)(memobj,
								   param_name,
								   param_value_size,
								   param_value,
								   param_value_size_ret);
}

cl_int
clRetainMemObject(cl_mem memobj)
{
	return (*p_clRetainMemObject)(memobj);
}

cl_int
clReleaseMemObject(cl_mem memobj)
{
	return (*p_clReleaseMemObject)(memobj);
}


cl_int
clSetMemObjectDestructorCallback(cl_mem memobj,
								 void (CL_CALLBACK  *pfn_notify)(
									 cl_mem memobj,
									 void *user_data),
								 void *user_data)
{
	return (*p_clSetMemObjectDestructorCallback)(memobj,
												 pfn_notify,
												 user_data);
}

/*
 * Sampler Objects
 */
static cl_sampler (*p_clCreateSampler)(
	cl_context context,
	cl_bool normalized_coords,
	cl_addressing_mode addressing_mode,
	cl_filter_mode filter_mode,
	cl_int *errcode_ret) = NULL;
static cl_int (*p_clRetainSampler)(cl_sampler sampler) = NULL;
static cl_int (*p_clReleaseSampler)(cl_sampler sampler) = NULL;
static cl_int (*p_clGetSamplerInfo)(
	cl_sampler sampler,
	cl_sampler_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret) = NULL;

cl_sampler
clCreateSampler(cl_context context,
				cl_bool normalized_coords,
				cl_addressing_mode addressing_mode,
				cl_filter_mode filter_mode,
				cl_int *errcode_ret)
{
	return (*p_clCreateSampler)(context,
								normalized_coords,
								addressing_mode,
								filter_mode,
								errcode_ret);
}

cl_int
clRetainSampler(cl_sampler sampler)
{
	return (*p_clRetainSampler)(sampler);
}

cl_int
clReleaseSampler(cl_sampler sampler)
{
	return (*p_clReleaseSampler)(sampler);
}

cl_int
clGetSamplerInfo(cl_sampler sampler,
				 cl_sampler_info param_name,
				 size_t param_value_size,
				 void *param_value,
				 size_t *param_value_size_ret)
{
	return (*p_clGetSamplerInfo)(sampler,
								 param_name,
								 param_value_size,
								 param_value,
								 param_value_size_ret);
}

/*
 * Program Objects
 */
static cl_program (*p_clCreateProgramWithSource)(
	cl_context context,
	cl_uint count,
	const char **strings,
	const size_t *lengths,
	cl_int *errcode_ret) = NULL;
static cl_int (*p_clRetainProgram)(cl_program program) = NULL;
static cl_int (*p_clReleaseProgram)(cl_program program) = NULL;
static cl_int (*p_clBuildProgram)(
	cl_program program,
	cl_uint num_devices,
	const cl_device_id *device_list,
	const char *options,
	void (CL_CALLBACK *pfn_notify)(
		cl_program program,
		void *user_data),
	void *user_data) = NULL;
static cl_int (*p_clGetProgramInfo)(
	cl_program program,
	cl_program_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret) = NULL;
static cl_int (*p_clGetProgramBuildInfo)(
	cl_program  program,
	cl_device_id  device,
	cl_program_build_info  param_name,
	size_t  param_value_size,
	void  *param_value,
	size_t  *param_value_size_ret) = NULL;

cl_program
clCreateProgramWithSource(cl_context context,
						  cl_uint count,
						  const char **strings,
						  const size_t *lengths,
						  cl_int *errcode_ret)
{
	return (*p_clCreateProgramWithSource)(context,
										  count,
										  strings,
										  lengths,
										  errcode_ret);
}

cl_int
clRetainProgram(cl_program program)
{
	return (*p_clRetainProgram)(program);
}

cl_int
clReleaseProgram(cl_program program)
{
	return (*p_clReleaseProgram)(program);
}

cl_int
clBuildProgram(cl_program program,
			   cl_uint num_devices,
			   const cl_device_id *device_list,
			   const char *options,
			   void (CL_CALLBACK *pfn_notify)(
				   cl_program program,
				   void *user_data),
			   void *user_data)
{
	return (*p_clBuildProgram)(program,
							   num_devices,
							   device_list,
							   options,
							   pfn_notify,
							   user_data);
}

cl_int
clGetProgramInfo(cl_program program,
				 cl_program_info param_name,
				 size_t param_value_size,
				 void *param_value,
				 size_t *param_value_size_ret)
{
	return (*p_clGetProgramInfo)(program,
								 param_name,
								 param_value_size,
								 param_value,
								 param_value_size_ret);
}

cl_int
clGetProgramBuildInfo(cl_program program,
					  cl_device_id device,
					  cl_program_build_info param_name,
					  size_t param_value_size,
					  void *param_value,
					  size_t *param_value_size_ret)
{
	return (*p_clGetProgramBuildInfo)(program,
									  device,
									  param_name,
									  param_value_size,
									  param_value,
									  param_value_size_ret);
}

/*
 * Kernel Objects
 */
static cl_kernel (*p_clCreateKernel)(
	cl_program program,
	const char *kernel_name,
	cl_int *errcode_ret) = NULL;
static cl_int (*p_clCreateKernelsInProgram)(
	cl_program  program,
	cl_uint num_kernels,
	cl_kernel *kernels,
	cl_uint *num_kernels_ret) = NULL;
static cl_int (*p_clRetainKernel)(cl_kernel kernel) = NULL;
static cl_int (*p_clReleaseKernel)(cl_kernel kernel) = NULL;
static cl_int (*p_clSetKernelArg)(
	cl_kernel kernel,
	cl_uint arg_index,
	size_t arg_size,
	const void *arg_value) = NULL;
static cl_int (*p_clGetKernelInfo)(
	cl_kernel kernel,
	cl_kernel_info param_name,
	size_t param_value_size,
	void  *param_value,
	size_t *param_value_size_ret) = NULL;
static cl_int (*p_clGetKernelWorkGroupInfo)(
	cl_kernel  kernel,
	cl_device_id  device,
	cl_kernel_work_group_info param_name,
	size_t  param_value_size,
	void  *param_value,
	size_t  *param_value_size_ret) = NULL;

cl_kernel
clCreateKernel(cl_program  program,
			   const char *kernel_name,
			   cl_int *errcode_ret)
{
	return (*p_clCreateKernel)(program,
							   kernel_name,
							   errcode_ret);
}

cl_int
clCreateKernelsInProgram(cl_program program,
						 cl_uint num_kernels,
						 cl_kernel *kernels,
						 cl_uint *num_kernels_ret)
{
	return (*p_clCreateKernelsInProgram)(program,
										 num_kernels,
										 kernels,
										 num_kernels_ret);
}

cl_int
clRetainKernel(cl_kernel kernel)
{
	return (*p_clRetainKernel)(kernel);
}

cl_int
clReleaseKernel(cl_kernel kernel)
{
	return (*p_clReleaseKernel)(kernel);
}

cl_int
clSetKernelArg(cl_kernel kernel,
			   cl_uint arg_index,
			   size_t arg_size,
			   const void *arg_value)
{
	return (*p_clSetKernelArg)(kernel,
							   arg_index,
							   arg_size,
							   arg_value);
}

cl_int
clGetKernelInfo(cl_kernel kernel,
				cl_kernel_info param_name,
				size_t param_value_size,
				void *param_value,
				size_t *param_value_size_ret)
{
	return (*p_clGetKernelInfo)(kernel,
								param_name,
								param_value_size,
								param_value,
								param_value_size_ret);
}

cl_int
clGetKernelWorkGroupInfo(cl_kernel kernel,
						 cl_device_id device,
						 cl_kernel_work_group_info param_name,
						 size_t param_value_size,
						 void *param_value,
						 size_t *param_value_size_ret)
{
	return (*p_clGetKernelWorkGroupInfo)(kernel,
										 device,
										 param_name,
										 param_value_size,
										 param_value,
										 param_value_size_ret);
}

/*
 * Executing Kernels
 */
static cl_int (*p_clEnqueueNDRangeKernel)(
	cl_command_queue command_queue,
	cl_kernel kernel,
	cl_uint work_dim,
	const size_t *global_work_offset,
	const size_t *global_work_size,
	const size_t *local_work_size,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event) = NULL;
static cl_int (*p_clEnqueueTask)(
	cl_command_queue command_queue,
	cl_kernel kernel,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event) = NULL;
static cl_int (*p_clEnqueueNativeKernel)(
	cl_command_queue command_queue,
	void (*user_func)(void *),
	void *args,
	size_t cb_args,
	cl_uint num_mem_objects,
	const cl_mem *mem_list,
	const void **args_mem_loc,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event) = NULL;

cl_int
clEnqueueNDRangeKernel(cl_command_queue command_queue,
					   cl_kernel kernel,
					   cl_uint work_dim,
					   const size_t *global_work_offset,
					   const size_t *global_work_size,
					   const size_t *local_work_size,
					   cl_uint num_events_in_wait_list,
					   const cl_event *event_wait_list,
					   cl_event *event)
{
	return (*p_clEnqueueNDRangeKernel)(command_queue,
									   kernel,
									   work_dim,
									   global_work_offset,
									   global_work_size,
									   local_work_size,
									   num_events_in_wait_list,
									   event_wait_list,
									   event);
}

cl_int
clEnqueueTask(cl_command_queue command_queue,
			  cl_kernel kernel,
			  cl_uint num_events_in_wait_list,
			  const cl_event *event_wait_list,
			  cl_event *event)
{
	return (*p_clEnqueueTask)(command_queue,
							  kernel,
							  num_events_in_wait_list,
							  event_wait_list,
							  event);
}

cl_int
clEnqueueNativeKernel(cl_command_queue command_queue,
					  void (*user_func)(void *),
					  void *args,
					  size_t cb_args,
					  cl_uint num_mem_objects,
					  const cl_mem *mem_list,
					  const void **args_mem_loc,
					  cl_uint num_events_in_wait_list,
					  const cl_event *event_wait_list,
					  cl_event *event)
{
	return (*p_clEnqueueNativeKernel)(command_queue,
									  user_func,
									  args,
									  cb_args,
									  num_mem_objects,
									  mem_list,
									  args_mem_loc,
									  num_events_in_wait_list,
									  event_wait_list,
									  event);
}

/* Event Objects */
static cl_event (*p_clCreateUserEvent)(
	cl_context context,
	cl_int *errcode_ret) = NULL;
static cl_int (*p_clSetUserEventStatus)(
	cl_event event,
	cl_int execution_status) = NULL;
static cl_int (*p_clWaitForEvents)(
	cl_uint num_events,
	const cl_event *event_list) = NULL;
static cl_int (*p_clGetEventInfo)(
	cl_event event,
	cl_event_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret) = NULL;
static cl_int (*p_clSetEventCallback)(
	cl_event event,
	cl_int  command_exec_callback_type ,
	void (CL_CALLBACK  *pfn_event_notify) (
		cl_event event,
		cl_int event_command_exec_status,
		void *user_data),
	void *user_data) = NULL;
static cl_int (*p_clRetainEvent)(cl_event event) = NULL;
static cl_int (*p_clReleaseEvent)(cl_event event) = NULL;

cl_event
clCreateUserEvent(cl_context context,
				  cl_int *errcode_ret)
{
	return (*p_clCreateUserEvent)(context,
								  errcode_ret);
}

cl_int
clSetUserEventStatus(cl_event event,
					 cl_int execution_status)
{
	return (*p_clSetUserEventStatus)(event,
									 execution_status);
}

cl_int
clWaitForEvents(cl_uint num_events,
				const cl_event *event_list)
{
	return (*p_clWaitForEvents)(num_events,
								event_list);
}

cl_int
clGetEventInfo(cl_event event,
			   cl_event_info param_name,
			   size_t param_value_size,
			   void *param_value,
			   size_t *param_value_size_ret)
{
	return (*p_clGetEventInfo)(event,
							   param_name,
							   param_value_size,
							   param_value,
							   param_value_size_ret);
}

cl_int
clSetEventCallback(cl_event event,
				   cl_int  command_exec_callback_type ,
				   void (CL_CALLBACK  *pfn_event_notify) (
					   cl_event event,
					   cl_int event_command_exec_status,
					   void *user_data),
				   void *user_data)
{
	return (*p_clSetEventCallback)(event,
								   command_exec_callback_type,
								   pfn_event_notify,
								   user_data);
}

cl_int
clRetainEvent(cl_event event)
{
	return (*p_clRetainEvent)(event);
}

cl_int
clReleaseEvent(cl_event event)
{
	return (*p_clReleaseEvent)(event);
}

#if 0
/*
 * Markers, Barriers, and Waiting
 *
 * NOTE: supported only opencl 1.2
 */
static cl_int (*p_clEnqueueMarkerWithWaitList)(
	cl_command_queue command_queue,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event) = NULL;
static cl_int (*p_clEnqueueBarrierWithWaitList)(
	cl_command_queue command_queue,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event) = NULL;

cl_int
clEnqueueMarkerWithWaitList(cl_command_queue command_queue,
							cl_uint num_events_in_wait_list,
							const cl_event *event_wait_list,
							cl_event *event)
{
	return (*p_clEnqueueMarkerWithWaitList)(command_queue,
											num_events_in_wait_list,
											event_wait_list,
											event);
}

cl_int
clEnqueueBarrierWithWaitList(cl_command_queue command_queue,
							 cl_uint num_events_in_wait_list,
							 const cl_event  *event_wait_list,
							 cl_event *event)
{
	return (*p_clEnqueueBarrierWithWaitList)(command_queue,
											 num_events_in_wait_list,
											 event_wait_list,
											 event);
}
#endif

/*
 * Profiling Operations on Memory Objects and Kernels
 */
static cl_int (*p_clGetEventProfilingInfo)(
	cl_event event,
	cl_profiling_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret) = NULL;

cl_int
clGetEventProfilingInfo(cl_event event,
						cl_profiling_info param_name,
						size_t param_value_size,
						void *param_value,
						size_t *param_value_size_ret)
{
	return (*p_clGetEventProfilingInfo)(event,
										param_name,
										param_value_size,
										param_value,
										param_value_size_ret);
}

/*
 * Flush and Finish
 */
static cl_int (*p_clFlush)(cl_command_queue  command_queue) = NULL;
static cl_int (*p_clFinish)(cl_command_queue command_queue) = NULL;

cl_int
clFlush(cl_command_queue command_queue)
{
	return (*p_clFlush)(command_queue);
}

cl_int
clFinish(cl_command_queue command_queue)
{
	return (*p_clFinish)(command_queue);
}

/*
 * Init OpenCL entrypoint
 */
static void *
lookup_opencl_function(void *handle, const char *func_name)
{
	void   *func_addr = dlsym(handle, func_name);

	if (!func_addr)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("could not find symbol \"%s\" - %s",
						func_name, dlerror())));
	return func_addr;
}

#define LOOKUP_OPENCL_FUNCTION(func_name)		\
	p_##func_name = lookup_opencl_function(handle, #func_name)

void
pgstrom_init_opencl_entry(void)
{
	void   *handle;

	handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
	if (!handle)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open OpenCL library: %s", dlerror())));
	PG_TRY();
	{
		/* Query Platform Info */
		LOOKUP_OPENCL_FUNCTION(clGetPlatformIDs);
		LOOKUP_OPENCL_FUNCTION(clGetPlatformInfo);
		/* Query Devices */
		LOOKUP_OPENCL_FUNCTION(clGetDeviceIDs);
		LOOKUP_OPENCL_FUNCTION(clGetDeviceInfo);
		/* Contexts */
		LOOKUP_OPENCL_FUNCTION(clCreateContext);
		LOOKUP_OPENCL_FUNCTION(clCreateContextFromType);
		LOOKUP_OPENCL_FUNCTION(clRetainContext);
		LOOKUP_OPENCL_FUNCTION(clReleaseContext);
		LOOKUP_OPENCL_FUNCTION(clGetContextInfo);
		/* Command Queues */
		LOOKUP_OPENCL_FUNCTION(clCreateCommandQueue);
		LOOKUP_OPENCL_FUNCTION(clRetainCommandQueue);
		LOOKUP_OPENCL_FUNCTION(clReleaseCommandQueue);
		LOOKUP_OPENCL_FUNCTION(clGetCommandQueueInfo);
		/* Buffer Objects */
		LOOKUP_OPENCL_FUNCTION(clCreateBuffer);
		LOOKUP_OPENCL_FUNCTION(clCreateSubBuffer);
		LOOKUP_OPENCL_FUNCTION(clEnqueueReadBuffer);
		LOOKUP_OPENCL_FUNCTION(clEnqueueWriteBuffer);
		//LOOKUP_OPENCL_FUNCTION(clEnqueueFillBuffer);
		LOOKUP_OPENCL_FUNCTION(clEnqueueCopyBuffer);
		LOOKUP_OPENCL_FUNCTION(clEnqueueMapBuffer);
		LOOKUP_OPENCL_FUNCTION(clEnqueueUnmapMemObject);
		LOOKUP_OPENCL_FUNCTION(clGetMemObjectInfo);
		LOOKUP_OPENCL_FUNCTION(clRetainMemObject);
		LOOKUP_OPENCL_FUNCTION(clReleaseMemObject);
		LOOKUP_OPENCL_FUNCTION(clSetMemObjectDestructorCallback);
		/* Sampler Objects */
		LOOKUP_OPENCL_FUNCTION(clCreateSampler);
		LOOKUP_OPENCL_FUNCTION(clRetainSampler);
		LOOKUP_OPENCL_FUNCTION(clReleaseSampler);
		LOOKUP_OPENCL_FUNCTION(clGetSamplerInfo);
		/* Program Objects */
		LOOKUP_OPENCL_FUNCTION(clCreateProgramWithSource);
		LOOKUP_OPENCL_FUNCTION(clRetainProgram);
		LOOKUP_OPENCL_FUNCTION(clReleaseProgram);
		LOOKUP_OPENCL_FUNCTION(clBuildProgram);
		LOOKUP_OPENCL_FUNCTION(clGetProgramInfo);
		LOOKUP_OPENCL_FUNCTION(clGetProgramBuildInfo);
		LOOKUP_OPENCL_FUNCTION(clCreateKernel);
		LOOKUP_OPENCL_FUNCTION(clCreateKernelsInProgram);
		LOOKUP_OPENCL_FUNCTION(clRetainKernel);
		LOOKUP_OPENCL_FUNCTION(clReleaseKernel);
		LOOKUP_OPENCL_FUNCTION(clSetKernelArg);
		LOOKUP_OPENCL_FUNCTION(clGetKernelInfo);
		LOOKUP_OPENCL_FUNCTION(clGetKernelWorkGroupInfo);
		/* Executing Kernels */
		LOOKUP_OPENCL_FUNCTION(clEnqueueNDRangeKernel);
		LOOKUP_OPENCL_FUNCTION(clEnqueueTask);
		LOOKUP_OPENCL_FUNCTION(clEnqueueNativeKernel);
		/* Event Objects */
		LOOKUP_OPENCL_FUNCTION(clCreateUserEvent);
		LOOKUP_OPENCL_FUNCTION(clSetUserEventStatus);
		LOOKUP_OPENCL_FUNCTION(clWaitForEvents);
		LOOKUP_OPENCL_FUNCTION(clGetEventInfo);
		LOOKUP_OPENCL_FUNCTION(clSetEventCallback);
		LOOKUP_OPENCL_FUNCTION(clRetainEvent);
		LOOKUP_OPENCL_FUNCTION(clReleaseEvent);
#if 0
		/* Markers, Barriers, and Waiting */
		LOOKUP_OPENCL_FUNCTION(clEnqueueMarkerWithWaitList);
		LOOKUP_OPENCL_FUNCTION(clEnqueueBarrierWithWaitList);
#endif
		/* Profiling Operations on Memory Objects and Kernels */
		LOOKUP_OPENCL_FUNCTION(clGetEventProfilingInfo);
		/* Flush and Finish */
		LOOKUP_OPENCL_FUNCTION(clFlush);
		LOOKUP_OPENCL_FUNCTION(clFinish);
	}
	PG_CATCH();
	{
		dlclose(handle);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * utility function to generate text form of OpenCL error code
 */
const char *
opencl_strerror(cl_int errcode)
{
	static char		unknown_buf[256];

	switch (errcode)
	{
		case CL_SUCCESS:
			return "success";
		case CL_DEVICE_NOT_FOUND:
			return "device not found";
		case CL_DEVICE_NOT_AVAILABLE:
			return "device not available";
		case CL_COMPILER_NOT_AVAILABLE:
			return "compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "memory object allocation failure";
		case CL_OUT_OF_RESOURCES:
			return "out of resources";
		case CL_OUT_OF_HOST_MEMORY:
			return "out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "profiling info not available";
		case CL_MEM_COPY_OVERLAP:
			return "memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH:
			return "image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "image format not supported";
		case CL_BUILD_PROGRAM_FAILURE:
			return "build program failure";
		case CL_MAP_FAILURE:
			return "map failure";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return "misaligned sub-buffer offset";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return "execution status error for event in wait list";
		case CL_INVALID_VALUE:
			return "invalid value";
		case CL_INVALID_DEVICE_TYPE:
			return "invalid device type";
		case CL_INVALID_PLATFORM:
			return "invalid platform";
		case CL_INVALID_DEVICE:
			return "invalid device";
		case CL_INVALID_CONTEXT:
			return "invalid context";
		case CL_INVALID_QUEUE_PROPERTIES:
			return "invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE:
			return "invalid command queue";
		case CL_INVALID_HOST_PTR:
			return "invalid host pointer";
		case CL_INVALID_MEM_OBJECT:
			return "invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE:
			return "invalid image size";
		case CL_INVALID_SAMPLER:
			return "invalid sampler";
		case CL_INVALID_BINARY:
			return "invalid binary";
		case CL_INVALID_BUILD_OPTIONS:
			return "invalid build options";
		case CL_INVALID_PROGRAM:
			return "invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "invalid program executable";
		case CL_INVALID_KERNEL_NAME:
			return "invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION:
			return "invalid kernel definition";
		case CL_INVALID_KERNEL:
			return "invalid kernel";
		case CL_INVALID_ARG_INDEX:
			return "invalid argument index";
		case CL_INVALID_ARG_VALUE:
			return "invalid argument value";
		case CL_INVALID_ARG_SIZE:
			return "invalid argument size";
		case CL_INVALID_KERNEL_ARGS:
			return "invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION:
			return "invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE:
			return "invalid group size";
		case CL_INVALID_WORK_ITEM_SIZE:
			return "invalid item size";
		case CL_INVALID_GLOBAL_OFFSET:
			return "invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "invalid wait list";
		case CL_INVALID_EVENT:
			return "invalid event";
		case CL_INVALID_OPERATION:
			return "invalid operation";
		case CL_INVALID_GL_OBJECT:
			return "invalid GL object";
		case CL_INVALID_BUFFER_SIZE:
			return "invalid buffer size";
		case CL_INVALID_MIP_LEVEL:
			return "invalid MIP level";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "invalid global work size";
		case CL_INVALID_PROPERTY:
			return "invalid property";
	}
	snprintf(unknown_buf, sizeof(unknown_buf),
			 "unknown opencl error (%d)", errcode);
	return unknown_buf;
}
