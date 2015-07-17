/*
 * gpuinfo.c
 *
 * GPU device properties collector.
 * ----
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
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
#include <stdio.h>
#include <unistd.h>
#include <libgen.h>
#include <cuda.h>

static void
__error_exit(const char *file_name, int lineno,
			 CUresult errcode, const char *message)
{
	const char *err_name;
	const char *err_string;

	cuGetErrorName(errcode, &err_name);
	cuGetErrorString(errcode, &err_string);

	fprintf(stderr, "%s:%d %s (%s:%s)\n",
			file_name, lineno, message, err_name, err_string);
	exit(1);
}

#define error_exit(errcode, message)			\
	__error_exit(__FILE__,__LINE__,(errcode),(message))

/*
 * attribute format class
 */
#define ATTRCLASS_INT			1
#define ATTRCLASS_BYTES			2
#define ATTRCLASS_KB			3
#define ATTRCLASS_MB			4
#define ATTRCLASS_KHZ			5
#define ATTRCLASS_COMPUTEMODE	6
#define ATTRCLASS_BOOL			7

#define ATTR_ENTRY(label,desc,class,is_detail)		\
	{ label, ATTRCLASS_##class, desc, #label, is_detail }

static struct
{
	CUdevice_attribute attnum;
	int			attclass;
	const char *attname_h;
	const char *attname_m;
	int			attdetail;	/* skip without -d option */
} attribute_catalog[] = {
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
			   "Max number of threads per block", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
			   "Maximum number of threads per block", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
			   "Maximum block dimension X", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
			   "Maximum block dimension Y", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
			   "Maximum block dimension Z", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
			   "Maximum grid dimension X", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
			   "Maximum grid dimension Y", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
			   "Maximum grid dimension Z", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
			   "Maximum shared memory available per block in bytes", KB, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
			   "Memory available on device for __constant__ variables",
			   BYTES, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_WARP_SIZE,
			   "Warp size in threads", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_PITCH,
			   "Maximum pitch in bytes allowed by memory copies", BYTES, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
			   "Maximum number of 32-bit registers available per block",
			   INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
			   "Typical clock frequency in kilohertz", KHZ, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
			   "Alignment requirement for textures", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
			   "Number of multiprocessors on device", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
			   "Specifies whether there is a run time limit on kernels",
			   INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_INTEGRATED,
			   "Device is integrated with host memory", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
			   "Device can map host memory into CUDA address space", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
			   "Compute mode (See CUcomputemode for details)", COMPUTEMODE, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
			   "Maximum 1D texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
			   "Maximum 2D texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
			   "Maximum 2D texture height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
			   "Maximum 3D texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
			   "Maximum 3D texture height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
			   "Maximum 3D texture depth", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
			   "Maximum 2D layered texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
			   "Maximum 2D layered texture height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
			   "Maximum layers in a 2D layered texture", INT,1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
			   "Alignment requirement for surfaces", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
			   "Device can possibly execute multiple kernels concurrently",
			   BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
			   "Device has ECC support enabled", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
			   "PCI bus ID of the device", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
			   "PCI device ID of the device", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
			   "Device is using TCC driver model", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
			   "Peak memory clock frequency in kilohertz", KHZ, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
			   "Global memory bus width in bits", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
			   "Size of L2 cache in bytes", BYTES, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
			   "Maximum resident threads per multiprocessor", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
			   "Number of asynchronous engines", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
			   "Device shares a unified address space with the host", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
			   "Maximum 1D layered texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
			   "Maximum layers in a 1D layered texture", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
			"Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set",
			   INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
			"Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set",
			   INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
			   "Alternate maximum 3D texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
			   "Alternate maximum 3D texture height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
			   "Alternate maximum 3D texture depth", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
			   "PCI domain ID of the device", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
			   "Pitch alignment requirement for textures", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
			   "Maximum cubemap texture width/height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
			   "Maximum cubemap layered texture width/height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
			   "Maximum layers in a cubemap layered texture", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
			   "Maximum 1D surface width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
			   "Maximum 2D surface width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
			   "Maximum 2D surface height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
			   "Maximum 3D surface width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
			   "Maximum 3D surface height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
			   "Maximum 3D surface depth", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
			   "Maximum 1D layered surface width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
			   "Maximum layers in a 1D layered surface", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
			   "Maximum 2D layered surface width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
			   "Maximum 2D layered surface height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
			   "Maximum layers in a 2D layered surface", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,
			   "Maximum cubemap surface width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
			   "Maximum cubemap layered surface width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
			   "Maximum layers in a cubemap layered surface", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
			   "Maximum 1D linear texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
			   "Maximum 2D linear texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
			   "Maximum 2D linear texture height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH,
			   "Maximum 2D linear texture pitch in bytes", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
			   "Maximum mipmapped 2D texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
			   "Maximum mipmapped 2D texture height", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
			   "Major compute capability version number", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
			   "Minor compute capability version number", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
			   "Maximum mipmapped 1D texture width", INT, 1),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
			   "Device supports stream priorities", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
			   "Device supports caching globals in L1", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
			   "Device supports caching locals in L1", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
			   "Maximum shared memory available per multiprocessor", BYTES, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
			   "Maximum number of 32bit registers per multiprocessor", INT, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
			   "Device can allocate managed memory on this system", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
			   "Device is on a multi-GPU board", BOOL, 0),
	ATTR_ENTRY(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
			   "Unique id for a group of devices on the same multi-GPU board",
			   INT, 0),
};

int main(int argc, char *argv[])
{
	CUdevice	device;
	CUresult	rc;
	int			i, count;
	int			opt;
	int			human_readable = -1;

	/*
	 * Parse options
	 */
	while ((opt = getopt(argc, argv, "mh")) != -1)
	{
		switch (opt)
		{
			case 'm':
				human_readable = 0;
				break;
			default:
				fprintf(stderr, "unknown option: %c\n", opt);
			case 'h':
				fprintf(stderr,
						"usage: %s [-m][-h]\n"
						"  -m : machine readable format\n"
						"  -h : shows this message\n",
						basename(argv[0]));
				return 1;
		}
	}

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuInit");

	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuDeviceGetCount");

	if (human_readable)
		printf("Number of devices: %d\n", count);
	else
		printf("CU_NUMBER_OF_DEVICE: %d\n", count);

	for (i=0; i < count; i++)
	{
		rc = cuDeviceGet(&device, i);
		if (rc != CUDA_SUCCESS)
			error_exit(rc, "failed on cuDeviceGet");



	}
	return 0;
}
