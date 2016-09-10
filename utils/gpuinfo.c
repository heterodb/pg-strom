/*
 * gpuinfo.c
 *
 * Utility program to collect GPU device properties.
 * 
 * NOTE: The reason why we have separate utility program is, to avoid
 * device memory leak (at least, it seems to us device memory is leaking
 * on the following situation).
 * We need to call cuInit() to initialize the CUDA runtime, to get device
 * properties during postmaster startup. Once we initialize the runtime,
 * it looks tp us, a small chunk of device memory is consumed.
 * After that, postmaster process fork(2) child process for each connection,
 * however, it looks to us (anyway, all the mysterious stuff was done in
 * the proprietary driver...) the above small chunk of device memory is
 * also duplicated but unmanaged by the child process, then it is still
 * kept after exit(2) of child process.
 * So, we launch gpuinfo command as a separated command to avoid to call
 * cuInit() by postmaster process.
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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

/*
 * command line options
 */
static int	machine_format = 0;
static int	detailed_output = 0;

#define lengthof(array)		(sizeof (array) / sizeof ((array)[0]))
	

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
	CUdevice_attribute attcode;
	int			attclass;
	const char *attname_h;
	const char *attname_m;
	int			attdetail;	/* skip without -d option */
} attribute_catalog[] = {
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

static void output_device(CUdevice device, int dev_id)
{
	char		dev_name[1024];
	size_t		dev_memsz;
	int			dev_prop;
	int			i;
	CUresult	rc;

	/* device identifier */
	if (!machine_format)
		printf("--------\nDevice Identifier: %d\n", dev_id);
	else
		printf("CU_DEVICE_ATTRIBUTE_DEVICE_ID=%d\n", dev_id);

	/* device name */
	rc = cuDeviceGetName(dev_name, sizeof(dev_name), device);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuDeviceGetName");
	if (!machine_format)
		printf("Device Name: %s\n", dev_name);
	else
		printf("CU_DEVICE_ATTRIBUTE_DEVICE_NAME=%s\n", dev_name);

	/* device RAM size */
	rc = cuDeviceTotalMem(&dev_memsz, device);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuDeviceTotalMem");
	if (!machine_format)
		printf("Global memory size: %zuMB\n", dev_memsz >> 20);
	else
		printf("CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_SIZE=%zu\n", dev_memsz);

	for (i=0; i < lengthof(attribute_catalog); i++)
	{
		CUdevice_attribute attcode = attribute_catalog[i].attcode;
		int         attclass  = attribute_catalog[i].attclass;
		const char *attname_h = attribute_catalog[i].attname_h;
		const char *attname_m = attribute_catalog[i].attname_m;
		int         attdetail = attribute_catalog[i].attdetail;
		const char *temp;

		/* no need to output, if not detailed mode */
		if (attdetail && !detailed_output)
			continue;

		rc = cuDeviceGetAttribute(&dev_prop, attcode, device);
		if (rc != CUDA_SUCCESS)
			error_exit(rc, "failed on cuDeviceGetAttribute");

		switch (attclass)
		{
			case ATTRCLASS_INT:
				if (!machine_format)
					printf("%s: %d\n", attname_h, dev_prop);
				else
					printf("%s=%d\n", attname_m, dev_prop);
				break;

			case ATTRCLASS_BYTES:
				if (!machine_format)
					printf("%s: %dbytes\n", attname_h, dev_prop);
				else
					printf("%s=%d\n", attname_m, dev_prop);
				break;

			case ATTRCLASS_KB:
				if (!machine_format)
					printf("%s: %dKB\n", attname_h, dev_prop);
				else
					printf("%s=%d\n", attname_m, dev_prop);
				break;

			case ATTRCLASS_MB:
				if (!machine_format)
					printf("%s: %dMB\n", attname_h, dev_prop);
				else
					printf("%s=%d\n", attname_m, dev_prop);
				break;

			case ATTRCLASS_KHZ:
				if (!machine_format)
					printf("%s: %dKHZ\n", attname_h, dev_prop);
				else
					printf("%s=%d\n", attname_m, dev_prop);
				break;

			case ATTRCLASS_COMPUTEMODE:
				switch (dev_prop)
				{
					case CU_COMPUTEMODE_DEFAULT:
						temp = "default";
						break;
#if CUDA_VERSION < 8000
					case CU_COMPUTEMODE_EXCLUSIVE:
						temp = "exclusive";
						break;
#endif
					case CU_COMPUTEMODE_PROHIBITED:
						temp = "prohibited";
						break;
					case CU_COMPUTEMODE_EXCLUSIVE_PROCESS:
						temp = "exclusive process";
						break;
					default:
						temp = "unknown";
						break;
				}
				if (!machine_format)
					printf("%s: %s\n", attname_h, temp);
				else
					printf("%s=%s\n", attname_m, temp);
				break;

			case ATTRCLASS_BOOL:
				if (!machine_format)
					printf("%s: %s\n", attname_h,
						   dev_prop ? "true" : "false");
				else
					printf("%s=%s\n", attname_m,
						   dev_prop ? "true" : "false");
				break;

			default:
				if (!machine_format)
					printf("%s: %d\n", attname_h, dev_prop);
				else
					printf("%s=%d\n", attname_m, dev_prop);
				break;
		}
	}
}

int main(int argc, char *argv[])
{
	CUdevice	device;
	CUresult	rc;
	int			version;
	int			i, count;
	int			opt;
	FILE	   *filp;

	/*
	 * Parse options
	 */
	while ((opt = getopt(argc, argv, "mdh")) != -1)
	{
		switch (opt)
		{
			case 'm':
				machine_format = 1;
				break;
			case 'd':
				detailed_output = 1;
				break;
			default:
				fprintf(stderr, "unknown option: %c\n", opt);
			case 'h':
				fprintf(stderr,
						"usage: %s [-d][-m][-h]\n"
						"  -d : detailed output\n"
						"  -m : machine readable format\n"
						"  -h : shows this message\n",
						basename(argv[0]));
				return 1;
		}
	}

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuInit");

	/*
	 * CUDA Runtime version
	 */
	rc = cuDriverGetVersion(&version);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuDriverGetVersion");
	if (!machine_format)
		printf("CUDA Runtime version: %d.%d.%d\n",
			   (version / 1000),
			   (version % 1000) / 10,
			   (version % 10));
	else
		printf("CU_PLATFORM_ATTRIBUTE_CUDA_RUNTIME_VERSION=%d.%d.%d\n",
			   (version / 1000),
               (version % 1000) / 10,
               (version % 10));
	/*
	 * NVIDIA driver version
	 */
	filp = fopen("/sys/module/nvidia/version", "rb");
	if (filp)
	{
		int		major;
		int		minor;

		if (fscanf(filp, "%d.%d", &major, &minor) == 2)
		{
			if (!machine_format)
				printf("NVIDIA Driver version: %d.%d\n", major, minor);
			else
				printf("CU_PLATFORM_ATTRIBUTE_NVIDIA_DRIVER_VERSION=%d.%d\n",
					   major, minor);
		}
		fclose(filp);
	}

	/*
	 * Number of devices available
	 */
	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuDeviceGetCount");

	if (!machine_format)
		printf("Number of devices: %d\n", count);
	else
		printf("CU_PLATFORM_ATTRIBUTE_NUMBER_OF_DEVICES=%d\n", count);

	for (i=0; i < count; i++)
	{
		rc = cuDeviceGet(&device, i);
		if (rc != CUDA_SUCCESS)
			error_exit(rc, "failed on cuDeviceGet");
		output_device(device, i);
	}
	return 0;
}
