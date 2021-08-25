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
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <assert.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda.h>
#include <nvml.h>
#include <heterodb_extra.h>

/*
 * command line options
 */
static int	machine_format = 0;
static int	print_license = 0;
static int	detailed_output = 0;

#define lengthof(array)			(sizeof (array) / sizeof ((array)[0]))
#define offsetof(type, field)	((long) &((type *)0)->field)

static const char *
cuErrorName(CUresult error_code)
{
	const char *error_name;

	if (cuGetErrorName(error_code, &error_name) != CUDA_SUCCESS)
		error_name = "unknown error";
	return error_name;
}

#define elog(fmt,...)								\
	do {											\
		fprintf(stderr, "gpuinfo:%d  " fmt "\n",	\
				__LINE__, ##__VA_ARGS__);			\
		exit(1);									\
	} while(0)

/*
 * HeteroDB's license checker
 */
static int  (*p_heterodb_license_reload)(void) = NULL;
static int  (*p_heterodb_validate_device)(int gpu_device_id,
										  const char *gpu_device_name,
										  const char *gpu_device_uuid) = NULL;
static int
__heterodb_extra_open(void)
{
#define __INVALID_HANDLE	((void *)(~0UL))
	static void *heterodb_extra_handle = NULL;
	void   *handle;

	if (heterodb_extra_handle)
	{
		if (heterodb_extra_handle == __INVALID_HANDLE)
			return -1;
		return 0;
	}
	handle = dlopen(HETERODB_EXTRA_FILENAME,
					RTLD_NOW | RTLD_LOCAL);
	if (!handle)
	{
		handle = dlopen(HETERODB_EXTRA_PATHNAME,
						RTLD_NOW | RTLD_LOCAL);
		if (!handle)
			goto error_0;
	}
	p_heterodb_license_reload = dlsym(handle, "heterodb_license_reload");
	if (!p_heterodb_license_reload)
		goto error_1;
	p_heterodb_validate_device = dlsym(handle, "heterodb_validate_device");
	if (!p_heterodb_validate_device)
		goto error_1;
	heterodb_extra_handle = handle;
	return 0;

error_1:
	p_heterodb_license_reload = NULL;
	p_heterodb_validate_device = NULL;
	dlclose(handle);
error_0:
	heterodb_extra_handle = __INVALID_HANDLE;
	return -1;
}

static int
__heterodbLicenseReload(void)
{
	if (__heterodb_extra_open() != 0)
		return -1;
	assert(p_heterodb_license_reload);
	return p_heterodb_license_reload();
}

static int
__heterodbValidateDevice(int gpu_device_id,
						 const char *gpu_device_name,
						 const char *gpu_device_uuid)
{
	if (gpu_device_id == 0)
		return 1;
	if (__heterodb_extra_open() == 0)
	{
		assert(p_heterodb_validate_device);
		return p_heterodb_validate_device(gpu_device_id,
										  gpu_device_name,
										  gpu_device_uuid);
	}
	return 0;
}

/*
 * attribute format class
 */
#define ATTRCLASS_INT			1
#define ATTRCLASS_BYTES			2
#define ATTRCLASS_KB			3
#define ATTRCLASS_KHZ			4
#define ATTRCLASS_COMPUTEMODE	5
#define ATTRCLASS_BOOL			6
#define ATTRCLASS_BITS			7

#define DEV_ATTR(label,class,is_minor,desc)			\
	{ CU_DEVICE_ATTRIBUTE_##label,					\
	  ATTRCLASS_##class, desc, #label, is_minor },

static struct
{
	CUdevice_attribute attcode;
	int			attclass;
	const char *attname_h;
	const char *attname_m;
	int			attisminor;		/* skip without -d option */
} attribute_catalog[] = {
#include "../src/device_attrs.h"
};
#undef DEV_ATTR

static inline int HEX2DIG(char c)
{
	if (isdigit(c))
		return c - '0';
	else if (c >= 'a' && c <= 'f')
		return c - 'a' + 10;
	else if (c >= 'A' && c <= 'F')
		return c - 'A' + 10;
	return -1;
}

static void output_device(CUdevice cuda_device,
						  nvmlDevice_t nvml_device,
						  const char *dev_name,
						  CUuuid *dev_uuid,
						  int dindex)
{
	size_t		dev_memsz;
	int			dev_prop;
	char		uuid[80];
	const char *label;
	nvmlBrandType_t nvml_brand;
	nvmlBAR1Memory_t nvml_bar1;
	int			i;
	CUresult	rc;
	nvmlReturn_t rv;

	/* device name */
	if (!machine_format)
		printf("Device Name: %s\n", dev_name);
	else
		printf("DEVICE%d:DEVICE_NAME=%s\n", dindex, dev_name);

	/* device Brand (by NVML) */
	rv = nvmlDeviceGetBrand(nvml_device, &nvml_brand);
	if (rv != NVML_SUCCESS)
		elog("failed on nvmlDeviceGetBrand: %s", nvmlErrorString(rv));
	switch (nvml_brand)
	{
		case NVML_BRAND_QUADRO:
			label = "QUADRO";
			break;
		case NVML_BRAND_TESLA:
			label = "TESLA";
			break;
		case NVML_BRAND_NVS:
			label = "NVS";
			break;
		case NVML_BRAND_GRID:
			label = "GRID";
			break;
		case NVML_BRAND_GEFORCE:
			label = "GEFORCE";
			break;
		case NVML_BRAND_TITAN:
			label = "TITAN";
			break;
		case NVML_BRAND_NVIDIA_VAPPS:
			label = "NVIDIA_VAPPS";
			break;
		case NVML_BRAND_NVIDIA_VPC:
			label = "NVIDIA_VPC";
			break;
		case NVML_BRAND_NVIDIA_VWS:
			label = "NVIDIA_VWS";
			break;
		case NVML_BRAND_NVIDIA_VGAMING:
			label = "NVIDIA_VGAMING";
			break;
		case NVML_BRAND_QUADRO_RTX:
			label = "QUADRO_RTX";
			break;
		case NVML_BRAND_NVIDIA_RTX:
			label = "NVIDIA_RTX";
			break;
		case NVML_BRAND_NVIDIA:
			label = "NVIDIA";
			break;
		case NVML_BRAND_GEFORCE_RTX:
			label = "GEFORCE_RTX";
			break;
		case NVML_BRAND_TITAN_RTX:
			label = "TITAN_RTX";
			break;
		default:
			label = "UNKNOWN";
			break;
	}
	if (!machine_format)
		printf("Device Brand: %s\n", label);
    else
		printf("DEVICE%d:DEVICE_BRAND=%s\n", dindex, label);

	/* device uuid */
	snprintf(uuid, sizeof(uuid),
			 "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
			 (unsigned char)dev_uuid->bytes[0],
			 (unsigned char)dev_uuid->bytes[1],
			 (unsigned char)dev_uuid->bytes[2],
			 (unsigned char)dev_uuid->bytes[3],
			 (unsigned char)dev_uuid->bytes[4],
			 (unsigned char)dev_uuid->bytes[5],
			 (unsigned char)dev_uuid->bytes[6],
			 (unsigned char)dev_uuid->bytes[7],
			 (unsigned char)dev_uuid->bytes[8],
			 (unsigned char)dev_uuid->bytes[9],
			 (unsigned char)dev_uuid->bytes[10],
			 (unsigned char)dev_uuid->bytes[11],
			 (unsigned char)dev_uuid->bytes[12],
			 (unsigned char)dev_uuid->bytes[13],
			 (unsigned char)dev_uuid->bytes[14],
			 (unsigned char)dev_uuid->bytes[15]);
	if (!machine_format)
		printf("Device UUID: %s\n", uuid);
	else
		printf("DEVICE%d:DEVICE_UUID=%s\n", dindex, uuid);

	/* device RAM size */
	rc = cuDeviceTotalMem(&dev_memsz, cuda_device);
	if (rc != CUDA_SUCCESS)
		elog("failed on cuDeviceTotalMem: %s", cuErrorName(rc));
	if (!machine_format)
		printf("Global memory size: %zuMB\n", dev_memsz >> 20);
	else
		printf("DEVICE%d:GLOBAL_MEMORY_SIZE=%zu\n", dindex, dev_memsz);

	/* device BAR1 size (by NVML) */
	rv = nvmlDeviceGetBAR1MemoryInfo(nvml_device, &nvml_bar1);
	if (rv != NVML_SUCCESS)
		elog("failed on nvmlDeviceGetBAR1MemoryInfo: %s",
			 nvmlErrorString(rv));
	if (!machine_format)
		printf("PCI Bar1 memory size: %lluMB\n", nvml_bar1.bar1Total >> 20);
	else
		printf("DEVICE%d:PCI_BAR1_MEMORY_SIZE=%llu\n", dindex, nvml_bar1.bar1Total);

	for (i=0; i < lengthof(attribute_catalog); i++)
	{
		CUdevice_attribute attcode = attribute_catalog[i].attcode;
		int         attclass   = attribute_catalog[i].attclass;
		const char *attname_h  = attribute_catalog[i].attname_h;
		const char *attname_m  = attribute_catalog[i].attname_m;
		int         attisminor = attribute_catalog[i].attisminor;

		/* no need to output, if not detailed mode */
		if (attisminor && !detailed_output)
			continue;

		rc = cuDeviceGetAttribute(&dev_prop, attcode, cuda_device);
		if (rc != CUDA_SUCCESS)
		{
			if (rc == CUDA_ERROR_INVALID_VALUE)
				continue;	/* likely, not supported at this runtime */
			elog("failed on cuDeviceGetAttribute: %s", cuErrorName(rc));
		}

		if (machine_format)
			printf("DEVICE%d:%s=%d\n", dindex, attname_m, dev_prop);
		else
		{
			switch (attclass)
			{
				case ATTRCLASS_INT:
					printf("%s: %d\n", attname_h, dev_prop);
					break;

			case ATTRCLASS_BYTES:
				if (dev_prop > (4UL << 30))
					printf("%s: %.1fGB\n", attname_h,
						   (double)dev_prop / (double)(1UL << 30));
				else if (dev_prop > (4UL << 20))
					printf("%s: %.1fMB\n", attname_h,
						   (double)dev_prop / (double)(1UL << 20));
				else if (dev_prop > (4UL << 10))
					printf("%s: %.1fKB\n", attname_h,
						   (double)dev_prop / (double)(1UL << 10));
				else
					printf("%s: %dbytes\n", attname_h, dev_prop);
				break;

			case ATTRCLASS_KB:
				if (dev_prop > (4UL << 20))
					printf("%s: %.1fGB\n", attname_h,
						   (double)dev_prop / (double)(1UL << 20));
				else if (dev_prop > (4UL << 10))
					printf("%s: %.1fMB\n", attname_h,
						   (double)dev_prop / (double)(1UL << 10));
				else
					printf("%s: %dKB\n", attname_h, dev_prop);
				break;

			case ATTRCLASS_KHZ:
				if (dev_prop > 4000000UL)
					printf("%s: %.1fGHz\n", attname_h,
						   (double)dev_prop / 1000000.0);
				else if (dev_prop > 4000UL)
					printf("%s: %.1fMHz\n", attname_h,
						   (double)dev_prop / 1000.0);
				else
					printf("%s: %dKHz\n", attname_h, dev_prop);
				break;

			case ATTRCLASS_COMPUTEMODE:
				switch (dev_prop)
				{
					case CU_COMPUTEMODE_DEFAULT:
						printf("%s: default\n", attname_h);
						break;
#if CUDA_VERSION < 8000
					case CU_COMPUTEMODE_EXCLUSIVE:
						printf("%s: exclusive\n", attname_h);
						break;
#endif
					case CU_COMPUTEMODE_PROHIBITED:
						printf("%s: prohibited\n", attname_h);
						break;
					case CU_COMPUTEMODE_EXCLUSIVE_PROCESS:
						printf("%s: exclusive process\n", attname_h);
						break;
					default:
						printf("%s: unknown\n", attname_h);
						break;
				}
				break;

			case ATTRCLASS_BOOL:
				printf("%s: %s\n", attname_h, dev_prop ? "true" : "false");
				break;

			default:
				printf("%s: %d\n", attname_h, dev_prop);
				break;
			}
		}
	}
}

int main(int argc, char *argv[])
{
	int		   *gpu_id = NULL;
	CUdevice   *cuda_devices = NULL;
	nvmlDevice_t *nvml_devices = NULL;
	char	  **cuda_dev_names = NULL;
	CUuuid	  **cuda_dev_uuids = NULL;
	CUresult	rc;
	nvmlReturn_t rv;
	int			version;
	int			i, j, count;
	int			nr_gpus = 0;
	int			opt;
	FILE	   *filp;

	/*
	 * Parse options
	 */
	while ((opt = getopt(argc, argv, "mldh")) != -1)
	{
		switch (opt)
		{
			case 'm':
				machine_format = 1;
				break;
			case 'l':
				print_license = 1;
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

	/*
	 * CUDA Runtime version
	 */
	rv = nvmlInit();
	if (rv != NVML_SUCCESS)
		elog("failed on nvmlInit: %s", nvmlErrorString(rv));

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog("failed on cuInit: %s", cuErrorName(rc));

	rc = cuDriverGetVersion(&version);
	if (rc != CUDA_SUCCESS)
		elog("failed on cuDriverGetVersion: %s", cuErrorName(rc));
	if (!machine_format)
		printf("CUDA Runtime version: %d.%d.%d\n",
			   (version / 1000),
			   (version % 1000) / 10,
			   (version % 10));
	else
		printf("PLATFORM:CUDA_RUNTIME_VERSION=%d.%d.%d\n",
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
				printf("PLATFORM:NVIDIA_DRIVER_VERSION=%d.%d\n",
					   major, minor);
		}
		fclose(filp);
	}

	/* commercial license validation (if any) */
	if (__heterodbLicenseReload() == 0)
		fprintf(stderr, "license format is not valid, or expired\n");

	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		elog("cuDeviceGetCount: %s", cuErrorName(rc));
	if (count > 0)
	{
		gpu_id = alloca(sizeof(int) * count);
		cuda_devices = alloca(sizeof(CUdevice) * count);
		nvml_devices = alloca(sizeof(nvmlDevice_t) * count);
		cuda_dev_names = alloca(sizeof(char *) * count);
		cuda_dev_uuids = alloca(sizeof(CUuuid *) * count);
		for (i=0; i < count; i++)
		{
			nvmlDevice_t	__nvml_device;
			CUdevice		__cuda_device;
			char			dev_name[1024];
			CUuuid			dev_uuid;

			rc = cuDeviceGet(&__cuda_device, i);
			if (rc != CUDA_SUCCESS)
				elog("failed on cuDeviceGet: %s", cuErrorName(rc));
			rv = nvmlDeviceGetHandleByIndex(i, &__nvml_device);
			if (rv != NVML_SUCCESS)
				elog("failed on nvmlDeviceGetHandleByIndex: %s",
					 nvmlErrorString(rv));
			rc = cuDeviceGetName(dev_name, sizeof(dev_name), __cuda_device);
			if (rc != CUDA_SUCCESS)
				elog("failed on cuDeviceGetName: %s", cuErrorName(rc));
			rc = cuDeviceGetUuid(&dev_uuid, __cuda_device);
			if (rc != CUDA_SUCCESS)
				elog("failed on cuDeviceGetUuid: %s", cuErrorName(rc));
			if (__heterodbValidateDevice(i, dev_name, dev_uuid.bytes) > 0)
			{
				void   *temp;
				size_t	sz;

				gpu_id[nr_gpus] = i;
				cuda_devices[nr_gpus] = __cuda_device;
				nvml_devices[nr_gpus] = __nvml_device;

				sz = strlen(dev_name) + 1;
				temp = alloca(sz);
				memcpy(temp, dev_name, sz);
				cuda_dev_names[nr_gpus] = temp;

				temp = alloca(sizeof(CUuuid));
				memcpy(temp, &dev_uuid, sizeof(CUuuid));
				cuda_dev_uuids[nr_gpus] = temp;

				nr_gpus++;
			}
		}
	}
	if (!machine_format)
		printf("Number of devices: %d\n", nr_gpus);
	else
		printf("PLATFORM:NUMBER_OF_DEVICES=%d\n", nr_gpus);

	for (i=0; i < nr_gpus; i++)
	{
		/* device identifier */
		if (!machine_format)
			printf("--------\nDevice Identifier: %d\n", gpu_id[i]);
		else
			printf("DEVICE%d:DEVICE_ID=%d\n", i, gpu_id[i]);

		output_device(cuda_devices[i],
					  nvml_devices[i],
					  cuda_dev_names[i],
					  cuda_dev_uuids[i], i);
	}
	return 0;
}
