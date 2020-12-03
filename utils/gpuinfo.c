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
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#include <assert.h>
#include <ctype.h>
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
#include "../src/nvme_strom.h"

/*
 * command line options
 */
static int	machine_format = 0;
static int	print_license = 0;
static int	detailed_output = 0;

#define lengthof(array)		(sizeof (array) / sizeof ((array)[0]))

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
 * commercial license validation if any
 */
static StromCmd__LicenseInfo *
commercial_license_validation(const char *license_filename)
{
	StromCmd__LicenseInfo *cmd;
	size_t		buffer_sz = 10000;
	int			nbytes;
	int			fdesc;
	int			retry_done = 0;
	int			i_year, i_mon, i_day;
	int			e_year, e_mon, e_day;
	static const char *months[] =
		{"Jan", "Feb", "Mar", "Apr", "May", "Jun",
		 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };

	/* Read the license file */
	cmd = calloc(1, sizeof(StromCmd__LicenseInfo) + buffer_sz);
	if (!cmd)
		elog("out of memory: %m");
	cmd->buffer_sz = buffer_sz;

	nbytes = read_heterodb_license_file(license_filename,
										cmd->u.buffer, buffer_sz,
										stderr);
	if (nbytes < 0)
		exit(1);
	if (nbytes == 0)
		return NULL;

	/* Load the license */
retry_open:
	fdesc = open(NVME_STROM_IOCTL_PATHNAME, O_RDONLY);
	if (fdesc < 0)
	{
		int		errcode = errno;

		if (errno == ENOENT)
		{
			if (retry_done == 0 &&
				system("/usr/bin/nvme_strom-modprobe") == 0)
			{
				retry_done = 1;
				goto retry_open;
			}
			fprintf(stderr, "nvme_strom kernel module is not installed.\n");
			return NULL;
		}
		elog("failed on open('%s'): %m\n", NVME_STROM_IOCTL_PATHNAME);
	}

	if (ioctl(fdesc, STROM_IOCTL__LICENSE_LOAD, cmd) != 0)
	{
		fprintf(stderr, "failed on ioctl(STROM_IOCTL__LICENSE_LOAD): %m\n");
		close(fdesc);
		return NULL;
	}
	close(fdesc);

	/* Print License Info */
	i_year = (cmd->issued_at / 10000);
	i_mon  = (cmd->issued_at / 100) % 100;
	i_day  = (cmd->issued_at % 100);
	e_year = (cmd->expired_at / 10000);
	e_mon  = (cmd->expired_at / 100) % 100;
	e_day  = (cmd->expired_at % 100);

	if (print_license)
	{
		int		i;

		if (machine_format)
		{
			printf("LICENSE_VERSION: %u\n", cmd->version);
			if (cmd->serial_nr)
				printf("LICENSE_SERIAL_NR: %s\n", cmd->serial_nr);
			printf("LICENSE_ISSUED_AT: %d-%s-%d\n"
				   "LICENSE_EXPIRED_AT: %d-%s-%d\n",
				   i_day, months[i_mon-1], i_year,
                   e_day, months[e_mon-1], e_year);
			if (cmd->licensee_org)
				printf("LICENSEE_ORG: %s\n", cmd->licensee_org);
			if (cmd->licensee_name)
				printf("LICENSEE_NAME: %s\n", cmd->licensee_name);
			if (cmd->licensee_mail)
				printf("LICENSEE_MAIL: %s\n", cmd->licensee_mail);
			if (cmd->description)
				printf("LICENSE_DESC: %s\n", cmd->description);
			printf("LICENSE_NR_GPUS: %d\n", cmd->nr_gpus);
		}
		else
		{
			printf("License Version: %u\n", cmd->version);
			if (cmd->serial_nr)
				printf("License Serial Number: %s\n", cmd->serial_nr);
			printf("License Issued at: %d-%s-%d\n"
				   "License Expired at: %d-%s-%d\n",
				   i_day, months[i_mon-1], i_year,
				   e_day, months[e_mon-1], e_year);
			if (cmd->licensee_org)
				printf("Licensee Organization: %s\n", cmd->licensee_org);
			if (cmd->licensee_name)
				printf("Licensee Name: %s\n", cmd->licensee_name);
			if (cmd->licensee_mail)
				printf("Licensee Mail: %s\n", cmd->licensee_mail);
			if (cmd->description)
				printf("License Description: %s\n", cmd->description);
			printf("License Num of GPUs: %d\n", cmd->nr_gpus);
		}
	}
	return cmd;
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

static int check_device(CUdevice device, StromCmd__LicenseInfo *li)
{
	CUresult	rc;
	CUuuid		dev_uuid;
	int			i, j=0;

	rc = cuDeviceGetUuid(&dev_uuid, device);
	if (rc != CUDA_SUCCESS)
		elog("failed on cuDeviceGetUuid: %s", cuErrorName(rc));

	for (i=0; i < li->nr_gpus; i++)
	{
		const char *c;
		char		uuid[16];

		for (c = li->u.gpus[i].uuid, j=0; *c != '\0' && j < 16; c++)
		{
			if (isxdigit(c[0]) && isxdigit(c[1]))
			{
				uuid[j++] = (HEX2DIG(c[0]) << 4) | (HEX2DIG(c[1]));
				c++;
			}
		}
		if (j == 16 && memcmp(dev_uuid.bytes, uuid, 16) == 0)
			return 1;
	}
	return 0;
}

static void output_device(CUdevice cuda_device,
						  nvmlDevice_t nvml_device,
						  int dindex)
{
	char		dev_name[1024];
	CUuuid		dev_uuid;
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
	rc = cuDeviceGetName(dev_name, sizeof(dev_name), cuda_device);
	if (rc != CUDA_SUCCESS)
		elog("failed on cuDeviceGetName: %s", cuErrorName(rc));
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
		case NVML_BRAND_QUADRO:  label = "QUADRO";  break;
		case NVML_BRAND_TESLA:   label = "TESLA";   break;
		case NVML_BRAND_NVS:     label = "NVS";     break;
		case NVML_BRAND_GRID:    label = "GRID";    break;
		case NVML_BRAND_GEFORCE: label = "GEFORCE"; break;
		case NVML_BRAND_TITAN:   label = "TITAN";   break;
		default:                 label = "UNKNOWN"; break;
	}
	if (!machine_format)
		printf("Device Brand: %s\n", label);
    else
		printf("DEVICE%d:DEVICE_BRAND=%s\n", dindex, label);

	/* device uuid */
	rc = cuDeviceGetUuid(&dev_uuid, cuda_device);
	if (rc != CUDA_SUCCESS)
		elog("failed on cuDeviceGetUuid: %s", cuErrorName(rc));
	snprintf(uuid, sizeof(uuid),
			 "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
			 (unsigned char)dev_uuid.bytes[0],
			 (unsigned char)dev_uuid.bytes[1],
			 (unsigned char)dev_uuid.bytes[2],
			 (unsigned char)dev_uuid.bytes[3],
			 (unsigned char)dev_uuid.bytes[4],
			 (unsigned char)dev_uuid.bytes[5],
			 (unsigned char)dev_uuid.bytes[6],
			 (unsigned char)dev_uuid.bytes[7],
			 (unsigned char)dev_uuid.bytes[8],
			 (unsigned char)dev_uuid.bytes[9],
			 (unsigned char)dev_uuid.bytes[10],
			 (unsigned char)dev_uuid.bytes[11],
			 (unsigned char)dev_uuid.bytes[12],
			 (unsigned char)dev_uuid.bytes[13],
			 (unsigned char)dev_uuid.bytes[14],
			 (unsigned char)dev_uuid.bytes[15]);
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
	StromCmd__LicenseInfo *li;
	int		   *gpu_id = NULL;
	CUdevice   *cuda_devices = NULL;
	nvmlDevice_t *nvml_devices = NULL;
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

	/*
	 * Commercial License Validation (if any)
	 *
	 * Memo: it must be called after cuInit(0), because the first call of
	 * CUDA driver setups UUID of GPU devices.
	 */
	li = commercial_license_validation(HETERODB_LICENSE_PATHNAME);

	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		elog("cuDeviceGetCount: %s", cuErrorName(rc));
	if (count > 0)
	{
		gpu_id = alloca(sizeof(int) * count);
		cuda_devices = alloca(sizeof(CUdevice) * count);
		nvml_devices = alloca(sizeof(nvmlDevice_t) * count);
		for (i=0; i < count; i++)
		{
			CUdevice		__cuda_device;
			nvmlDevice_t	__nvml_device;

			rc = cuDeviceGet(&__cuda_device, i);
			if (rc != CUDA_SUCCESS)
				elog("failed on cuDeviceGet: %s", cuErrorName(rc));
			rv = nvmlDeviceGetHandleByIndex(i, &__nvml_device);
			if (rv != NVML_SUCCESS)
				elog("failed on nvmlDeviceGetHandleByIndex: %s",
					 nvmlErrorString(rv));
			if (!li || check_device(__cuda_device, li))
			{
				gpu_id[nr_gpus] = i;
				cuda_devices[nr_gpus] = __cuda_device;
				nvml_devices[nr_gpus] = __nvml_device;
				nr_gpus++;
				if (!li)
					break;
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
					  nvml_devices[i], i);
	}
	return 0;
}
