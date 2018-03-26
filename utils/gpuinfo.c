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
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
#include "../src/nvme_strom.h"

/*
 * command line options
 */
static int	machine_format = 0;
static int	print_license = 0;
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
 * commercial license validation if any
 */
static int
commercial_license_validation(int *nr_gpus)
{
	StromCmd__LicenseInfo cmd;
	int			fdesc;
	int			i_year, i_mon, i_day;
	int			e_year, e_mon, e_day;
	static const char *months[] =
		{"Jan", "Feb", "Mar", "Apr", "May", "Jun",
		 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };

	/* Read the license file */
	memset(&cmd, 0, sizeof(StromCmd__LicenseInfo));
	if (read_heterodb_license_file(&cmd, HETERODB_LICENSE_PATHNAME, stderr))
		return 1;
	cmd.validation = 1;

	/* License validation */
	fdesc = open(NVME_STROM_IOCTL_PATHNAME, O_RDONLY);
	if (fdesc < 0)
	{
		fprintf(stderr, "failed on open('%s'): %m\n",
				NVME_STROM_IOCTL_PATHNAME);
		return 1;
	}
	if (ioctl(fdesc, STROM_IOCTL__LICENSE_ADMIN, &cmd) != 0)
	{
		fprintf(stderr, "failed on ioctl(STROM_IOCTL__LICENSE_ADMIN): %m\n");
		return 1;
	}
	close(fdesc);

	/* Print License Info */
	i_year = (cmd.issued_at / 10000);
	i_mon  = (cmd.issued_at / 100) % 100;
	i_day  = (cmd.issued_at % 100);
	e_year = (cmd.expired_at / 10000);
	e_mon  = (cmd.expired_at / 100) % 100;
	e_day  = (cmd.expired_at % 100);

	if (print_license)
	{
		if (machine_format)
		{
			printf("LICENSE_VERSION: %u\n"
				   "LICENSE_SERIAL_NR: %s\n"
				   "LICENSE_ISSUED_AT: %d-%s-%d\n"
				   "LICENSE_EXPIRED_AT: %d-%s-%d\n"
				   "LICENSE_NR_GPUS: %d\n"
				   "LICENSEE_ORG: %s\n"
				   "LICENSEE_NAME: %s\n"
				   "LICENSEE_MAIL: %s\n"
				   "LICENSE_DESC: %s\n",
				   cmd.version,
				   cmd.serial_nr,
				   i_day, months[i_mon-1], i_year,
				   e_day, months[e_mon-1], e_year,
				   cmd.nr_gpus,
				   cmd.licensee_org,
				   cmd.licensee_name,
				   cmd.licensee_mail,
				   cmd.license_desc ? cmd.license_desc : "");
		}
		else
		{
			printf("License Version: %u\n"
				   "License Serial Number: %s\n"
				   "License Issued at: %d-%s-%d\n"
				   "License Expired at: %d-%s-%d\n"
				   "License Num of GPUs: %d\n"
				   "Licensee Organization: %s\n"
				   "Licensee Name: %s\n"
				   "Licensee Mail: %s\n",
				   cmd.version,
				   cmd.serial_nr,
				   i_day, months[i_mon-1], i_year,
				   e_day, months[e_mon-1], e_year,
				   cmd.nr_gpus,
				   cmd.licensee_org,
				   cmd.licensee_name,
				   cmd.licensee_mail);
			if (cmd.license_desc)
				printf("License Description: %s\n", cmd.license_desc);
		}
	}
	*nr_gpus = cmd.nr_gpus;

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
		printf("DEVICE%d:DEVICE_ID=%d\n", dev_id, dev_id);

	/* device name */
	rc = cuDeviceGetName(dev_name, sizeof(dev_name), device);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuDeviceGetName");
	if (!machine_format)
		printf("Device Name: %s\n", dev_name);
	else
		printf("DEVICE%d:DEVICE_NAME=%s\n", dev_id, dev_name);

	/* device RAM size */
	rc = cuDeviceTotalMem(&dev_memsz, device);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuDeviceTotalMem");
	if (!machine_format)
		printf("Global memory size: %zuMB\n", dev_memsz >> 20);
	else
		printf("DEVICE%d:GLOBAL_MEMORY_SIZE=%zu\n", dev_id, dev_memsz);

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

		rc = cuDeviceGetAttribute(&dev_prop, attcode, device);
		if (rc != CUDA_SUCCESS)
			error_exit(rc, "failed on cuDeviceGetAttribute");

		if (machine_format)
			printf("DEVICE%d:%s=%d\n", dev_id, attname_m, dev_prop);
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
	CUdevice	device;
	CUresult	rc;
	int			version;
	int			i, count;
	int			nr_gpus = 1;
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
	/* Commercial License Validation (if any) */
	commercial_license_validation(&nr_gpus);

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
	 * Number of devices available
	 */
	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		error_exit(rc, "failed on cuDeviceGetCount");
	if (nr_gpus > count)
		nr_gpus = count;
	if (!machine_format)
		printf("Number of devices: %d\n", nr_gpus);
	else
		printf("PLATFORM:NUMBER_OF_DEVICES=%d\n", nr_gpus);

	for (i=0; i < nr_gpus; i++)
	{
		rc = cuDeviceGet(&device, i);
		if (rc != CUDA_SUCCESS)
			error_exit(rc, "failed on cuDeviceGet");
		output_device(device, i);
	}
	return 0;
}
