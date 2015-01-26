/*
 * cuda_control.c
 *
 * Overall logic to control cuda context and devices.
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
#include "postgres.h"
#include "pg_strom.h"

/* available devices set by postmaster startup */
static List		   *cuda_device_ordinals = NIL;

/* stuffs related to GpuContext */
static slock_t		gcontext_lock;
static dlist_head	gcontext_hash[100];
static GpuContext  *gcontext_last = NULL;


static inline int
gpucontext_hash_index(ResourceOwner resowner)
{
	pg_crc32	crc;

	INIT_CRC32C(crc);
	COMP_CRC32C(crc, resowner, sizeof(ResourceOwner));
	FIN_CRC32C(crc);

	return crc % lengthof(gcontext_hash);
}


static GpuContext *
pgstrom_create_gpucontext(void)
{}

GpuContext *
pgstrom_get_gpucontext(void)
{}

void
pgstrom_put_gpucontext(GpuContext *gcontext)
{}


static void
pgstrom_cleanup_gpucontext(ResourceReleasePhase phase,
						   bool is_commit,
						   bool is_toplevel,
						   void *arg)
{

}






bool
pgstrom_check_device_capability(int orginal, CUdevice device)
{
	bool		result = true;
	char		dev_name[256];
	size_t		dev_mem_sz;
	int			dev_mem_clk;
	int			dev_mem_width;
	int			dev_l2_sz;
	int			dev_cap_major;
	int			dev_cap_minor;
	int			dev_mpu_nums;
	int			dev_mpu_clk;
	CUresult	rc;
	CUdevice_attribute attrib;

	rc = cuDeviceGetName(dev_name, sizeof(dev_name), device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetName: %s", cuda_strerror(rc));

	rc = cuDeviceTotalMem(&dev_memsz, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceTotalMem: %s", cuda_strerror(rc));

	attrib = CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
	rc = cuDeviceGetAttribute(&dev_mem_clk, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", cuda_strerror(rc));

	attrib = CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
	rc = cuDeviceGetAttribute(&dev_mem_width, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", cuda_strerror(rc));

	attrib = CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
	rc = cuDeviceGetAttribute(&dev_l2_sz, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", cuda_strerror(rc));

	attrib = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
	rc = cuDeviceGetAttribute(&dev_cap_major, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", cuda_strerror(rc));

	attrib = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
	rc = cuDeviceGetAttribute(&dev_cap_minor, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", cuda_strerror(rc));

	attrib = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
	rc = cuDeviceGetAttribute(&dev_mpu_nums, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", cuda_strerror(rc));

	attrib = CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
	rc = cuDeviceGetAttribute(&dev_mpu_clk, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", cuda_strerror(rc));

	/*
	 * device older than Kepler is not supported
	 */
	if (dev_cap_major < 3)
		result = false;

	elog(LOG, "CUDA device[%d] %s (%d of SMs (%dMHz), L2 %dKB, RAM %zuMB (%dbits, %dKHz), computing capability %d.%d%s",
		 ordinal,
		 dev_name,
		 dev_mpu_nums,
		 dev_mpu_clk / 1000,
		 dev_l2_sz >> 10,
		 dev_mem_sz >> 20,
		 dev_mem_width,
		 dev_mem_clk / 1000,
		 dev_cap_major,
		 dev_cap_minor,
		 !result ? ", NOT SUPPORTED" : "");

	return result;
}

void
pgstrom_init_cuda_control(void)
{
	CUrevice	device;
	CUresult	rc;
	int			i, count;

	/*
	 * initialization of CUDA runtime
	 */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit(%s)", cuda_strerror(rc));

	/*
	 * construct a list of available devices
	 */
	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetCount(%s)", cuda_strerror(rc));

	for (i=0; i < count; i++)
	{
		rc = cuDeviceGet(&device, i);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet(%s)", cuda_strerror(rc));
		if (pgstrom_check_device_capability(i, device))
			cuda_device_ordinals = lappend_int(cuda_device_ordinals, i);
	}
	if (cuda_device_ordinals == NIL)
		elog(ERROR, "no CUDA device found on the system");

	/*
	 * initialization of GpuContext related stuff
	 */
	SpinLockInit(&gcontext_lock);
	for (i=0; i < lengthof(gcontext_hash); i++)
		dlist_init(&gcontext_hash[i]);
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);
}

/*
 * cuda_strerror
 *
 * translation from cuda error code to text representation
 */
const char *
cuda_strerror(CUresult errcode)
{
	__thread static char buffer[512];
	const char *error_val;
	const char *error_str;

	if (cuGetErrorName(errcode, &error_val) == CUDA_SUCCESS &&
		cuGetErrorString(errcode, &error_val) == CUDA_SUCCESS)
		snprintf(buffer, sizeof(buffer), "%s - %s", error_val, error_str);
	else
		snprintf(buffer, sizeof(buffer), "%d - unknown", (int)errcode);

	return buffer;
}
