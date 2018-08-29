/*
 * xpack.c
 *
 * Support routines for the extra package of PG-Strom
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
#include "pg_strom.h"

static cl_ulong		PG_fnoid_gpulz_compress = InvalidOid;
static cl_ulong		PG_fnoid_gpulz_decompress = InvalidOid;

/*
 * xpack_callback_on_procoid
 */
static void
xpack_callback_on_procoid(Datum arg, int cacheid, uint32 hashvalue)
{
	Assert(cacheid == PROCOID);
	PG_fnoid_gpulz_compress = InvalidOid;
	PG_fnoid_gpulz_decompress = InvalidOid;
}

/*
 * pgstrom_gpulz_compression
 */
#define OidVectorSize(n)	offsetof(oidvector, values[(n)])

struct varlena *
pgstrom_gpulz_compression(void *buffer, size_t nbytes)
{
	FmgrInfo	flinfo;
	FunctionCallInfoData fcinfo;
	Datum		result;

	if (PG_fnoid_gpulz_compress == InvalidOid)
	{
		Oid		namespace_oid = get_namespace_oid("pgstrom", false);
		Oid		func_oid;
		union {
			oidvector	func_args;
			char		__buf[OidVectorSize(2)];
		} u;

		memset(&u, 0, sizeof(u));
		SET_VARSIZE(&u, OidVectorSize(2));
		u.func_args.ndim = 1;
		u.func_args.dataoffset = 0;
		u.func_args.elemtype = OIDOID;
		u.func_args.dim1 = 2;
		u.func_args.lbound1 = 0;
		u.func_args.values[0] = INTERNALOID;
		u.func_args.values[1] = INT8OID;

		func_oid = GetSysCacheOid3(PROCNAMEARGSNSP,
								   PointerGetDatum("gpulz_compress"),
								   PointerGetDatum(&u.func_args),
								   ObjectIdGetDatum(namespace_oid));
		if (OidIsValid(func_oid))
			PG_fnoid_gpulz_compress = func_oid;
		else
			PG_fnoid_gpulz_compress = ULONG_MAX;
	}
	if (PG_fnoid_gpulz_compress == ULONG_MAX)
		return NULL;		/* not supported */
	/*
	 * pgstrom.gpulz_compression(buffer, nbytes) can return NULL,
	 * so unable to use OidFunctionCall2() here.
	 */
	fmgr_info(PG_fnoid_gpulz_compress, &flinfo);
	InitFunctionCallInfoData(fcinfo, &flinfo, 2, InvalidOid, NULL, NULL);
	fcinfo.arg[0] = PointerGetDatum(buffer);
	fcinfo.arg[1] = Int64GetDatum(nbytes);
	fcinfo.argnull[0] = false;
	fcinfo.argnull[1] = false;
	result = FunctionCallInvoke(&fcinfo);

	if (fcinfo.isnull)
		return NULL;
	return (struct varlena *)PG_DETOAST_DATUM(result);
}

/*
 * pgstrom_gpulz_decompression
 */
struct varlena *
pgstrom_gpulz_decompression(struct varlena *compressed)
{
	Datum		result;

	if (PG_fnoid_gpulz_decompress == InvalidOid)
	{
		Oid		namespace_oid = get_namespace_oid("pgstrom", false);
		Oid		func_oid;
		union {
			oidvector	func_args;
			char		__buf[OidVectorSize(1)];
		} u;

		memset(&u, 0, sizeof(u));
		SET_VARSIZE(&u, OidVectorSize(1));
		u.func_args.ndim = 1;
		u.func_args.dataoffset = 0;
		u.func_args.elemtype = OIDOID;
		u.func_args.dim1 = 1;
		u.func_args.lbound1 = 0;
		u.func_args.values[0] = BYTEAOID;

		func_oid = GetSysCacheOid3(PROCNAMEARGSNSP,
								   PointerGetDatum("gpulz_decompress"),
								   PointerGetDatum(&u.func_args),
								   ObjectIdGetDatum(namespace_oid));
		if (OidIsValid(func_oid))
			PG_fnoid_gpulz_decompress = func_oid;
		else
			PG_fnoid_gpulz_decompress = ULONG_MAX;
	}
	if (PG_fnoid_gpulz_decompress == ULONG_MAX)
		return NULL;		/* not supported */

	result = OidFunctionCall1(PG_fnoid_gpulz_decompress,
							  PointerGetDatum(compressed));
	return DatumGetByteaP(result);
}

/*
 * kernel_gpulz_decompression
 */
void
kernel_gpulz_decompression(CUmodule cuda_module,
						   kern_errorbuf *errbuf,
						   kern_data_store *kds_col)
{
	const char *kern_fname = "kernel_gpulz_decompression";
	CUdeviceptr	m_errbuf = (CUdeviceptr)errbuf;
	CUdeviceptr	m_kds_col = (CUdeviceptr)kds_col;
	CUfunction	kern_decompress;
	CUresult	rc;
	void	   *kern_args[4];
	int			grid_sz;
	int			block_sz;

	Assert(GpuWorkerCurrentContext != NULL);
	Assert(kds_col->format == KDS_FORMAT_COLUMN);
	Assert(kds_col->ncols > 0);

	rc = cuModuleGetFunction(&kern_decompress,
							 cuda_module,
							 kern_fname);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction('%s'): %s",
			   kern_fname, errorText(rc));

	rc = gpuLargestBlockSize(&grid_sz,
							 &block_sz,
							 kern_decompress,
							 CU_DEVICE_PER_THREAD,
							 sizeof(cl_uint) * MAXTHREADS_PER_BLOCK,
							 0);
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuLargestBlockSize: %s", errorText(rc));
	grid_sz = Min(grid_sz, kds_col->ncols);

	kern_args[0] = &m_errbuf;
	kern_args[1] = &m_kds_col;

	rc = cuLaunchKernel(kern_decompress,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						sizeof(cl_uint) * MAXTHREADS_PER_BLOCK,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s", errorText(rc));
}

/*
 * pgstrom_init_xpack
 */
void
pgstrom_init_xpack(void)
{
	CacheRegisterSyscacheCallback(PROCOID, xpack_callback_on_procoid, 0);
}
