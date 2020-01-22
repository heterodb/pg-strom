/*
 * largeobject.c
 *
 * Routines to support data exchange between GPU memory and PG largeobjects.
 * of PG-Strom.
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
#include "pg_strom.h"

Datum pgstrom_lo_import_gpu(PG_FUNCTION_ARGS);
Datum pgstrom_lo_export_gpu(PG_FUNCTION_ARGS);

/*
 * oid pgstrom_lo_import_gpu(
 *         int    cuda_dindex, -- index of the source GPU device
 *         bytea  ipc_handle,  -- identifier of the GPU memory block
 *         bigint offset,      -- offset from head of the GPU memory block
 *         bigint length,      -- length of the GPU memory block
 *         oid    loid = 0)    -- (optional) OID of the new largeobject
 *
 * This routine imports content of the GPU memory region into new or existing
 * PG largeobject. GPU memory regision is identified with (ipc_handle + offset
 * + length).
 */
Datum
pgstrom_lo_import_gpu(PG_FUNCTION_ARGS)
{
	int			cuda_dindex = PG_GETARG_INT32(0);
	bytea	   *handle = PG_GETARG_BYTEA_PP(1);
	int64		offset = PG_GETARG_INT64(2);
	int64		length = PG_GETARG_INT64(3);
	Oid			loid = PG_GETARG_OID(4);
	int			lo_fd;
	char	   *hbuffer;
	char	   *pos;
	Datum		datum;
	CUipcMemHandle ipc_mhandle;

	/* sanity checks */
	if (cuda_dindex < 0 || cuda_dindex >= numDevAttrs)
		elog(ERROR, "unknown GPU device index: %d", cuda_dindex);

	if (VARSIZE_ANY_EXHDR(handle) != sizeof(CUipcMemHandle))
		elog(ERROR, "length of ipc_handle mismatch (%zu of %zu bytes)",
			 VARSIZE_ANY_EXHDR(handle), sizeof(CUipcMemHandle));
	memcpy(&ipc_mhandle, VARDATA_ANY(handle), sizeof(CUipcMemHandle));

	if (offset <= 0)
		elog(ERROR, "wrong offset of GPU memory block: %ld", offset);
	if (length <= 0)
		elog(ERROR, "wrong length of GPU memory block: %ld", length);

	hbuffer = MemoryContextAllocHuge(CurrentMemoryContext, length);
	gpuIpcMemCopyToHost(hbuffer,
						cuda_dindex,
						ipc_mhandle,
						offset,
						length);
	/*
	 * Try to create a new largeobject, if loid is not valid.
	 * Then, open the largeobject and truncate it if any.
	 */
	if (!OidIsValid(loid))
	{
		datum = DirectFunctionCall1(be_lo_create,
									ObjectIdGetDatum(InvalidOid));
		loid = DatumGetObjectId(datum);
	}
	datum = DirectFunctionCall2(be_lo_open,
								ObjectIdGetDatum(loid),
								Int32GetDatum(INV_WRITE));
	lo_fd = DatumGetInt32(datum);
	DirectFunctionCall2(be_lo_truncate64,
						Int32GetDatum(lo_fd),
						Int64GetDatum(0));
	/*
	 * Write out the buffer to largeobject
	 */
	pos = hbuffer;
	while (length > 0)
	{
		int		nbytes = Min(length, (1U << 30));	/* up to 1GB at once */
		int		nwritten;

		nwritten = lo_write(lo_fd, pos, nbytes);
		pos += nwritten;
		length -= nwritten;
	}

	/* close the largeobject */
	DirectFunctionCall1(be_lo_close,
						Int32GetDatum(lo_fd));
	pfree(hbuffer);

	PG_RETURN_OID(loid);
}
PG_FUNCTION_INFO_V1(pgstrom_lo_import_gpu);

/*
 * bigint pgstrom_lo_export_gpu(
 *            oid    loid,        -- OID of the PG largeobject to export
 *            int    cuda_dindex, -- index of the destination GPU device
 *            bytea  ipc_handle,  -- identifier of the GPU memory block
 *            bigint offset,      -- offset from head of the GPU memory block
 *            bigint length)      -- length of the GPU memory block
 *
 * This routine exports content of the PG largeobject to the specified GPU
 * memory region.
 */
Datum
pgstrom_lo_export_gpu(PG_FUNCTION_ARGS)
{
	Oid			loid = PG_GETARG_OID(0);
	int			cuda_dindex = PG_GETARG_INT32(1);
	bytea	   *handle = PG_GETARG_BYTEA_PP(2);
	int64		offset = PG_GETARG_INT64(3);
	int64		length = PG_GETARG_INT64(4);
	int			lo_fd;
	size_t		lo_size;
	size_t		lo_offset;
	Datum		datum;
	char	   *hbuffer;
	CUipcMemHandle ipc_mhandle;

	/* sanity checks */
	if (cuda_dindex < 0 || cuda_dindex >= numDevAttrs)
		elog(ERROR, "unknown GPU device index: %d", cuda_dindex);

	if (VARSIZE_ANY_EXHDR(handle) != sizeof(CUipcMemHandle))
		elog(ERROR, "length of ipc_handle mismatch (%zu of %zu bytes)",
			 VARSIZE_ANY_EXHDR(handle), sizeof(CUipcMemHandle));
	memcpy(&ipc_mhandle, VARDATA_ANY(handle), sizeof(CUipcMemHandle));

	if (offset <= 0)
		elog(ERROR, "wrong offset of GPU memory block: %ld", offset);
	if (length <= 0)
		elog(ERROR, "wrong length of GPU memory block: %ld", length);
	hbuffer = MemoryContextAllocHuge(CurrentMemoryContext, length);

	/* get length of the largeobject */
	datum = DirectFunctionCall2(be_lo_open,
								ObjectIdGetDatum(loid),
								Int32GetDatum(INV_READ));
	lo_fd = DatumGetInt32(datum);


	datum = DirectFunctionCall3(be_lo_lseek64,
								Int32GetDatum(lo_fd),
								Int64GetDatum(0),
								Int32GetDatum(SEEK_END));
	lo_size = DatumGetInt64(datum);
	/* rewind to the head, then read the large object */
	DirectFunctionCall3(be_lo_lseek64,
						lo_fd,
						Int64GetDatum(0),
						Int32GetDatum(SEEK_SET));
	lo_offset = 0;
	while (lo_offset < lo_size)
	{
		int		nbytes = Min(lo_size - lo_offset, (1U << 30));
		int		nread;

		nread = lo_read(lo_fd, hbuffer + lo_offset, nbytes);
		lo_offset += nread;
	}
	if (lo_size < length)
		memset(hbuffer + lo_size, 0, length - lo_size);

	/* send to GPU memory chunk */
	gpuIpcMemCopyFromHost(cuda_dindex,
						  ipc_mhandle,
						  offset,
						  hbuffer,
						  length);

	/* release resources */
	DirectFunctionCall1(be_lo_close,
						Int32GetDatum(lo_fd));
	pfree(hbuffer);

	PG_RETURN_INT64(lo_size);
}
PG_FUNCTION_INFO_V1(pgstrom_lo_export_gpu);
