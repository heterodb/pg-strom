/*
 * largeobject.c
 *
 * Routines to support data exchange between GPU memory and PG largeobjects.
 * of PG-Strom.
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
	char	   *hbuffer;
	char	   *pos;
	CUipcMemHandle ipc_mhandle;
	LargeObjectDesc *lo_desc = NULL;
	MemoryContext ccxt_saved = CurrentMemoryContext;

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

	/* try to open largeobject if valid loid is supplied */
	PG_TRY();
	{
		if (OidIsValid(loid))
			lo_desc = inv_open(loid, INV_WRITE, CurrentMemoryContext);
	}
	PG_CATCH();
	{
		MemoryContext ecxt = MemoryContextSwitchTo(ccxt_saved);
		ErrorData	 *edata;

		edata = CopyErrorData();
		if (edata->sqlerrcode != ERRCODE_UNDEFINED_OBJECT)
		{
			MemoryContextSwitchTo(ecxt);
			PG_RE_THROW();
		}
		FlushErrorState();
		Assert(lo_desc == NULL);
	}
	PG_END_TRY();

	PG_TRY();
	{
		if (lo_desc)
		{
			/* once truncate existing largeobject */
			inv_truncate(lo_desc, 0);
		}
		else
		{
			/* create a new empty largeobject */
			loid = inv_create(loid);

			lo_desc = inv_open(loid, INV_WRITE, CurrentMemoryContext);
			if (!lo_desc)
				elog(ERROR, "failed to open a new largeobject");
		}

		pos = hbuffer;
		while (length > 0)
		{
			int		nbytes = Min(length, (1U << 30));	/* up to 1GB at once */
			int		nwritten;

			nwritten = inv_write(lo_desc, pos, nbytes);
			pos += nwritten;
			length -= nwritten;
		}
	}
	PG_CATCH();
	{
		if (lo_desc)
			inv_close(lo_desc);
		PG_RE_THROW();
	}
	PG_END_TRY();

	inv_close(lo_desc);
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
	int64		lo_size;
	int64		lo_offset;
	char	   *hbuffer;
	CUipcMemHandle ipc_mhandle;
	LargeObjectDesc *lo_desc = NULL;

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
	lo_desc = inv_open(loid, INV_READ, CurrentMemoryContext);
	PG_TRY();
	{
		lo_size = inv_seek(lo_desc, 0, SEEK_END);

		/* rewind to the head, then read large object */
		inv_seek(lo_desc, 0, SEEK_SET);
		lo_offset = 0;
		while (lo_offset < lo_size)
		{
			int		nbytes = Min(lo_size - lo_offset, (1U << 30));
			int		nread;

			nread = inv_read(lo_desc, hbuffer + lo_offset, nbytes);
			lo_offset += nread;
		}
		if (lo_size < length)
			memset(hbuffer, 0, length - lo_size);

		/* send to GPU memory chunk */
		gpuIpcMemCopyFromHost(cuda_dindex,
							  ipc_mhandle,
							  offset,
							  hbuffer,
							  length);
	}
	PG_CATCH();
	{
		inv_close(lo_desc);
		PG_RE_THROW();
	}
	PG_END_TRY();
	inv_close(lo_desc);
	pfree(hbuffer);

	PG_RETURN_INT64(lo_size);
}
PG_FUNCTION_INFO_V1(pgstrom_lo_export_gpu);
