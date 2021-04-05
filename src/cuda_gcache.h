/*
 * cuda_gcache.h
 *
 * CUDA device code specific to GPU Cache
 * --
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
#ifndef CUDA_GCACHE_H
#define CUDA_GCACHE_H

#define GCACHE_TX_LOG__MAGIC		0xEBAD7C00
#define GCACHE_TX_LOG__INSERT		(GCACHE_TX_LOG__MAGIC | 'I')
#define GCACHE_TX_LOG__DELETE		(GCACHE_TX_LOG__MAGIC | 'D')
#define GCACHE_TX_LOG__COMMIT		(GCACHE_TX_LOG__MAGIC | 'C')

typedef struct {
	cl_uint		type;
	cl_uint		length;
	cl_ulong	timestamp;
	char		data[1];		/* variable length */
} GCacheTxLogCommon;

typedef struct {
	cl_uint		type;
	cl_uint		length;
	cl_ulong	timestamp;
	cl_uint		rowid;
	HeapTupleHeaderData htup __attribute__((aligned(8)));
} GCacheTxLogInsert;

typedef struct {
	cl_uint		type;
	cl_uint		length;
	cl_ulong	timestamp;
	cl_uint		rowid;
	cl_uint		xid;
} GCacheTxLogDelete;

/*
 * COMMIT/ABORT
 */
#define GCACHE_TX_LOG_COMMIT_ALLOCSZ	96
typedef struct {
	cl_uint		type;
	cl_uint		length;
	cl_ulong	timestamp;
	cl_uint		xid;
	cl_ushort	nitems;
	char		data[1];		/* variable length */
} GCacheTxLogCommit;

/*
 * GpuCacheSysattr
 *
 * A fixed-length system attribute for each row.
 */
struct GpuCacheSysattr
{
	cl_uint		xmin;
	cl_uint		xmax;
	/* get_global_id() of the thread who tries to update the row */
	cl_uint		owner_id;
};
typedef struct GpuCacheSysattr	GpuCacheSysattr;

#ifdef __CUDACC_RTC__
DEVICE_INLINE(cl_int)
pg_sysattr_ctid_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	dclass = DATUM_CLASS__NULL;
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_oid_fetch_column(kern_context *kcxt,
							kern_data_store *kds,
							cl_uint rowid,
							cl_char &dclass,
							Datum   &value)
{
	dclass = DATUM_CLASS__NULL;
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_xmin_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	dclass = DATUM_CLASS__NULL;
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_xmax_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	dclass = DATUM_CLASS__NULL;
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_cmin_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	dclass = DATUM_CLASS__NULL;
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_cmax_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	dclass = DATUM_CLASS__NULL;
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_tableoid_fetch_column(kern_context *kcxt,
								 kern_data_store *kds,
								 cl_uint rowid,
								 cl_char &dclass,
								 Datum   &value)
{
	if (!kds)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = kds->table_oid;
	}
	return 0;
}
#endif

/*
 * kern_gpucache_redolog
 */
typedef struct
{
	kern_errorbuf	kerror;
	size_t			length;
	cl_uint			nrooms;
	cl_uint			nitems;
	cl_uint			log_index[FLEXIBLE_ARRAY_MEMBER];
} kern_gpucache_redolog;

#endif /* CUDA_GCACHE_H */
