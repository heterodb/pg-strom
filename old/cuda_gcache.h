/*
 * cuda_gcache.h
 *
 * CUDA device code specific to GPU Cache
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef CUDA_GCACHE_H
#define CUDA_GCACHE_H

#define GCACHE_TX_LOG__MAGIC		0xEBAD7C00
#define GCACHE_TX_LOG__INSERT		(GCACHE_TX_LOG__MAGIC | 'I')
#define GCACHE_TX_LOG__DELETE		(GCACHE_TX_LOG__MAGIC | 'D')
#define GCACHE_TX_LOG__XACT			(GCACHE_TX_LOG__MAGIC | 'X')

typedef struct {
	cl_uint		type;
	cl_uint		length;
	char		data[1];		/* variable length */
} GCacheTxLogCommon;

typedef struct {
	cl_uint		type;
	cl_uint		length;
	cl_uint		rowid;			/* set by GPU kernel */
	cl_bool		rowid_found;	/* set by GPU kernel */
	HeapTupleHeaderData htup __attribute__((aligned(8)));
} GCacheTxLogInsert;

typedef struct {
	cl_uint		type;
	cl_uint		length;
	cl_uint		xid;
	cl_uint		rowid;
	cl_bool		rowid_found;
	ItemPointerData ctid;
} GCacheTxLogDelete;

/*
 * COMMIT/ABORT
 */
typedef struct {
	cl_uint		type;
	cl_uint		length;
	cl_uint		xid;
	cl_uint		rowid;
	cl_bool		rowid_found;
	cl_char		tag;
	ItemPointerData ctid;
} GCacheTxLogXact;

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
	ItemPointerData ctid;
	cl_ushort	__padding__;
};
typedef struct GpuCacheSysattr	GpuCacheSysattr;

/*
 * kds_get_column_sysattr
 */
STATIC_INLINE(GpuCacheSysattr *)
kds_get_column_sysattr(kern_data_store *kds, cl_uint rowid)
{
	kern_colmeta   *cmeta = &kds->colmeta[kds->nr_colmeta - 1];
	char		   *addr;

	assert(cmeta->attbyval &&
		   cmeta->attalign == sizeof(cl_uint) &&
		   cmeta->attlen == sizeof(GpuCacheSysattr) &&
		   cmeta->nullmap_offset == 0);
	addr = (char *)kds + __kds_unpack(cmeta->values_offset);
	if (rowid < kds->nrooms)
		return ((GpuCacheSysattr *)addr) + rowid;
	return NULL;
}

#ifdef __CUDACC_RTC__
DEVICE_INLINE(cl_int)
pg_sysattr_ctid_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	GpuCacheSysattr *sysattr = kds_get_column_sysattr(kds, rowid);
	void	   *temp;

	if (!sysattr)
	{
		dclass = DATUM_CLASS__NULL;
	}
	else
	{
		temp = kern_context_alloc(kcxt, sizeof(ItemPointerData));
		if (temp)
		{
			memcpy(temp, &sysattr->ctid, sizeof(ItemPointerData));
			dclass = DATUM_CLASS__NORMAL;
			value = PointerGetDatum(temp);

			return sizeof(ItemPointerData);
		}
		dclass = DATUM_CLASS__NULL;
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
	}
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
	GpuCacheSysattr *sysattr = kds_get_column_sysattr(kds, rowid);

	if (!sysattr)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = sysattr->xmin;
	}
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_xmax_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	GpuCacheSysattr *sysattr = kds_get_column_sysattr(kds, rowid);

	if (!sysattr)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = sysattr->xmax;
	}
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_cmin_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	dclass = DATUM_CLASS__NORMAL;
	value  = rowid;
	return 0;
}

DEVICE_INLINE(cl_int)
pg_sysattr_cmax_fetch_column(kern_context *kcxt,
							 kern_data_store *kds,
							 cl_uint rowid,
							 cl_char &dclass,
							 Datum   &value)
{
	dclass = DATUM_CLASS__NORMAL;
	value  = rowid;
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
 * kern_gpucache_rowhash
 *
 * CTID-->RowId lookup table
 */
#define KERN_GPUCACHE_ROWHASH_MAGIC		0xcafebabeU
#define KERN_GPUCACHE_FREE_WIDTH		80
typedef struct
{
	cl_uint		magic;		/* =KERN_GPUCACHE_ROWHASH_MAGIC */
	cl_uint		nslots;
	cl_uint		nrooms;
	cl_uint		freelist[KERN_GPUCACHE_FREE_WIDTH];
	struct {
		cl_uint	lock;
		cl_uint	rowid;
	} slots[1];
	/*
	 * Note that:
	 * ((cl_uint *)&rowhash->slots[nslots]) is an array of rowmap
	 */
} kern_gpucache_rowhash;

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
