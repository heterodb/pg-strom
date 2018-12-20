/*
 * gstore_buf.c
 *
 * Buffer management for Gstore_Fdw
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

/*
 * GpuStoreChunk - shared structure
 */
typedef struct
{
	dlist_node		chain;
	cl_uint			revision;
	pg_crc32		hash;
	Oid				database_oid;
	Oid				table_oid;
	TransactionId	xmax;
	TransactionId	xmin;
	bool			xmax_committed;
	bool			xmin_committed;
	cl_int			pinning;	/* CUDA device index */
	cl_int			format;		/* one of GSTORE_FDW_FORMAT__* */
	size_t			rawsize;	/* rawsize regardless of the internal format */
	size_t			nitems;		/* nitems regardless of the internal format */
	CUipcMemHandle	ipc_mhandle;
	dsm_handle		dsm_mhandle;
} GpuStoreChunk;

/*
 * GpuStoreHead - shared structure
 */
#define GSTORE_CHUNK_HASH_NSLOTS	97
typedef struct
{
	pg_atomic_uint32 revision_seed;
	pg_atomic_uint32 has_warm_chunks;
	slock_t			lock;
	dlist_head		free_chunks;
	dlist_head		active_chunks[GSTORE_CHUNK_HASH_NSLOTS];
	GpuStoreChunk	gs_chunks[FLEXIBLE_ARRAY_MEMBER];
} GpuStoreHead;

typedef struct
{
	TransactionId	xmin;
	TransactionId	xmax;
	CommandId		cid;
	bool			not_all_visible;
	bool			xmin_committed;
	bool			xmax_committed;
} MVCCAttrs;

struct GpuStoreBuffer
{
	Oid			table_oid;	/* oid of the gstore_fdw */
	cl_int		pinning;	/* CUDA device index */
	cl_int		format;		/* one of GSTORE_FDW_FORMAT__* */
	cl_uint		revision;	/* revision number of the buffer */
	bool		read_only;	/* true, if read-write buffer is not ready */
	bool		is_dirty;	/* true, if any updates happen on the read-
							 * write buffer, thus read-only buffer is
							 * not uptodata any more. */
	MemoryContext memcxt;	/* memory context of read-write buffer */
	/* read-only buffer */
	size_t		rawsize;
	dsm_segment	*h_seg;
	union {
		kern_data_store *kds;
		void   *buffer;		/* copy of GPU device memory if any */
	} h;
	CUipcMemHandle ipc_mhandle;
	/* read-write buffer */
	int			nattrs;
	size_t		nitems;
	size_t		nrooms;
	bool	   *hasnull;
	bits8	  **nullmap;
	void	  **values;
	HTAB	  **vl_dict;		/* for varlena columns  */
	int		   *vl_compress;	/* one of GSTORE_COMPRESSION__* */
	size_t	   *extra_sz;
	MVCCAttrs  *gs_mvcc;		/* MVCC attributes */
};

/*
 * vl_dict_key - dictionary of varlena datum
 */
typedef struct
{
	cl_uint		offset;		/* to be used later */
	struct varlena *vl_datum;
} vl_dict_key;

/* static variables */
static int				gstore_max_relations;	/* GUC */
static object_access_hook_type object_access_next;
static shmem_startup_hook_type shmem_startup_next;
static GpuStoreHead	   *gstore_head = NULL;
static HTAB			   *gstore_buffer_htab = NULL;

/* SQL functions */
Datum pgstrom_gstore_fdw_chunk_info(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_format(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_nitems(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_nattrs(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_rawsize(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_export_ipchandle(PG_FUNCTION_ARGS);

/*
 * gstore_buf_chunk_visibility - equivalent to HeapTupleSatisfiesMVCC,
 * but simplified for GpuStoreChunk because only commited chunks are written
 * to the shared memory object.
 */
static bool
gstore_buf_chunk_visibility(GpuStoreChunk *gs_chunk, Snapshot snapshot)
{
	/* xmin is committed, but maybe not according to our snapshot */
	if (gs_chunk->xmin != FrozenTransactionId &&
		XidInMVCCSnapshot(gs_chunk->xmin, snapshot))
		return false;       /* treat as still in progress */
	/* by here, the inserting transaction has committed */
	if (!TransactionIdIsValid(gs_chunk->xmax))
		return true;    /* nobody deleted yet */
	/* xmax is committed, but maybe not according to our snapshot */
	if (XidInMVCCSnapshot(gs_chunk->xmax, snapshot))
		return true;
	/* xmax transaction committed */
	return false;
}

/*
 * gstore_fdw_tuple_visibility
 */
static bool
gstore_buf_tuple_visibility(MVCCAttrs *mvcc, Snapshot snapshot)
{
	if (!mvcc->xmin_committed)
	{
		if (mvcc->xmin == InvalidTransactionId)
			return false;
		if (TransactionIdIsCurrentTransactionId(mvcc->xmin))
		{
			if (mvcc->cid >= snapshot->curcid)
				return false;	/* inserted after scan started */
			if (mvcc->xmax == InvalidTransactionId)
				return true;
			if (!TransactionIdIsCurrentTransactionId(mvcc->xmax))
			{
				/* deleting subtransaction must have aborted */
				mvcc->xmax = InvalidTransactionId;
				mvcc->xmax_committed = false;
				return true;
			}
			if (mvcc->cid >= snapshot->curcid)
				return true;	/* deleted after scan started */
			else
				return false;	/* deleted before scan started */
		}
		else if (XidInMVCCSnapshot(mvcc->xmin, snapshot))
			return false;
        else if (TransactionIdDidCommit(mvcc->xmin))
			mvcc->xmin_committed = true;
        else
        {
            /* it must have aborted or crashed */
			mvcc->xmin = InvalidTransactionId;
			return false;
		}
	}
	else
	{
		/* xmin is committed, but maybe not according to our snapshot */
		if (mvcc->xmin != FrozenTransactionId &&
			XidInMVCCSnapshot(mvcc->xmin, snapshot))
			return false;	/* treat as still in progress */
	}

	/* by here, the inserting transaction has committed */
	if (mvcc->xmax == InvalidTransactionId)
	{
		Assert(!mvcc->xmax_committed);
		return true;
	}
	if (!mvcc->xmax_committed)
	{
		if (TransactionIdIsCurrentTransactionId(mvcc->xmax))
		{
			if (mvcc->cid >= snapshot->curcid)
				return true;	/* deleted after scan started */
			else
				return false;	/* deleted before scan started */
		}
		if (XidInMVCCSnapshot(mvcc->xmax, snapshot))
			return true;
		if (!TransactionIdDidCommit(mvcc->xmax))
		{
			mvcc->xmax = InvalidTransactionId;
			mvcc->xmax_committed = false;
			return true;
		}
		/* xmax transaction committed*/
		mvcc->xmax_committed = true;
	}
	else
	{
		/* xmax is committed, but maybe not according to our snapshot */
		if (XidInMVCCSnapshot(mvcc->xmax, snapshot))
			return true;
	}
	/* xmax transaction committed */
	return false;
}

/*
 * gstore_buf_visibility_bitmap
 */
static bits8 *
gstore_buf_visibility_bitmap(GpuStoreBuffer *gs_buffer, size_t *p_nrooms)
{
	size_t		i, nitems = gs_buffer->nitems;
	size_t		nrooms = 0;
	bits8	   *rowmap;
	bool		has_invisible = false;

	if (nitems == 0)
	{
		*p_nrooms = 0;
		return NULL;
	}

	rowmap = palloc0(BITMAPLEN(nitems));
	for (i=0; i < nitems; i++)
	{
		MVCCAttrs  *mvcc = &gs_buffer->gs_mvcc[i];

		if (!TransactionIdIsCurrentTransactionId(mvcc->xmax))
		{
			/*
			 * Row is exist on the initial load (it means somebody others
			 * inserted or updated, then committed. gstore_fdw always takes
			 * exclusive lock towards concurrent writer operations (INSERT/
			 * UPDATE/DELETE), so no need to pay attention for the updates
			 * by the concurrent transactions.
			 */
			rowmap[(i >> 3)] |= (1 << (i & 7));
			nrooms++;
		}
		else
			has_invisible = true;
	}
	*p_nrooms = nrooms;
	if (has_invisible)
		return rowmap;
	pfree(rowmap);
	return NULL;	/* all-visible */
}

/*
 * gstore_buf_chunk_hashvalue
 */
static inline pg_crc32
gstore_buf_chunk_hashvalue(Oid ftable_oid)
{
	pg_crc32		hash;

	INIT_LEGACY_CRC32(hash);
	COMP_LEGACY_CRC32(hash, &MyDatabaseId, sizeof(Oid));
	COMP_LEGACY_CRC32(hash, &ftable_oid, sizeof(Oid));
	FIN_LEGACY_CRC32(hash);

	return hash;
}

/*
 * gstore_buf_lookup_chunk
 */
static GpuStoreChunk *
gstore_buf_lookup_chunk(Oid ftable_oid, Snapshot snapshot)
{
	GpuStoreChunk  *gs_chunk = NULL;

	SpinLockAcquire(&gstore_head->lock);
	PG_TRY();
	{
		pg_crc32	hash = gstore_buf_chunk_hashvalue(ftable_oid);
		int			index = hash % GSTORE_CHUNK_HASH_NSLOTS;
		dlist_iter	iter;

		dlist_foreach(iter, &gstore_head->active_chunks[index])
		{
			GpuStoreChunk  *gs_temp = dlist_container(GpuStoreChunk,
													  chain, iter.cur);
			if (gs_temp->hash == hash &&
				gs_temp->database_oid == MyDatabaseId &&
				gs_temp->table_oid == ftable_oid &&
				gstore_buf_chunk_visibility(gs_temp, snapshot))
			{
				if (!gs_chunk)
					gs_chunk = gs_temp;
				else
					elog(ERROR, "Bug? multiple GpuStoreChunks are visible");
			}
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&gstore_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&gstore_head->lock);

	return gs_chunk;
}

/*
 * gstore_buf_insert_chunk
 */
static void
gstore_buf_insert_chunk(GpuStoreBuffer *gs_buffer,
						size_t nrooms,
						CUipcMemHandle ipc_mhandle,
						dsm_handle dsm_mhandle)
{
	dlist_node	   *dnode;
	GpuStoreChunk  *gs_chunk;
	int				index;
	dlist_iter		iter;

	Assert(gs_buffer->pinning < numDevAttrs);
	Assert(gs_buffer->read_only &&
		   gs_buffer->h_seg != NULL &&
		   gs_buffer->h.buffer == dsm_segment_address(gs_buffer->h_seg));
	/* setup GpuStoreChunk */
	SpinLockAcquire(&gstore_head->lock);
	if (dlist_is_empty(&gstore_head->free_chunks))
	{
		SpinLockRelease(&gstore_head->lock);
		elog(ERROR, "gstore_fdw: out of GpuStoreChunk strucure");
	}
	dnode = dlist_pop_head_node(&gstore_head->free_chunks);
	gs_chunk = dlist_container(GpuStoreChunk, chain, dnode);
	SpinLockRelease(&gstore_head->lock);

	gs_chunk->revision
		= pg_atomic_add_fetch_u32(&gstore_head->revision_seed, 1);
	gs_chunk->hash = gstore_buf_chunk_hashvalue(gs_buffer->table_oid);
	gs_chunk->database_oid = MyDatabaseId;
	gs_chunk->table_oid = gs_buffer->table_oid;
	gs_chunk->xmax = InvalidTransactionId;
	gs_chunk->xmin = GetCurrentTransactionId();
	gs_chunk->pinning = gs_buffer->pinning;
	gs_chunk->format = gs_buffer->format;
	gs_chunk->rawsize = gs_buffer->rawsize;
	gs_chunk->nitems = nrooms;
	gs_chunk->ipc_mhandle = ipc_mhandle;
	gs_chunk->dsm_mhandle = dsm_mhandle;
	/* remember the revision when buffer is built */
	gs_buffer->revision = gs_chunk->revision;

	/* add GpuStoreChunk to the shared hash table */
	index = gs_chunk->hash % GSTORE_CHUNK_HASH_NSLOTS;
	SpinLockAcquire(&gstore_head->lock);
	dlist_foreach(iter, &gstore_head->active_chunks[index])
	{
		GpuStoreChunk  *gs_temp = dlist_container(GpuStoreChunk,
												  chain, iter.cur);
		if (gs_temp->hash == gs_chunk->hash &&
			gs_temp->database_oid == gs_chunk->database_oid &&
			gs_temp->table_oid == gs_chunk->table_oid &&
			gs_temp->xmax == InvalidTransactionId)
		{
			gs_temp->xmax = gs_chunk->xmin;
		}
	}
	dlist_push_head(&gstore_head->active_chunks[index],
					&gs_chunk->chain);
	pg_atomic_add_fetch_u32(&gstore_head->has_warm_chunks, 1);
	SpinLockRelease(&gstore_head->lock);
}

/*
 * gstore_buf_release_chunk
 *
 * memo: must be called under the 'gstore_head->lock'
 */
static void
gstore_buf_release_chunk(GpuStoreChunk *gs_chunk)
{
	dlist_delete(&gs_chunk->chain);
	gpuMemFreePreserved(gs_chunk->pinning,
						gs_chunk->ipc_mhandle);
	memset(gs_chunk, 0, sizeof(GpuStoreChunk));
	dlist_push_head(&gstore_head->free_chunks,
					&gs_chunk->chain);
}

/*
 * vl_dict_hash_value - hash value of varlena dictionary
 */
static uint32
vl_dict_hash_value(const void *__key, Size keysize)
{
	const vl_dict_key *key = __key;
	pg_crc32	crc;

	if (VARATT_IS_EXTERNAL(key->vl_datum))
		elog(ERROR, "unexpected external toast datum");

	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, key->vl_datum, VARSIZE_ANY(key->vl_datum));
	FIN_LEGACY_CRC32(crc);

	return (uint32) crc;
}

/*
 * vl_dict_compare - comparison of varlena dictionary
 */
static int
vl_dict_compare(const void *__key1, const void *__key2, Size keysize)
{
	const vl_dict_key *key1 = __key1;
	const vl_dict_key *key2 = __key2;

	if (VARATT_IS_EXTERNAL(key1->vl_datum) ||
		VARATT_IS_EXTERNAL(key2->vl_datum))
		elog(ERROR, "unexpected external toast datum");

	if (VARATT_IS_COMPRESSED(key1->vl_datum) ==
		VARATT_IS_COMPRESSED(key2->vl_datum))
	{
		/*
		 * Both of the varlena datum are compressed or uncompressed,
		 * so binary comparison is sufficient to check full-match.
		 */
		if (VARSIZE_ANY_EXHDR(key1->vl_datum) ==
			VARSIZE_ANY_EXHDR(key2->vl_datum))
			return memcmp(VARDATA_ANY(key1->vl_datum),
						  VARDATA_ANY(key2->vl_datum),
						  VARSIZE_ANY_EXHDR(key1->vl_datum));
	}
	else
	{
		struct varlena *temp1 = pg_detoast_datum(key1->vl_datum);
		struct varlena *temp2 = pg_detoast_datum(key2->vl_datum);
		int			result;

		if (VARSIZE_ANY_EXHDR(temp1) == VARSIZE_ANY_EXHDR(temp2))
		{
			result = memcmp(VARDATA_ANY(temp1),
							VARDATA_ANY(temp2),
							VARSIZE_ANY_EXHDR(temp1));
			if (temp1 != key1->vl_datum)
				pfree(temp1);
			if (temp2 != key2->vl_datum)
				pfree(temp2);
			return result;
		}
		if (temp1 != key1->vl_datum)
			pfree(temp1);
		if (temp2 != key2->vl_datum)
			pfree(temp2);
	}
	return 1;
}

/*
 * vl_datum_compression - makes varlena compressed
 */
static struct varlena *
vl_datum_compression(void *datum, int vl_comression)
{
	struct varlena *vl;

	if (vl_comression == GSTORE_COMPRESSION__NONE)
	{
		/* not compressed */
		vl = PG_DETOAST_DATUM(datum);
	}
	else if (vl_comression == GSTORE_COMPRESSION__PGLZ)
	{
		/* try pglz comression */
		if (VARATT_IS_COMPRESSED(datum))
			vl = (struct varlena *)DatumGetPointer(datum);
		else
		{
			struct varlena *temp = PG_DETOAST_DATUM(datum);
			struct varlena *comp;

			comp = (struct varlena *)
				toast_compress_datum(PointerGetDatum(temp));
			if (!comp)
				vl = temp;
			else
			{
				vl = comp;
				pfree(temp);
			}
		}
	}
	else
		elog(ERROR, "unknown format %d", vl_comression);

	return vl;
}

/*
 * GpuStoreBufferCopyFromKDS
 *
 * It fills up the initial read-write buffer by the read-only KDS.
 */
static void
GpuStoreBufferCopyFromKDS(GpuStoreBuffer *gs_buffer,
						  TupleDesc tupdesc,
						  kern_data_store *kds)
{
	size_t		i, nitems = kds->nitems;
	cl_uint		j;
	MemoryContext oldcxt;

	Assert(kds->ncols == tupdesc->natts &&
		   kds->ncols == gs_buffer->nattrs);
	if (kds->nitems > gs_buffer->nrooms)
		elog(ERROR, "lack of GpuStoreBuffer rooms");

	oldcxt = MemoryContextSwitchTo(gs_buffer->memcxt);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
		kern_colmeta *cmeta = &kds->colmeta[j];
		ssize_t		va_offset;
		ssize_t		va_length;
		void	   *addr;

		/* sanity check */
		Assert(cmeta->attbyval  == attr->attbyval &&
			   cmeta->attalign  == att_align_nominal(1, attr->attalign) &&
			   cmeta->attlen    == attr->attlen &&
			   cmeta->attnum    == attr->attnum &&
			   cmeta->atttypid  == attr->atttypid &&
			   cmeta->atttypmod == attr->atttypmod);
		/* skip if already dropped */
		if (attr->attisdropped)
			continue;

		va_offset = __kds_unpack(cmeta->va_offset);
		va_length = __kds_unpack(cmeta->va_length);
		/* considered as all-null */
		if (va_offset == 0)
		{
			if (cmeta->attlen < 0)
			{
				Assert(gs_buffer->nullmap[j] == NULL);
				gs_buffer->hasnull[j] = true;
				memset(gs_buffer->values[j],
					   0, sizeof(vl_dict_key *) * nitems);
				Assert(gs_buffer->vl_dict[j] != NULL);
				gs_buffer->extra_sz[j] = 0;
			}
			else
			{
				memset(gs_buffer->nullmap[j], 0, BITMAPLEN(nitems));
				gs_buffer->hasnull[j] = true;
				Assert(gs_buffer->vl_dict[j] == NULL);
				gs_buffer->extra_sz[j] = 0;
			}
			continue;
		}
		addr = (char *)kds + va_offset;
		if (cmeta->attlen < 0)
		{
			vl_dict_key	  **vl_array = (vl_dict_key **)gs_buffer->values[j];
			cl_int			vl_compress	= gs_buffer->vl_compress[j];

			for (i=0; i < kds->nitems; i++)
			{
				vl_dict_key	key, *entry;
				size_t		offset;
				void	   *datum;
				bool		found;

				offset = __kds_unpack(((cl_uint *)addr)[i]);
				if (offset == 0)
				{
					vl_array[i] = NULL;
					continue;
				}
				datum = (char *)addr + offset;
				key.offset = 0;
				key.vl_datum = (struct varlena *)datum;
				entry = hash_search(gs_buffer->vl_dict[j],
									&key,
									HASH_ENTER,
									&found);
				if (!found)
				{
					struct varlena *vl
						= vl_datum_compression(datum, vl_compress);
					if (vl == datum)
					{
						struct varlena *temp = palloc(VARSIZE_ANY(vl));

						memcpy(temp, vl, VARSIZE_ANY(vl));
						vl = temp;
					}
					entry->vl_datum = vl;
					gs_buffer->extra_sz[j] += MAXALIGN(VARSIZE(vl));
				}
				if (MAXALIGN(sizeof(cl_uint) * nitems) +
					gs_buffer->extra_sz[j] >= KDS_OFFSET_MAX_SIZE)
					elog(ERROR, "too much vl_dictionary consumption");
				vl_array[i] = entry;
			}
		}
		else
		{
			int		unitsz = TYPEALIGN(cmeta->attalign,
									   cmeta->attlen);
			size_t	extra_sz = va_length - MAXALIGN(unitsz * nitems);

			if (extra_sz > 0)
			{
				Assert(extra_sz == MAXALIGN(BITMAPLEN(nitems)));
				memcpy(gs_buffer->nullmap[j],
					   (char *)addr + MAXALIGN(unitsz * nitems),
					   BITMAPLEN(nitems));
				gs_buffer->hasnull[j] = true;
			}
			else
			{
				memset(gs_buffer->nullmap[j], ~0, BITMAPLEN(nitems));
				gs_buffer->hasnull[j] = false;
			}
			memcpy(gs_buffer->values[j], addr, unitsz * nitems);
			Assert(gs_buffer->vl_dict[j] == NULL);
			gs_buffer->extra_sz[j] = 0;
		}
	}
	MemoryContextSwitchTo(oldcxt);
}

/*
 * GpuStoreBufferCopyToKDS - setup KDS by the read-write buffer
 */
static void
GpuStoreBufferCopyToKDS(kern_data_store *kds,
						GpuStoreBuffer *gs_buffer,
						TupleDesc tupdesc,
						bits8 *rowmap, size_t visible_nitems)
{
	size_t	nrooms = (!rowmap ? gs_buffer->nitems : visible_nitems);
	char   *pos;
	long	i, j, k;

	init_kernel_data_store(kds,
						   tupdesc,
						   SIZE_MAX,	/* to be set later */
						   KDS_FORMAT_COLUMN,
						   nrooms,
						   true);
	Assert(gs_buffer->nattrs == tupdesc->natts);

	pos = KERN_DATA_STORE_BODY(kds);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
		kern_colmeta   *cmeta = &kds->colmeta[j];
		size_t			offset;
		size_t			nbytes;

		/* skip dropped / empty columns */
		if (attr->attisdropped || !gs_buffer->values[j])
			continue;

		offset = ((char *)pos - (char *)kds);
		cmeta->va_offset = __kds_packed(offset);
		if (cmeta->attlen < 0)
		{
			cl_uint	   *base = (cl_uint *)pos;
			char	   *extra = pos + MAXALIGN(sizeof(cl_uint) * nrooms);
			vl_dict_key **vl_entries = (vl_dict_key **)gs_buffer->values[j];
			vl_dict_key *entry;

			//TODO: sort physical location of varlena entries
			//to simplify GPU sorting
			for (i=0, k=0; i < gs_buffer->nitems; i++)
			{
				/* only visible rows */
				if (rowmap && att_isnull(i, rowmap))
					continue;
				entry = vl_entries[i];
				if (!entry)
					base[k] = 0;
				else
				{
					if (entry->offset == 0)
					{
						offset = (size_t)(extra - (char *)base);
						entry->offset = __kds_packed(offset);
						nbytes = VARSIZE_ANY(entry->vl_datum);
						Assert(nbytes > 0);
						memcpy(extra, entry->vl_datum, nbytes);
						extra += MAXALIGN(nbytes);
					}
					base[k] = entry->offset;
				}
				k++;
			}
			Assert(k == nrooms);
			nbytes = ((char *)extra - (char *)base);
			pos += nbytes;
			cmeta->va_length = __kds_packed(nbytes);
		}
		else if (!rowmap)
		{
			/* all-visible fixed-length attribute */
			char	   *base = pos;

			cmeta->va_offset = __kds_packed(pos - (char *)kds);
			nbytes = MAXALIGN(TYPEALIGN(cmeta->attalign,
										cmeta->attlen) * nrooms);
			memcpy(pos, gs_buffer->values[j], nbytes);
			pos += nbytes;
			/* null bitmap, if any */
			if (gs_buffer->hasnull[j])
			{
				nbytes = MAXALIGN(BITMAPLEN(nrooms));
				memcpy(pos, gs_buffer->nullmap[j], nbytes);
				pos += nbytes;
			}
			nbytes = (pos - base);
			cmeta->va_length = __kds_packed(nbytes);
		}
		else
		{
			char	   *base = pos;
			bool		meet_null = false;
			char	   *src = gs_buffer->values[j];
			bits8	   *d_nullmap = NULL;
			bits8	   *s_nullmap = NULL;
			int			unitsz = TYPEALIGN(cmeta->attalign, cmeta->attlen);

			/* fixed-length attribute with row-visibility map */
			cmeta->va_offset = __kds_packed(pos - (char *)kds);
			nbytes = MAXALIGN(TYPEALIGN(cmeta->attalign,
										cmeta->attlen) * nrooms);
			if (gs_buffer->hasnull[j])
			{
				d_nullmap = (bits8 *)(pos + nbytes);
				s_nullmap = gs_buffer->nullmap[j];
			}

			for (i=0, k=0; i < gs_buffer->nitems; i++)
			{
				/* only visible rows */
				if (att_isnull(i, rowmap))
					continue;

				if (s_nullmap && att_isnull(i, s_nullmap))
				{
					Assert(d_nullmap != NULL);
					d_nullmap[k>>3] &= ~(1 << (k & (BITS_PER_BYTE - 1)));
					meet_null = true;
				}
				else
				{
					if (d_nullmap)
						d_nullmap[k>>3] |=  (1 << (k & (BITS_PER_BYTE - 1)));
					memcpy(pos + unitsz * k, src + unitsz * i, unitsz);
				}
				k++;
			}
			Assert(k == nrooms);
			pos += MAXALIGN(unitsz * nrooms);
			if (meet_null)
				pos += MAXALIGN(BITMAPLEN(nrooms));
			nbytes = pos - base;
			cmeta->va_length = __kds_packed(nbytes);
		}
	}
	kds->nitems = nrooms;
	kds->usage = __kds_packed((char *)pos - (char *)kds);
	kds->length = (char *)pos - (char *)kds;
}

/*
 * GpuStoreBufferMakeReadOnly
 */
static void
GpuStoreBufferMakeReadOnly(GpuStoreBuffer *gs_buffer)
{
	/* release read-write buffer */
	MemoryContextReset(gs_buffer->memcxt);
	gs_buffer->nitems   = 0;
	gs_buffer->nrooms   = 0;
	gs_buffer->hasnull  = NULL;
	gs_buffer->nullmap  = NULL;
	gs_buffer->values   = NULL;
	gs_buffer->vl_dict  = NULL;
	gs_buffer->vl_compress = NULL;
	gs_buffer->extra_sz = NULL;
	gs_buffer->gs_mvcc  = NULL;

	/* then, mark the buffer read-only with no dirty */
	gs_buffer->read_only = true;
	gs_buffer->is_dirty = false;
	/* sanity checks */
	Assert(gs_buffer->h_seg != NULL &&
		   gs_buffer->rawsize <= dsm_segment_map_length(gs_buffer->h_seg) &&
		   gs_buffer->h.buffer == dsm_segment_address(gs_buffer->h_seg));
}

/*
 * GpuStoreBufferAllocRW
 */
static void
GpuStoreBufferAllocRW(GpuStoreBuffer *gs_buffer,
					  TupleDesc tupdesc, size_t nrooms)
{
	cl_int		j, nattrs = tupdesc->natts;
	MemoryContext oldcxt;

	oldcxt = MemoryContextSwitchTo(gs_buffer->memcxt);
	gs_buffer->nattrs = nattrs;
	gs_buffer->nitems = 0;
	gs_buffer->nrooms = nrooms;
	gs_buffer->vl_compress = palloc0(sizeof(int) * nattrs);
	gs_buffer->vl_dict = palloc0(sizeof(HTAB *) * nattrs);
	gs_buffer->extra_sz = palloc0(sizeof(size_t) * nattrs);
	gs_buffer->hasnull = palloc0(sizeof(bool) * nattrs);
	gs_buffer->nullmap = palloc0(sizeof(bits8 *) * nattrs);
	gs_buffer->values = palloc0(sizeof(void *) * nattrs);

	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		if (attr->attisdropped)
			continue;
		if (attr->attlen < 0)
		{
			HASHCTL		hctl;
			cl_int		vl_compress;

			gstore_fdw_column_options(attr->attrelid, attr->attnum,
									  &vl_compress);

			memset(&hctl, 0, sizeof(HASHCTL));
			hctl.hash = vl_dict_hash_value;
			hctl.match = vl_dict_compare;
			hctl.keysize = sizeof(vl_dict_key);
			hctl.hcxt = gs_buffer->memcxt;

			gs_buffer->vl_dict[j] = hash_create("varlena dictionary",
												Max(nrooms / 5, 4096),
												&hctl,
												HASH_FUNCTION |
												HASH_COMPARE |
												HASH_CONTEXT);
			gs_buffer->values[j] = palloc_huge(sizeof(vl_dict_key *) * nrooms);
			gs_buffer->vl_compress[j] = vl_compress;
		}
		else
		{
			int		unitsz = att_align_nominal(attr->attlen,
											   attr->attalign);
			gs_buffer->nullmap[j] = palloc_huge(BITMAPLEN(nrooms));
			gs_buffer->values[j] = palloc_huge(unitsz * nrooms);
		}
	}
	gs_buffer->gs_mvcc = palloc_huge(sizeof(MVCCAttrs) * nrooms);
	MemoryContextSwitchTo(oldcxt);
}

/*
 * GpuStoreBufferMakeWritable
 */
static void
GpuStoreBufferMakeWritable(GpuStoreBuffer *gs_buffer, TupleDesc tupdesc)
{
	size_t			nrooms;
	size_t			nitems;
	size_t			i;

	/* already done? */
	if (!gs_buffer->read_only)
		return;
	/* calculation of nrooms */
	if (!gs_buffer->h_seg)
	{
		Assert(!gs_buffer->h.buffer);
		nitems = 0;
		nrooms = 10000;
	}
	else if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
	{
		nitems = gs_buffer->h.kds->nitems;
		nrooms = gs_buffer->h.kds->nitems + 10000;
	}
	else
		elog(ERROR, "gstore_fdw: Bug? unknown buffer format: %d",
			 gs_buffer->format);
	/* allocation of read-write buffer */
	GpuStoreBufferAllocRW(gs_buffer, tupdesc, nrooms);

	/* extract the read-only buffer if any */
	gs_buffer->nitems = nitems;
	if (nitems > 0)
	{
		MVCCAttrs		all_visible;

		if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
		{
			GpuStoreBufferCopyFromKDS(gs_buffer, tupdesc,
									  gs_buffer->h.kds);
		}
		else
			elog(ERROR, "gstore_fdw: Bug? unknown buffer format: %d",
				 gs_buffer->format);
		/* read-only tuples are all visible at first */
		memset(&all_visible, 0, sizeof(MVCCAttrs));
		all_visible.xmin = FrozenTransactionId;
		all_visible.xmax = InvalidTransactionId;
		all_visible.cid  = 0;
		all_visible.xmin_committed = true;
		all_visible.xmax_committed = false;

		/* initial tuples are all visible */
		for (i=0; i < nitems; i++)
			gs_buffer->gs_mvcc[i] = all_visible;
	}

	if (gs_buffer->h_seg)
	{
		Assert(gs_buffer->rawsize > 0 && gs_buffer->h.buffer != NULL);
		dsm_detach(gs_buffer->h_seg);

		gs_buffer->rawsize = 0;
		gs_buffer->h_seg = NULL;
		gs_buffer->h.buffer = NULL;
		memset(&gs_buffer->ipc_mhandle, 0, sizeof(CUipcMemHandle));
	}
	gs_buffer->read_only = false;
}

/*
 * GpuStoreBufferCreate - make a local buffer of GpuStoreBuffer
 */
GpuStoreBuffer *
GpuStoreBufferCreate(Relation frel, Snapshot snapshot)
{
	GpuStoreBuffer *gs_buffer = NULL;
	GpuStoreChunk  *gs_chunk = NULL;
	MemoryContext	memcxt = NULL;
	bool			found;

	if (!gstore_buffer_htab)
	{
		HASHCTL	hctl;
		int		flags;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(GpuStoreBuffer);
		hctl.hcxt = CacheMemoryContext;
		flags = HASH_ELEM | HASH_BLOBS | HASH_CONTEXT;

		gstore_buffer_htab = hash_create("GpuStoreBuffer HTAB",
										 100, &hctl, flags);
	}
	gs_buffer = hash_search(gstore_buffer_htab,
							&RelationGetRelid(frel),
							HASH_ENTER,
							&found);
	if (found)
	{
		Assert(gs_buffer->table_oid == RelationGetRelid(frel));
		gs_chunk = gstore_buf_lookup_chunk(RelationGetRelid(frel), snapshot);
		if (!gs_chunk)
		{
			if (gs_buffer->revision == 0)
				return gs_buffer;	/* no gs_chunk right now */
		}
		else if (gs_buffer->revision == gs_chunk->revision)
			return gs_buffer;		/* ok local buffer is up to date */
		/*
		 * Oops, local buffer is not up-to-date, older than in-GPU image.
		 * So, GpuStoreBuffer must be reconstructed based on the latest
		 * image.
		 */
		MemoryContextDelete(gs_buffer->memcxt);
		memset(gs_buffer, 0, sizeof(GpuStoreBuffer));
		gs_buffer->table_oid = RelationGetRelid(frel);
	}
	else
	{
		gs_chunk = gstore_buf_lookup_chunk(RelationGetRelid(frel), snapshot);
	}

	/*
	 * Local buffer is not found, or invalid. So, re-initialize it again.
	 */
	PG_TRY();
	{
		memcxt = AllocSetContextCreate(CacheMemoryContext,
									   "GpuStoreBuffer",
									   ALLOCSET_DEFAULT_SIZES);
		if (!gs_chunk)
		{
			cl_int		pinning;
			cl_int		format;

			gstore_fdw_table_options(RelationGetRelid(frel),
									 &pinning, &format);
			gs_buffer->pinning   = pinning;
			gs_buffer->format    = format;
			gs_buffer->revision  = 0;
			gs_buffer->read_only = true;
			gs_buffer->is_dirty  = false;
			gs_buffer->memcxt    = memcxt;
			gs_buffer->rawsize   = 0;
			gs_buffer->h_seg     = NULL;
			gs_buffer->h.buffer  = NULL;
			memset(&gs_buffer->ipc_mhandle, 0, sizeof(CUipcMemHandle));
			GpuStoreBufferMakeWritable(gs_buffer, RelationGetDescr(frel));
		}
		else
		{
			Assert(gs_buffer->table_oid == RelationGetRelid(frel));
			gs_buffer->pinning   = gs_chunk->pinning;
			gs_buffer->format    = gs_chunk->format;
			gs_buffer->revision  = gs_chunk->revision;
			gs_buffer->read_only = true;
			gs_buffer->is_dirty  = false;
			gs_buffer->memcxt    = memcxt;
			gs_buffer->rawsize   = gs_chunk->rawsize;
			gs_buffer->h_seg     = dsm_attach(gs_chunk->dsm_mhandle);
			gs_buffer->h.buffer  = dsm_segment_address(gs_buffer->h_seg);
			gs_buffer->ipc_mhandle = gs_chunk->ipc_mhandle;
			/* DSM mapping will alive more than transaction duration */
			dsm_pin_mapping(gs_buffer->h_seg);
		}
	}
	PG_CATCH();
	{
		if (gs_buffer)
		{
			hash_search(gstore_buffer_htab,
						&RelationGetRelid(frel),
						HASH_REMOVE,
						&found);
			Assert(found);
		}
		if (memcxt)
			MemoryContextDelete(memcxt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return gs_buffer;
}

/*
 * GpuStoreBufferOpenDevPtr
 */
CUdeviceptr
GpuStoreBufferOpenDevPtr(GpuContext *gcontext,
						 GpuStoreBuffer *gs_buffer)
{
	CUdeviceptr	m_devptr;
	CUresult	rc;

	if (!gs_buffer->read_only)
		elog(ERROR, "Gstore_Fdw has uncommitted changes");
	Assert(gs_buffer->h_seg != NULL &&
		   gs_buffer->h.buffer == dsm_segment_address(gs_buffer->h_seg));
	rc = gpuIpcOpenMemHandle(gcontext,
							 &m_devptr,
							 gs_buffer->ipc_mhandle,
							 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuIpcOpenMemHandle: %s",
			 errorText(rc));
	return m_devptr;
}

/*
 * GpuStoreBufferExpand
 */
static void
GpuStoreBufferExpand(GpuStoreBuffer *gs_buffer, TupleDesc tupdesc)
{
	size_t		j, nrooms = 2 * gs_buffer->nrooms + 20000;
	MemoryContext oldcxt;

	oldcxt = MemoryContextSwitchTo(gs_buffer->memcxt);
	Assert(tupdesc->natts == gs_buffer->nattrs);
	for (j=0; j < gs_buffer->nattrs; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		if (attr->attisdropped)
			continue;
		if (attr->attlen < 0)
		{
			Assert(!gs_buffer->nullmap[j]);
			gs_buffer->values[j] =
				repalloc_huge(gs_buffer->values[j],
							  sizeof(vl_dict_key *) * nrooms);
			Assert(gs_buffer->vl_dict[j] != NULL);
		}
		else if (gs_buffer->values[j] != NULL)
		{
			int		unitsz = att_align_nominal(attr->attlen,
											   attr->attalign);
			gs_buffer->nullmap[j] = repalloc_huge(gs_buffer->nullmap[j],
												  BITMAPLEN(nrooms));
			gs_buffer->values[j] = repalloc_huge(gs_buffer->values[j],
												 unitsz * nrooms);
			Assert(gs_buffer->vl_dict[j] == NULL);
		}
	}
	gs_buffer->gs_mvcc = repalloc_huge(gs_buffer->gs_mvcc,
									   sizeof(MVCCAttrs) * nrooms);
	gs_buffer->nrooms = nrooms;
	MemoryContextSwitchTo(oldcxt);
}

/*
 * GpuStoreBufferGetTuple
 */
int
GpuStoreBufferGetTuple(Relation frel,
					   Snapshot snapshot,
					   TupleTableSlot *slot,
					   GpuStoreBuffer *gs_buffer,
					   size_t row_index,
					   bool needs_system_columns)
{
	TupleDesc	tupdesc = RelationGetDescr(frel);

	ExecClearTuple(slot);
	if (gs_buffer->read_only)
	{
		/* read from the read-only buffer */
		if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
		{
			if (!KDS_fetch_tuple_column(slot,
										gs_buffer->h.kds,
										row_index))
				return -1;
		}
		else
		{
			elog(ERROR, "Gstore_Fdw: unexpected format: %d",
				 gs_buffer->format);
		}
	}
	else if (row_index < gs_buffer->nitems)
	{
		cl_int			j;

		if (!gstore_buf_tuple_visibility(&gs_buffer->gs_mvcc[row_index],
										 snapshot))
			return 1;		/* try next */

		/* OK, tuple is visible */
		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
			int			unitsz;
			void	   *addr;

			if (att_isnull(j, gs_buffer->nullmap[j]))
			{
				slot->tts_isnull[j] = true;
				continue;
			}
			slot->tts_isnull[j] = false;
			if (attr->attlen < 0)
			{
				vl_dict_key	*vkey
					= ((vl_dict_key **)gs_buffer->values[j])[row_index];
				slot->tts_values[j] = PointerGetDatum(vkey->vl_datum);
			}
			else
			{
				unitsz = att_align_nominal(attr->attlen,
										   attr->attalign);
				addr = (char *)gs_buffer->values[j] + unitsz * row_index;
				if (!attr->attbyval)
					slot->tts_values[j] = PointerGetDatum(addr);
				else if (attr->attlen == sizeof(cl_char))
					slot->tts_values[j] = CharGetDatum(*((cl_char *)addr));
				else if (attr->attlen == sizeof(cl_short))
					slot->tts_values[j] = Int16GetDatum(*((cl_short *)addr));
				else if (attr->attlen == sizeof(cl_int))
					slot->tts_values[j] = Int32GetDatum(*((cl_int *)addr));
				else if (attr->attlen == sizeof(cl_long))
					slot->tts_values[j] = Int64GetDatum(*((cl_long *)addr));
				else
					elog(ERROR, "gstore_buf: unexpected attlen: %d",
						 attr->attlen);
			}
		}
		ExecStoreVirtualTuple(slot);
	}
	else
		return -1;

	/* put system column information if needed */
	if (needs_system_columns)
	{
		HeapTuple   tup = ExecMaterializeSlot(slot);

		tup->t_self.ip_blkid.bi_hi = (row_index >> 32) & 0x0000ffff;
		tup->t_self.ip_blkid.bi_lo = (row_index >> 16) & 0x0000ffff;
		tup->t_self.ip_posid       = (row_index & 0x0000ffff);
		tup->t_tableOid = RelationGetRelid(frel);
		if (gs_buffer->read_only)
		{
			tup->t_data->t_choice.t_heap.t_xmin = FrozenTransactionId;
			tup->t_data->t_choice.t_heap.t_xmax = InvalidTransactionId;
			tup->t_data->t_choice.t_heap.t_field3.t_cid = 0;
		}
		else
		{
			MVCCAttrs  *mvcc = &gs_buffer->gs_mvcc[row_index];
			tup->t_data->t_choice.t_heap.t_xmin = mvcc->xmin;
			tup->t_data->t_choice.t_heap.t_xmax = mvcc->xmax;
			tup->t_data->t_choice.t_heap.t_field3.t_cid = mvcc->cid;
		}
	}
	return 0;
}

/*
 * GpuStoreBufferAppendRow
 */
void
GpuStoreBufferAppendRow(GpuStoreBuffer *gs_buffer,
						TupleDesc tupdesc,
						Snapshot snapshot,
						TupleTableSlot *slot)
{
	MemoryContext	oldcxt;
	size_t			index = gs_buffer->nitems++;
	cl_uint			j;
	MVCCAttrs	   *mvcc;

	/* ensure the buffer is read-writable */
	if (gs_buffer->read_only)
		GpuStoreBufferMakeWritable(gs_buffer, tupdesc);
	/* expand the buffer on demand */
	while (index >= gs_buffer->nrooms)
		GpuStoreBufferExpand(gs_buffer, tupdesc);

	/* write out the new tuple */
	slot_getallattrs(slot);
	oldcxt = MemoryContextSwitchTo(gs_buffer->memcxt);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
		bool	isnull = slot->tts_isnull[j];
		Datum	datum  = slot->tts_values[j];

		if (attr->attisdropped)
			continue;

		if (attr->attlen < 0)
		{
			vl_dict_key *entry = NULL;

			if (!isnull)
			{
				struct varlena *vl;
				vl_dict_key key;
				bool		found;
				size_t		usage;

				vl = vl_datum_compression(DatumGetPointer(datum),
										  gs_buffer->vl_compress[j]);
				key.offset = 0;
				key.vl_datum = vl;
				entry = hash_search(gs_buffer->vl_dict[j],
									&key,
									HASH_ENTER,
									&found);
				if (!found)
				{
					entry->offset = 0;
					if (PointerGetDatum(vl) == datum)
					{
						size_t		len = VARSIZE_ANY(datum);

						vl = (struct varlena *) palloc(len);
						memcpy(vl, DatumGetPointer(datum), len);
					}
					entry->vl_datum = vl;
					gs_buffer->extra_sz[j] += MAXALIGN(VARSIZE_ANY(vl));
					Assert(gs_buffer->memcxt == GetMemoryChunkContext(vl));

					usage = (MAXALIGN(sizeof(cl_uint) * index) +
							 gs_buffer->extra_sz[j]);
					if (usage >= KDS_OFFSET_MAX_SIZE)
						elog(ERROR, "attribute \"%s\" consumed too much",
							 NameStr(attr->attname));
				}
			}
			((vl_dict_key **)gs_buffer->values[j])[index] = entry;
		}
		else
		{
			bits8  *nullmap = gs_buffer->nullmap[j];
			char   *base = gs_buffer->values[j];

			if (isnull)
			{
				gs_buffer->hasnull[j] = true;
				nullmap[index >> 3] &= ~(1 << (index & 7));
			}
			else if (!attr->attbyval)
			{
				nullmap[index >> 3] |= (1 << (index & 7));
				base += att_align_nominal(attr->attlen,
										  attr->attalign) * index;
				memcpy(base, DatumGetPointer(datum), attr->attlen);
			}
			else
			{
				nullmap[index >> 3] |= (1 << (index & 7));
				base += att_align_nominal(attr->attlen,
										  attr->attalign) * index;
				memcpy(base, &datum, attr->attlen);
			}
		}
	}
	mvcc = &gs_buffer->gs_mvcc[index];
	memset(mvcc, 0, sizeof(MVCCAttrs));
	mvcc->xmin = GetCurrentTransactionId();
    mvcc->xmax = InvalidTransactionId;
    mvcc->cid  = snapshot->curcid;
	/*
	 * mark the buffer is dirty, and read-only buffer is not valid any more.
	 */
	gs_buffer->is_dirty = true;
	if (gs_buffer->h.buffer)
	{
		pfree(gs_buffer->h.buffer);
		gs_buffer->h.buffer = NULL;
	}
	MemoryContextSwitchTo(oldcxt);
}

void
GpuStoreBufferRemoveRow(GpuStoreBuffer *gs_buffer,
						TupleDesc tupdesc,
						Snapshot snapshot,
						size_t old_index)
{
	MVCCAttrs  *mvcc;

	if (gs_buffer->read_only)
		GpuStoreBufferMakeWritable(gs_buffer, tupdesc);
	/* remove the old version */
	if (old_index >= gs_buffer->nitems)
		elog(ERROR, "gstore_buf: UPDATE row out of range (%lu of %zu)",
			 old_index, gs_buffer->nitems);
	mvcc = &gs_buffer->gs_mvcc[old_index];
	mvcc->xmax = GetCurrentTransactionId();
	mvcc->cid  = snapshot->curcid;

	/*
	 * mark the buffer is dirty, and read-only buffer is not valid any more.
	 */
	gs_buffer->is_dirty = true;
	if (gs_buffer->h.buffer)
	{
		pfree(gs_buffer->h.buffer);
		gs_buffer->h.buffer = NULL;
	}
}

/*
 * GpuStoreBufferEstimateSize
 *
 * XXX needs rename?
 */
static size_t
GpuStoreBufferEstimateSize(Relation frel,
						   GpuStoreBuffer *gs_buffer, size_t nrooms)
{
	TupleDesc	tupdesc = RelationGetDescr(frel);
	size_t		rawsize;
	int			j;

	Assert(gs_buffer->table_oid == RelationGetRelid(frel) &&
		   gs_buffer->nattrs == tupdesc->natts);
	if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
	{
		rawsize = KDS_CALCULATE_HEAD_LENGTH(tupdesc->natts, true);
		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

			if (attr->attisdropped)
				continue;
			if (attr->attlen < 0)
			{
				rawsize += (MAXALIGN(sizeof(cl_uint) * nrooms) +
							MAXALIGN(gs_buffer->extra_sz[j]));
			}
			else
			{
				size_t		unitsz = att_align_nominal(attr->attlen,
													   attr->attalign);
				rawsize += MAXALIGN(unitsz * nrooms);
				if (gs_buffer->hasnull[j])
					rawsize += MAXALIGN(BITMAPLEN(nrooms));
			}
		}
	}
	else
	{
		elog(ERROR, "Gstore_Fdw: unknown format %d", gs_buffer->format);
	}
	return rawsize;
}

/*
 * GpuStoreBufferGetSize
 */
void
GpuStoreBufferGetSize(Oid ftable_oid, Snapshot snapshot,
					  Size *p_rawsize,
					  Size *p_nitems)
{
	GpuStoreBuffer *gs_buffer;
	GpuStoreChunk  *gs_chunk;
	Size			rawsize = 0;
	Size			nitems = 0;

	if (gstore_buffer_htab)
	{
		gs_buffer = hash_search(gstore_buffer_htab,
								&ftable_oid,
								HASH_FIND,
								NULL);
		if (gs_buffer)
		{
			if (gs_buffer->read_only)
			{
				switch (gs_buffer->format)
				{
					case GSTORE_FDW_FORMAT__PGSTROM:
						rawsize = gs_buffer->h.kds->length;
						nitems  = gs_buffer->h.kds->nitems;
						break;
					default:
						elog(ERROR, "Unknown Gstore_Fdw format: %d",
							 gs_buffer->format);
						break;
				}
			}
			else
			{
				HeapTuple	tup;
				Form_pg_attribute attr;
				cl_int		j;

				nitems = gs_buffer->nitems;
				for (j=0; j < gs_buffer->nattrs; j++)
				{
					tup = SearchSysCache2(ATTNUM,
										  ObjectIdGetDatum(ftable_oid),
										  Int16GetDatum(j+1));
					if (!HeapTupleIsValid(tup))
						elog(ERROR, "cache lookup failed for attribute %d of relation %u", j+1, ftable_oid);
					attr = (Form_pg_attribute) GETSTRUCT(tup);
					if (attr->attisdropped)
					{
						/* nothing to count */
					}
					else if (attr->attlen > 0)
					{
						int		unitsz = att_align_nominal(attr->attlen,
														   attr->attalign);
						rawsize += MAXALIGN(unitsz * nitems);
						if (gs_buffer->hasnull[j])
							rawsize += MAXALIGN(BITMAPLEN(nitems));
					}
					else if (attr->attlen == -1)
					{
						rawsize += (MAXALIGN(sizeof(cl_uint) * nitems)
									+ gs_buffer->extra_sz[j]);
					}
					else
					{
						elog(ERROR, "unexpected type length (=%d) of %s",
							 attr->attlen, format_type_be(attr->atttypid));
					}
					ReleaseSysCache(tup);
				}
			}
			goto out;
		}
	}

	gs_chunk = gstore_buf_lookup_chunk(ftable_oid, snapshot);
	if (gs_chunk)
	{
		rawsize = gs_chunk->rawsize;
		nitems = gs_chunk->nitems;
	}
out:
	if (p_rawsize)
		*p_rawsize = rawsize;
	if (p_nitems)
		*p_nitems  = nitems;
}

/*
 * GpuStoreBufferGetNitems
 */
size_t
GpuStoreBufferGetNitems(GpuStoreBuffer *gs_buffer)
{
	size_t		nitems;

	if (gs_buffer->read_only)
	{
		Assert(gs_buffer->h_seg &&
			   gs_buffer->h.buffer == dsm_segment_address(gs_buffer->h_seg));
		if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
			nitems = gs_buffer->h.kds->nitems;
		else
			elog(ERROR, "Gstore_Fdw has unknown format: %d",
				 gs_buffer->format);
	}
	else
	{
		nitems = gs_buffer->nitems;
	}
	return nitems;
}





/*
 * gstoreXactCallbackOnPreCommit
 */
static void
gstoreXactCallbackOnPreCommit(void)
{
	HASH_SEQ_STATUS	status;
	GpuStoreBuffer *gs_buffer;

	if (!gstore_buffer_htab)
		return;

	hash_seq_init(&status, gstore_buffer_htab);
	while ((gs_buffer = hash_seq_search(&status)) != NULL)
	{
		Relation		frel;
		size_t			rawsize;
		bits8		   *rowmap;
		size_t			nrooms = gs_buffer->nitems;
		CUresult		rc;
		CUipcMemHandle	ipc_mhandle;
		dsm_handle		dsm_mhandle;

		/* any writes happen? */
		if (!gs_buffer->is_dirty)
			continue;
		/* check visibility for each rows (if any) */
		rowmap = gstore_buf_visibility_bitmap(gs_buffer, &nrooms);

		/*
		 * once all the rows are removed from the gstore_fdw, we don't
		 * add new version of GpuStoreChunk/GpuStoreBuffer.
		 * Older version will be removed when it becomes invisible from
		 * all the transactions.
		 */
		if (nrooms == 0)
		{
			Oid			gstore_oid = gs_buffer->table_oid;
			pg_crc32	hash = gstore_buf_chunk_hashvalue(gstore_oid);
			int			index = hash % GSTORE_CHUNK_HASH_NSLOTS;
			dlist_iter	iter;
			bool		found;

			SpinLockAcquire(&gstore_head->lock);
			dlist_foreach(iter, &gstore_head->active_chunks[index])
			{
				GpuStoreChunk  *gs_temp = dlist_container(GpuStoreChunk,
														  chain, iter.cur);
				if (gs_temp->hash == hash &&
					gs_temp->database_oid == MyDatabaseId &&
					gs_temp->table_oid == gstore_oid &&
					gs_temp->xmax == InvalidTransactionId)
				{
					gs_temp->xmax = GetCurrentTransactionId();
				}
			}
			pg_atomic_add_fetch_u32(&gstore_head->has_warm_chunks, 1);
			SpinLockRelease(&gstore_head->lock);
			/* also remove the buffer */
			MemoryContextDelete(gs_buffer->memcxt);
			hash_search(gstore_buffer_htab,
						&gstore_oid,
						HASH_REMOVE,
						&found);
			Assert(found);
			continue;
		}

		/*
		 * construction of new version of GPU device memory image
		 */
		frel = heap_open(gs_buffer->table_oid, NoLock);
		rawsize = GpuStoreBufferEstimateSize(frel, gs_buffer, nrooms);
		rc = gpuMemAllocPreserved(gs_buffer->pinning,
								  &ipc_mhandle,
								  &dsm_mhandle,
								  rawsize);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAllocPreserved: %s", errorText(rc));
		PG_TRY();
		{
			dsm_segment	   *h_seg = dsm_attach(dsm_mhandle);

			if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
			{
				kern_data_store	   *kds = dsm_segment_address(h_seg);

				GpuStoreBufferCopyToKDS(kds, gs_buffer,
										RelationGetDescr(frel),
										rowmap, nrooms);
				Assert(kds->length == rawsize);
			}
			else
				elog(ERROR, "Gstore_Fdw: unknown format %d",
					 gs_buffer->format);
			/* load the read-only buffer to GPU device */
			rc = gpuMemLoadPreserved(gs_buffer->pinning, ipc_mhandle);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on gpuMemLoadPreserved: %s",
					 errorText(rc));
			/* register the new version of chunk */
			gs_buffer->rawsize  = rawsize;
			gs_buffer->h_seg    = h_seg;
			gs_buffer->h.buffer = dsm_segment_address(h_seg);
			gs_buffer->ipc_mhandle = ipc_mhandle;
			/* mark the buffer read-only again */
			GpuStoreBufferMakeReadOnly(gs_buffer);
			/* keep DSM mapping */
			dsm_pin_mapping(h_seg);
			gstore_buf_insert_chunk(gs_buffer, nrooms, ipc_mhandle,
									dsm_segment_handle(h_seg));
		}
		PG_CATCH();
		{
			gpuMemFreePreserved(gs_buffer->pinning, ipc_mhandle);
			gs_buffer->rawsize  = 0;
			gs_buffer->h_seg    = NULL;
			gs_buffer->h.buffer = NULL;
			PG_RE_THROW();
		}
		PG_END_TRY();

		heap_close(frel, NoLock);
	}
}

/*
 * gstoreXactCallbackOnAbort - clear all the local buffers
 */
static void
gstoreXactCallbackOnAbort(void)
{
	HASH_SEQ_STATUS	status;
	GpuStoreBuffer *gs_buffer;

	if (gstore_buffer_htab)
	{
		hash_seq_init(&status, gstore_buffer_htab);
		while ((gs_buffer = hash_seq_search(&status)) != NULL)
		{
			MemoryContextDelete(gs_buffer->memcxt);
			if (gs_buffer->h_seg)
				dsm_detach(gs_buffer->h_seg);
		}
		hash_destroy(gstore_buffer_htab);
		gstore_buffer_htab = NULL;
	}
}

/*
 * gstoreXactCallbackPerChunk
 */
static bool
gstoreOnXactCallbackPerChunk(bool is_commit,
							 GpuStoreChunk *gs_chunk,
							 TransactionId oldestXmin)
{
	if (TransactionIdIsCurrentTransactionId(gs_chunk->xmax))
	{
		if (is_commit)
			gs_chunk->xmax_committed = true;
		else
			gs_chunk->xmax = InvalidTransactionId;
	}
	if (TransactionIdIsCurrentTransactionId(gs_chunk->xmin))
	{
		if (is_commit)
			gs_chunk->xmin_committed = true;
		else
		{
			gstore_buf_release_chunk(gs_chunk);
			return false;
		}
	}

	if (TransactionIdIsValid(gs_chunk->xmax))
	{
		/* someone tried to delete chunk, but not commited yet */
		if (!gs_chunk->xmax_committed)
			return true;
		/*
		 * chunk deletion is commited, but some open transactions may
		 * still reference the chunk
		 */
		if (!TransactionIdPrecedes(gs_chunk->xmax, oldestXmin))
			return true;

		/* Otherwise, GpuStoreChunk can be released immediately */
		gstore_buf_release_chunk(gs_chunk);
	}
	else if (TransactionIdIsNormal(gs_chunk->xmin))
	{
		/* someone tried to insert chunk, but not commited yet */
		if (!gs_chunk->xmin_committed)
			return true;
		/*
		 * chunk insertion is commited, but some open transaction may
		 * need MVCC style visibility control
		 */
		if (!TransactionIdPrecedes(gs_chunk->xmin, oldestXmin))
			return true;

		/* Otherwise, GpuStoreChunk can be visible to everybody */
		gs_chunk->xmin = FrozenTransactionId;
	}
	else if (!TransactionIdIsValid(gs_chunk->xmin))
	{
		/* GpuChunk insertion aborted */
		gstore_buf_release_chunk(gs_chunk);
	}
	return false;
}

/*
 * gstoreXactCallback
 */
static void
gstoreXactCallback(XactEvent event, void *arg)
{
	TransactionId oldestXmin;
	bool		is_commit;
	bool		meet_warm_chunks = false;
	cl_int		i;

	switch (event)
	{
		case XACT_EVENT_PRE_COMMIT:
			gstoreXactCallbackOnPreCommit();
			return;
		case XACT_EVENT_COMMIT:
			is_commit = true;
			break;
		case XACT_EVENT_ABORT:
			gstoreXactCallbackOnAbort();
			is_commit = false;
			break;
		default:
			/* do nothing */
			return;
	}
#if 0
	elog(INFO, "gstoreXactCallback xid=%u (oldestXmin=%u)",
		 GetCurrentTransactionIdIfAny(), oldestXmin);
#endif
	if (pg_atomic_read_u32(&gstore_head->has_warm_chunks) == 0)
		return;

	oldestXmin = GetOldestXmin(NULL, true);
	SpinLockAcquire(&gstore_head->lock);
	for (i=0; i < GSTORE_CHUNK_HASH_NSLOTS; i++)
	{
		dlist_mutable_iter	iter;

		dlist_foreach_modify(iter, &gstore_head->active_chunks[i])
		{
			GpuStoreChunk  *gs_chunk
				= dlist_container(GpuStoreChunk, chain, iter.cur);

			if (gstoreOnXactCallbackPerChunk(is_commit, gs_chunk, oldestXmin))
				meet_warm_chunks = true;
		}
	}
	if (!meet_warm_chunks)
		pg_atomic_write_u32(&gstore_head->has_warm_chunks, 0);
	SpinLockRelease(&gstore_head->lock);
}

#if 0
/*
 * gstoreSubXactCallback - just for debug
 */
static void
gstoreSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
					  SubTransactionId parentSubid, void *arg)
{
	elog(INFO, "gstoreSubXactCallback event=%s my_xid=%u pr_xid=%u",
		 (event == SUBXACT_EVENT_START_SUB ? "StartSub" :
		  event == SUBXACT_EVENT_COMMIT_SUB ? "CommitSub" :
		  event == SUBXACT_EVENT_ABORT_SUB ? "AbortSub" :
		  event == SUBXACT_EVENT_PRE_COMMIT_SUB ? "PreCommitSub" : "???"),
		 mySubid, parentSubid);
}
#endif

/*
 * gstore_fdw_post_alter
 *
 * It prevents ALTER FOREIGN TABLE if Gstore_Fdw is not empty.
 */
static void
gstore_fdw_post_alter(Oid relid, AttrNumber attnum)
{
	GpuStoreBuffer *gs_buffer;
	GpuStoreChunk  *gs_chunk;
	bool			found;

	/* not a gstore_fdw foreign-table */
	if (!relation_is_gstore_fdw(relid))
		return;

	/* we don't allow ALTER FOREIGN TABLE onto non-empty gstore_fdw */
	if (gstore_buffer_htab)
	{
		gs_buffer = hash_search(gstore_buffer_htab,
								&relid,
								HASH_FIND,
								&found);
		if (gs_buffer)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("gstore_fdw: unable to run ALTER FOREIGN TABLE for non-empty gstore_fdw table")));
	}

	gs_chunk = gstore_buf_lookup_chunk(relid, GetActiveSnapshot());
	if (gs_chunk)
	{
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("gstore_fdw: unable to run ALTER FOREIGN TABLE for non-empty gstore_fdw table")));
	}
}

/*
 * gstore_fdw_post_drop
 *
 * It marks Gstore_Fdw is removed and invisible to the later transaction
 */
static void
gstore_fdw_post_drop(Oid relid, AttrNumber attnum)
{
	GpuStoreChunk *gs_chunk;
	pg_crc32	hash = gstore_buf_chunk_hashvalue(relid);
	int			index = hash % GSTORE_CHUNK_HASH_NSLOTS;
	dlist_iter	iter;

	SpinLockAcquire(&gstore_head->lock);
	dlist_foreach(iter, &gstore_head->active_chunks[index])
	{
		gs_chunk = dlist_container(GpuStoreChunk, chain, iter.cur);

		if (gs_chunk->hash == hash &&
			gs_chunk->database_oid == MyDatabaseId &&
			gs_chunk->table_oid == relid &&
			gs_chunk->xmax == InvalidTransactionId)
		{
			gs_chunk->xmax = GetCurrentTransactionId();
		}
	}
	pg_atomic_add_fetch_u32(&gstore_head->has_warm_chunks, 1);
	SpinLockRelease(&gstore_head->lock);
}

/*
 * gstore_fdw_object_access
 */
static void
gstore_fdw_object_access(ObjectAccessType access,
						 Oid classId,
						 Oid objectId,
						 int subId,
						 void *__arg)
{
	if (object_access_next)
		(*object_access_next)(access, classId, objectId, subId, __arg);

	switch (access)
	{
		case OAT_POST_CREATE:
			if (classId == RelationRelationId)
			{
				ObjectAccessPostCreate *arg = __arg;

				if (arg->is_internal)
					break;
				/* A new gstore_fdw table is obviously empty */
				if (subId != 0)
					gstore_fdw_post_alter(objectId, subId);
			}
			break;

		case OAT_POST_ALTER:
			if (classId == RelationRelationId)
			{
				ObjectAccessPostAlter  *arg = __arg;

				if (arg->is_internal)
					break;
				gstore_fdw_post_alter(objectId, subId);
			}
			break;

		case OAT_DROP:
			if (classId == RelationRelationId)
			{
				ObjectAccessDrop	   *arg = __arg;

				if ((arg->dropflags & PERFORM_DELETION_INTERNAL) != 0)
					break;

				if (subId == 0)
				{
					if (relation_is_gstore_fdw(objectId))
						gstore_fdw_post_drop(objectId, subId);
				}
				else
					gstore_fdw_post_alter(objectId, subId);
			}
			break;

		default:
			/* do nothing */
			break;
	}
}

/*
 * pgstrom_startup_gstore_buf
 */
static void
pgstrom_startup_gstore_buf(void)
{
	bool		found;
	int			i;

	if (shmem_startup_next)
		(*shmem_startup_next)();
	gstore_head = ShmemInitStruct("GPU Store Control Structure",
								  offsetof(GpuStoreHead,
										   gs_chunks[gstore_max_relations]),
								  &found);
	if (found)
		elog(ERROR, "Bug? shared memory for gstore_fdw already exist");

	pg_atomic_init_u32(&gstore_head->revision_seed, 1);
	SpinLockInit(&gstore_head->lock);
	dlist_init(&gstore_head->free_chunks);
	for (i=0; i < GSTORE_CHUNK_HASH_NSLOTS; i++)
		dlist_init(&gstore_head->active_chunks[i]);
	for (i=0; i < gstore_max_relations; i++)
	{
		GpuStoreChunk  *gs_chunk = &gstore_head->gs_chunks[i];

		memset(gs_chunk, 0, sizeof(GpuStoreChunk));
		dlist_push_tail(&gstore_head->free_chunks, &gs_chunk->chain);
	}
}

/*
 * pgstrom_init_gstore_buf
 */
void
pgstrom_init_gstore_buf(void)
{
	size_t		required;

	DefineCustomIntVariable("pg_strom.gstore_max_relations",
							"maximum number of gstore_fdw relations",
							NULL,
							&gstore_max_relations,
							512,
							1,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	required = offsetof(GpuStoreHead, gs_chunks[gstore_max_relations]);
	RequestAddinShmemSpace(MAXALIGN(required));

	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gstore_buf;

	object_access_next = object_access_hook;
	object_access_hook = gstore_fdw_object_access;

	RegisterXactCallback(gstoreXactCallback, NULL);
	//RegisterSubXactCallback(gstoreSubXactCallback, NULL);
}

/*
 * pgstrom_gstore_fdw_format
 */
Datum
pgstrom_gstore_fdw_format(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	GpuStoreChunk  *gs_chunk;

	if (!relation_is_gstore_fdw(gstore_oid))
		PG_RETURN_NULL();
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	gs_chunk = gstore_buf_lookup_chunk(gstore_oid, GetActiveSnapshot());
	if (!gs_chunk)
		PG_RETURN_NULL();

	/* currently, only 'pgstrom' is the supported format */
	PG_RETURN_TEXT_P(cstring_to_text("pgstrom"));
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_format);

/*
 * pgstrom_gstore_fdw_nitems
 */
Datum
pgstrom_gstore_fdw_nitems(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	GpuStoreChunk  *gs_chunk;
	int64			retval = 0;

	if (!relation_is_gstore_fdw(gstore_oid))
		PG_RETURN_NULL();
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	gs_chunk = gstore_buf_lookup_chunk(gstore_oid, GetActiveSnapshot());
	if (gs_chunk)
		retval = gs_chunk->nitems;

	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_nitems);

/*
 * pgstrom_gstore_fdw_nattrs
 */
Datum
pgstrom_gstore_fdw_nattrs(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	Relation		frel;
	int64			retval = 0;

	if (!relation_is_gstore_fdw(gstore_oid))
		PG_RETURN_NULL();
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	frel = heap_open(gstore_oid, AccessShareLock);
	retval = RelationGetNumberOfAttributes(frel);
	heap_close(frel, NoLock);

	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_nattrs);

/*
 * pgstrom_gstore_fdw_rawsize
 */
Datum
pgstrom_gstore_fdw_rawsize(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	GpuStoreChunk  *gs_chunk;
	int64			retval = 0;

	if (!relation_is_gstore_fdw(gstore_oid))
		PG_RETURN_NULL();
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	gs_chunk = gstore_buf_lookup_chunk(gstore_oid, GetActiveSnapshot());
	if (gs_chunk)
		retval = gs_chunk->rawsize;

	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_rawsize);

/*
 * pgstrom_gstore_fdw_chunk_info
 */
Datum
pgstrom_gstore_fdw_chunk_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	GpuStoreChunk  *gs_chunk;
	GpuStoreChunk  *gs_temp;
	List	   *chunks_list;
	Datum		values[9];
	bool		isnull[9];
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(9, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "database_oid",
						   OIDOID, -1, 0);
        TupleDescInitEntry(tupdesc, (AttrNumber) 2, "table_oid",
						   REGCLASSOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "revision",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "xmin",
						   XIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "xmax",
						   XIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "pinning",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 7, "format",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 8, "rawsize",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 9, "nitems",
						   INT8OID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		chunks_list = NIL;
		SpinLockAcquire(&gstore_head->lock);
		PG_TRY();
		{
			dlist_iter	iter;
			int			i;

			for (i=0; i < GSTORE_CHUNK_HASH_NSLOTS; i++)
			{
				dlist_foreach(iter, &gstore_head->active_chunks[i])
				{
					gs_chunk = dlist_container(GpuStoreChunk, chain, iter.cur);
					if (!superuser())
					{
						if (gs_chunk->database_oid != MyDatabaseId)
							continue;
						if (pg_class_aclcheck(gs_chunk->table_oid,
											  GetUserId(),
											  ACL_SELECT) != ACLCHECK_OK)
							continue;
					}
					gs_temp = palloc(sizeof(GpuStoreChunk));
					memcpy(gs_temp, gs_chunk, sizeof(GpuStoreChunk));

					chunks_list = lappend(chunks_list, gs_temp);
				}
			}
		}
		PG_CATCH();
		{
			SpinLockRelease(&gstore_head->lock);
			PG_RE_THROW();
		}
		PG_END_TRY();
		SpinLockRelease(&gstore_head->lock);

		fncxt->user_fctx = chunks_list;
		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	chunks_list = fncxt->user_fctx;
	if (chunks_list == NIL)
		SRF_RETURN_DONE(fncxt);
	gs_chunk = linitial(chunks_list);
	Assert(gs_chunk != NULL);
	fncxt->user_fctx = list_delete_first(chunks_list);

	memset(isnull, 0, sizeof(isnull));
	values[0] = ObjectIdGetDatum(gs_chunk->database_oid);
	values[1] = ObjectIdGetDatum(gs_chunk->table_oid);
	values[2] = Int32GetDatum(gs_chunk->revision);
	values[3] = TransactionIdGetDatum(gs_chunk->xmin);
	values[4] = TransactionIdGetDatum(gs_chunk->xmax);
	values[5] = Int32GetDatum(gs_chunk->pinning);
	if (gs_chunk->format == GSTORE_FDW_FORMAT__PGSTROM)
		values[6] = CStringGetTextDatum("pgstrom");
	else
		values[6] = CStringGetTextDatum(psprintf("unknown - %u",
												 gs_chunk->format));
	values[7] = Int64GetDatum(gs_chunk->rawsize);
	values[8] = Int64GetDatum(gs_chunk->nitems);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_chunk_info);


/*
 * pgstrom_gstore_export_ipchandle
 */
GstoreIpcHandle *
__pgstrom_gstore_export_ipchandle(Oid ftable_oid)
{
	cl_int			pinning;
	GpuStoreChunk  *gs_chunk;
	GstoreIpcHandle *result;

	if (!relation_is_gstore_fdw(ftable_oid))
		elog(ERROR, "relation %u is not gstore_fdw foreign table",
			 ftable_oid);
	strom_foreign_table_aclcheck(ftable_oid, GetUserId(), ACL_SELECT);

	gstore_fdw_table_options(ftable_oid, &pinning, NULL);
	if (pinning < 0)
		elog(ERROR, "gstore_fdw: \"%s\" is not pinned on GPU devices",
			 get_rel_name(ftable_oid));
	if (pinning >= numDevAttrs)
		elog(ERROR, "gstore_fdw: \"%s\" is not pinned on valid GPU device",
			 get_rel_name(ftable_oid));

	gs_chunk = gstore_buf_lookup_chunk(ftable_oid, GetActiveSnapshot());
	if (!gs_chunk)
		return NULL;

	result = palloc0(sizeof(GstoreIpcHandle));
	result->device_id = devAttrs[pinning].DEV_ID;
	result->format = gs_chunk->format;
	result->rawsize = gs_chunk->rawsize;
	memcpy(&result->ipc_mhandle.d,
		   &gs_chunk->ipc_mhandle,
		   sizeof(CUipcMemHandle));
	SET_VARSIZE(result, sizeof(GstoreIpcHandle));

	return result;
}

Datum
pgstrom_gstore_export_ipchandle(PG_FUNCTION_ARGS)
{
	GstoreIpcHandle *handle;

	handle = __pgstrom_gstore_export_ipchandle(PG_GETARG_OID(0));
	if (!handle)
		PG_RETURN_NULL();
	PG_RETURN_POINTER(handle);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_export_ipchandle);
