/*
 * cuda_gcache.cu
 *
 * GPU device code to manage GPU Cache
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"
#include "cuda_gcache.h"

#define KERN_CONTEXT_VARLENA_BUFSZ		0
#define KERN_CONTEXT_STACK_LIMIT		0

STATIC_FUNCTION(char *)
kern_datum_get_reusable_varlena(kern_data_store *kds,
								kern_data_extra *extra,
								cl_uint colidx,
								cl_uint rowidx,
								char *newvar)
{
	kern_colmeta   *cmeta = &kds->colmeta[colidx];
	GpuCacheSysattr *sysattr;
	cl_uint		   *vindex;
	size_t			offset;
	char		   *addr;
	cl_uint			sz;

	assert(colidx < kds->ncols);
	if (rowidx >= kds->nitems)
		return NULL;
	if (cmeta->attbyval || cmeta->attlen != -1)
		return NULL;
	if (cmeta->nullmap_offset != 0)
	{
		cl_uint	   *nullmap = (cl_uint *)
			((char *)kds + __kds_unpack(cmeta->nullmap_offset));
		if ((nullmap[rowidx>>5] & (1U<<(rowidx & 0x1f))) == 0)
			return NULL;
	}
	/*
	 * In case of concurrent apply of REDO log, we cannot avoid
	 * race condition of the extra buffer. So, as a second best,
	 * we reuse the varlena datum only if it is committed.
	 */
	sysattr = kds_get_column_sysattr(kds, rowidx);
	if (!sysattr || sysattr->xmin != FrozenTransactionId)
		return NULL;
	assert(extra != NULL);

	vindex = (cl_uint *)((char *)kds + __kds_unpack(cmeta->values_offset));
	offset = __kds_unpack(vindex[rowidx]);
	sz = VARSIZE_ANY(newvar);
	if (offset < offsetof(kern_data_extra, data) &&
		offset + sz < extra->length)
	{
		addr = (char *)extra + offset;
		if (VARATT_IS_COMPRESSED(addr) || VARATT_IS_EXTERNAL(addr))
			return NULL;	/* usually, should not happen */

		sz = VARSIZE_ANY_EXHDR(newvar);
		if (VARSIZE_ANY_EXHDR(addr) == sz &&
			__memcmp(VARDATA_ANY(newvar), VARDATA_ANY(addr), sz) == 0)
			return addr;
	}
	return NULL;
}

/*
 * MEMO: The base chunk of GpuCache contains the following three portions
 * 1. KDS (column-format), 2. Hash-slot of RowId, 3. RowId link list.
 *
 * +----------------------------------------------+  ----+
 * | 1. KDS (column)                              |      |
 * | +--------------------------------------------+      |
 * | | KDS header + columns metadata array        |  kds->lentrh
 * | | Last colmeta (kds->colmeta[nr_colmeta-1])  |      |
 * | | points the system attribute array          |      v
 * +-+--------------------------------------------+  ---------
 * | 2. Hash-slot of RowId                        |      ^
 * | kern_gpucache_rowhash allows to lookup       |      |
 * | rowid by CTID in the GPU kernel.             | offsetof(kern_gpucache_rowhash,
 * | rowhash->slots[].rowid points the first item |          slots[nslots])
 * | of the rowid in use. Also, freelist points   |      |
 * | the first item of the rowid                  |      v
 * +----------------------------------------------+  ----------
 * | 3. RowId link list                           |   ^
 * | This rowid array points the next item in     |   | sizeof(cl_uint) * nrooms
 * | either of the hash-slot or freelist.         |   |
 * | Elsewhere, if it the list tail if UINT_MAX   |   v
 * +----------------------------------------------+  ----------
 */
#define DECL_ROWID_HASH_AND_MAP(KDS)								\
	kern_gpucache_rowhash *rowhash =								\
		(kern_gpucache_rowhash *)((char *)(KDS) + (KDS)->length);	\
	cl_uint *rowmap = (cl_uint *)(&rowhash->slots[(KDS)->nslots])

/*
 * kern_gpucache_initialize_empty
 */
KERNEL_FUNCTION(void)
kern_gpucache_init_empty(kern_data_store *kds,
						 kern_data_extra *extra)
{
	DECL_ROWID_HASH_AND_MAP(kds);
	cl_uint		index;
	cl_uint		nrooms = kds->nrooms;
	cl_uint		nslots = kds->nslots;

	if (get_global_id() == 0)
	{
		kds->nitems = 0;

		if (extra)
			extra->usage = offsetof(kern_data_extra, data);

		rowhash->magic  = KERN_GPUCACHE_ROWHASH_MAGIC;
		rowhash->nslots = nslots;
		rowhash->nrooms = nrooms;
	}
	if (get_global_id() < KERN_GPUCACHE_FREE_WIDTH)
	{
		rowhash->freelist[get_global_id()]
			= (get_global_id() < nrooms ? get_global_id() : UINT_MAX);
	}
	/* setup rowhash */
	for (index = get_global_id(); index < nslots; index += get_global_size())
	{
		rowhash->slots[index].lock  = UINT_MAX;	/* unlocked */
		rowhash->slots[index].rowid = UINT_MAX;
	}
	/* setup rowmap */
	for (index = get_global_id(); index < nrooms; index += get_global_size())
	{
		if (index + KERN_GPUCACHE_FREE_WIDTH < nrooms)
			rowmap[index] = index + KERN_GPUCACHE_FREE_WIDTH;
		else
			rowmap[index] = UINT_MAX;	/* terminator */
	}
}

/*
 * gpucache_ctid_hash
 */
STATIC_INLINE(cl_uint)
gpucache_ctid_hash(ItemPointerData *ctid)
{
	cl_ulong	prime = 0x9e3779b97f4a7c13UL;
	cl_ulong	hash;

	hash = ((cl_ulong)ctid->ip_blkid.bi_hi << 32 |
			(cl_ulong)ctid->ip_blkid.bi_lo << 16 |
			(cl_ulong)ctid->ip_posid) * prime;
	return (cl_uint)((hash >> 20) & 0xffffffffU);
}

/*
 * gpucache_lookup_rowid_nolock
 */
STATIC_FUNCTION(cl_uint)
gpucache_lookup_rowid_nolock(kern_data_store *kds,
							 ItemPointerData *t_ctid)
{
	DECL_ROWID_HASH_AND_MAP(kds);
	GpuCacheSysattr *sysattr;
	cl_uint		hindex;
	cl_uint		rowid;

	hindex = gpucache_ctid_hash(t_ctid) % rowhash->nslots;
	for (rowid = rowhash->slots[hindex].rowid;
		 rowid != UINT_MAX;
		 rowid = rowmap[rowid])
	{
		assert(rowid < rowhash->nrooms);
		sysattr = kds_get_column_sysattr(kds, rowid);
		if (ItemPointerEquals(&sysattr->ctid, t_ctid))
			break;
	}
	return rowid;
}

/*
 * __gcache_alloc_rowid
 *
 * NOTE: See the chapter of "Volatile Qualifier" in the CUDA C++ Programming Guide.
 * It says that the compiler is free to optimize reads and writes to global or
 * shared memory, thus, memory access without volatile qualifier might not be
 * visible to other threads. Thus, we explicitly puts volatile access macros to
 * reference or modify the rowid hash-list to ensure the changes are visible to
 * any other concurrent threads.
 * Also, __threadfence() prior to the unlock makes any changes (including writes
 * without volatile qualifier) visible to other concurrent threads.
 */
STATIC_FUNCTION(cl_bool)
__gcache_alloc_rowid(kern_data_store *kds,
					 ItemPointerData *t_ctid,
					 cl_uint *p_rowid, cl_bool *found)
{
	DECL_ROWID_HASH_AND_MAP(kds);
	GpuCacheSysattr *sysattr;
	cl_uint		hindex;
	cl_uint		findex;
	cl_uint		rowid;
	cl_uint		next;
	cl_uint		lval __attribute__((unused));

	/* lookup hash slot */
	hindex = gpucache_ctid_hash(t_ctid) % rowhash->nslots;
	if (atomicCAS(&rowhash->slots[hindex].lock,
				  UINT_MAX,
				  get_global_id()) != UINT_MAX)
		return false;	/* try again */

	for (rowid = __volatileRead(&rowhash->slots[hindex].rowid);
		 rowid != UINT_MAX;
		 rowid = __volatileRead(&rowmap[rowid]))
	{
		assert(rowid < rowhash->nrooms);
		sysattr = kds_get_column_sysattr(kds, rowid);
		if (ItemPointerEquals(&sysattr->ctid, t_ctid))
		{
			/* already exists */
			*found = true;
			goto out_unlock;
		}
	}

	/* not found, so try to allocate a new one */
	findex = get_global_id() % KERN_GPUCACHE_FREE_WIDTH;
	do {
		rowid = __volatileRead(&rowhash->freelist[findex]);
		if (rowid == UINT_MAX)
		{
			/* no more free rowid? */
			*found = false;
			goto out_unlock;
		}
		assert(rowid < rowhash->nrooms);
		next = __volatileRead(&rowmap[rowid]);
		assert(next == UINT_MAX || next <  rowhash->nrooms);
	} while (atomicCAS(&rowhash->freelist[findex],
					   rowid,
					   next) != rowid);

	/* ok, rowid (a new row) was successfull allocated */
	sysattr = kds_get_column_sysattr(kds, rowid);
	sysattr->xmin = InvalidTransactionId;
	sysattr->xmax = InvalidTransactionId;
	sysattr->owner_id = 0;
	memcpy(&sysattr->ctid, t_ctid, sizeof(ItemPointerData));

	next = __volatileRead(&rowhash->slots[hindex].rowid);
	assert(next == UINT_MAX || next < rowhash->nrooms);
	__volatileWrite(&rowmap[rowid], next);
	__volatileWrite(&rowhash->slots[hindex].rowid, rowid);
#ifdef PGSTROM_DEBUG_BUILD
	printf("gid=%u rowid=%u allocated for ctid=(%u,%u)\n",
		   get_global_id(),
		   rowid,
		   (cl_uint)t_ctid->ip_blkid.bi_hi << 16 |
		   (cl_uint)t_ctid->ip_blkid.bi_lo,
		   (cl_uint)t_ctid->ip_posid);
#endif
	*found = false;
out_unlock:
	*p_rowid = rowid;
	__threadfence();
	lval = atomicExch(&rowhash->slots[hindex].lock, UINT_MAX);
	assert(lval == get_global_id());

	return true;
}

/*
 * gpucache_apply_insert
 */
STATIC_FUNCTION(cl_bool)
gpucache_apply_insert(kern_context *kcxt,
					  kern_data_store *kds,
					  kern_data_extra *extra,
					  GpuCacheSysattr *sysattr,
					  cl_uint rowid,
					  HeapTupleHeaderData *htup)
{
	char	   *pos = (char *)htup + htup->t_hoff;
	cl_bool		tup_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	cl_int		j, natts = (htup->t_infomask2 & HEAP_NATTS_MASK);
	cl_uint		oldid;

	assert(pos == (char *)MAXALIGN(pos));
	/*
	 * Note that nobody will modify the row-id hash table in this phase,
	 * so we can lookup the hash table without lock.
	 */
	oldid = gpucache_lookup_rowid_nolock(kds, &htup->t_ctid);
	for (j=0; j < kds->ncols; j++)
	{
		kern_colmeta   *cmeta = &kds->colmeta[j];
		char		   *dest = ((char *)kds + __kds_unpack(cmeta->values_offset));
		cl_uint		   *nullmap;

		if (j >= natts || (tup_hasnull && att_isnull(j, htup->t_bits)))
		{
			assert(cmeta->nullmap_offset != 0);
			nullmap = (cl_uint *)((char *)kds + __kds_unpack(cmeta->nullmap_offset));
			atomicAnd(&nullmap[rowid>>5], ~(1U << (rowid & 0x1f)));
			continue;
		}
		else if (cmeta->nullmap_offset != 0)
		{
			nullmap = (cl_uint *)((char *)kds + __kds_unpack(cmeta->nullmap_offset));
			atomicOr(&nullmap[rowid>>5], (1U << (rowid & 0x1f)));
		}

		if (cmeta->attlen > 0)
		{
			pos = (char *)TYPEALIGN(cmeta->attalign, pos);
			dest += TYPEALIGN(cmeta->attalign,
							  cmeta->attlen) * rowid;
			memcpy(dest, pos, cmeta->attlen);
			pos += cmeta->attlen;
		}
		else
		{
			cl_uint		sz;
			char	   *var;
			char	   *datum;
			cl_ulong	offset;

			assert(cmeta->attlen == -1);
			if (!VARATT_NOT_PAD_BYTE(pos))
				pos = (char *)TYPEALIGN(cmeta->attalign, pos);
			assert(!VARATT_IS_COMPRESSED(pos) && !VARATT_IS_EXTERNAL(pos));
			sz = VARSIZE_ANY(pos);
			var = pos;
			pos += sz;

			/* try to check whether it is actually updated */
			if (oldid != UINT_MAX)
			{
				datum = kern_datum_get_reusable_varlena(kds, extra, j, oldid, var);
				if (datum)
				{
					/* ok, this attribute is not updated */
					offset = (char *)datum - (char *)extra;
					assert(offset >= offsetof(kern_data_extra, data) &&
						   offset < extra->length);
					goto reuse_extra;
				}
			}
			/* allocation of extra buffer on demand */
			offset = atomicAdd(&extra->usage, MAXALIGN(sz));
			if (offset + MAXALIGN(sz) > extra->length)
			{
				STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY,
							  "out of extra buffer");
				return false;
			}
			memcpy((char *)extra + offset, var, sz);
		reuse_extra:
			((cl_uint *)dest)[rowid] = __kds_packed(offset);
		}
	}
	/* and, system attributes */
	sysattr->xmin = htup->t_choice.t_heap.t_xmin;
	sysattr->xmax = InvalidTransactionId;
	atomicMax(&kds->nitems, rowid+1);

	return true;
}

/*
 * gpucache_alloc_rowid
 */
STATIC_FUNCTION(void)
gpucache_alloc_rowid(kern_context *kcxt,
					 kern_gpucache_redolog *redo,
					 kern_data_store *kds,
					 kern_data_extra *extra)
{
	int		nloops;

	nloops = (redo->nitems + get_global_size() - 1) / get_global_size();
	for (int loop=0; loop < nloops; loop++)
	{
		cl_uint		owner_id = loop * get_global_size() + get_global_id();
		cl_bool		try_again = (owner_id < redo->nitems);

		do {
			if (try_again)
			{
				GCacheTxLogCommon *tx_log;
				cl_uint		offset = redo->log_index[owner_id];

				tx_log = (GCacheTxLogCommon *)
					((char *)redo + __kds_unpack(offset));
				if (tx_log->type == GCACHE_TX_LOG__INSERT)
				{
					GCacheTxLogInsert *i_log = (GCacheTxLogInsert *)tx_log;

					if (__gcache_alloc_rowid(kds,
											 &i_log->htup.t_ctid,
											 &i_log->rowid,
											 &i_log->rowid_found))
					{
						if (i_log->rowid == UINT_MAX)
							STROM_EREPORT(kcxt, ERRCODE_INVALID_NAME,
										  "no more rowid allocatable");
						try_again = false;
					}
				}
				else if (tx_log->type == GCACHE_TX_LOG__DELETE)
				{
					GCacheTxLogDelete *d_log = (GCacheTxLogDelete *)tx_log;

					if (__gcache_alloc_rowid(kds,
											 &d_log->ctid,
											 &d_log->rowid,
											 &d_log->rowid_found))
					{
						if (d_log->rowid == UINT_MAX)
							STROM_EREPORT(kcxt, ERRCODE_INVALID_NAME,
										  "no more rowid allocatable");
						try_again = false;
					}
				}
				else if (tx_log->type == GCACHE_TX_LOG__XACT)
				{
					GCacheTxLogXact *x_log = (GCacheTxLogXact *)tx_log;
				
					if (__gcache_alloc_rowid(kds,
											 &x_log->ctid,
											 &x_log->rowid,
											 &x_log->rowid_found))
					{
						if (x_log->rowid == UINT_MAX)
							STROM_EREPORT(kcxt, ERRCODE_INVALID_NAME,
										  "no more rowid allocatable");
						try_again = false;
					}
				}
				else
				{
					/* ignore other log types, if any */
					try_again = false;
				}
			}
		} while (__syncthreads_count(try_again) != 0);
	}
}

/*
 * gpucache_cleanup_owner
 */
STATIC_FUNCTION(void)
gpucache_cleanup_owner(kern_context *kcxt,
					   kern_gpucache_redolog *redo,
					   kern_data_store *kds)
{
	cl_uint		owner_id;

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GpuCacheSysattr *sysattr;
		GCacheTxLogCommon *tx_log;
		cl_uint		offset = redo->log_index[owner_id];
		cl_uint		rowid;

		tx_log = (GCacheTxLogCommon *)
			((char *)redo + __kds_unpack(offset));
		if (tx_log->type == GCACHE_TX_LOG__INSERT)
		{
			rowid = ((GCacheTxLogInsert *)tx_log)->rowid;
			assert(rowid < kds->nrooms);
			sysattr = kds_get_column_sysattr(kds, rowid);
			sysattr->owner_id = 0;
		}
		else if (tx_log->type == GCACHE_TX_LOG__DELETE)
		{
			rowid = ((GCacheTxLogDelete *)tx_log)->rowid;
			assert(rowid < kds->nrooms);
			sysattr = kds_get_column_sysattr(kds, rowid);
			sysattr->owner_id = 0;
		}
		else if (tx_log->type == GCACHE_TX_LOG__XACT)
		{
			rowid = ((GCacheTxLogXact *)tx_log)->rowid;
			assert(rowid < kds->nrooms);
			sysattr = kds_get_column_sysattr(kds, rowid);
			sysattr->owner_id = 0;
		}
		else
		{
			printf("unknown GCacheTxLog type '%08x'\n", tx_log->type);
		}
	}
}

/*
 * gpucache_setup_owner
 */
STATIC_FUNCTION(void)
gpucache_setup_owner(kern_context *kcxt,
					 kern_gpucache_redolog *redo,
					 kern_data_store *kds,
					 cl_bool skip_xact_log)
{
	cl_uint		owner_id;

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GpuCacheSysattr *sysattr;
		GCacheTxLogCommon *tx_log;
		cl_uint		offset = redo->log_index[owner_id];
		cl_uint		rowid;

		tx_log = (GCacheTxLogCommon *)((char *)redo + __kds_unpack(offset));
		if (tx_log->type == GCACHE_TX_LOG__INSERT)
		{
			rowid = ((GCacheTxLogInsert *)tx_log)->rowid;
			assert(rowid < kds->nrooms);
			sysattr = kds_get_column_sysattr(kds, rowid);
			atomicMax(&sysattr->owner_id, owner_id);
		}
		else if (tx_log->type == GCACHE_TX_LOG__DELETE)
		{
			rowid = ((GCacheTxLogDelete *)tx_log)->rowid;
			assert(rowid < kds->nrooms);
			sysattr = kds_get_column_sysattr(kds, rowid);
			atomicMax(&sysattr->owner_id, owner_id);
		}
		else if (tx_log->type == GCACHE_TX_LOG__XACT)
		{
			if (!skip_xact_log)
			{
				rowid = ((GCacheTxLogXact *)tx_log)->rowid;
				assert(rowid < kds->nrooms);
				sysattr = kds_get_column_sysattr(kds, rowid);
				atomicMax(&sysattr->owner_id, owner_id);
			}
		}
		else
		{
			printf("unknown GCacheTxLog type '%08x'\n", tx_log->type);
		}
	}
}

/*
 * gpucache_apply_redo
 */
STATIC_FUNCTION(void)
gpucache_apply_redo(kern_context *kcxt,
					kern_gpucache_redolog *redo,
					kern_data_store *kds,
					kern_data_extra *extra)
{
	cl_uint		owner_id;

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GCacheTxLogCommon *tx_log;
		GpuCacheSysattr *sysattr;
		cl_uint		offset = redo->log_index[owner_id];

		tx_log = (GCacheTxLogCommon *)
			((char *)redo + __kds_unpack(offset));
		if (tx_log->type == GCACHE_TX_LOG__INSERT)
		{
			GCacheTxLogInsert *i_log = (GCacheTxLogInsert *)tx_log;

			assert(i_log->rowid < kds->nrooms);
			sysattr = kds_get_column_sysattr(kds, i_log->rowid);
			if (sysattr->owner_id == owner_id)
			{
				gpucache_apply_insert(kcxt, kds, extra, sysattr,
									  i_log->rowid, &i_log->htup);
			}
		}
		else if (tx_log->type == GCACHE_TX_LOG__DELETE)
		{
			GCacheTxLogDelete *d_log = (GCacheTxLogDelete *)tx_log;

			assert(d_log->rowid < kds->nrooms);
			sysattr = kds_get_column_sysattr(kds, d_log->rowid);
			if (sysattr->owner_id == owner_id)
			{
				sysattr->xmax = d_log->xid;
			}
		}
	}
}

/*
 * __gpucache_release_rowid
 */
STATIC_FUNCTION(cl_bool)
__gpucache_release_rowid(kern_data_store *kds,
						 GCacheTxLogXact *x_log)
{
	DECL_ROWID_HASH_AND_MAP(kds);
	GpuCacheSysattr *sysattr;
	cl_uint		hindex;
	cl_uint		findex;
	cl_uint		rowid;
	cl_uint		next;
	cl_uint		lval __attribute__((unused));
	volatile cl_uint *prev;

	/* lock and lookup the hash-slot */
	hindex = gpucache_ctid_hash(&x_log->ctid) % rowhash->nslots;
	if (atomicCAS(&rowhash->slots[hindex].lock,
				  UINT_MAX,
				  get_global_id()) != UINT_MAX)
		return false;		/* try again */

	for (prev = &rowhash->slots[hindex].rowid, rowid = *prev;
		 rowid != UINT_MAX;
		 prev = &rowmap[rowid], rowid = *prev)
	{
		assert(rowid < rowhash->nrooms);
		sysattr = kds_get_column_sysattr(kds, rowid);
		if (ItemPointerEquals(&sysattr->ctid, &x_log->ctid))
		{
			assert(rowid == x_log->rowid);
			/* detach rowid from the hash table */
			next = __volatileRead(&rowmap[rowid]);
			assert(next == UINT_MAX || next < rowhash->nrooms);
			*prev = next;

			/* attach rowid to the freelist */
			findex = rowid % KERN_GPUCACHE_FREE_WIDTH;
			do {
				next = __volatileRead(&rowhash->freelist[findex]);
				__volatileWrite(&rowmap[rowid], next);
			} while (atomicCAS(&rowhash->freelist[findex],
							   next,
							   rowid) != next);
#ifdef PGSTROM_DEBUG_BUILD
			printf("__gpucache: rowid=%u ctid=(%u,%u) released\n",
				   rowid,
				   (cl_uint)x_log->ctid.ip_blkid.bi_hi << 16 |
				   (cl_uint)x_log->ctid.ip_blkid.bi_lo,
				   (cl_uint)x_log->ctid.ip_posid);
#endif
			goto out_unlock;
		}
	}
	printf("__gpucache_release_rowid: rowid=%u ctid=(%u,%u) not found\n",
		   x_log->rowid,
		   (cl_uint)x_log->ctid.ip_blkid.bi_hi << 16 |
		   (cl_uint)x_log->ctid.ip_blkid.bi_lo,
		   (cl_uint)x_log->ctid.ip_posid);
out_unlock:
	__threadfence();
	lval = atomicExch(&rowhash->slots[hindex].lock, UINT_MAX);
	assert(lval == get_global_id());

	return true;
}

/*
 * gpucache_apply_xact
 */
STATIC_FUNCTION(void)
gpucache_apply_xact(kern_context *kcxt,
					kern_gpucache_redolog *redo,
					kern_data_store *kds)
{
	cl_uint		nloops;

	nloops = (redo->nitems + get_global_size() - 1) / get_global_size();
	for (int loop=0; loop < nloops; loop++)
	{
		cl_uint		owner_id = loop * get_global_size() + get_global_id();
		int			try_again = (owner_id < redo->nitems);

		do {
			if (try_again)
			{
				GCacheTxLogXact *x_log;
				GpuCacheSysattr *sysattr;
				cl_uint		offset = redo->log_index[owner_id];

				x_log = (GCacheTxLogXact *)
					((char *)redo + __kds_unpack(offset));
				if (x_log->type == GCACHE_TX_LOG__XACT)
				{
					assert(x_log->rowid < kds->nrooms);
					sysattr = kds_get_column_sysattr(kds, x_log->rowid);
					if (x_log->tag == 'I')			/* COMMIT INSERT */
					{
						if (sysattr->owner_id == owner_id)
							sysattr->xmin = FrozenTransactionId;
						try_again = false;
					}
					else if (x_log->tag == 'D' ||	/* COMMIT DELETE */
							 x_log->tag == 'i')		/* ROLLBACK INSERT */
					{
						if (sysattr->owner_id == owner_id)
						{
							if (__gpucache_release_rowid(kds, x_log))
							{
								sysattr->xmin = InvalidTransactionId;
								sysattr->xmax = InvalidTransactionId;
								try_again = false;
							}
						}
						else
						{
							try_again = false;
						}
					}
					else if (x_log->tag == 'd')		/* ROLLBACK DELETE */
					{
						if (sysattr->owner_id == owner_id)
							sysattr->xmax = InvalidTransactionId;
						try_again = false;
					}
					else
					{
						try_again = false;
					}
				}
				else
				{
					try_again = false;
				}
			}
		} while (__syncthreads_count(try_again) != 0);
	}
}

/*
 * gpucache_fixup_rowid
 */
STATIC_FUNCTION(void)
gpucache_fixup_rowid(kern_context *kcxt,
					 kern_gpucache_redolog *redo,
					 kern_data_store *kds)
{


}

KERNEL_FUNCTION(void)
kern_gpucache_apply_redo(kern_gpucache_redolog *redo,
						 kern_data_store *kds,
						 kern_data_extra *extra,
						 int phase)
{
	kern_context kcxt;

	/* bailout if any errors */
	if (__syncthreads_count(redo->kerror.errcode) > 0)
		return;

	INIT_KERNEL_CONTEXT(&kcxt, NULL);	/* no kparams */
	switch (phase)
	{
		case 0:		/* assign rowid for each log entries */
			gpucache_alloc_rowid(&kcxt, redo, kds, extra);
			break;
		case 1:		/* clear the owner_id field of log entries */
			gpucache_cleanup_owner(&kcxt, redo, kds);
			break;
		case 2:		/* assign the largest owner_id of INS/DEL log entries */
			gpucache_setup_owner(&kcxt, redo, kds, true);
			break;
		case 3:		/* apply INS/DEL log entries */
			gpucache_apply_redo(&kcxt, redo, kds, extra);
			break;
		case 4:		/* assign the largest owner_id of XACT log entries */
			gpucache_setup_owner(&kcxt, redo, kds, false);
			break;
		case 5:		/* apply XACT log entries */
			gpucache_apply_xact(&kcxt, redo, kds);
			break;
		default:	/* release rowid if any errors */
			gpucache_fixup_rowid(&kcxt, redo, kds);
			break;
	}
	kern_writeback_error_status(&redo->kerror, &kcxt);
}

KERNEL_FUNCTION(void)
kern_gpucache_compaction(kern_data_store *kds,
						 kern_data_extra *old_extra,
						 kern_data_extra *new_extra)
{
	
	__shared__ cl_uint required;
	__shared__ cl_ulong extra_base;
	cl_uint		nloops;

	nloops = (kds->nitems + get_global_size() - 1) / get_global_size();
	for (int j=0; j < kds->ncols; j++)
	{
		kern_colmeta *cmeta = &kds->colmeta[j];

		if (cmeta->attbyval || cmeta->attlen != -1)
			continue;		/* not varlena */
		for (int loop=0; loop < nloops; loop++)
		{
			GpuCacheSysattr *sysattr;
			cl_uint		rowid = loop * get_global_size() + get_global_id();
			cl_bool		isnull = false;
			cl_uint	   *values;
			char	   *orig = NULL;
			char	   *dest;
			cl_uint		sz, l_off = 0;

			sysattr = kds_get_column_sysattr(kds, rowid);
			if (rowid >= kds->nitems)
				isnull = true;		/* out of range */
			else if (sysattr->xmin == InvalidTransactionId ||
					 sysattr->xmax == FrozenTransactionId)
				isnull = true;		/* row is already removed */
			else if (cmeta->nullmap_offset != 0)
			{
				cl_uint	   *nullmap = (cl_uint *)
					((char *)kds + __kds_unpack(cmeta->nullmap_offset));
				if ((nullmap[rowid>>5] & (1U << (rowid & 0x1f))) == 0)
					isnull = true;
			}
			values = (cl_uint *)((char *)kds + __kds_unpack(cmeta->values_offset));
			/* copy the varlena to new extra buffer */
			if (get_local_id() == 0)
				required = 0;
			__syncthreads();
			if (!isnull)
			{
				orig = (char *)old_extra + __kds_unpack(values[rowid]);
				sz = VARSIZE_ANY(orig);
				assert(orig > (char *)old_extra &&
					   orig + sz <= (char *)old_extra + old_extra->length);
				l_off = atomicAdd(&required, MAXALIGN(sz));
			}
			__syncthreads();
			if (get_local_id() == 0)
				extra_base = atomicAdd(&new_extra->usage, required);
			__syncthreads();
			dest = (char *)new_extra + extra_base + l_off;
			if (isnull)
			{
				if (rowid < kds->nitems)
					values[rowid] = 0;
			}
			else if (dest + sz <= (char *)new_extra + new_extra->length)
			{
				memcpy(dest, orig, sz);
				values[rowid] = __kds_packed((char *)dest - (char *)new_extra);
			}
			else
			{
				assert(new_extra->length == 0);
			}
		}
	}
}
