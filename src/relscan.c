/*
 * relscan.c
 *
 * Routines related to outer relation scan
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* ----------------------------------------------------------------
 *
 * Routines to support optimization / path or plan construction
 *
 * ----------------------------------------------------------------
 */
Bitmapset *
pickup_outer_referenced(PlannerInfo *root,
						RelOptInfo *base_rel,
						Bitmapset *referenced)
{
	ListCell   *lc;
	int			j, k;

	if (base_rel->reloptkind == RELOPT_BASEREL)
	{
		for (j=base_rel->min_attr; j <= base_rel->max_attr; j++)
		{
			if (j <= 0 || !base_rel->attr_needed[j - base_rel->min_attr])
				continue;
			k = j - FirstLowInvalidHeapAttributeNumber;
			referenced = bms_add_member(referenced, k);
		}
	}
	else if (base_rel->reloptkind == RELOPT_OTHER_MEMBER_REL)
	{
		foreach (lc, root->append_rel_list)
		{
			AppendRelInfo  *apinfo = lfirst(lc);
			RelOptInfo	   *parent_rel;
			Bitmapset	   *parent_refs;
			Var			   *var;

			if (apinfo->child_relid != base_rel->relid)
				continue;
			Assert(apinfo->parent_relid < root->simple_rel_array_size);
			parent_rel = root->simple_rel_array[apinfo->parent_relid];
			parent_refs = pickup_outer_referenced(root, parent_rel, NULL);

			for (k = bms_next_member(parent_refs, -1);
				 k >= 0;
				 k = bms_next_member(parent_refs, k))
			{
				j = k + FirstLowInvalidHeapAttributeNumber;
				if (j <= 0)
					bms_add_member(referenced, k);
				else if (j > list_length(apinfo->translated_vars))
					elog(ERROR, "Bug? column reference out of range");
				else
				{
					var = list_nth(apinfo->translated_vars, j-1);
					Assert(IsA(var, Var));
					j = var->varattno - FirstLowInvalidHeapAttributeNumber;
					referenced = bms_add_member(referenced, j);
				}
			}
			break;
		}
		if (!lc)
			elog(ERROR, "Bug? AppendRelInfo not found (relid=%u)",
				 base_rel->relid);
	}
	else
	{
		elog(ERROR, "Bug? outer relation is not a simple relation");
	}
	return referenced;
}

/* ----------------------------------------------------------------
 *
 * Routines to setup kern_data_store
 *
 * ----------------------------------------------------------------
 */
int
count_num_of_subfields(Oid type_oid)
{
	TypeCacheEntry *tcache;
	int		j, count = 0;

	tcache = lookup_type_cache(type_oid, TYPECACHE_TUPDESC);
	if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
		/* array type */
		count = 1 + count_num_of_subfields(tcache->typelem);
	}
	else if (tcache->tupDesc)
	{
		/* composite type */
		TupleDesc	tupdesc = tcache->tupDesc;

		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

			if (attr->attisdropped)
				continue;
			count += 1 + count_num_of_subfields(attr->atttypid);
		}
	}
	return count;
}

static void
__setup_kern_colmeta(kern_data_store *kds,
					 int column_index,
					 const char *attname,
					 int attnum,
					 bool attbyval,
					 char attalign,
					 int16 attlen,
					 Oid atttypid,
					 int atttypmod,
					 int *p_attcacheoff)
{
	kern_colmeta   *cmeta = &kds->colmeta[column_index];
	devtype_info   *dtype;
	TypeCacheEntry *tcache;

	memset(cmeta, 0, sizeof(kern_colmeta));
	cmeta->attbyval	= attbyval;
	cmeta->attalign	= typealign_get_width(attalign);
	cmeta->attlen	= attlen;
	if (attlen == 0 || attlen < -1)
		elog(ERROR, "attribute %s has unexpected length (%d)", attname, attlen);
	else if (attlen == -1)
		kds->has_varlena = true;
	cmeta->attnum	= attnum;

	if (!p_attcacheoff || *p_attcacheoff < 0)
		cmeta->attcacheoff = -1;
	else if (attlen > 0)
	{
		cmeta->attcacheoff = att_align_nominal(*p_attcacheoff, attalign);
		*p_attcacheoff = cmeta->attcacheoff + attlen;
	}
	else if (attlen == -1)
	{
		/*
		 * Note that attcacheoff is also available on varlena datum
		 * only if it appeared at the first, and its offset is aligned.
		 * Elsewhere, we cannot utilize the attcacheoff for varlena
		 */
		uint32_t	__off = att_align_nominal(*p_attcacheoff, attalign);

		if (*p_attcacheoff == __off)
			cmeta->attcacheoff = __off;
		else
			cmeta->attcacheoff = -1;
		*p_attcacheoff = -1;
	}
	else
	{
		cmeta->attcacheoff = *p_attcacheoff = -1;
	}
	cmeta->atttypid = atttypid;
	cmeta->atttypmod = atttypmod;
	strncpy(cmeta->attname, attname, NAMEDATALEN);

	if (!OidIsValid(atttypid) ||
		!(tcache = lookup_type_cache(atttypid, TYPECACHE_TUPDESC)))
	{
		/* corner case: column might be already dropped */
		cmeta->atttypkind = TYPE_KIND__NULL;
	}
	else if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
		/* array type */
		char		elem_name[NAMEDATALEN+10];
		int16		elem_len;
		bool		elem_byval;
		char		elem_align;

		cmeta->atttypkind = TYPE_KIND__ARRAY;
		cmeta->idx_subattrs = kds->nr_colmeta++;
		cmeta->num_subattrs = 1;

		snprintf(elem_name, sizeof(elem_name), "__%s", attname);
		get_typlenbyvalalign(tcache->typelem,
							 &elem_len,
							 &elem_byval,
							 &elem_align);
		__setup_kern_colmeta(kds,
							 cmeta->idx_subattrs,
							 elem_name,			/* attname */
							 1,					/* attnum */
							 elem_byval,		/* attbyval */
							 elem_align,		/* attalign */
							 elem_len,			/* attlen */
							 tcache->typelem,	/* atttypid */
							 -1,				/* atttypmod */
							 NULL);				/* attcacheoff */
	}
	else if (tcache->tupDesc)
	{
		/* composite type */
		TupleDesc	tupdesc = tcache->tupDesc;
		int			j, attcacheoff = -1;

		cmeta->atttypkind = TYPE_KIND__COMPOSITE;
		cmeta->idx_subattrs = kds->nr_colmeta;
		cmeta->num_subattrs = tupdesc->natts;
		kds->nr_colmeta += tupdesc->natts;

		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

			__setup_kern_colmeta(kds,
								 cmeta->idx_subattrs + j,
								 NameStr(attr->attname),
								 attr->attnum,
								 attr->attbyval,
								 attr->attalign,
								 attr->attlen,
								 attr->atttypid,
								 attr->atttypmod,
								 &attcacheoff);
		}
	}
	else
	{
		switch (tcache->typtype)
		{
			case TYPTYPE_BASE:
				cmeta->atttypkind = TYPE_KIND__BASE;
				break;
			case TYPTYPE_DOMAIN:
				cmeta->atttypkind = TYPE_KIND__DOMAIN;
				break;
			case TYPTYPE_ENUM:
				cmeta->atttypkind = TYPE_KIND__ENUM;
				break;
			case TYPTYPE_PSEUDO:
				cmeta->atttypkind = TYPE_KIND__PSEUDO;
				break;
			case TYPTYPE_RANGE:
				cmeta->atttypkind = TYPE_KIND__RANGE;
				break;
			default:
				elog(ERROR, "Unexpected typtype ('%c')", tcache->typtype);
				break;
		}
	}
	/*
	 * for the reverse references to KDS
	 */
	dtype = pgstrom_devtype_lookup(atttypid);
	if (dtype)
		cmeta->dtype_sizeof = dtype->type_sizeof;
	cmeta->kds_format = kds->format;
	cmeta->kds_offset = (char *)cmeta - (char *)kds;
}

size_t
setup_kern_data_store(kern_data_store *kds,
					  TupleDesc tupdesc,
					  size_t length,
					  char format)
{
	int		j, attcacheoff = -1;

	/*
	 * lp_items[] is declared as uint32 (and uint64 has no benefit because
	 * PGSTROM_CHUNK_SIZE is much smaller), so BLOCK format length must be
	 * 32bit range.
	 */
	Assert(format != KDS_FORMAT_BLOCK || length < UINT_MAX);

	memset(kds, 0, offsetof(kern_data_store, colmeta));
	kds->length		= length;
	kds->usage		= 0;
	kds->nitems		= 0;
	kds->ncols		= tupdesc->natts;
	kds->format		= format;
	kds->tdhasoid	= false;	/* PG12 removed 'oid' system column */
	kds->tdtypeid	= tupdesc->tdtypeid;
	kds->tdtypmod	= tupdesc->tdtypmod;
	kds->table_oid	= InvalidOid;	/* to be set by the caller */
	kds->hash_nslots = 0;			/* to be set by the caller, if any */
	kds->nr_colmeta	= tupdesc->natts;

	if (format == KDS_FORMAT_ROW  ||
		format == KDS_FORMAT_HASH ||
		format == KDS_FORMAT_BLOCK)
		attcacheoff = 0;

	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		__setup_kern_colmeta(kds, j,
							 NameStr(attr->attname),
							 attr->attnum,
							 attr->attbyval,
							 attr->attalign,
							 attr->attlen,
							 attr->atttypid,
							 attr->atttypmod,
							 &attcacheoff);
	}
	/* internal system attribute */
	if (format == KDS_FORMAT_COLUMN)
	{
		kern_colmeta *cmeta = &kds->colmeta[kds->nr_colmeta++];

		memset(cmeta, 0, sizeof(kern_colmeta));
		cmeta->attbyval = false;
		cmeta->attalign = sizeof(int32_t);
		cmeta->attlen = sizeof(GpuCacheSysattr);
		cmeta->attnum = -1;
		cmeta->attcacheoff = -1;
		cmeta->atttypid = InvalidOid;
		cmeta->atttypmod = -1;
		cmeta->atttypkind = TYPE_KIND__BASE;
		cmeta->kds_format = kds->format;
		cmeta->kds_offset = (uint32_t)((char *)cmeta - (char *)kds);
		strcpy(cmeta->attname, "__gcache_sysattr__");
	}
	return MAXALIGN(offsetof(kern_data_store, colmeta[kds->nr_colmeta]));
}

/*
 * estimate_kern_data_store
 *
 * NOTE: This function estimates required buffer size for the KDS that
 *       follows the given TupleDesc, but may not be exact size.
 *       setup_kern_data_store() shall return exact header size.
 */
size_t
estimate_kern_data_store(TupleDesc tupdesc)
{
	int		j, nr_colmeta = tupdesc->natts;

	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		if (attr->attisdropped)
			continue;
		nr_colmeta += count_num_of_subfields(attr->atttypid);
	}
	/* internal system attribute if KDS_FORMAT_COLUMN */
	nr_colmeta++;
	return MAXALIGN(offsetof(kern_data_store, colmeta[nr_colmeta]));
}

/* ----------------------------------------------------------------
 *
 * Routines to load chunks from storage
 *
 * ----------------------------------------------------------------
 */
#define __XCMD_KDS_SRC_OFFSET(buf)							\
	(((XpuCommand *)((buf)->data))->u.task.kds_src_offset)
#define __XCMD_GET_KDS_SRC(buf)								\
	((kern_data_store *)((buf)->data + __XCMD_KDS_SRC_OFFSET(buf)))

static void
__relScanDirectFallbackBlock(pgstromTaskState *pts,
							 kern_data_store *kds,
							 BlockNumber block_num)
{
	pgstromSharedState *ps_state = pts->ps_state;
	Relation	relation = pts->css.ss.ss_currentRelation;
	HeapScanDesc h_scan = (HeapScanDesc)pts->css.ss.ss_currentScanDesc;
	Snapshot	snapshot = pts->css.ss.ps.state->es_snapshot;
	Buffer		buffer;
	Page		page;
	int			lines;
	OffsetNumber lineoff;
	ItemId		lpp;

	buffer = ReadBufferExtended(relation,
								MAIN_FORKNUM,
								block_num,
								RBM_NORMAL,
								h_scan->rs_strategy);
	/* just like heapgetpage() */
	heap_page_prune_opt(relation, buffer);
	/* pick up valid tuples from the target page */
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buffer);
	lines = PageGetMaxOffsetNumber(page);
	for (lineoff = FirstOffsetNumber, lpp = PageGetItemId(page, lineoff);
		 lineoff <= lines;
		 lineoff++, lpp++)
	{
		HeapTupleData htup;
		bool		valid;

		if (!ItemIdIsNormal(lpp))
			continue;

		htup.t_tableOid = RelationGetRelid(relation);
		htup.t_data = (HeapTupleHeader) PageGetItem((Page)page, lpp);
		htup.t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&htup.t_self, block_num, lineoff);

		valid = HeapTupleSatisfiesVisibility(&htup, snapshot, buffer);
		HeapCheckForSerializableConflictOut(valid, relation, &htup,
											buffer, snapshot);
		if (valid)
			execCpuFallbackBaseTuple(pts, &htup);
	}
	UnlockReleaseBuffer(buffer);
	pg_atomic_fetch_add_u64(&ps_state->npages_buffer_read, PAGES_PER_BLOCK);
}

static void
__relScanDirectCachedBlock(pgstromTaskState *pts, BlockNumber block_num)
{
	Relation	relation = pts->css.ss.ss_currentRelation;
	HeapScanDesc h_scan = (HeapScanDesc)pts->css.ss.ss_currentScanDesc;
	Snapshot	snapshot = pts->css.ss.ps.state->es_snapshot;
	kern_data_store *kds;
	Buffer		buffer;
	Page		spage;
	Page		dpage;
	bool		has_valid_tuples = false;

	/*
	 * Load the source buffer with synchronous read
	 */
	buffer = ReadBufferExtended(relation,
								MAIN_FORKNUM,
								block_num,
								RBM_NORMAL,
								h_scan->rs_strategy);
	/* prune the old items, if any */
	heap_page_prune_opt(relation, buffer);
	/* let's check tuples visibility for each */
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	spage = (Page) BufferGetPage(buffer);
	appendBinaryStringInfo(&pts->xcmd_buf, (const char *)spage, BLCKSZ);

	kds = __XCMD_GET_KDS_SRC(&pts->xcmd_buf);
	dpage = (Page) KDS_BLOCK_PGPAGE(kds, kds->block_nloaded);
	Assert(dpage >= pts->xcmd_buf.data &&
		   dpage + BLCKSZ <= pts->xcmd_buf.data + pts->xcmd_buf.len);
	KDS_BLOCK_BLCKNR(kds, kds->block_nloaded) = block_num;

	/*
	 * Logic is almost equivalent as heapgetpage() doing.
	 * We have to invalidate tuples prior to GPU kernel
	 * execution, if not all-visible.
	 */
	if (!PageIsAllVisible(spage) || snapshot->takenDuringRecovery)
	{
		int		lines = PageGetMaxOffsetNumber(spage);
		ItemId	lpp;
		OffsetNumber lineoff;

		Assert(lines == PageGetMaxOffsetNumber(dpage));
		for (lineoff = FirstOffsetNumber, lpp = PageGetItemId(spage, lineoff);
			 lineoff <= lines;
			 lineoff++, lpp++)
		{
			HeapTupleData htup;
			bool	valid;

			if (!ItemIdIsNormal(lpp))
				continue;
			htup.t_tableOid = RelationGetRelid(relation);
			htup.t_data = (HeapTupleHeader) PageGetItem((Page) spage, lpp);
			Assert((((uintptr_t)htup.t_data - (uintptr_t)spage) & 7) == 0);
			htup.t_len = ItemIdGetLength(lpp);
			ItemPointerSet(&htup.t_self, block_num, lineoff);

			valid = HeapTupleSatisfiesVisibility(&htup, snapshot, buffer);
			HeapCheckForSerializableConflictOut(valid, relation, &htup,
												buffer, snapshot);
			if (valid)
				has_valid_tuples = true;
			else
			{
				/* also invalidate the duplicated page */
				ItemIdSetUnused(PageGetItemId(dpage, lineoff));
			}
		}
	}
	else
	{
		has_valid_tuples = true;
	}
	UnlockReleaseBuffer(buffer);

	/*
	 * If no tuples in this block are visible, we don't need to load
	 * them to xPU device (just wast of memory and bandwidth),
	 * so it shall be reverted from the xcmd-buffer.
	 */
	if (!has_valid_tuples)
	{
		pts->xcmd_buf.len -= BLCKSZ;
		return;
	}
	/* dpage became all-visible also */
	PageSetAllVisible(dpage);
	kds->nitems++;
	kds->block_nloaded++;
}

static bool
__relScanDirectCheckBufferClean(SMgrRelation smgr, BlockNumber block_num)
{
	BufferTag	bufTag;
	uint32_t	bufHash;
	LWLock	   *bufLock;
	Buffer		buffer;
	BufferDesc *bufDesc;
	uint32_t	bufState;

	smgr_init_buffer_tag(&bufTag, smgr, MAIN_FORKNUM, block_num);
	bufHash = BufTableHashCode(&bufTag);
	bufLock = BufMappingPartitionLock(bufHash);

	/* check whether the block exists on the shared buffer? */
	LWLockAcquire(bufLock, LW_SHARED);
	buffer = BufTableLookup(&bufTag, bufHash);
	if (buffer < 0)
	{
		LWLockRelease(bufLock);
		return true;		/* OK, block is not buffered */
	}
	bufDesc = GetBufferDescriptor(buffer);
	bufState = pg_atomic_read_u32(&bufDesc->state);
	LWLockRelease(bufLock);

	return (bufState & BM_DIRTY) == 0;
}

XpuCommand *
pgstromRelScanChunkDirect(pgstromTaskState *pts,
						  struct iovec *xcmd_iov, int *xcmd_iovcnt)
{
	pgstromSharedState *ps_state = pts->ps_state;
	Relation		relation = pts->css.ss.ss_currentRelation;
	SMgrRelation	smgr = RelationGetSmgr(relation);
	XpuCommand	   *xcmd;
	kern_data_store *kds;
	unsigned long	m_offset = 0UL;
	BlockNumber		segment_id = InvalidBlockNumber;
	strom_io_vector *strom_iovec;
	strom_io_chunk *strom_ioc = NULL;
	BlockNumber	   *strom_blknums;
	uint32_t		strom_nblocks = 0;
	uint32_t		kds_src_pathname = 0;
	uint32_t		kds_src_iovec = 0;
	uint32_t		kds_nrooms;
	int32_t			scan_repeat_id = -1;

	kds = __XCMD_GET_KDS_SRC(&pts->xcmd_buf);
	kds_nrooms = (PGSTROM_CHUNK_SIZE -
				  KDS_HEAD_LENGTH(kds)) / (sizeof(BlockNumber) + BLCKSZ);
	kds->nitems = 0;
	kds->usage = 0;
	kds->block_offset = (KDS_HEAD_LENGTH(kds) +
						 MAXALIGN(sizeof(BlockNumber) * kds_nrooms));
	kds->block_nloaded = 0;
	pts->xcmd_buf.len = __XCMD_KDS_SRC_OFFSET(&pts->xcmd_buf) + kds->block_offset;
	Assert(pts->xcmd_buf.len == MAXALIGN(pts->xcmd_buf.len));
	enlargeStringInfo(&pts->xcmd_buf, BLCKSZ * kds_nrooms);
	kds = __XCMD_GET_KDS_SRC(&pts->xcmd_buf);

	strom_iovec = alloca(offsetof(strom_io_vector, ioc[kds_nrooms]));
	strom_iovec->nr_chunks = 0;
	strom_blknums = alloca(sizeof(BlockNumber) * kds_nrooms);
	strom_nblocks = 0;

	while (!pts->scan_done)
	{
		while (pts->curr_block_num < pts->curr_block_tail &&
			   kds->nitems < kds_nrooms)
		{
			BlockNumber		block_num
				= (pts->curr_block_num +
				   ps_state->scan_block_start) % ps_state->scan_block_nums;
			/*
			 * MEMO: Usually, CPU is (much) more powerful than DPUs.
			 * In case when the source cache is already on the shared-
			 * buffer, it makes no sense to handle this page on the
			 * DPU device.
			 */
			if (pts->ds_entry && !pgstrom_dpu_handle_cached_pages)
			{
				BufferTag	bufTag;
				uint32		bufHash;
				LWLock	   *bufLock;
				int			buf_id;

				smgr_init_buffer_tag(&bufTag, smgr, MAIN_FORKNUM, block_num);
				bufHash = BufTableHashCode(&bufTag);
				bufLock = BufMappingPartitionLock(bufHash);

				/* check whether the block exists on the shared buffer? */
				LWLockAcquire(bufLock, LW_SHARED);
				buf_id = BufTableLookup(&bufTag, bufHash);
				if (buf_id >= 0)
				{
					LWLockRelease(bufLock);
					__relScanDirectFallbackBlock(pts, kds, block_num);
					pts->curr_block_num++;
					continue;
				}
				LWLockRelease(bufLock);
			}

			/*
			 * MEMO: When multiple scans are needed (pts->num_scan_repeats > 0),
			 * kds_src should not mixture the blocks come from different scans,
			 * because it shall be JOIN'ed on different partitions.
			 */
			if (scan_repeat_id < 0)
				scan_repeat_id = pts->curr_block_num / ps_state->scan_block_nums;
			else if (scan_repeat_id != pts->curr_block_num / ps_state->scan_block_nums)
				goto out;

			/*
			 * MEMO: right now, we allow GPU Direct SQL for the all-visible
			 * pages only, due to the restrictions about MVCC checks.
			 * However, it is too strict for the purpose. If we would have
			 * a mechanism to perform MVCC checks without commit logs.
			 * In other words, if all the tuples in a certain page have
			 * HEAP_XMIN_* or HEAP_XMAX_* flags correctly, we can have MVCC
			 * logic in the device code.
			 */
			if (VM_ALL_VISIBLE(relation, block_num, &pts->curr_vm_buffer) &&
				__relScanDirectCheckBufferClean(smgr, block_num))
			{
				/*
				 * We don't allow xPU Direct SQL across multiple heap
				 * segments (for the code simplification). So, once
				 * relation scan is broken out, then restart with new
				 * KDS buffer.
				 */
				unsigned int	fchunk_id;

				if (segment_id == InvalidBlockNumber)
					segment_id = block_num / RELSEG_SIZE;
				else if (segment_id != block_num / RELSEG_SIZE)
					goto out;

				fchunk_id = (block_num % RELSEG_SIZE) * PAGES_PER_BLOCK;
				if (strom_ioc != NULL && (strom_ioc->fchunk_id +
										  strom_ioc->nr_pages) == fchunk_id)
				{
					/* expand the iovec entry */
					strom_ioc->nr_pages += PAGES_PER_BLOCK;
				}
				else
				{
					/* add the next iovec entry */
					strom_ioc = &strom_iovec->ioc[strom_iovec->nr_chunks++];
					strom_ioc->m_offset  = m_offset;
					strom_ioc->fchunk_id = fchunk_id;
					strom_ioc->nr_pages  = PAGES_PER_BLOCK;
				}
				kds->nitems++;
				strom_blknums[strom_nblocks++] = block_num;
				m_offset += BLCKSZ;
			}
			else if (pts->ds_entry)
			{
				/*
				 * For DPU devices, it makes no sense to move the data blocks
				 * to the (relatively) poor performance devices instead of CPUs.
				 * So, we run CPU fallback for the tuples in dirty pages.
				 */
				__relScanDirectFallbackBlock(pts, kds, block_num);
			}
			else
			{
				__relScanDirectCachedBlock(pts, block_num);
			}
			pts->curr_block_num++;
		}

		if (kds->nitems >= kds_nrooms)
		{
			/* ok, we cannot load more pages in this chunk */
			break;
		}
		else if (pts->br_state)
		{
			int		__next_repeat_id = pgstromBrinIndexNextChunk(pts);

			if (__next_repeat_id < 0)
				pts->scan_done = true;
			else if (scan_repeat_id < 0)
				scan_repeat_id = __next_repeat_id;
			else if (scan_repeat_id != __next_repeat_id)
				break;
		}
		else
		{
			/* full table scan */
			uint32_t	num_blocks = kds_nrooms - kds->nitems;
			uint64_t	scan_block_limit = (ps_state->scan_block_nums *
											pts->num_scan_repeats);

			pts->curr_block_num  = pg_atomic_fetch_add_u64(&ps_state->scan_block_count,
														   num_blocks);
			pts->curr_block_tail = pts->curr_block_num + num_blocks;
			if (pts->curr_block_num >= scan_block_limit)
				pts->scan_done = true;
			if (pts->curr_block_tail > scan_block_limit)
				pts->curr_block_tail = scan_block_limit;
		}
	}
out:
	Assert(kds->nitems == kds->block_nloaded + strom_nblocks);
	pg_atomic_fetch_add_u64(&ps_state->npages_buffer_read,
							kds->block_nloaded * PAGES_PER_BLOCK);
	kds->length = kds->block_offset + BLCKSZ * kds->nitems;
	if (kds->nitems == 0)
		return NULL;
	Assert(scan_repeat_id >= 0);
	/* XXX - debug message */
	if (scan_repeat_id > 0 && scan_repeat_id != pts->last_repeat_id)
		elog(NOTICE, "direct scan on '%s' moved into %dth loop for inner-buffer partitions (pid: %u)  scan_block_count=%lu scan_block_nums=%u scan_block_start=%u num_scan_repeats=%u scan_repeat_id=%d curr_block_num=%lu curr_block_tail=%lu",
			 RelationGetRelationName(pts->css.ss.ss_currentRelation),
			 scan_repeat_id+1, MyProcPid,
			 pg_atomic_read_u64(&ps_state->scan_block_count),
			 ps_state->scan_block_nums,
			 ps_state->scan_block_start,
			 pts->num_scan_repeats,
			 scan_repeat_id,
			 pts->curr_block_num,
			 pts->curr_block_tail);
	pts->last_repeat_id = scan_repeat_id;

	if (strom_nblocks > 0)
	{
		memcpy(&KDS_BLOCK_BLCKNR(kds, kds->block_nloaded),
			   strom_blknums,
			   sizeof(BlockNumber) * strom_nblocks);
	}
	Assert(kds->nitems == kds->block_nloaded + strom_nblocks);

	if (strom_iovec->nr_chunks > 0)
	{
		size_t		sz;

		kds_src_pathname = pts->xcmd_buf.len;
		appendStringInfoString(&pts->xcmd_buf, pts->kds_pathname);
		if (segment_id > 0)
			appendStringInfo(&pts->xcmd_buf, ".%u", segment_id);
		appendStringInfoChar(&pts->xcmd_buf, '\0');

		sz = offsetof(strom_io_vector, ioc[strom_iovec->nr_chunks]);
		kds_src_iovec = __appendBinaryStringInfo(&pts->xcmd_buf,
												 (const char *)strom_iovec, sz);
	}
	else
	{
		Assert(segment_id == InvalidBlockNumber);
	}
	xcmd = (XpuCommand *)pts->xcmd_buf.data;
	xcmd->u.task.kds_src_pathname = kds_src_pathname;
	xcmd->u.task.kds_src_iovec = kds_src_iovec;
	xcmd->repeat_id = scan_repeat_id;
	xcmd->length = pts->xcmd_buf.len;

	xcmd_iov[0].iov_base = xcmd;
	xcmd_iov[0].iov_len  = xcmd->length;
	*xcmd_iovcnt = 1;

	return xcmd;
}

void
pgstrom_init_relscan(void)
{
	/* nothing to do */
}
