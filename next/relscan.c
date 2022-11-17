/*
 * relscan.c
 *
 * Routines related to outer relation scan
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
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
static int
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

			count += count_num_of_subfields(attr->atttypid);
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
	TypeCacheEntry *tcache;

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

	/* array? composite type? */
	tcache = lookup_type_cache(atttypid, TYPECACHE_TUPDESC);
	if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
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

	memset(kds, 0, offsetof(kern_data_store, colmeta));
	kds->length		= length;
	kds->nitems		= 0;
	kds->usage		= 0;
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
		cmeta->attbyval = true;
		cmeta->attalign = sizeof(int32_t);
		cmeta->attlen = sizeof(GpuCacheSysattr);
		cmeta->attnum = -1;
		cmeta->attcacheoff = -1;
		cmeta->atttypid = InvalidOid;
		cmeta->atttypmod = -1;
		cmeta->atttypkind = TYPE_KIND__BASE;
		strcpy(cmeta->attname, "__gcache_sysattr__");
	}
	return MAXALIGN(offsetof(kern_data_store, colmeta[kds->nr_colmeta]));
}

size_t
estimate_kern_data_store(TupleDesc tupdesc)
{
	int		j, nr_colmeta = tupdesc->natts;

	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		nr_colmeta += count_num_of_subfields(attr->atttypid);
	}
	return MAXALIGN(offsetof(kern_data_store, colmeta[nr_colmeta]));
}

/* ----------------------------------------------------------------
 *
 * Routines to load chunks from storage
 *
 * ----------------------------------------------------------------
 */
#define __XCMD_KDS_SRC_OFFSET(buf)							\
	(((XpuCommand *)((buf)->data))->u.scan.kds_src_offset)
#define __XCMD_GET_KDS_SRC(buf)								\
	((kern_data_store *)((buf)->data + __XCMD_KDS_SRC_OFFSET(buf)))

static void
__relScanDirectFallbackBlock(pgstromTaskState *pts, BlockNumber block_num)
{
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
			pts->cb_cpu_fallback(pts, &htup);
	}
	UnlockReleaseBuffer(buffer);
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
	UnlockReleaseBuffer(buffer);
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
	if (!PageIsAllVisible(dpage) || snapshot->takenDuringRecovery)
	{
		int		lines = PageGetMaxOffsetNumber(dpage);
		ItemId	lpp;
		OffsetNumber lineoff;


		for (lineoff = FirstOffsetNumber, lpp = PageGetItemId(dpage, lineoff);
			 lineoff <= lines;
			 lineoff++, lpp++)
		{
			HeapTupleData htup;
			bool	valid;

			if (!ItemIdIsNormal(lpp))
				continue;
			htup.t_tableOid = RelationGetRelid(relation);
			htup.t_data = (HeapTupleHeader) PageGetItem((Page) dpage, lpp);
			Assert((((uintptr_t)htup.t_data - (uintptr_t)dpage) & 7) == 0);
			htup.t_len = ItemIdGetLength(lpp);
			ItemPointerSet(&htup.t_self, block_num, lineoff);

			valid = HeapTupleSatisfiesVisibility(&htup, snapshot, buffer);
			HeapCheckForSerializableConflictOut(valid, relation, &htup,
												buffer, snapshot);
			if (valid)
				has_valid_tuples = true;
			else
				ItemIdSetUnused(lpp);
		}
	}
	else
	{
		has_valid_tuples = true;
	}

	/*
	 * If no tuples in this block are visible, we don't need to load
	 * them to xPU device (just wast of memory and bandwidth),
	 * so it shall be reverted from the xcmd-buffer.
	 */
	if (has_valid_tuples)
	{
		pts->xcmd_buf.len -= BLCKSZ;
		return;
	}
	/* dpage became all-visible also */
	PageSetAllVisible(dpage);
	kds->nitems++;
	kds->block_nloaded++;
}

/*
 * __relScanDirectAssignFilename
 */
static void
__relScanDirectAssignFilename(StringInfo buf,
							  RelFileNodeBackend *smgr_rnode,
							  BlockNumber segment_id,
							  uint32_t *p_fullpath_offset,
							  uint32_t *p_pathname_offset)
{
	/* see logic in GetRelationPath */
	Assert(segment_id != InvalidBlockNumber);
		
	if (smgr_rnode->node.spcNode == GLOBALTABLESPACE_OID)
	{
		Assert(smgr_rnode->node.dbNode == InvalidOid &&
			   smgr_rnode->backend == InvalidBackendId);
		*p_fullpath_offset = *p_pathname_offset = buf->len;
		appendStringInfo(buf, "global/%u",
						 smgr_rnode->node.relNode);
	}
	else if (smgr_rnode->node.spcNode == DEFAULTTABLESPACE_OID)
	{
		*p_fullpath_offset = *p_pathname_offset = buf->len;
		if (smgr_rnode->backend == InvalidBackendId)
			appendStringInfo(buf, "base/%u/%u",
							 smgr_rnode->node.dbNode,
							 smgr_rnode->node.relNode);
		else
			appendStringInfo(buf, "base/%u/t%d_%u",
							 smgr_rnode->node.dbNode,
							 smgr_rnode->backend,
							 smgr_rnode->node.relNode);
	}
	else
	{
		*p_fullpath_offset = buf->len;
		appendStringInfo(buf, "pg_tblspc/%u/",
						 smgr_rnode->node.spcNode);
		*p_pathname_offset = buf->len;
		if (smgr_rnode->backend == InvalidBackendId)
			appendStringInfo(buf, "%s/%u/%u",
							 TABLESPACE_VERSION_DIRECTORY,
							 smgr_rnode->node.dbNode,
							 smgr_rnode->node.relNode);
		else
			appendStringInfo(buf, "%s/%u/t%d_%u",
							 TABLESPACE_VERSION_DIRECTORY,
							 smgr_rnode->node.dbNode,
							 smgr_rnode->backend,
							 smgr_rnode->node.relNode);
	}
	if (segment_id != 0)
		appendStringInfo(buf, ".%u", segment_id);
}

XpuCommand *
pgstromRelScanChunkDirect(pgstromTaskState *pts,
						  struct iovec *xcmd_iov, int *xcmd_iovcnt)
{
	Relation		relation = pts->css.ss.ss_currentRelation;
	HeapScanDesc    h_scan = (HeapScanDesc)pts->css.ss.ss_currentScanDesc;
	/* NOTE: 'smgr_rnode' always locates on the head of SMgrRelationData */
	RelFileNodeBackend *smgr_rnode = (RelFileNodeBackend *)RelationGetSmgr(relation);
	XpuCommand	   *xcmd;
	kern_data_store *kds;
	unsigned long	m_offset = 0UL;
	BlockNumber		segment_id = InvalidBlockNumber;
	strom_io_vector *strom_iovec;
	strom_io_chunk *strom_ioc = NULL;
	BlockNumber	   *strom_blknums;
	uint32_t		strom_nblocks = 0;
	uint32_t		kds_src_fullpath = 0;
	uint32_t		kds_src_pathname = 0;
	uint32_t		kds_src_iovec = 0;
	uint32_t		kds_nrooms;

	kds = __XCMD_GET_KDS_SRC(&pts->xcmd_buf);
	kds_nrooms = (PGSTROM_CHUNK_SIZE -
				  KDS_HEAD_LENGTH(kds)) / (sizeof(BlockNumber) + BLCKSZ);
	kds->nitems  = 0;
	kds->usage   = 0;
	kds->block_offset = (KDS_HEAD_LENGTH(kds) +
						 MAXALIGN(sizeof(BlockNumber) * kds_nrooms));
	kds->block_nloaded = 0;
	pts->xcmd_buf.len = __XCMD_KDS_SRC_OFFSET(&pts->xcmd_buf) + kds->block_offset;
	Assert(pts->xcmd_buf.len == MAXALIGN(pts->xcmd_buf.len));
	enlargeStringInfo(&pts->xcmd_buf, 0);
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
				= (pts->curr_block_num + h_scan->rs_startblock) % h_scan->rs_nblocks;
			/*
			 * MEMO: Usually, CPU is more powerful device than DPU.
			 * In case when the source cache is already on the shared-
			 * buffer, it makes no sense to handle this page on the
			 * DPU device.
			 */
			if (pts->ds_entry && pgstrom_dpu_handle_cached_pages)
			{
				BufferTag	bufTag;
				uint32		bufHash;
				LWLock	   *bufLock;
				int			buf_id;

				INIT_BUFFERTAG(bufTag, smgr_rnode->node, MAIN_FORKNUM, block_num);
				bufHash = BufTableHashCode(&bufTag);
				bufLock = BufMappingPartitionLock(bufHash);

				/* check whether the block exists on the shared buffer? */
				LWLockAcquire(bufLock, LW_SHARED);
				buf_id = BufTableLookup(&bufTag, bufHash);
				if (buf_id >= 0)
				{
					LWLockRelease(bufLock);
					__relScanDirectFallbackBlock(pts, block_num);
					pts->curr_block_num++;
					continue;
				}
				LWLockRelease(bufLock);
			}
			
			/*
			 * MEMO: right now, we allow GPU Direct SQL for the all-visible
			 * pages only, due to the restrictions about MVCC checks.
			 * However, it is too strict for the purpose. If we would have
			 * a mechanism to perform MVCC checks without commit logs.
			 * In other words, if all the tuples in a certain page have
			 * HEAP_XMIN_* or HEAP_XMAX_* flags correctly, we can have MVCC
			 * logic in the device code.
			 */
			if (VM_ALL_VISIBLE(relation, block_num, &pts->curr_vm_buffer))
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
				__relScanDirectFallbackBlock(pts, block_num);
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
			if (!pgstromBrinIndexNextChunk(pts))
				pts->scan_done = true;
		}
		else if (!h_scan->rs_base.rs_parallel)
		{
			/* single process scan */
			BlockNumber		num_blocks = kds_nrooms - kds->nitems;

			if (!h_scan->rs_inited)
			{
				h_scan->rs_cblock = 0;
				h_scan->rs_inited = true;
			}
			pts->curr_block_num = h_scan->rs_cblock;
			if (pts->curr_block_num >= h_scan->rs_nblocks)
				pts->scan_done = true;
			else if (pts->curr_block_num + num_blocks > h_scan->rs_nblocks)
				num_blocks = h_scan->rs_nblocks - pts->curr_block_num;
			h_scan->rs_cblock += num_blocks;
			pts->curr_block_tail = pts->curr_block_num + num_blocks;
		}
		else
		{
			/* parallel processes scan */
			ParallelBlockTableScanDesc pb_scan =
				(ParallelBlockTableScanDesc)h_scan->rs_base.rs_parallel;
			BlockNumber		num_blocks = kds_nrooms - kds->nitems;

			if (!h_scan->rs_inited)
			{
				/* see table_block_parallelscan_startblock_init */
				BlockNumber	start_block = InvalidBlockNumber;

			retry_parallel_init:
				SpinLockAcquire(&pb_scan->phs_mutex);
				if (pb_scan->phs_startblock == InvalidBlockNumber)
				{
					if (!pb_scan->base.phs_syncscan)
						pb_scan->phs_startblock = 0;
					else if (start_block != InvalidBlockNumber)
						pb_scan->phs_startblock = start_block;
					else
					{
						SpinLockRelease(&pb_scan->phs_mutex);
						start_block = ss_get_location(relation, pb_scan->phs_nblocks);
						goto retry_parallel_init;
					}
				}
				h_scan->rs_nblocks = pb_scan->phs_nblocks;
				h_scan->rs_startblock = pb_scan->phs_startblock;
				SpinLockRelease(&pb_scan->phs_mutex);
				h_scan->rs_inited = true;
			}
			pts->curr_block_num = pg_atomic_fetch_add_u64(&pb_scan->phs_nallocated,
														  num_blocks);
			if (pts->curr_block_num >= h_scan->rs_nblocks)
				pts->scan_done = true;
			else if (pts->curr_block_num + num_blocks > h_scan->rs_nblocks)
				num_blocks = h_scan->rs_nblocks - pts->curr_block_num;
			pts->curr_block_tail = pts->curr_block_num + num_blocks;
		}
	}
out:
	Assert(kds->nitems == kds->block_nloaded + strom_nblocks);
	kds->length = kds->block_offset + BLCKSZ * kds->nitems;
	if (kds->nitems == 0)
		return NULL;
	if (strom_iovec->nr_chunks > 0)
	{
		size_t		sz;

		__relScanDirectAssignFilename(&pts->xcmd_buf,
									  smgr_rnode,
									  segment_id,
									  &kds_src_fullpath,
									  &kds_src_pathname);
		sz = offsetof(strom_io_vector, ioc[strom_iovec->nr_chunks]);
		kds_src_iovec = __appendBinaryStringInfo(&pts->xcmd_buf,
												 (const char *)strom_iovec, sz);
	}
	else
	{
		Assert(segment_id == InvalidBlockNumber);
	}
	xcmd = (XpuCommand *)pts->xcmd_buf.data;
	xcmd->u.scan.kds_src_fullpath = kds_src_fullpath;
	xcmd->u.scan.kds_src_pathname = kds_src_pathname;
	xcmd->u.scan.kds_src_iovec = kds_src_iovec;
	xcmd->length = pts->xcmd_buf.len;

	xcmd_iov[0].iov_base = xcmd;
	xcmd_iov[0].iov_len  = xcmd->length;
	*xcmd_iovcnt = 1;
	
	return xcmd;
}

static bool
__kds_row_insert_tuple(kern_data_store *kds, TupleTableSlot *slot)
{
	uint32_t   *rowindex = KDS_GET_ROWINDEX(kds);
	HeapTuple	tuple;
	size_t		sz, __usage;
	bool		should_free;
	kern_tupitem *titem;

	Assert(kds->format == KDS_FORMAT_ROW && kds->hash_nslots == 0);
	tuple = ExecFetchSlotHeapTuple(slot, false, &should_free);

	__usage = (__kds_unpack(kds->usage) +
			   MAXALIGN(offsetof(kern_tupitem, htup) + tuple->t_len));
	sz = KDS_HEAD_LENGTH(kds) + sizeof(uint32_t) * (kds->nitems + 1) + __usage;
	if (sz > kds->length)
		return false;	/* no more items! */
	titem = (kern_tupitem *)((char *)kds + kds->length - __usage);
	titem->t_len = tuple->t_len;
	titem->rowid = kds->nitems;
	memcpy(&titem->htup, tuple->t_data, tuple->t_len);
	kds->usage = rowindex[kds->nitems++] = __kds_packed(__usage);

	if (should_free)
		heap_freetuple(tuple);
	ExecClearTuple(slot);

	return true;
}

XpuCommand *
pgstromRelScanChunkNormal(pgstromTaskState *pts,
						  struct iovec *xcmd_iov, int *xcmd_iovcnt)
{
	EState		   *estate = pts->css.ss.ps.state;
	TableScanDesc	scan = pts->css.ss.ss_currentScanDesc;
	TupleTableSlot *slot = pts->base_slot;
	kern_data_store *kds;
	XpuCommand	   *xcmd;
	size_t			sz1, sz2;

	pts->xcmd_buf.len = __XCMD_KDS_SRC_OFFSET(&pts->xcmd_buf) + PGSTROM_CHUNK_SIZE;
	enlargeStringInfo(&pts->xcmd_buf, 0);
	kds = __XCMD_GET_KDS_SRC(&pts->xcmd_buf);
	kds->nitems = 0;
	kds->usage  = 0;
	kds->length = PGSTROM_CHUNK_SIZE;

	if (pts->br_state)
	{
		/* scan by BRIN index */
		while (!pts->scan_done)
		{
			if (!pts->curr_tbm)
			{
				TBMIterateResult *next_tbm = pgstromBrinIndexNextBlock(pts);

				if (!next_tbm)
				{
					pts->scan_done = true;
					break;
				}
				if (!table_scan_bitmap_next_block(scan, next_tbm))
					elog(ERROR, "failed on table_scan_bitmap_next_block");
				pts->curr_tbm = next_tbm;
			}
			if (!TTS_EMPTY(slot) &&
				!__kds_row_insert_tuple(kds, slot))
				break;
			if (!table_scan_bitmap_next_tuple(scan, pts->curr_tbm, slot))
				pts->curr_tbm = NULL;
			else if (!__kds_row_insert_tuple(kds, slot))
				break;
		}
	}
	else
	{
		/* full table scan */
		while (!pts->scan_done)
		{
			if (!TTS_EMPTY(slot) &&
				!__kds_row_insert_tuple(kds, slot))
				break;
			if (!table_scan_getnextslot(scan, estate->es_direction, slot))
			{
				pts->scan_done = true;
				break;
			}
			if (!__kds_row_insert_tuple(kds, slot))
				break;
		}
	}

	if (kds->nitems == 0)
		return NULL;

	/* setup iovec that may skip the hole between row-index and tuples-buffer */
	sz1 = ((KDS_BODY_ADDR(kds) - pts->xcmd_buf.data) +
		   MAXALIGN(sizeof(uint32_t) * kds->nitems));
	sz2 = __kds_unpack(kds->usage);
	Assert(sz1 + sz2 <= pts->xcmd_buf.len);
	kds->length = (KDS_HEAD_LENGTH(kds) +
				   MAXALIGN(sizeof(uint32_t) * kds->nitems) + sz2);
	xcmd = (XpuCommand *)pts->xcmd_buf.data;
	xcmd->length = sz1 + sz2;
	xcmd_iov[0].iov_base = xcmd;
	xcmd_iov[0].iov_len  = sz1;
	xcmd_iov[1].iov_base = (pts->xcmd_buf.data + pts->xcmd_buf.len - sz2);
	xcmd_iov[1].iov_len  = sz2;
	*xcmd_iovcnt = 2;

	return xcmd;
}

/*
 * pgstromSharedStateEstimateDSM
 */
Size
pgstromSharedStateEstimateDSM(pgstromTaskState *pts)
{
	EState	   *estate = pts->css.ss.ps.state;
	Snapshot	snapshot = estate->es_snapshot;
	Relation	relation = pts->css.ss.ss_currentRelation;
	Size		len = 0;

	if (pts->br_state)
		len += pgstromBrinIndexEstimateDSM(pts);
	len += MAXALIGN(sizeof(pgstromSharedState) +
					table_parallelscan_estimate(relation, snapshot));
	return len;
}

/*
 * pgstromSharedStateInitDSM
 */
void
pgstromSharedStateInitDSM(pgstromTaskState *pts, char *dsm_addr)
{
	pgstromSharedState *ps_state;
	Relation	relation = pts->css.ss.ss_currentRelation;
	EState	   *estate = pts->css.ss.ps.state;
	TableScanDesc scan;

	if (pts->br_state)
		dsm_addr += pgstromBrinIndexInitDSM(pts, dsm_addr);

	Assert(!pts->css.ss.ss_currentScanDesc);
	if (dsm_addr)
	{
		ps_state = (pgstromSharedState *)dsm_addr;
		memset(ps_state, 0, offsetof(pgstromSharedState, bpscan));
		table_parallelscan_initialize(relation, &ps_state->bpscan.base,
									  estate->es_snapshot);
		scan = table_beginscan_parallel(relation, &ps_state->bpscan.base);
	}
	else
	{
		ps_state = MemoryContextAllocZero(estate->es_query_cxt,
										  sizeof(pgstromSharedState));
		scan = table_beginscan(relation, estate->es_snapshot, 0, NULL);
	}
	pts->ps_state = ps_state;
	pts->css.ss.ss_currentScanDesc = scan;
}

/*
 * pgstromSharedStateReInitDSM
 */
void
pgstromSharedStateReInitDSM(pgstromTaskState *pts)
{
	//pgstromSharedState *ps_state = pts->ps_state;
	if (pts->br_state)
		pgstromBrinIndexReInitDSM(pts);
}

/*
 * pgstromSharedStateAttachDSM
 */
void
pgstromSharedStateAttachDSM(pgstromTaskState *pts, char *dsm_addr)
{
	if (pts->br_state)
		dsm_addr += pgstromBrinIndexAttachDSM(pts, dsm_addr);
	pts->ps_state = (pgstromSharedState *)dsm_addr;
	pts->css.ss.ss_currentScanDesc =
		table_beginscan_parallel(pts->css.ss.ss_currentRelation,
								 &pts->ps_state->bpscan.base);	
}

/*
 * pgstromSharedStateShutdownDSM
 */
void
pgstromSharedStateShutdownDSM(pgstromTaskState *pts)
{
	pgstromSharedState *src_state = pts->ps_state;
	pgstromSharedState *dst_state;
	EState	   *estate = pts->css.ss.ps.state;

	if (pts->br_state)
		pgstromBrinIndexShutdownDSM(pts);
	if (src_state)
	{
		dst_state = MemoryContextAllocZero(estate->es_query_cxt,
										   sizeof(pgstromSharedState));
		memcpy(dst_state, src_state, sizeof(pgstromSharedState));
		pts->ps_state = dst_state;
	}
}

void
pgstrom_init_relscan(void)
{
	/* nothing to do */
}
