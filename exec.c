/*
 * exec.c
 *
 * Executor routines of PG-Strom
 *
 *
 *
 */
#include "postgres.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/pg_lzcompress.h"
#include "utils/rel.h"
#include "utils/resowner.h"
#include "pg_strom.h"

typedef struct {
	int				chkbh_exec_cnt;	/* counter of in-exec chunks */
	ShmsegList		chkbh_chain;	/* chain to local_chunk_buffer_head */
	ShmsegQueue		chkbh_recvq;	/* receive queue of this scan */
	ResourceOwner	chkbh_owner;	/* resource owner of this scan */
} ChunkBufferHead;

typedef struct {
	/* parameters come from planner stage */
	bytea		   *gpu_cmds;		/* command series for OpenCL */
	bytea		   *cpu_cmds;		/* command series for OpenMP */
	Bitmapset	   *gpu_cols;		/* columns used for OpenCL */
	Bitmapset	   *cpu_cols;		/* columns used for OpenMP */
	Bitmapset	   *required_cols;	/* columns to be returned */

	/* shadow tables, indexes and scan descriptors */
	Relation		ft_rel;			/* foreign table managed by PG-Strom */
	Relation		id_rel;			/* shadow rowid table */
	Relation	   *cs_rels;		/* shadow column-store tables */
	Relation	   *cs_idxs;		/* shadow column-store indexes */
	HeapScanDesc	id_scan;		/* scan on rowid table */
	IndexScanDesc  *cs_scan;		/* scan on column-store with index */
	MemoryContext	es_qrycxt;		/* per-query memory context */

	ChunkBufferHead	*chkb_head;		/* reference to shared memory object */
	ShmsegList		chkb_ready_list;/* local link to ready chunks */
	ShmsegList		chkb_free_list;	/* local link to free chunks */

	/* stuff related to scan */
	ChunkBuffer	   *curr_chunk;
	int				curr_index;
	bytea		  **curr_cs_isnull;
	bytea		  **curr_cs_values;
	int64		   *curr_cs_rowid_min;
	int64		   *curr_cs_rowid_max;
} PgStromExecState;

/*
 * Local declarations
 */
static ShmsegList	local_chunk_buffer_head;
static int			pgstrom_max_async_chunks;
static int			pgstrom_min_async_chunks;
static bool			pgstrom_exec_profile;

/*
 * An error handler of the executor of PG-Strom. All the ChunkBuffer related
 * to the current resource owner shall be released at this timing.
 */
static void
pgstrom_release_resources(ResourceReleasePhase phase,
						  bool isCommit,
						  bool isTopLevel,
						  void *arg)
{
	ChunkBufferHead	*curr;
	ChunkBufferHead	*next;

	if (phase != RESOURCE_RELEASE_AFTER_LOCKS)
		return;

	pgstrom_shmseg_list_foreach_safe(curr, next, chkbh_chain,
									 &local_chunk_buffer_head)
	{
		if (curr->chkbh_owner == CurrentResourceOwner)
		{
			pgstrom_shmseg_list_delete(&curr->chkbh_chain);

			if (curr->chkbh_exec_cnt > 0)
			{
				/* TODO: completion of in-exec chunks */
			}
			pgstrom_shmseg_free(curr);
		}
	}
}

/*
 * deconstruct_cs_bytea
 *
 * It deconstruct the supplied bytea varlena data on the supplied
 * destination address. In case of the varlena being compressed,
 * this routine also decompress the source data.
 * Unlike pg_detoast_datum(), it does not require a buffer to
 * decompress, so it allows to extract the compressed array on
 * page-locked buffer that is available to copy by DMA.
 */
static void
deconstruct_cs_bytea(void *dest, Datum cs_bytea, uint32 length_be)
{
	bytea  *temp = (bytea *)DatumGetPointer(cs_bytea);

	/* we don't allow to save contents of column-store externally */
	Assert(!VARATT_IS_EXTERNAL(temp));

	if (VARATT_IS_COMPRESSED(temp))
	{
		PGLZ_Header *lzhd = (PGLZ_Header *) temp;
		Assert(PGLZ_RAW_SIZE(lzhd) == length_be);
		pglz_decompress(lzhd, dest);
	}
	else
	{
		Assert(VARSIZE_ANY_EXHDR(temp) == length_be);
		memcpy(dest, VARDATA_ANY(temp), VARSIZE_ANY_EXHDR(temp));
	}
}

static void
pgstrom_load_column_store(PgStromExecState *sestate,
						  ChunkBuffer *chunk, AttrNumber attno)
{
	ScanKeyData	skeys[2];
	TupleDesc	tupdesc;
	HeapTuple	tup;
	int			attlen;

	attlen = RelationGetDescr(sestate->ft_rel)->attrs[attno - 1]->attlen;
	Assert(attlen > 0);

	/*
	 * Null-bitmap shall be initialized as if all the values are NULL
	 */
	memset(chunk_cs_isnull(chunk, attno), -1, chunk->nitems / BITS_PER_BYTE);

	/*
	 * Scan the column store with cs_rowid between rowid and (rowid + nitems)
	 */
	ScanKeyInit(&skeys[0],
				Anum_pg_strom_cs_rowid,
				BTGreaterEqualStrategyNumber, F_INT8GE,
				Int64GetDatum(chunk->rowid));
	ScanKeyInit(&skeys[1],
				Anum_pg_strom_cs_rowid,
				BTLessStrategyNumber, F_INT8LT,
				Int64GetDatum(chunk->rowid + chunk->nitems));

	index_rescan(sestate->cs_scan[attno-1], skeys, 2, NULL, 0);

	tupdesc = RelationGetDescr(sestate->cs_rels[attno - 1]);

	while (HeapTupleIsValid(tup = index_getnext(sestate->cs_scan[attno-1],
												ForwardScanDirection)))
	{
		Datum	values[Natts_pg_strom_cs];
		bool	isnull[Natts_pg_strom_cs];
		int64	curr_rowid;
		int32	curr_nitems;
		Size	offset;

		heap_deform_tuple(tup, tupdesc, values, isnull);
		Assert(!isnull[Anum_pg_strom_cs_rowid - 1] &&
			   !isnull[Anum_pg_strom_cs_nitems - 1] &&
			   !isnull[Anum_pg_strom_cs_values - 1]);

		curr_rowid = DatumGetInt64(values[Anum_pg_strom_cs_rowid - 1]);
		curr_nitems = DatumGetInt32(values[Anum_pg_strom_cs_nitems - 1]);
		offset = curr_rowid - chunk->rowid;

		Assert(curr_nitems % BITS_PER_BYTE == 0);
		Assert(offset + curr_nitems <= chunk->nitems);

		if (!isnull[Anum_pg_strom_cs_isnull - 1])
		{
			deconstruct_cs_bytea(chunk_cs_isnull(chunk, attno)
								 + offset / BITS_PER_BYTE,
								 values[Anum_pg_strom_cs_isnull - 1],
								 curr_nitems / BITS_PER_BYTE);
		}
		else
		{
			/*
			 * In case of 'isnull' == NULL, it means all the items
			 * in the 'values' are valid, so we clear the related
			 * scope of ths null-bitmap.
			 */
			memset(chunk_cs_isnull(chunk, attno) + offset / BITS_PER_BYTE,
				   0, curr_nitems / BITS_PER_BYTE);
		}
		deconstruct_cs_bytea(chunk_cs_values(chunk, attno)
							 + offset * attlen,
							 values[Anum_pg_strom_cs_values - 1],
							 curr_nitems * attlen);
	}
}

static bool
pgstrom_load_chunk(PgStromExecState *sestate)
{
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	Datum		values[Natts_pg_strom_rmap];
	bool		isnull[Natts_pg_strom_rmap];
	int64		rowid;
	uint32		nitems;
	Datum		rowmap;
	Size		offset;
	Bitmapset  *tempset;
	AttrNumber	attno;
	int			attlen;
	ChunkBuffer	*chunk;

	/* no chunks to read any more */
	if (!sestate->id_scan)
		return false;

	tuple = heap_getnext(sestate->id_scan, ForwardScanDirection);
	if (!HeapTupleIsValid(tuple))
	{
		heap_endscan(sestate->id_scan);
		sestate->id_scan = NULL;
		return false;
	}

	tupdesc = RelationGetDescr(sestate->id_rel);
	heap_deform_tuple(tuple, tupdesc, values, isnull);
	Assert(!isnull[Anum_pg_strom_rmap_rowid - 1] &&
		   !isnull[Anum_pg_strom_rmap_nitems - 1] &&
		   !isnull[Anum_pg_strom_rmap_rowmap - 1]);

	rowid = DatumGetInt64(values[Anum_pg_strom_rmap_rowid - 1]);
	nitems = DatumGetUInt32(values[Anum_pg_strom_rmap_nitems - 1]);
	rowmap = values[Anum_pg_strom_rmap_rowmap - 1];

	Assert(nitems % BITS_PER_BYTE == 0);

	/* caller must not invoke this routine without free chunks */
	Assert(!pgstrom_shmseg_list_empty(&sestate->chkb_free_list));
	chunk = container_of(sestate->chkb_free_list.next, ChunkBuffer, chain);
	pgstrom_shmseg_list_delete(&chunk->chain);

	/*
	 * Setup chunk buffer
	 */
	Assert(chunk->status == CHUNKBUF_STATUS_FREE);
	chunk->recv_cmdq = &sestate->chkb_head->chkbh_recvq;
	chunk->nattrs = RelationGetNumberOfAttributes(sestate->ft_rel);
	chunk->rowid = rowid;
	chunk->nitems = nitems;

	memset(chunk->cs_isnull, 0, sizeof(int) * chunk->nattrs);
	memset(chunk->cs_values, 0, sizeof(int) * chunk->nattrs);

	tupdesc = RelationGetDescr(sestate->ft_rel);

	offset = MAXALIGN(chunk->nitems / BITS_PER_BYTE);

	tempset = bms_copy(sestate->gpu_cols);
	while ((attno = bms_first_member(tempset)) > 0)
	{
		attlen = tupdesc->attrs[attno - 1]->attlen;
		Assert(attlen > 0);

		chunk->cs_values[attno - 1] = offset;
		offset += MAXALIGN(chunk->nitems * attlen);
		chunk->cs_isnull[attno - 1] = offset;
		offset += MAXALIGN(chunk->nitems / BITS_PER_BYTE);
	}
	bms_free(tempset);

	chunk->dma_length = MAXALIGN(chunk->gpu_cmds_len) +
		MAXALIGN(sizeof(int) * chunk->nattrs) +
		MAXALIGN(sizeof(int) * chunk->nattrs) +
		MAXALIGN(offset);

	tempset = bms_copy(sestate->cpu_cols);
	while ((attno = bms_first_member(tempset)) > 0)
	{
		if (bms_is_member(attno, sestate->gpu_cols))
			continue;

		attlen = tupdesc->attrs[attno - 1]->attlen;
		Assert(attlen > 0);

		chunk->cs_values[attno - 1] = offset;
		offset += MAXALIGN(chunk->nitems * attlen);
		chunk->cs_isnull[attno - 1] = offset;
		offset += MAXALIGN(chunk->nitems / BITS_PER_BYTE);
	}
	bms_free(tempset);

	/*
	 * Load referenced column store
	 */
	deconstruct_cs_bytea(chunk->cs_rowmap, rowmap,
						 chunk->nitems / BITS_PER_BYTE);
	tempset = bms_union(sestate->gpu_cols, sestate->cpu_cols);
	while ((attno = bms_first_member(tempset)) > 0)
		pgstrom_load_column_store(sestate, chunk, attno);
	bms_free(tempset);

	/*
	 * Enqueue the chunk-buffer for asynchronous execution
	 */
	if (chunk->gpu_cmds)
	{
		chunk->status = CHUNKBUF_STATUS_EXEC;
		sestate->chkb_head->chkbh_exec_cnt++;
		pgstrom_gpu_enqueue_chunk(chunk);
	}
	else if (chunk->cpu_cmds)
	{
		chunk->status = CHUNKBUF_STATUS_EXEC;
        sestate->chkb_head->chkbh_exec_cnt++;
		pgstrom_openmp_enqueue_chunk(chunk);
	}
	else
	{
		chunk->status = CHUNKBUF_STATUS_READY;
		pgstrom_shmseg_list_add(&sestate->chkb_ready_list, &chunk->chain);
	}

	return true;
}

#if SIZEOF_DATUM == 8
#define SHIFT_PER_DATUM		6
#elif SIZEOF_DATUM == 4
#define SHIFT_PER_DATUM		5
#else
#error "sizeof(Datum) should be either 32bits or 64bits"
#endif

static void
pgstrom_scan_column_store(PgStromExecState *sestate, TupleTableSlot *slot,
						  AttrNumber attno, int64 rowid)
{
	MemoryContext	oldcxt;
	Datum  *nullmap;
	int		index;
	int		index_l;
	int		index_h;

	if (rowid < sestate->curr_cs_rowid_min[attno-1] ||
		rowid > sestate->curr_cs_rowid_max[attno-1])
	{
		ScanKeyData	skey;
		TupleDesc	tupdesc;
		HeapTuple	tuple;
		Datum		values[Natts_pg_strom_cs];
		bool		isnull[Natts_pg_strom_cs];
		int64		curr_rowid;
		int			curr_nitems;

		/*
		 * Reset cached values
		 */
		if (sestate->curr_cs_values[attno-1])
		{
			pfree(sestate->curr_cs_values[attno-1]);
			if (sestate->curr_cs_isnull[attno-1])
				pfree(sestate->curr_cs_isnull[attno-1]);
			sestate->curr_cs_values[attno-1] = NULL;
			sestate->curr_cs_isnull[attno-1] = NULL;
			sestate->curr_cs_rowid_min[attno-1] = -1;
			sestate->curr_cs_rowid_max[attno-1] = -1;
		}

		/*
		 * Rewind the current index scan to fetch a tuple of column-store
		 * that contins the required rowid.
		 */
		ScanKeyInit(&skey,
					Anum_pg_strom_cs_rowid,
					BTLessEqualStrategyNumber, F_INT8LE,
					Int64GetDatum(rowid));
		index_rescan(sestate->cs_scan[attno-1], &skey, 1, NULL, 0);

		tuple = index_getnext(sestate->cs_scan[attno-1],
							  BackwardScanDirection);
		if (!HeapTupleIsValid(tuple))
		{
			slot->tts_isnull[attno-1] = true;
			slot->tts_values[attno-1] = (Datum) 0;
			return;
		}

		tupdesc = RelationGetDescr(sestate->cs_rels[attno-1]);
		heap_deform_tuple(tuple, tupdesc, values, isnull);
		Assert(!isnull[Anum_pg_strom_cs_rowid - 1] &&
			   !isnull[Anum_pg_strom_cs_nitems - 1] &&
			   !isnull[Anum_pg_strom_cs_values - 1]);

		curr_rowid = DatumGetInt64(values[Anum_pg_strom_cs_rowid - 1]);
		curr_nitems = DatumGetInt32(values[Anum_pg_strom_cs_nitems - 1]);
		if (rowid < curr_rowid || rowid >= curr_rowid + curr_nitems)
		{
			slot->tts_isnull[attno-1] = true;
			slot->tts_values[attno-1] = (Datum) 0;
			return;
		}

		/*
		 * Found the required tuple, so save it on the exec-state
		 */
		oldcxt = MemoryContextSwitchTo(sestate->es_qrycxt);
		if (!isnull[Anum_pg_strom_cs_isnull - 1])
			sestate->curr_cs_isnull[attno-1] =
				PG_DETOAST_DATUM_COPY(values[Anum_pg_strom_cs_isnull - 1]);
		else
			sestate->curr_cs_isnull[attno-1] = NULL;
		sestate->curr_cs_values[attno-1] =
			PG_DETOAST_DATUM_COPY(values[Anum_pg_strom_cs_values - 1]);
		sestate->curr_cs_rowid_min[attno-1] = curr_rowid;
		sestate->curr_cs_rowid_max[attno-1] = curr_rowid + curr_nitems - 1;
		MemoryContextSwitchTo(oldcxt);

		Assert(rowid >= sestate->curr_cs_rowid_min[attno-1] &&
			   rowid <= sestate->curr_cs_rowid_max[attno-1]);
	}
	index = rowid - sestate->curr_cs_rowid_min[attno-1];
	index_h = index >> SHIFT_PER_DATUM;
	index_l = index & ((1 << SHIFT_PER_DATUM) - 1);

	Assert(sestate->curr_cs_values[attno-1] != NULL);

	if (sestate->curr_cs_isnull[attno-1])
		nullmap = (Datum *)VARDATA(sestate->curr_cs_isnull[attno-1]);
	else
		nullmap = NULL;

	if (nullmap == NULL || (nullmap[index_h] & (1 << index_l)) == 0)
	{
		Form_pg_attribute	attr
			= slot->tts_tupleDescriptor->attrs[attno-1];

		if (attr->attlen > 0)
		{
			slot->tts_values[attno-1]
				= fetch_att(VARDATA(sestate->curr_cs_values[attno-1]) +
							attr->attlen * index,
							attr->attbyval, attr->attlen);
		}
		else
		{
			char   *temp = VARDATA(sestate->curr_cs_values[attno-1]);
			int		offset = ((uint16 *)temp)[index];

			Assert(offset > 0);
			slot->tts_values[attno-1] = PointerGetDatum(temp + offset);
		}
		slot->tts_isnull[attno-1] = false;
	}
	else
	{
		slot->tts_isnull[attno-1] = true;
        slot->tts_values[attno-1] = (Datum) 0;
	}
}

static bool
pgstrom_scan_chunk(PgStromExecState *sestate, TupleTableSlot *slot)
{
	ChunkBuffer	*chunk = sestate->curr_chunk;
	int			index;
	int			index_h;
	int			index_l;
	Datum		rowmap;
	AttrNumber	attno;

	while (sestate->curr_index < chunk->nitems)
	{
		index_h = (sestate->curr_index >> SHIFT_PER_DATUM);
		index_l = (sestate->curr_index & ((1 << SHIFT_PER_DATUM) - 1));

		rowmap = ((Datum *)chunk->cs_rowmap)[index_h];
#if 0
		index_l = ffsl(~(rowmap | ((1 << index_l) - 1)));
		if (index_l == 0)
		{
			sestate->curr_index = (index_h + 1) << SHIFT_PER_DATUM;
			continue;
		}
		index = (index_h << SHIFT_PER_DATUM) | (index_l - 1);
		if (index >= chunk->nitems)
			return false;
#else
		index = sestate->curr_index;
		if ((rowmap & (1UL << index_l)) != 0)
		{
			sestate->curr_index++;
			continue;
		}
		//elog(INFO, "chunk = %p index = %d rowmap = %016lx", chunk, sestate->curr_index, rowmap);
#endif

		for (attno = 1; attno <= chunk->nattrs; attno++)
		{
			/*
			 * No need to set a valid datum on unreferenced column
			 */
			if (!bms_is_member(attno, sestate->required_cols))
			{
				slot->tts_isnull[attno-1] = true;
				slot->tts_values[attno-1] = (Datum) 0;
				continue;
			}

			/*
			 * No need to scan the column-store again, if this column
			 * was already loaded onto the chunk-buffer. All we need to
			 * do is picking up an appropriate value from the chunk-
			 * buffer.
			 */
			if (chunk->cs_values[attno-1] > 0)
			{
				Datum  *nullmap = (Datum *) chunk_cs_isnull(chunk, attno);

				if ((nullmap[index_h] & (1 << index_l)) == 0)
				{
					Form_pg_attribute	attr
						= slot->tts_tupleDescriptor->attrs[attno-1];
					slot->tts_isnull[attno-1] = false;
					slot->tts_values[attno-1] =
						fetchatt(attr, (chunk_cs_values(chunk,attno) +
										index * attr->attlen));
				}
				else
				{
					slot->tts_isnull[attno-1] = true;
					slot->tts_values[attno-1] = (Datum) 0;
				}
				continue;
			}
			/*
			 * Elsewhere, we scan the column-store with the current rowid
			 */
			pgstrom_scan_column_store(sestate, slot, attno,
									  chunk->rowid + index);
		}
		ExecStoreVirtualTuple(slot);

		/* update next index to be fetched */
		sestate->curr_index = index + 1;
		return true;
	}
	return false;
}

static void
pgstrom_init_chunk_buffers(PgStromExecState *sestate)
{
	ChunkBufferHead	*chunk_head;
	AttrNumber	attno, nattrs;
	Bitmapset  *tempset;
	Size		buffer_sz;
	Size		chunk_sz;
	char	   *pos;
	int			i, num_chunks = pgstrom_max_async_chunks;

	chunk_sz = MAXALIGN(sizeof(ChunkBuffer));
	if (sestate->gpu_cmds)
		chunk_sz += MAXALIGN(VARSIZE_ANY_EXHDR(sestate->gpu_cmds));
	if (sestate->cpu_cmds)
		chunk_sz += MAXALIGN(VARSIZE_ANY_EXHDR(sestate->cpu_cmds));
	nattrs = RelationGetNumberOfAttributes(sestate->ft_rel);
	chunk_sz += 2 * MAXALIGN(sizeof(int) * nattrs);

	buffer_sz = MAXALIGN(PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
	tempset = bms_union(sestate->gpu_cols, sestate->cpu_cols);
	while ((attno = bms_first_member(tempset)) > 0)
	{
		Form_pg_attribute	attr
			= RelationGetDescr(sestate->ft_rel)->attrs[attno - 1];
		Assert(attr->attlen > 0);
		buffer_sz += MAXALIGN(PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
		buffer_sz += MAXALIGN(attr->attlen * PGSTROM_CHUNK_SIZE);
	}
	bms_free(tempset);
	chunk_sz += buffer_sz;

	chunk_head = pgstrom_shmseg_alloc(MAXALIGN(sizeof(ChunkBufferHead)) +
									  MAXALIGN(chunk_sz) * num_chunks);
	if (!chunk_head)
		elog(ERROR, "PG-Strom: out of shared memory");
	if (!pgstrom_shmqueue_init(&chunk_head->chkbh_recvq))
	{
		pgstrom_shmseg_free(chunk_head);
		elog(ERROR, "PG-Strom: failed to init shmqueue");
	}
	chunk_head->chkbh_exec_cnt = 0;
	chunk_head->chkbh_owner = CurrentResourceOwner;

	sestate->chkb_head = chunk_head;

	pgstrom_shmseg_list_add(&local_chunk_buffer_head,
							&chunk_head->chkbh_chain);

	pos = ((char *)chunk_head) + MAXALIGN(sizeof(ChunkBufferHead));
	for (i=0; i < num_chunks; i++)
	{
		ChunkBuffer *chunk = (ChunkBuffer *) MAXALIGN(pos);

		pos += MAXALIGN(sizeof(ChunkBuffer));

		if (sestate->cpu_cmds)
		{
			chunk->cpu_cmds = (int *) pos;
			chunk->cpu_cmds_len = VARSIZE_ANY_EXHDR(sestate->cpu_cmds);
			memcpy(chunk->cpu_cmds, VARDATA(sestate->cpu_cmds),
				   chunk->cpu_cmds_len);
			pos += MAXALIGN(chunk->cpu_cmds_len);			
		}
		else
		{
			chunk->cpu_cmds = NULL;
			chunk->cpu_cmds_len = 0;
		}

		/*
		 * To minimize times of DMA operations, we stores gpu_cmds, cs_isnull,
		 * cs_values and cs_rowmap in a contenious region.
		 * Thus, the base address of DMA shall locate in front of the GPU
		 * command sequence.
		 */
		chunk->dma_buffer = (char *)pos;
		chunk->dma_length = -1;		/* to be set later */

		if (sestate->gpu_cmds)
		{
			chunk->gpu_cmds = (int *) pos;
			chunk->gpu_cmds_len = VARSIZE_ANY_EXHDR(sestate->gpu_cmds);
			memcpy(chunk->gpu_cmds, VARDATA(sestate->gpu_cmds),
				   chunk->gpu_cmds_len);
			pos += MAXALIGN(chunk->gpu_cmds_len);
		}
		else
		{
			chunk->gpu_cmds = NULL;
			chunk->gpu_cmds_len = 0;
		}

		chunk->status = CHUNKBUF_STATUS_FREE;
		chunk->nattrs = nattrs;
		chunk->rowid = -1;
		chunk->nitems = -1;

		chunk->cs_isnull = (int *) pos;
		pos += MAXALIGN(sizeof(int) * nattrs);

		chunk->cs_values = (int *) pos;
		pos += MAXALIGN(sizeof(int) * nattrs);

		chunk->cs_rowmap = pos;
		pos += MAXALIGN(buffer_sz);

		pgstrom_shmseg_list_add(&sestate->chkb_free_list, &chunk->chain);

		Assert(pos <= ((char *)chunk + chunk_sz));
	}
}

static PgStromExecState *
pgstrom_init_exec_state(ForeignScanState *fss)
{
	ForeignScan	   *fscan = (ForeignScan *) fss->ss.ps.plan;
	bytea		   *gpu_cmds = NULL;
	bytea		   *cpu_cmds = NULL;
	Bitmapset	   *gpu_cols = NULL;
	Bitmapset	   *cpu_cols = NULL;
	Bitmapset	   *required_cols = NULL;
	Bitmapset	   *tempset;
	AttrNumber		nattrs, attno;
	ListCell	   *cell;
	PgStromExecState   *sestate;

	foreach (cell, fscan->fdw_private)
	{
		DefElem	   *defel = (DefElem *) lfirst(cell);

		if (strcmp(defel->defname, "gpu_cmds") == 0)
			gpu_cmds = DatumGetByteaP(((Const *)defel->arg)->constvalue);
		else if (strcmp(defel->defname, "cpu_cmds") == 0)
			cpu_cmds = DatumGetByteaP(((Const *)defel->arg)->constvalue);
		else if (strcmp(defel->defname, "gpu_cols") == 0)
			gpu_cols = bms_add_member(gpu_cols, intVal(defel->arg));
		else if (strcmp(defel->defname, "cpu_cols") == 0)
			cpu_cols = bms_add_member(cpu_cols, intVal(defel->arg));
		else if (strcmp(defel->defname, "required_cols") == 0)
			required_cols = bms_add_member(required_cols, intVal(defel->arg));
		else
			elog(ERROR, "unexpected private plan token: %s", defel->defname);
	}

	/*
	 * Setup of PgStromExecState
	 */
	sestate = palloc0(sizeof(PgStromExecState));
	sestate->gpu_cmds = gpu_cmds;
	sestate->cpu_cmds = cpu_cmds;
	sestate->gpu_cols = gpu_cols;
	sestate->cpu_cols = cpu_cols;
	sestate->required_cols = required_cols;

	sestate->ft_rel = fss->ss.ss_currentRelation;
	nattrs = RelationGetNumberOfAttributes(sestate->ft_rel);
	sestate->id_rel = pgstrom_open_rowid_map(sestate->ft_rel, AccessShareLock);
	sestate->cs_rels = palloc0(sizeof(Relation) * nattrs);
	sestate->cs_idxs = palloc0(sizeof(Relation) * nattrs);
	sestate->cs_scan = palloc0(sizeof(IndexScanDesc) * nattrs);
	sestate->es_qrycxt = fss->ss.ps.state->es_query_cxt;

	tempset = bms_union(required_cols, bms_union(gpu_cols, cpu_cols));
	while ((attno = bms_first_member(tempset)) > 0)
	{
		sestate->cs_rels[attno - 1]
			= pgstrom_open_cs_table(sestate->ft_rel, attno, AccessShareLock);
		sestate->cs_idxs[attno - 1]
			= pgstrom_open_cs_index(sestate->ft_rel, attno, AccessShareLock);
	}
	bms_free(tempset);

	sestate->chkb_head = NULL;		/* should be set later */
	pgstrom_shmseg_list_init(&sestate->chkb_ready_list);
	pgstrom_shmseg_list_init(&sestate->chkb_free_list);

	sestate->curr_chunk = NULL;
	sestate->curr_index = 0;
	sestate->curr_cs_isnull = palloc0(sizeof(bytea *) * nattrs);
	sestate->curr_cs_values = palloc0(sizeof(bytea *) * nattrs);
	sestate->curr_cs_rowid_min = palloc(sizeof(int64) * nattrs);
	memset(sestate->curr_cs_rowid_min, -1, sizeof(int64) * nattrs);
	sestate->curr_cs_rowid_max = palloc(sizeof(int64) * nattrs);
	memset(sestate->curr_cs_rowid_max, -1, sizeof(int64) * nattrs);

	/*
	 * Allocate chunk buffers on shared memory
	 */
	pgstrom_init_chunk_buffers(sestate);

	return sestate;
}

void
pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags)
{
	Snapshot	snapshot = fss->ss.ps.state->es_snapshot;
	AttrNumber	attno, nattrs;
	PgStromExecState *sestate;

	/* Do nothing for EXPLAIN or ANALYZE cases */
	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;

	sestate = pgstrom_init_exec_state(fss);

	sestate->id_scan = heap_beginscan(sestate->id_rel, snapshot, 0, NULL);
	nattrs = RelationGetNumberOfAttributes(sestate->ft_rel);
	for (attno = 1; attno <= nattrs; attno++)
	{
		/*
		 * Columns referenced in qualifier-clause required two keys on
		 * index-scan both lower and higher limit with rowid.
		 * Elsewhere, only its upper limit is needed on index-scan.
		 */
		if (bms_is_member(attno, sestate->gpu_cols) ||
			bms_is_member(attno, sestate->cpu_cols))
		{
			sestate->cs_scan[attno - 1]
				= index_beginscan(sestate->cs_rels[attno - 1],
								  sestate->cs_idxs[attno - 1],
								  snapshot, 2, 0);
		}
		else if (bms_is_member(attno, sestate->required_cols))
		{
			sestate->cs_scan[attno - 1]
				= index_beginscan(sestate->cs_rels[attno - 1],
								  sestate->cs_idxs[attno - 1],
								  snapshot, 1, 0);
		}
	}
    fss->fdw_state = sestate;
}

TupleTableSlot*
pgstrom_iterate_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	TupleTableSlot	   *slot = fss->ss.ss_ScanTupleSlot;

	ExecClearTuple(slot);

	while (!sestate->curr_chunk ||
		   !pgstrom_scan_chunk(sestate, slot))
	{
		ShmsegQueue	*recvq = &sestate->chkb_head->chkbh_recvq;
		ChunkBuffer	*chunk;
		ShmsegList	*q;

		/*
		 * Release the current chunk-buffer already scanned
		 */
		if (sestate->curr_chunk)
		{
			sestate->curr_chunk->status = CHUNKBUF_STATUS_FREE;
			pgstrom_shmseg_list_add(&sestate->chkb_free_list,
									&sestate->curr_chunk->chain);
			sestate->curr_chunk = NULL;			
		}
		Assert(!pgstrom_shmseg_list_empty(&sestate->chkb_free_list));

		/*
		 * Move any chunks being already complete its execution
		 */
		while ((q = pgstrom_shmqueue_trydequeue(recvq)) != NULL)
		{
			chunk = container_of(q, ChunkBuffer, chain);
			Assert(chunk->status == CHUNKBUF_STATUS_READY ||
				   chunk->status == CHUNKBUF_STATUS_ERROR);
			sestate->chkb_head->chkbh_exec_cnt--;
			if (chunk->status == CHUNKBUF_STATUS_ERROR)
				elog(ERROR, "%s", chunk->error_msg);
			pgstrom_shmseg_list_add(&sestate->chkb_ready_list, &chunk->chain);
		}
		Assert(sestate->chkb_head->chkbh_exec_cnt >= 0);

		/*
		 * Try to keep num of chunks being executed overs
		 * pgstrom_min_async_chunks
		 */
		while (sestate->id_scan != NULL &&
			   !pgstrom_shmseg_list_empty(&sestate->chkb_free_list))
		{
			/*
			 * If we still have ready chunks and num of chunks in execution
			 * exceeds pgstrom_min_async_chunks, break the chunk load.
			 */
			if (!pgstrom_shmseg_list_empty(&sestate->chkb_ready_list) &&
				sestate->chkb_head->chkbh_exec_cnt >= pgstrom_min_async_chunks)
				break;

			/*
			 * Load and execute chunks
			 */
			if (!pgstrom_load_chunk(sestate))
				break;
		}

		if (!pgstrom_shmseg_list_empty(&sestate->chkb_ready_list))
		{
			chunk = container_of(sestate->chkb_ready_list.next,
								 ChunkBuffer, chain);

			Assert(chunk->status == CHUNKBUF_STATUS_READY ||
				   chunk->status == CHUNKBUF_STATUS_ERROR);
			pgstrom_shmseg_list_delete(&chunk->chain);
			sestate->curr_chunk = chunk;
			sestate->curr_index = 0;
		}
		else if (sestate->chkb_head->chkbh_exec_cnt > 0)
		{
			q = pgstrom_shmqueue_dequeue(recvq);

			chunk = container_of(q, ChunkBuffer, chain);

			Assert(chunk->status == CHUNKBUF_STATUS_READY ||
				   chunk->status == CHUNKBUF_STATUS_ERROR);
			sestate->chkb_head->chkbh_exec_cnt--;
			if (chunk->status == CHUNKBUF_STATUS_ERROR)
				elog(ERROR, "%s", chunk->error_msg);
			sestate->curr_chunk = chunk;
			sestate->curr_index = 0;
		}
		else
			break;
	}
	return slot;
}

void
pgstrom_rescan_foreign_scan(ForeignScanState *fss)
{
	/* TODO: implement it later */
}

void
pgstrom_end_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	AttrNumber			i, nattrs;

	/* if sestate is NULL, we are in EXPLAIN; nothing to do */
	if (!sestate)
		return;

	/*
	 * Close the scan descriptor and relation
	 */
	nattrs = RelationGetNumberOfAttributes(sestate->ft_rel);
	for (i=0; i < nattrs; i++)
	{
		if (sestate->cs_scan[i])
			index_endscan(sestate->cs_scan[i]);
	}
	if (sestate->id_scan)
		heap_endscan(sestate->id_scan);

	for (i=0; i < nattrs; i++)
	{
		if (sestate->cs_rels[i])
			relation_close(sestate->cs_rels[i], AccessShareLock);
		if (sestate->cs_idxs[i])
			relation_close(sestate->cs_idxs[i], AccessShareLock);
	}
	relation_close(sestate->id_rel, AccessShareLock);

	/* TODO: Wait for completion of in-exec chunks */

	pgstrom_shmseg_list_delete(&sestate->chkb_head->chkbh_chain);
	pgstrom_shmseg_free(sestate->chkb_head);
	bms_free(sestate->gpu_cols);
	bms_free(sestate->cpu_cols);
	bms_free(sestate->required_cols);
	pfree(sestate->cs_rels);
	pfree(sestate->cs_idxs);
	pfree(sestate->cs_scan);
	pfree(sestate);
}

void
pgstrom_executor_init(void)
{
	/* resource cleanup hook towards shared memory segment */
	pgstrom_shmseg_list_init(&local_chunk_buffer_head);
	RegisterResourceReleaseCallback(pgstrom_release_resources, NULL);

	DefineCustomIntVariable("pg_strom.max_async_chunks",
							"max number of chunk to be executed concurrently",
							NULL,
							&pgstrom_max_async_chunks,
							6 * pgstrom_gpu_num_devices(),
							1,
							32,
							PGC_USERSET,
							0,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.min_async_chunks",
							"min number of chunks to be kept in execution",
							NULL,
							&pgstrom_min_async_chunks,
							1 * pgstrom_gpu_num_devices(),
							1,
							8,
							PGC_USERSET,
							0,
							NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.exec_profile",
							 "print execution profile information",
							 NULL,
							 &pgstrom_exec_profile,
							 false,
							 PGC_USERSET,
							 0,
							 NULL, NULL, NULL);
}
