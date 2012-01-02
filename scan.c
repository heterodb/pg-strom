/*
 * scan.c
 *
 * Routines to scan column based data store with stream processing
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "access/relscan.h"
#include "catalog/namespace.h"
#include "catalog/pg_class.h"
#include "catalog/pg_type.h"
#include "foreign/foreign.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "utils/array.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/resowner.h"
#include "utils/varbit.h"
#include "pg_strom.h"

/*
 * Declarations
 */
//static cl_context	pgstrom_device_context = NULL;
static int		pgstrom_max_async_chunks;
static int		pgstrom_work_group_size;

typedef struct {
	int64		rowid;
	int			nattrs;
	int			cs_rownums;
	bits8	   *cs_rowmap;	/* also, head of the page locked memory */
	int		   *cs_nulls;	/* offset from the cs_rowmap, or 0 */
	int		   *cs_values;	/* offset from the cs_rowmap, or 0 */
	CUdeviceptr	devmem;
	CUstream	stream;
} PgStromChunkBuf;

#define chunk_cs_nulls(chunk,csidx)		\
	((bits8 *)((chunk)->cs_rowmap + (chunk)->cs_nulls[(csidx)]))
#define chunk_cs_values(chunk,csidx)	\
	((char *)((chunk)->cs_rowmap + (chunk)->cs_values[(csidx)]))

typedef struct {
	RelationSet		relset;

	/* parameters come from planner */
	bool			nevermatch;		/* true, if no items shall be matched */
	Bitmapset	   *required_cols;	/* columns being returned to executor */
	Bitmapset	   *clause_cols;	/* columns being copied to device */
	const char	   *kernel_source;	/* source of kernel code */

	/* copy from EState */
	Relation		es_relation;	/* copy from ScanState */
	Snapshot		es_snapshot;	/* copy from EState */
	MemoryContext	es_memcxt;		/* per-query memory context */
	//ErrorContextCallback es_errcxt;	/* callback context on error */

	/* scan descriptors */
	HeapScanDesc	ri_scan;		/* scan on rowid map */
	IndexScanDesc  *cs_scan;		/* scan on column store */
	ArrayType	  **cs_cur_values;
	int64		   *cs_cur_rowid_min;
	int64		   *cs_cur_rowid_max;

	/* list of the chunk */
	List		   *chunk_exec_list;	/* chunks in device execution */
	List		   *chunk_ready_list;	/* chunks in ready to scaning */
	ListCell	   *curr_chunk;
	int				curr_index;

	/* CUDA related stuff */
	CUmodule		dev_module;
	CUfunction		dev_function;
} PgStromExecState;









static void
pgstrom_cleanup_exec_state(PgStromExecState *sestate)
{
	elog(NOTICE, "pgstrom_release_exec_state called: %p", sestate);

	if (sestate->dev_module)
		cuModuleUnload(sestate->dev_module);
}

static void
pgstrom_load_column_store(PgStromExecState *sestate,
						  PgStromChunkBuf *chunk, int csidx)
{
	Form_pg_attribute	attr;
	IndexScanDesc	iscan;
	ScanKeyData		skeys[2];
	HeapTuple		tup;

	/*
	 * XXX - Because this column shall be copied to device to execute
	 * kernel function, variable length value should not be appeared
	 * in this stage.
	 */
	attr = RelationGetDescr(sestate->es_relation)->attrs[csidx];
	Assert(attr->attlen > 0);

	/*
	 * Null-bitmap shall be initialized as if all the values are NULL
	 */
	memset(chunk_cs_nulls(chunk,csidx), -1,
		   PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);

	/*
	 * Try to scan column store with cs_rowid betweem rowid and
	 * (rowid + PGSTROM_CHUNK_SIZE)
	 */
	ScanKeyInit(&skeys[0],
				(AttrNumber) 1,
				BTGreaterEqualStrategyNumber, F_INT8GE,
				Int64GetDatum(chunk->rowid));
	ScanKeyInit(&skeys[1],
				(AttrNumber) 1,
				BTLessStrategyNumber, F_INT8LT,
				Int64GetDatum(chunk->rowid + PGSTROM_CHUNK_SIZE));

	iscan = index_beginscan(sestate->relset->cs_rel[csidx],
							sestate->relset->cs_idx[csidx],
							sestate->es_snapshot, 2, 0);
	index_rescan(iscan, skeys, 2, NULL, 0);

	while (HeapTupleIsValid(tup = index_getnext(iscan, ForwardScanDirection)))
	{
		TupleDesc	tupdesc;
		Datum		values[2];
		bool		nulls[2];
		int64		cur_rowid;
		ArrayType  *cur_array;
		bits8	   *nullbitmap;
		int			offset;
		int			nitems;

		tupdesc = RelationGetDescr(sestate->relset->cs_rel[csidx]);
		heap_deform_tuple(tup, tupdesc, values, nulls);
		Assert(!nulls[0] && !nulls[1]);

		cur_rowid = Int64GetDatum(values[0]);
		cur_array = DatumGetArrayTypeP(values[1]);

		offset = cur_rowid - chunk->rowid;
		Assert(offset >= 0 && offset < PGSTROM_CHUNK_SIZE);
		Assert((offset & (BITS_PER_BYTE - 1)) == 0);
		Assert(ARR_NDIM(cur_array) == 1);
		Assert(ARR_LBOUND(cur_array)[0] == 0);
		Assert((ARR_DIMS(cur_array)[0] & (BITS_PER_BYTE - 1)) == 0);
		Assert(ARR_ELEMTYPE(cur_array) == attr->atttypid);

		nitems = ARR_DIMS(cur_array)[0];
		memcpy(chunk_cs_values(chunk,csidx) + offset * attr->attlen,
			   ARR_DATA_PTR(cur_array),
			   nitems * attr->attlen);
		nullbitmap = ARR_NULLBITMAP(cur_array);
		if (nullbitmap)
		{
			memcpy(chunk_cs_nulls(chunk,csidx) + offset / BITS_PER_BYTE,
				   nullbitmap,
				   (nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		}
		else
		{
			/* Clear nullbitmap, if all items are not NULL */
			memset(chunk_cs_nulls(chunk,csidx) + offset / BITS_PER_BYTE,
				   0,
				   (nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		}
	}
	index_endscan(iscan);
}

static CUresult
pgstrom_exec_kernel_qual(PgStromExecState *sestate, PgStromChunkBuf *chunk)
{
	CUresult	ret;

	//ret = cuStreamCreate(&chunk->stream, 0);
	//if (ret != CUDA_SUCCESS)
	//	return ret;

	



	return CUDA_SUCCESS;
}

static int
pgstrom_load_chunk_buffer(PgStromExecState *sestate, int num_chunks)
{
	int		loaded_chunks;

	if (sestate->ri_scan == NULL)
		return -1;

	for (loaded_chunks = 0; loaded_chunks < num_chunks; loaded_chunks++)
	{
		TupleDesc	tupdesc;
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2];
		int			rowid;
		VarBit	   *rowmap;
		MemoryContext oldcxt;
		PgStromChunkBuf	*chunk;

		tuple = heap_getnext(sestate->ri_scan, ForwardScanDirection);
		if (!HeapTupleIsValid(tuple))
		{
			/* No any tuples, close the sequential rowid scan */
			heap_endscan(sestate->ri_scan);
			sestate->ri_scan = NULL;
			break;
		}

		tupdesc = RelationGetDescr(sestate->relset->rowid_rel);
		heap_deform_tuple(tuple, tupdesc, values, nulls);
		Assert(!nulls[0] && !nulls[1]);

		rowid = DatumGetInt64(values[0]);
		rowmap = DatumGetVarBitP(values[1]);

		oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
		chunk = palloc0(sizeof(PgStromChunkBuf));
		chunk->rowid = rowid;
		chunk->nattrs = RelationGetNumberOfAttributes(sestate->es_relation);
		chunk->cs_nulls = palloc0(sizeof(int) * chunk->nattrs);
		chunk->cs_values = palloc0(sizeof(int) * chunk->nattrs);
		MemoryContextSwitchTo(oldcxt);

		if (sestate->dev_function)
		{
			Bitmapset  *temp;
			AttrNumber	attnum;
			uint16		attlen;
			char	   *dma_buffer;
			size_t		dma_offset;
			CUresult	ret;

			/*
			 * Compute and allocate required size of column store
			 */
			dma_offset = PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;

			tupdesc = RelationGetDescr(sestate->es_relation);
			temp = bms_copy(sestate->clause_cols);
			while ((attnum = bms_first_member(temp)) > 0)
			{
				attlen = tupdesc->attrs[attnum-1]->attlen;
				Assert(attlen > 0);

				chunk->cs_values[attnum-1] = dma_offset;
				dma_offset += PGSTROM_CHUNK_SIZE * attlen;
				chunk->cs_nulls[attnum-1] = dma_offset;
				dma_offset += PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;
			}
			bms_free(temp);
			ret = cuMemAllocHost((void **)&dma_buffer, dma_offset);
			if (ret != CUDA_SUCCESS)
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("cuda: failed to page-locked memory : %s",
								cuda_error_to_string(ret))));
			/*
			 * Load necessary column store
			 */
			chunk->cs_rowmap = (uint8 *)dma_buffer;
			chunk->cs_rownums = VARBITLEN(rowmap);
			if (VARBITLEN(rowmap) != PGSTROM_CHUNK_SIZE)
				memset(chunk->cs_rowmap, 0,
					   PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
			memcpy(chunk->cs_rowmap, VARBITS(rowmap), VARBITBYTES(rowmap));
			dma_offset = PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;

			temp = bms_copy(sestate->clause_cols);
			while ((attnum = bms_first_member(temp)) > 0)
				pgstrom_load_column_store(sestate, chunk, attnum-1);
			bms_free(temp);

			/*
			 * Asynchronous execution of kernel code on this chunk
			 */
			ret = pgstrom_exec_kernel_qual(sestate, chunk);
			if (ret != CUDA_SUCCESS)
			{
				cuMemFreeHost(chunk->cs_rowmap);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("cuda: failed to execute kernel code : %s",
								cuda_error_to_string(ret))));
			}

			/*
			 * XXX - Do we need to pay attention of the case when
			 * lappend raises an error?
			 */
			oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
			sestate->chunk_ready_list
				= lappend(sestate->chunk_ready_list, chunk);
			MemoryContextSwitchTo(oldcxt);
		}
		else
		{
			/*
			 * In the case when the supplied plan has no qualifier,
			 * all the chunks are ready to scan using rowid.
			 */
			oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
			chunk->cs_rownums = VARBITLEN(rowmap);
			chunk->cs_rowmap = palloc(PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
			memcpy(chunk->cs_rowmap, VARBITS(rowmap),
				   PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
			sestate->chunk_ready_list
				= lappend(sestate->chunk_ready_list, chunk);
			MemoryContextSwitchTo(oldcxt);
		}
	}
	return loaded_chunks;
}

static void
pgstrom_scan_column_store(PgStromExecState *sestate,
						  int csidx, int64 rowid,
						  TupleTableSlot *slot)
{
	ScanKeyData		skey;
	TupleDesc		tupdesc;
	HeapTuple		tuple;
	Datum			values[2];
	bool			nulls[2];
	int64			cur_rowid;
	ArrayType	   *cur_values;
	int				index;
	MemoryContext	oldcxt;
	Form_pg_attribute	attr;

	if (!sestate->cs_cur_values[csidx] ||
		rowid < sestate->cs_cur_rowid_min[csidx] ||
		rowid > sestate->cs_cur_rowid_max[csidx])
	{
		/*
		 * XXX - Just our heuristic, when the supplied rowid is located
		 * enought near range with the previous array, it will give us
		 * performance gain to just pick up next tuple according to the
		 * current index scan.
		 * Right now, we decide its threshold as a range between
		 * cs_cur_rowid_max and cs_cur_rowid_max + 2 * (cs_cur_rowid_max
		 * - cs_cur_rowid_min).
		 */
		if (sestate->cs_cur_values[csidx] &&
			rowid > sestate->cs_cur_rowid_max[csidx] &&
			rowid < (sestate->cs_cur_rowid_max[csidx] +
					 2 * (sestate->cs_cur_rowid_max[csidx] -
						  sestate->cs_cur_rowid_min[csidx])))
		{
			int		count = 2;

			while (count-- > 0)
			{
				tuple = index_getnext(sestate->cs_scan[csidx],
									  ForwardScanDirection);
				if (!HeapTupleIsValid(tuple))
					break;

				tupdesc = RelationGetDescr(sestate->relset->cs_rel[csidx]);
				heap_deform_tuple(tuple, tupdesc, values, nulls);
				Assert(!nulls[0] && !nulls[1]);

				oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
				cur_rowid = Int64GetDatum(values[0]);
				cur_values = DatumGetArrayTypePCopy(values[1]);
				MemoryContextSwitchTo(oldcxt);

				/* Hit! */
				if (rowid >= cur_rowid &&
					rowid <= cur_rowid + ARR_DIMS(cur_values)[0])
				{
					pfree(sestate->cs_cur_values[csidx]);

					sestate->cs_cur_values[csidx] = cur_values;
					sestate->cs_cur_rowid_min[csidx] = cur_rowid;
					sestate->cs_cur_rowid_max[csidx]
						= cur_rowid + ARR_DIMS(cur_values)[0] - 1;

					goto out;
				}
#ifndef USE_FLOAT8_BYVAL
				pfree(cur_rowid);
#endif
				pfree(cur_values);
			}
		}

		/*
		 * Rewind the index scan again, the fetch tuple that contains
		 * the supplied rowid.
		 */
		if (sestate->cs_cur_values[csidx])
		{
			pfree(sestate->cs_cur_values[csidx]);
			sestate->cs_cur_values[csidx] = NULL;
			sestate->cs_cur_rowid_min[csidx] = -1;
			sestate->cs_cur_rowid_max[csidx] = -1;
		}

		ScanKeyInit(&skey,
					(AttrNumber) 1,
					BTLessEqualStrategyNumber, F_INT8LE,
					Int64GetDatum(rowid));
		index_rescan(sestate->cs_scan[csidx], &skey, 1, NULL, 0);

		tuple = index_getnext(sestate->cs_scan[csidx],
							  BackwardScanDirection);
		if (!HeapTupleIsValid(tuple))
		{
			slot->tts_isnull[csidx] = true;
			slot->tts_values[csidx] = (Datum) 0;
			return;
		}

		tupdesc = RelationGetDescr(sestate->relset->cs_rel[csidx]);
		heap_deform_tuple(tuple, tupdesc, values, nulls);
		Assert(!nulls[0] && !nulls[1]);

		oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
		cur_rowid = Int64GetDatum(values[0]);
		cur_values = DatumGetArrayTypePCopy(values[1]);
		MemoryContextSwitchTo(oldcxt);

		sestate->cs_cur_values[csidx] = cur_values;
		sestate->cs_cur_rowid_min[csidx] = cur_rowid;
		sestate->cs_cur_rowid_max[csidx]
			= cur_rowid + ARR_DIMS(cur_values)[0] - 1;

		Assert(rowid >= sestate->cs_cur_rowid_min[csidx] &&
			   rowid <= sestate->cs_cur_rowid_max[csidx]);

		/*
		 * XXX - For the above heuristic shortcut, it resets direction
		 * and condition of index scan.
		 */
		ScanKeyInit(&skey,
					(AttrNumber) 1,
					BTGreaterStrategyNumber, F_INT8GT,
					Int64GetDatum(sestate->cs_cur_rowid_max[csidx]));
		index_rescan(sestate->cs_scan[csidx], &skey, 1, NULL, 0);
	}
out:
	attr = slot->tts_tupleDescriptor->attrs[csidx];
	index = rowid - sestate->cs_cur_rowid_min[csidx];
	slot->tts_values[csidx] = array_ref(sestate->cs_cur_values[csidx],
										1,
										&index,
										-1,	/* varlena array */
										attr->attlen,
										attr->attbyval,
										attr->attalign,
										&slot->tts_isnull[csidx]);
}

static bool
pgstrom_scan_chunk_buffer(PgStromExecState *sestate, TupleTableSlot *slot)
{
	PgStromChunkBuf	*chunk = lfirst(sestate->curr_chunk);
	int		index;

	for (index = sestate->curr_index; index < chunk->cs_rownums; index++)
	{
		int		index_h = (index / BITS_PER_BYTE);
		int		index_l = (index & (BITS_PER_BYTE - 1));
		int		csidx;
		int64	rowid;

		if ((chunk->cs_rowmap[index_h] & (1 << index_l)) == 0)
			continue;

		rowid = chunk->rowid + index;
		for (csidx=0; csidx < chunk->nattrs; csidx++)
		{
			/*
			 * No need to back actual value of unreferenced column.
			 */
			if (!bms_is_member(csidx+1, sestate->required_cols))
			{
				slot->tts_isnull[csidx] = true;
				slot->tts_values[csidx] = (Datum) 0;
				continue;
			}

			/*
			 * No need to scan column-store again, if this column was
			 * already loaded on the previous stage. All we need to do
			 * is pick up an appropriate value from chunk buffer.
			 */
			if (chunk->cs_values[csidx] > 0)
			{
				bits8  *nullbitmap = chunk_cs_nulls(chunk,csidx);

				if ((nullbitmap[index_h] & (1 << index_l)) == 0)
				{
					Form_pg_attribute	attr
						= slot->tts_tupleDescriptor->attrs[csidx];
					slot->tts_isnull[csidx] = false;
					slot->tts_values[csidx] =
						fetchatt(attr, (chunk_cs_values(chunk,csidx) +
										index * attr->attlen));
				}
				else
				{
					slot->tts_isnull[csidx] = true;
					slot->tts_values[csidx] = (Datum) 0;
				}
				continue;
			}
			/*
			 * Elsewhere, we scan the column-store with the current
			 * rowid.
			 */
			pgstrom_scan_column_store(sestate, csidx, rowid, slot);
		}
		ExecStoreVirtualTuple(slot);
		/* update next index to be fetched */
		sestate->curr_index = index + 1;
		return true;
	}
	return false;	/* end of chunk, need next chunk! */
}

static PgStromExecState *
pgstrom_init_exec_state(ForeignScanState *fss)
{
	ForeignScan		   *fscan = (ForeignScan *) fss->ss.ps.plan;
	PgStromExecState   *sestate;
	ListCell		   *l;
	AttrNumber			nattrs;

	nattrs = RelationGetNumberOfAttributes(fss->ss.ss_currentRelation);
	sestate = palloc0(sizeof(PgStromExecState));
	sestate->cs_scan = palloc0(sizeof(IndexScanDesc) * nattrs);
	sestate->cs_cur_values = palloc0(sizeof(ArrayType *) * nattrs);
	sestate->cs_cur_rowid_min = palloc0(sizeof(int64) * nattrs);
	sestate->cs_cur_rowid_max = palloc0(sizeof(int64) * nattrs);

	sestate->es_relation = fss->ss.ss_currentRelation;
    sestate->es_snapshot = fss->ss.ps.state->es_snapshot;
	sestate->es_memcxt = fss->ss.ps.ps_ExprContext->ecxt_per_query_memory;

	foreach (l, fscan->fdwplan->fdw_private)
	{
		DefElem	   *defel = (DefElem *)lfirst(l);

		if (strcmp(defel->defname, "nevermatch") == 0)
		{
			sestate->nevermatch = intVal(defel->arg);
		}
		else if (strcmp(defel->defname, "kernel_source") == 0)
		{
			sestate->kernel_source = strVal(defel->arg);
		}
		else if (strcmp(defel->defname, "clause_cols") == 0)
		{
			int		csidx = (intVal(defel->arg));

			Assert(csidx > 0);

			sestate->clause_cols
				= bms_add_member(sestate->clause_cols, csidx);
		}
		else if (strcmp(defel->defname, "required_cols") == 0)
		{
			int		csidx = (intVal(defel->arg));

			if (csidx > 0)
				sestate->required_cols
					= bms_add_member(sestate->required_cols, csidx);
			else
				Assert(fscan->fsSystemCol);
		}
		else
			elog(ERROR, "pg_strom: unexpected private plan information: %s",
				 defel->defname);
	}

	if (!sestate->nevermatch &&
		sestate->kernel_source != NULL)
	{
		CUresult	ret;
		void	   *image;

		/*
		 * XXX - we should handle multiple GPU devices
		 */
		pgstrom_set_device_context(0);

		/*
		 * Build the kernel source code
		 */
		image = pgstrom_nvcc_kernel_build(sestate->kernel_source);

		ret = cuModuleLoadData(&sestate->dev_module, image);
		if (ret != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("cuda: failed to load executable module : %s",
							cuda_error_to_string(ret))));

		ret = cuModuleGetFunction(&sestate->dev_function,
								  sestate->dev_module,
								  "pgstrom_qual");
		if (ret != CUDA_SUCCESS)
		{
			cuModuleUnload(sestate->dev_module);
			ereport(ERROR,
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("cuda: failed to get device function : %s",
							cuda_error_to_string(ret))));
		}
	}
	sestate->chunk_exec_list = NIL;
	sestate->chunk_ready_list = NIL;
	sestate->curr_chunk = NULL;
	sestate->curr_index = 0;

	return sestate;
}

void
pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags)
{
	Relation			base_rel = fss->ss.ss_currentRelation;
	PgStromExecState   *sestate;
	Bitmapset		   *tempset;
	AttrNumber			attnum;

	/*
	 * Do nothing for EXPLAIN or ANALYZE cases
	 */
	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;

	sestate = pgstrom_init_exec_state(fss);

	/*
	 * Begin the scan
	 */
	sestate->relset = pgstrom_open_relation_set(base_rel,
												AccessShareLock, true);
	sestate->ri_scan = heap_beginscan(sestate->relset->rowid_rel,
									  sestate->es_snapshot,
									  0, NULL);

	tempset = bms_copy(sestate->required_cols);
	while ((attnum = bms_first_member(tempset)) > 0)
	{
		/*
		 * Clause cols should be loaded prior to scan, so no need to
		 * scan it again using rowid.
		 */
		if (bms_is_member(attnum, sestate->clause_cols))
			continue;

		sestate->cs_scan[attnum - 1]
			= index_beginscan(sestate->relset->cs_rel[attnum - 1],
							  sestate->relset->cs_idx[attnum - 1],
							  sestate->es_snapshot, 1, 0);
	}
	bms_free(tempset);

	fss->fdw_state = sestate;
}

TupleTableSlot*
pgstrom_iterate_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	TupleTableSlot	   *slot = fss->ss.ss_ScanTupleSlot;
	int					num_chunks;

	ExecClearTuple(slot);
	if (sestate->nevermatch)
		return slot;

	/* Is it the first call? */
	if (sestate->curr_chunk == NULL)
	{
		num_chunks = (pgstrom_max_async_chunks -
					  list_length(sestate->chunk_exec_list));
		if (pgstrom_load_chunk_buffer(sestate, num_chunks) < 1)
			return slot;

		//pgstrom_sync_kernel_qual(sestate);
		Assert(sestate->chunk_ready_list != NIL);
		sestate->curr_chunk = list_head(sestate->chunk_ready_list);
		sestate->curr_index = 0;
	}
retry:
	if (!pgstrom_scan_chunk_buffer(sestate, slot))
	{
		PgStromChunkBuf	*chunk = lfirst(sestate->curr_chunk);

		sestate->chunk_ready_list
			= list_delete(sestate->chunk_ready_list, chunk);
		/*
		 * release chunk being scanned already
		 */
		if (sestate->dev_function)
			cuMemFreeHost(chunk->cs_rowmap);
		else
			pfree(chunk->cs_rowmap);
		pfree(chunk->cs_nulls);
		pfree(chunk->cs_values);
		pfree(chunk);

		//pgstrom_sync_kernel_qual(sestate);

		num_chunks = (pgstrom_max_async_chunks -
					  list_length(sestate->chunk_exec_list));
		num_chunks = pgstrom_load_chunk_buffer(sestate, num_chunks);
		//if (sestate->chunk_ready_list == NIL)
		//	pgstrom_sync_kernel_qual(sestate);

		/* no more chunks any more */
		if (sestate->chunk_ready_list == NIL)
			return slot;
		sestate->curr_chunk = list_head(sestate->chunk_ready_list);
		sestate->curr_index = 0;
		goto retry;
	}
	return slot;
}

void
pgstrom_rescan_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;

	/* Rewind rowid scan */
	if (sestate->ri_scan != NULL)
		heap_endscan(sestate->ri_scan);
	sestate->ri_scan = heap_beginscan(sestate->relset->rowid_rel,
									  sestate->es_snapshot,
									  0, NULL);
	/*
	 * XXX - chunk buffers to be relased
	 */

	/* Clear current chunk pointer */
	sestate->curr_chunk = NULL;
	sestate->curr_index = 0;
}

void
pgstrom_end_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	PgStromChunkBuf	   *chunk;
	ListCell		   *cell;
	int					nattrs, i;

	/* if sestate is NULL, we are in EXPLAIN; nothing to do */
	if (!sestate)
		return;

	/*
	 * End the rowid scan
	 */
	nattrs = RelationGetNumberOfAttributes(fss->ss.ss_currentRelation);
	for (i=0; i < nattrs; i++)
	{
		if (sestate->cs_scan[i])
			index_endscan(sestate->cs_scan[i]);
	}
	if (sestate->ri_scan != NULL)
		heap_endscan(sestate->ri_scan);

	pgstrom_close_relation_set(sestate->relset, AccessShareLock);

	/*
	 * cleanup stuff related to OpenCL to prevent leaks
	 */
	//while (sestate->chunk_exec_list != NIL)
	//	pgstrom_sync_kernel_qual(sestate);
	pgstrom_cleanup_exec_state(sestate);
}

/*
 * pgstrom_scan_init
 *
 * Initialize stuff related to scan.c
 */
void
pgstrom_scan_init(void)
{
	DefineCustomIntVariable("pg_strom.max_async_chunks",
							"max number of concurrency to exec async kernels",
							NULL,
							&pgstrom_max_async_chunks,
							32,
							1,
							1024,
							PGC_USERSET,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("pg_strom.work_group_size",
							"size of work group on execution of kernel code",
							NULL,
							&pgstrom_work_group_size,
							32,
							1,
							PGSTROM_CHUNK_SIZE / BITS_PER_BYTE,
							PGC_USERSET,
							0,
							NULL, NULL, NULL);
}
