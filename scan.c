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
#include "access/xact.h"
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

typedef struct {
	int64		rowid;
	int			nattrs;
	int			cs_rownums;
	bits8	   *cs_rowmap;	/* also, head of the page locked memory */
	int		   *cs_nulls;	/* offset from the cs_rowmap, or 0 */
	int		   *cs_values;	/* offset from the cs_rowmap, or 0 */
	size_t		devmem_size;
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
	char		   *kernel_source;	/* source of kernel code */

	/* copy from EState */
	ResourceOwner	es_owner;		/* copy from CurrentResourceOwner */
	Relation		es_relation;	/* copy from ScanState */
	Snapshot		es_snapshot;	/* copy from EState */
	MemoryContext	es_memcxt;		/* per-query memory context */

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

/*
 * Declarations
 */
static MemoryContext	pgstrom_scan_memcxt;
static List	   *pgstrom_exec_state_list = NIL;
static int		pgstrom_max_async_chunks;


static void
pgstrom_release_chunk_buffer(PgStromExecState *sestate,
							 PgStromChunkBuf *chunk)
{
	if (chunk->stream != NULL)
	{
		/*
		 * If the supplied chunk buffer is under execution of kernel code,
		 * we try to synchronize its completion, then release its resources,
		 * because asynchronous memory copy may be ready to run...
		 */
		if (cuStreamQuery(chunk->stream) == CUDA_ERROR_NOT_READY)
			cuStreamSynchronize(chunk->stream);
		cuStreamDestroy(chunk->stream);
	}
	if (sestate->dev_module)
		cuMemFreeHost(chunk->cs_rowmap);
	else
		pfree(chunk->cs_rowmap);
	pfree(chunk->cs_nulls);
	pfree(chunk->cs_values);
	pfree(chunk);
}

static void
pgstrom_release_exec_state(PgStromExecState *sestate)
{
	ListCell   *cell;

	pgstrom_exec_state_list
		= list_delete_ptr(pgstrom_exec_state_list, sestate);

	foreach (cell, sestate->chunk_exec_list)
		pgstrom_release_chunk_buffer(sestate, lfirst(cell));
	list_free(sestate->chunk_exec_list);

	foreach (cell, sestate->chunk_ready_list)
		pgstrom_release_chunk_buffer(sestate, lfirst(cell));
	list_free(sestate->chunk_ready_list);

	if (sestate->dev_module)
		cuModuleUnload(sestate->dev_module);
	if (sestate->required_cols)
		bms_free(sestate->required_cols);
	if (sestate->clause_cols)
		bms_free(sestate->clause_cols);
	if (sestate->kernel_source)
		pfree(sestate->kernel_source);
	if (sestate->cs_scan)
		pfree(sestate->cs_scan);
	if (sestate->cs_cur_values)
		pfree(sestate->cs_cur_values);
	if (sestate->cs_cur_rowid_min)
		pfree(sestate->cs_cur_rowid_min);
	if ( sestate->cs_cur_rowid_max)
		pfree( sestate->cs_cur_rowid_max);
	pfree(sestate);
}

static void
pgstrom_release_resources(ResourceReleasePhase phase,
						   bool isCommit,
						   bool isTopLevel,
						   void *arg)
{
	ListCell   *cell;
	ListCell   *next;

	if (phase != RESOURCE_RELEASE_AFTER_LOCKS)
		return;

	for (cell = list_head(pgstrom_exec_state_list); cell; cell = next)
	{
		PgStromExecState   *sestate = lfirst(cell);

		next = lnext(cell);

		if (sestate->es_owner == CurrentResourceOwner)
			pgstrom_release_exec_state(sestate);
	}
}

static CUresult
pgstrom_exec_kernel_qual(PgStromExecState *sestate, PgStromChunkBuf *chunk)
{
	CUdeviceptr	   *kernel_data;
	void		  **kernel_args;
	CUresult		ret;
	int				i, j;

	ret = cuStreamCreate(&chunk->stream, 0);
	if (ret != CUDA_SUCCESS)
		goto error_1;

	ret = cuMemAlloc(&chunk->devmem, chunk->devmem_size);
	if (ret != CUDA_SUCCESS)
		goto error_2;

	ret = cuMemcpyHtoDAsync(chunk->devmem,
							chunk->cs_rowmap,
							chunk->devmem_size,
							chunk->stream);
	if (ret != CUDA_SUCCESS)
        goto error_3;

	kernel_data = alloca((1 + 2 * chunk->nattrs) * sizeof(CUdeviceptr));
	kernel_args = alloca((1 + 2 * chunk->nattrs) * sizeof(void *));
	kernel_data[0] = chunk->devmem;
	kernel_args[0] = &kernel_data[0];
	for (i=0, j=1; i < chunk->nattrs; i++)
	{
		if (chunk->cs_values[i] > 0)
		{
			kernel_data[j] = chunk->devmem + chunk->cs_values[i];
			kernel_data[j+1] = chunk->devmem + chunk->cs_nulls[i];
			Assert(chunk->cs_nulls[i] > 0);
			kernel_args[j] = &kernel_data[j];
			kernel_args[j+1] = &kernel_data[j+1];
			j += 2;
		}
	}
	ret = cuLaunchKernel(sestate->dev_function,
						 (chunk->cs_rownums / BITS_PER_BYTE + 29) / 30,
						 1,
						 1,
						 30,
						 1,
						 1,
						 0,
						 chunk->stream,
						 kernel_args,
						 NULL);
	if (ret != CUDA_SUCCESS)
        goto error_4;

	ret = cuMemcpyDtoHAsync(chunk->cs_rowmap,
							chunk->devmem,
							PGSTROM_CHUNK_SIZE / BITS_PER_BYTE,
							chunk->stream);
	if (ret != CUDA_SUCCESS)
		goto error_4;

	return CUDA_SUCCESS;

error_4:
	cuStreamSynchronize(chunk->stream);
error_3:
	cuMemFree(chunk->devmem);
error_2:
	cuStreamDestroy(chunk->stream);
error_1:
	return ret;
}

static void
pgstrom_sync_kernel_exec(PgStromExecState *sestate)
{
	ListCell   *cell;
	ListCell   *next;
	ListCell   *prev;
	CUstream	first_stream = NULL;
	CUresult	ret;

	if (sestate->chunk_exec_list == NIL)
		return;

retry:
	prev = NULL;
	for (cell = list_head(sestate->chunk_exec_list); cell; cell = next)
	{
		PgStromChunkBuf	*chunk = lfirst(cell);

		next = lnext(cell);

		ret = cuStreamQuery(chunk->stream);
		if (ret == CUDA_SUCCESS)
		{
			MemoryContext	oldcxt;

			sestate->chunk_exec_list
				= list_delete_cell(sestate->chunk_exec_list, cell, prev);
			oldcxt = MemoryContextSwitchTo(pgstrom_scan_memcxt);
			sestate->chunk_ready_list
				= lappend(sestate->chunk_ready_list, chunk);
			MemoryContextSwitchTo(oldcxt);
		}
		else if (ret == CUDA_ERROR_NOT_READY)
		{
			if (!first_stream)
				first_stream = chunk->stream;
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("cuda: failed on query status of stream : %s",
							cuda_error_to_string(ret))));
		}
	}

	if (sestate->chunk_ready_list == NIL)
	{
		Assert(first_stream != NULL);
		cuStreamSynchronize(first_stream);
		goto retry;
	}
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

		oldcxt = MemoryContextSwitchTo(pgstrom_scan_memcxt);
		chunk = palloc0(sizeof(PgStromChunkBuf));
		chunk->rowid = rowid;
		chunk->nattrs = RelationGetNumberOfAttributes(sestate->es_relation);
		chunk->cs_nulls = palloc0(sizeof(int) * chunk->nattrs);
		chunk->cs_values = palloc0(sizeof(int) * chunk->nattrs);
		MemoryContextSwitchTo(oldcxt);

		if (sestate->dev_module)
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
			chunk->devmem_size = dma_offset;

			/*
			 * Load necessary column store
			 */
			chunk->cs_rowmap = (uint8 *)dma_buffer;
			chunk->cs_rownums = VARBITLEN(rowmap);
			if (VARBITLEN(rowmap) != PGSTROM_CHUNK_SIZE)
				memset(chunk->cs_rowmap, 0,
					   PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
			memcpy(chunk->cs_rowmap, VARBITS(rowmap), VARBITBYTES(rowmap));

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
			oldcxt = MemoryContextSwitchTo(pgstrom_scan_memcxt);
			sestate->chunk_exec_list
				= lappend(sestate->chunk_exec_list, chunk);
			MemoryContextSwitchTo(oldcxt);
		}
		else
		{
			/*
			 * In the case when the supplied plan has no qualifier,
			 * all the chunks are ready to scan using rowid.
			 */
			chunk->cs_rownums = VARBITLEN(rowmap);
			chunk->cs_rowmap
				= MemoryContextAllocZero(pgstrom_scan_memcxt,
										 PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
			memcpy(chunk->cs_rowmap, VARBITS(rowmap), VARBITBYTES(rowmap));

			oldcxt = MemoryContextSwitchTo(pgstrom_scan_memcxt);
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
	ListCell		   *cell;
	AttrNumber			nattrs;
	MemoryContext		oldcxt;
	bool				nevermatch = false;
	char			   *kernel_source = NULL;
	Bitmapset		   *clause_cols = NULL;
	Bitmapset		   *required_cols = NULL;

	nattrs = RelationGetNumberOfAttributes(fss->ss.ss_currentRelation);

	foreach (cell, fscan->fdwplan->fdw_private)
	{
		DefElem	   *defel = (DefElem *)lfirst(cell);

		if (strcmp(defel->defname, "nevermatch") == 0)
		{
			nevermatch = intVal(defel->arg);
		}
		else if (strcmp(defel->defname, "kernel_source") == 0)
		{
			kernel_source = strVal(defel->arg);
		}
		else if (strcmp(defel->defname, "clause_cols") == 0)
		{
			int		csidx = (intVal(defel->arg));

			Assert(csidx > 0);

			clause_cols = bms_add_member(clause_cols, csidx);
		}
		else if (strcmp(defel->defname, "required_cols") == 0)
		{
			int		csidx = (intVal(defel->arg));

			if (csidx > 0)
				required_cols = bms_add_member(required_cols, csidx);
			else
				Assert(fscan->fsSystemCol);
		}
		else
			elog(ERROR, "pg_strom: unexpected private plan information: %s",
				 defel->defname);
	}

	/*
	 * Allocate PgStromExecState object within pgstrom_scan_memcxt
	 */
	oldcxt = MemoryContextSwitchTo(pgstrom_scan_memcxt);

	sestate = palloc0(sizeof(PgStromExecState));

	sestate->nevermatch = nevermatch;
	sestate->required_cols = bms_copy(required_cols);
	sestate->clause_cols = bms_copy(clause_cols);
	if (kernel_source)
		sestate->kernel_source = pstrdup(kernel_source);

	sestate->es_owner = CurrentResourceOwner;
	sestate->es_relation = fss->ss.ss_currentRelation;
    sestate->es_snapshot = fss->ss.ps.state->es_snapshot;
	sestate->es_memcxt = fss->ss.ps.ps_ExprContext->ecxt_per_query_memory;

	sestate->cs_scan = palloc0(sizeof(IndexScanDesc) * nattrs);
	sestate->cs_cur_values = palloc0(sizeof(ArrayType *) * nattrs);
	sestate->cs_cur_rowid_min = palloc0(sizeof(int64) * nattrs);
	sestate->cs_cur_rowid_max = palloc0(sizeof(int64) * nattrs);

	sestate->chunk_exec_list = NIL;
	sestate->chunk_ready_list = NIL;
	sestate->curr_chunk = NULL;
	sestate->curr_index = 0;

	pgstrom_exec_state_list
		= lappend(pgstrom_exec_state_list, sestate);
	MemoryContextSwitchTo(oldcxt);

	if (!sestate->nevermatch && sestate->kernel_source != NULL)
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

		ret = cuFuncSetCacheConfig(sestate->dev_function,
								   CU_FUNC_CACHE_PREFER_L1);
		if (ret != CUDA_SUCCESS)
        {
			cuModuleUnload(sestate->dev_module);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("cuda: failed to set L1 cache setting : %s",
                            cuda_error_to_string(ret))));
		}
	}
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

		pgstrom_sync_kernel_exec(sestate);
		/*
		 * XXX - at least one chunk should be synchronized.
		 */
		Assert(sestate->chunk_ready_list != NIL);
		sestate->curr_chunk = list_head(sestate->chunk_ready_list);
		sestate->curr_index = 0;
	}
retry:
	if (!pgstrom_scan_chunk_buffer(sestate, slot))
	{
		PgStromChunkBuf	*chunk = lfirst(sestate->curr_chunk);

		/*
		 * Release the current chunk being already scanned
		 */
		sestate->chunk_ready_list
			= list_delete(sestate->chunk_ready_list, chunk);
		pgstrom_release_chunk_buffer(sestate, chunk);

		/*
		 * Is the concurrent chunks ready now?
		 */
		pgstrom_sync_kernel_exec(sestate);

		num_chunks = (pgstrom_max_async_chunks -
					  list_length(sestate->chunk_exec_list));
		num_chunks = pgstrom_load_chunk_buffer(sestate, num_chunks);

		if (sestate->chunk_ready_list == NIL)
		{
			if (num_chunks < 1)
				return slot;
			else
				pgstrom_sync_kernel_exec(sestate);
		}
		Assert(sestate->chunk_ready_list != NIL);
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
	ListCell		   *cell;

	/*
	 * Rewind rowid scan
	 */
	if (sestate->ri_scan != NULL)
		heap_endscan(sestate->ri_scan);
	sestate->ri_scan = heap_beginscan(sestate->relset->rowid_rel,
									  sestate->es_snapshot,
									  0, NULL);
	/*
	 * Chunk buffers being released
	 */
	foreach (cell, sestate->chunk_exec_list)
		pgstrom_release_chunk_buffer(sestate, lfirst(cell));
    list_free(sestate->chunk_exec_list);

    foreach (cell, sestate->chunk_ready_list)
        pgstrom_release_chunk_buffer(sestate, lfirst(cell));
    list_free(sestate->chunk_ready_list);

	/*
	 * Clear current chunk pointer
	 */
	sestate->curr_chunk = NULL;
	sestate->curr_index = 0;
}

void
pgstrom_end_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
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
	pgstrom_release_exec_state(sestate);
}

/*
 * pgstrom_scan_init
 *
 * Initialize stuff related to scan.c
 */
void
pgstrom_scan_init(void)
{
	/*
	 * exec-state and chunk-buffer
	 */
	pgstrom_scan_memcxt
		= AllocSetContextCreate(TopMemoryContext,
								"pg_strom exec-state",
								ALLOCSET_DEFAULT_MINSIZE,
								ALLOCSET_DEFAULT_INITSIZE,
								ALLOCSET_DEFAULT_MAXSIZE);
	RegisterResourceReleaseCallback(pgstrom_release_resources, NULL);

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
}
