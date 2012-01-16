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
#include "utils/int8.h"
#include "utils/lsyscache.h"
#include "utils/pg_lzcompress.h"
#include "utils/rel.h"
#include "utils/resowner.h"
#include "utils/varbit.h"
#include "pg_strom.h"

typedef struct {
	int64		rowid;
	int			nattrs;
	int			nitems;		/* number of items */
	bits8	   *cs_rowmap;	/* also, head of the page locked memory */
	int		   *cs_nulls;	/* offset from the cs_rowmap, or 0 */
	int		   *cs_values;	/* offset from the cs_rowmap, or 0 */
	size_t		devmem_size;
	CUdeviceptr	devmem;
	CUstream	stream;
	CUevent		events[4];
} PgStromChunkBuf;

#define chunk_cs_nulls(chunk,csidx)		\
	((bits8 *)((chunk)->cs_rowmap + (chunk)->cs_nulls[(csidx)]))
#define chunk_cs_values(chunk,csidx)	\
	((char *)((chunk)->cs_rowmap + (chunk)->cs_values[(csidx)]))

typedef struct {
	int			dev_index;
	CUmodule	dev_module;
	CUfunction	dev_function;
	uint32		dev_nthreads;
} PgStromDevContext;

typedef struct {
	/* Parameters come from planner */
	bool			nevermatch;		/* true, if no items shall be matched */
	Bitmapset	   *required_cols;	/* columns being returned to executor */
	Bitmapset	   *clause_cols;	/* columns being copied to device */

	/* copy from EState */
	ResourceOwner	es_owner;		/* copy from CurrentResourceOwner */
	Relation		es_relation;	/* copy from ScanState */
	Snapshot		es_snapshot;	/* copy from EState */
	MemoryContext	es_memcxt;		/* memory context contains this object */

	/* shadow tables and indexes */
	Relation		id_rel;			/* shadow rowid table */
	Relation	   *cs_rels;		/* shadow column-store table */
	Relation	   *cs_idxs;		/* shadow column-store index */

	/* scan descriptors */
	HeapScanDesc	id_scan;		/* scan on rowid map */
	IndexScanDesc  *cs_scan;		/* scan on column store */
	bytea		  **cs_cur_isnull;
	bytea		  **cs_cur_values;
	int64		   *cs_cur_rowid_min;
	int64		   *cs_cur_rowid_max;

	/* list of the chunk */
	List		   *chunk_exec_list;	/* chunks in device execution */
	List		   *chunk_ready_list;	/* chunks in ready to scaning */
	ListCell	   *curr_chunk;
	int				curr_index;

	/* CUDA related stuff */
	List		   *dev_list;		/* list of PgStromDevContext */
	ListCell	   *dev_curr;		/* currently used item of dev_list */

	/* Profiling stuff [us] */
	uint64			pf_pgstrom_total;	/* total time in this module */
	uint64			pf_jit_compile;		/* time to jit compile */
	uint64			pf_device_init;		/* time to device initialization */
	uint64			pf_async_memcpy;	/* time to async memcpy */
	uint64			pf_async_kernel;	/* time to async kernel exec */
	uint64			pf_synchronization;	/* time to synchronization */
	uint64			pf_load_chunk;		/* time to load chunks */
} PgStromExecState;

#define TIMEVAL_ELAPSED(tv1,tv2)					\
	(((tv2)->tv_sec  - (tv1)->tv_sec) * 1000000 +	\
	 ((tv2)->tv_usec - (tv1)->tv_usec))

/*
 * Declarations
 */
static MemoryContext	pgstrom_scan_memcxt;
static List	   *pgstrom_exec_state_list = NIL;
static int		pgstrom_max_async_chunks;
static bool		pgstrom_exec_profile;

static void
pgstrom_release_chunk_buffer(PgStromExecState *sestate,
							 PgStromChunkBuf *chunk)
{
	/*
	 * If the supplied chunk buffer is under execution of kernel code,
	 * we need to synchronize its completion, then release its resources,
	 * because asynchronous memory copy may be ready to run...
	 */
	if (chunk->stream != NULL)
	{
		if (cuStreamQuery(chunk->stream) == CUDA_ERROR_NOT_READY)
			cuStreamSynchronize(chunk->stream);
		cuStreamDestroy(chunk->stream);
	}
	if (chunk->devmem)
		cuMemFree(chunk->devmem);
	if (sestate->dev_list != NIL)
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

	foreach (cell, sestate->chunk_ready_list)
		pgstrom_release_chunk_buffer(sestate, lfirst(cell));

	foreach (cell, sestate->dev_list)
	{
		PgStromDevContext *dev_cxt = lfirst(cell);
		if (dev_cxt->dev_module)
			cuModuleUnload(dev_cxt->dev_module);
	}
	MemoryContextDelete(sestate->es_memcxt);
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

static void
pgstrom_exec_kernel_qual(PgStromDevContext *dev_cxt, PgStromChunkBuf *chunk)
{
	CUdeviceptr	   *kernel_data;
	void		  **kernel_args;
	CUresult		ret;
	unsigned int	n_blocks;
	unsigned int	n_threads;
	int				i, j;

	PG_TRY();
	{
		ret = cuStreamCreate(&chunk->stream, 0);
		if (ret != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to create a stream: %s",
							cuda_error_to_string(ret))));

		ret = cuMemAlloc(&chunk->devmem, chunk->devmem_size);
		if (ret != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to alloc device memory: %s",
							cuda_error_to_string(ret))));
		/*
		 * Create event object to track elapsed time
		 */
		if (pgstrom_exec_profile)
		{
			int		i;

			for (i = 0; i < lengthof(chunk->events); i++)
			{
				ret = cuEventCreate(&chunk->events[i], CU_EVENT_DEFAULT);
				if (ret != CUDA_SUCCESS)
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("Failed to create CUDA event: %s",
									cuda_error_to_string(ret))));
			}
		}

		/*
		 * Arguments copy from host to device
		 */
		if (pgstrom_exec_profile)
		{
			ret = cuEventRecord(chunk->events[0], chunk->stream);
			if (ret != CUDA_SUCCESS)
				elog(ERROR, "Failed to enqueue an event: %s",
					 cuda_error_to_string(ret));
		}

		ret = cuMemcpyHtoDAsync(chunk->devmem,
								chunk->cs_rowmap,
								chunk->devmem_size,
								chunk->stream);
		if (ret != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to enqueue asynchronous memcpy: %s",
							cuda_error_to_string(ret))));

		if (pgstrom_exec_profile)
		{
			ret = cuEventRecord(chunk->events[1], chunk->stream);
			if (ret != CUDA_SUCCESS)
				elog(ERROR, "Failed to enqueue an event: %s",
					 cuda_error_to_string(ret));
		}

		/*
		 * Setup kernel arguments
		 */
		kernel_data = alloca((2 + 2 * chunk->nattrs) * sizeof(CUdeviceptr));
		kernel_args = alloca((2 + 2 * chunk->nattrs) * sizeof(void *));
		kernel_data[0] = chunk->nitems;
		kernel_args[0] = &kernel_data[0];
		kernel_data[1] = chunk->devmem;
		kernel_args[1] = &kernel_data[1];
		for (i=0, j=2; i < chunk->nattrs; i++)
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

		/*
		 * Launch kernel function
		 */
		n_threads = dev_cxt->dev_nthreads;
		n_blocks = (chunk->nitems + n_threads * BITS_PER_BYTE - 1)
			/ (BITS_PER_BYTE * n_threads);
		ret = cuLaunchKernel(dev_cxt->dev_function,
							 n_blocks, 1, 1,
							 n_threads, 1, 1,
							 0,
							 chunk->stream,
							 kernel_args,
							 NULL);
		if (ret != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to enqueue kernel execution: %s",
							cuda_error_to_string(ret))));

		/*
		 * Write back the result from device to host
		 */
		if (pgstrom_exec_profile)
		{
			ret = cuEventRecord(chunk->events[2], chunk->stream);
			if (ret != CUDA_SUCCESS)
				elog(ERROR, "Failed to enqueue an event: %s",
					 cuda_error_to_string(ret));
		}

		ret = cuMemcpyDtoHAsync(chunk->cs_rowmap,
								chunk->devmem,
								chunk->nitems / BITS_PER_BYTE,
								chunk->stream);
		if (ret != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to enqueue asynchronous memcpy: %s",
							cuda_error_to_string(ret))));

		if (pgstrom_exec_profile)
		{
			ret = cuEventRecord(chunk->events[3], chunk->stream);
			if (ret != CUDA_SUCCESS)
				elog(ERROR, "Failed to enqueue an event: %s",
					 cuda_error_to_string(ret));
		}
	}
	PG_CATCH();
	{
		int		i;

		if (chunk->stream)
		{
			cuStreamSynchronize(chunk->stream);
			cuStreamDestroy(chunk->stream);
		}
		for (i = 0; i < lengthof(chunk->events); i++)
		{
			if (chunk->events[i])
				cuEventDestroy(chunk->events[i]);
		}
		if (chunk->devmem)
			cuMemFree(chunk->devmem);
		cuMemFreeHost(chunk->cs_rowmap);

		PG_RE_THROW();
	}
	PG_END_TRY();
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

			if (pgstrom_exec_profile)
			{
				float	elapsed;

				Assert(chunk->events[0] != NULL &&
					   chunk->events[1] != NULL &&
					   chunk->events[2] != NULL &&
					   chunk->events[3] != NULL);

				ret = cuEventElapsedTime(&elapsed,
										 chunk->events[0],
										 chunk->events[1]);
				if (ret != CUDA_SUCCESS)
					elog(ERROR, "Failed to get elapsed time: %s",
						 cuda_error_to_string(ret));
				sestate->pf_async_memcpy += (uint64)(elapsed * 1000.0);

				ret = cuEventElapsedTime(&elapsed,
										 chunk->events[1],
										 chunk->events[2]);
				if (ret != CUDA_SUCCESS)
					elog(ERROR, "Failed to get elapsed time: %s",
						 cuda_error_to_string(ret));
				sestate->pf_async_kernel += (uint64)(elapsed * 1000.0);

				ret = cuEventElapsedTime(&elapsed,
										 chunk->events[2],
										 chunk->events[3]);
				if (ret != CUDA_SUCCESS)
					elog(ERROR, "Failed to get elapsed time: %s",
						 cuda_error_to_string(ret));
				sestate->pf_async_memcpy += (uint64)(elapsed * 1000.0);
			}
			oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
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
		struct timeval tv1, tv2;

		Assert(first_stream != NULL);
		if (pgstrom_exec_profile)
			gettimeofday(&tv1, NULL);

		cuStreamSynchronize(first_stream);

		if (pgstrom_exec_profile)
		{
			gettimeofday(&tv2, NULL);
			sestate->pf_synchronization += TIMEVAL_ELAPSED(&tv1, &tv2);
		}
		goto retry;
	}
}

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
						  PgStromChunkBuf *chunk, int csidx)
{
	Form_pg_attribute	attr;
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
	memset(chunk_cs_nulls(chunk,csidx), -1, chunk->nitems / BITS_PER_BYTE);

	/*
	 * Try to scan column store with cs_rowid betweem rowid and
	 * (rowid + chunk->nitems)
	 */
	ScanKeyInit(&skeys[0],
				Anum_pg_strom_rowid,
				BTGreaterEqualStrategyNumber, F_INT8GE,
				Int64GetDatum(chunk->rowid));
	ScanKeyInit(&skeys[1],
				Anum_pg_strom_rowid,
				BTLessStrategyNumber, F_INT8LT,
				Int64GetDatum(chunk->rowid + chunk->nitems));

	index_rescan(sestate->cs_scan[csidx], skeys, 2, NULL, 0);

	while (HeapTupleIsValid(tup = index_getnext(sestate->cs_scan[csidx],
												ForwardScanDirection)))
	{
		TupleDesc	tupdesc;
		Datum		values[Natts_pg_strom];
		bool		isnull[Natts_pg_strom];
		int64		cur_rowid;
		uint32		cur_nitems;
		int64		offset;

		tupdesc = RelationGetDescr(sestate->cs_rels[csidx]);
		heap_deform_tuple(tup, tupdesc, values, isnull);
		Assert(!isnull[Anum_pg_strom_rowid - 1] &&
			   !isnull[Anum_pg_strom_nitems - 1] &&
			   !isnull[Anum_pg_strom_values - 1]);

		cur_rowid = DatumGetInt64(values[Anum_pg_strom_rowid-1]);
		cur_nitems = DatumGetUInt32(values[Anum_pg_strom_nitems-1]);
		offset = cur_rowid - chunk->rowid;

		Assert(cur_nitems % BITS_PER_BYTE == 0);
	    Assert(offset + cur_nitems <= chunk->nitems);

		if (!isnull[Anum_pg_strom_isnull - 1])
		{
			deconstruct_cs_bytea(chunk_cs_nulls(chunk,csidx) +
								 offset / BITS_PER_BYTE,
								 values[Anum_pg_strom_isnull - 1],
								 cur_nitems / BITS_PER_BYTE);
		}
		else
		{
			/*
			 * 'isnull' == NULL means; all the items within 'values' array
			 * is not null, so clear the corresponding scope of null bitmap.
			 */
			memset(chunk_cs_nulls(chunk,csidx) + offset / BITS_PER_BYTE,
				   0,
				   cur_nitems / BITS_PER_BYTE);
		}
		deconstruct_cs_bytea(chunk_cs_values(chunk,csidx) +
							 offset * attr->attlen,
							 values[Anum_pg_strom_values - 1],
							 cur_nitems * attr->attlen);
	}
}

static int
pgstrom_load_chunk_buffer(PgStromExecState *sestate, int num_chunks)
{
	int		loaded_chunks;

	if (sestate->id_scan == NULL)
		return -1;

	for (loaded_chunks = 0; loaded_chunks < num_chunks; loaded_chunks++)
	{
		TupleDesc	tupdesc;
		HeapTuple	tuple;
		Datum		values[Natts_pg_strom];
		bool		isnull[Natts_pg_strom];
		int64		rowid;
		uint32		nitems;
		MemoryContext oldcxt;
		PgStromChunkBuf	*chunk;
		struct timeval tv1, tv2;

		if (pgstrom_exec_profile)
			gettimeofday(&tv1, NULL);

		tuple = heap_getnext(sestate->id_scan, ForwardScanDirection);
		if (!HeapTupleIsValid(tuple))
		{
			/* No any tuples, close the sequential rowid scan */
			heap_endscan(sestate->id_scan);
			sestate->id_scan = NULL;
			break;
		}

		tupdesc = RelationGetDescr(sestate->id_rel);
		heap_deform_tuple(tuple, tupdesc, values, isnull);
		Assert(!isnull[Anum_pg_strom_rowid - 1] &&
			   !isnull[Anum_pg_strom_nitems - 1] &&
			   !isnull[Anum_pg_strom_isnull - 1]);

		rowid = DatumGetInt64(values[Anum_pg_strom_rowid - 1]);
		nitems = DatumGetUInt32(values[Anum_pg_strom_nitems - 1]);

		oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
		chunk = palloc0(sizeof(PgStromChunkBuf));
		chunk->rowid = rowid;
		chunk->nattrs = RelationGetNumberOfAttributes(sestate->es_relation);
		chunk->nitems = nitems;
		chunk->cs_nulls = palloc0(sizeof(int) * chunk->nattrs);
		chunk->cs_values = palloc0(sizeof(int) * chunk->nattrs);
		MemoryContextSwitchTo(oldcxt);
		Assert(chunk->nitems % BITS_PER_BYTE == 0);

		if (sestate->dev_list != NIL)
		{
			PgStromDevContext *dev_cxt;
			Bitmapset  *temp;
			AttrNumber	csidx;
			uint16		attlen;
			char	   *dma_buffer;
			size_t		dma_offset;
			CUresult	ret;

			/*
			 * Switch to the next GPU device to be used
			 */
			do {
				if (!sestate->dev_curr)
					sestate->dev_curr = list_head(sestate->dev_list);
				else
					sestate->dev_curr = lnext(sestate->dev_curr);
			} while (!sestate->dev_curr);
			dev_cxt = lfirst(sestate->dev_curr);
			pgstrom_set_device_context(dev_cxt->dev_index);

			/*
			 * Compute and allocate required size of column store
			 */
			dma_offset = (chunk->nitems / BITS_PER_BYTE);

			tupdesc = RelationGetDescr(sestate->es_relation);
			temp = bms_copy(sestate->clause_cols);
			while ((csidx = (bms_first_member(temp)-1)) >= 0)
			{
				attlen = tupdesc->attrs[csidx]->attlen;
				Assert(attlen > 0);

				chunk->cs_values[csidx] = dma_offset;
				dma_offset += chunk->nitems * attlen;
				chunk->cs_nulls[csidx] = dma_offset;
				dma_offset += chunk->nitems / BITS_PER_BYTE;
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
			deconstruct_cs_bytea(chunk->cs_rowmap,
								 values[Anum_pg_strom_isnull - 1],
								 chunk->nitems / BITS_PER_BYTE);

			temp = bms_copy(sestate->clause_cols);
			while ((csidx = (bms_first_member(temp)-1)) >= 0)
				pgstrom_load_column_store(sestate, chunk, csidx);
			bms_free(temp);

			if (pgstrom_exec_profile)
			{
				gettimeofday(&tv2, NULL);
				sestate->pf_load_chunk += TIMEVAL_ELAPSED(&tv1, &tv2);
			}

			/*
			 * Asynchronous execution of kernel code on this chunk
			 */
			pgstrom_exec_kernel_qual(dev_cxt, chunk);

			/*
			 * XXX - Do we need to pay attention of the case when
			 * lappend raises an error?
			 */
			oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
			sestate->chunk_exec_list
				= lappend(sestate->chunk_exec_list, chunk);
			MemoryContextSwitchTo(oldcxt);
		}
		else
		{
			/*
			 * In case of the supplied plan without no qualifiers,
			 * all the chunks are ready to scan using rowid.
			 */
			oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
			chunk->cs_rowmap = palloc0(chunk->nitems / BITS_PER_BYTE);
			deconstruct_cs_bytea(chunk->cs_rowmap,
								 values[Anum_pg_strom_isnull - 1],
								 chunk->nitems / BITS_PER_BYTE);
			if (pgstrom_exec_profile)
			{
				gettimeofday(&tv2, NULL);
				sestate->pf_load_chunk += TIMEVAL_ELAPSED(&tv1, &tv2);
			}
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
	ScanKeyData	skey;
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	Datum		values[Natts_pg_strom];
	bool		isnull[Natts_pg_strom];
	int64		cur_rowid;
	uint32		cur_nitems;
	uint8	   *nullmap;
	int			index;
	MemoryContext	oldcxt;

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
			int		count = 2;	/* try twice at maximun */

			while (count-- > 0)
			{
				tuple = index_getnext(sestate->cs_scan[csidx],
									  ForwardScanDirection);
				if (!HeapTupleIsValid(tuple))
					break;

				tupdesc = RelationGetDescr(sestate->cs_rels[csidx]);
				heap_deform_tuple(tuple, tupdesc, values, isnull);
				Assert(!isnull[Anum_pg_strom_rowid - 1] &&
					   !isnull[Anum_pg_strom_nitems - 1] &&
					   !isnull[Anum_pg_strom_values - 1]);

				cur_rowid = DatumGetInt64(values[Anum_pg_strom_rowid - 1]);
				cur_nitems = DatumGetUInt32(values[Anum_pg_strom_nitems - 1]);

				/* Hit! */
				if (rowid >= cur_rowid &&
					rowid <= cur_rowid + cur_nitems - 1)
				{
					Datum	cur_isnull = values[Anum_pg_strom_isnull - 1];
					Datum	cur_values = values[Anum_pg_strom_values - 1];

					if (sestate->cs_cur_isnull[csidx])
						pfree(sestate->cs_cur_isnull[csidx]);
					if (sestate->cs_cur_values[csidx])
						pfree(sestate->cs_cur_values[csidx]);

					oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
					sestate->cs_cur_isnull[csidx]
						= (!isnull[Anum_pg_strom_isnull - 1] ?
						   PG_DETOAST_DATUM_COPY(cur_isnull) : NULL);
					sestate->cs_cur_values[csidx]
						= PG_DETOAST_DATUM_COPY(cur_values);
					MemoryContextSwitchTo(oldcxt);
					sestate->cs_cur_rowid_min[csidx] = cur_rowid;
					sestate->cs_cur_rowid_max[csidx]
						= cur_rowid + cur_nitems - 1;
					goto out;
				}
			}
		}

		/* Reset cached values */
		if (sestate->cs_cur_values[csidx])
		{
			if (sestate->cs_cur_isnull[csidx])
				pfree(sestate->cs_cur_isnull[csidx]);
			pfree(sestate->cs_cur_values[csidx]);
			sestate->cs_cur_isnull[csidx] = NULL;
			sestate->cs_cur_values[csidx] = NULL;
			sestate->cs_cur_rowid_min[csidx] = -1;
			sestate->cs_cur_rowid_max[csidx] = -1;
		}

		/*
		 * Rewind the current index scan to fetch a cs-tuple with
		 * required rowid.
		 */
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

		tupdesc = RelationGetDescr(sestate->cs_rels[csidx]);
		heap_deform_tuple(tuple, tupdesc, values, isnull);
		Assert(!isnull[Anum_pg_strom_rowid - 1] &&
			   !isnull[Anum_pg_strom_nitems - 1] &&
			   !isnull[Anum_pg_strom_values - 1]);

		cur_rowid = DatumGetInt64(values[Anum_pg_strom_rowid - 1]);
		cur_nitems = DatumGetUInt32(values[Anum_pg_strom_nitems - 1]);
		if (rowid < cur_rowid || rowid >= cur_rowid + cur_nitems)
		{
			slot->tts_isnull[csidx] = true;
			slot->tts_values[csidx] = (Datum) 0;
			return;
		}
		oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
		sestate->cs_cur_isnull[csidx] =
			(!isnull[Anum_pg_strom_isnull - 1] ?
			 PG_DETOAST_DATUM_COPY(values[Anum_pg_strom_isnull - 1]) : NULL);
		sestate->cs_cur_values[csidx] =
			PG_DETOAST_DATUM_COPY(values[Anum_pg_strom_values - 1]);
		sestate->cs_cur_rowid_min[csidx] = cur_rowid;
		sestate->cs_cur_rowid_max[csidx] = cur_rowid + cur_nitems - 1;
		MemoryContextSwitchTo(oldcxt);

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
	index = rowid - sestate->cs_cur_rowid_min[csidx];
	nullmap = (!sestate->cs_cur_isnull[csidx] ?
			   NULL : (uint8 *)VARDATA(sestate->cs_cur_isnull[csidx]));
	if (!nullmap ||
		(nullmap[index / BITS_PER_BYTE] & (1<<(index % BITS_PER_BYTE))) == 0)
	{
		Form_pg_attribute	attr = slot->tts_tupleDescriptor->attrs[csidx];

		if (attr->attlen > 0)
			slot->tts_values[csidx]
				= fetch_att(VARDATA(sestate->cs_cur_values[csidx]) +
							attr->attlen * index,
							attr->attbyval, attr->attlen);
		else
		{
			char   *temp = VARDATA(sestate->cs_cur_values[csidx]);
			int		offset = ((uint16 *)temp)[index];

			Assert(offset > 0);
			slot->tts_values[csidx] = PointerGetDatum(temp + offset);
		}
		slot->tts_isnull[csidx] = false;
	}
	else
	{
		slot->tts_isnull[csidx] = true;
		slot->tts_values[csidx] = (Datum) 0;
	}
}

#define BITS_PER_DATUM	(sizeof(Datum) * BITS_PER_BYTE)
static bool
pgstrom_scan_chunk_buffer(PgStromExecState *sestate, TupleTableSlot *slot)
{
	PgStromChunkBuf	*chunk = lfirst(sestate->curr_chunk);
	int		index;

	index = sestate->curr_index;
	while (index < chunk->nitems)
	{
		int		index_h;
		int		index_l;
		int		csidx;
		int64	rowid;

		if ((index & (BITS_PER_DATUM - 1)) == 0 &&
			((Datum *)chunk->cs_rowmap)[index / BITS_PER_DATUM] == -1)
		{
			index += BITS_PER_DATUM;
			continue;
		}
		index_h = index / BITS_PER_BYTE;
		index_l = index % BITS_PER_BYTE;
		if ((chunk->cs_rowmap[index_h] & (1 << index_l)) != 0)
		{
			index++;
			continue;
		}

		rowid = chunk->rowid + index;
		for (csidx=0; csidx < chunk->nattrs; csidx++)
		{
			struct timeval	tv1, tv2;

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
			if (pgstrom_exec_profile)
				gettimeofday(&tv1, NULL);

			pgstrom_scan_column_store(sestate, csidx, rowid, slot);

			if (pgstrom_exec_profile)
			{
				gettimeofday(&tv2, NULL);
				sestate->pf_load_chunk += TIMEVAL_ELAPSED(&tv1, &tv2);
			}
		}
		ExecStoreVirtualTuple(slot);
		/* update next index to be fetched */
		sestate->curr_index = index + 1;
		return true;
	}
	return false;	/* end of chunk, need next chunk! */
}

static PgStromDevContext *
pgstrom_init_exec_device(void *image, int dev_index,
						 const int64 const_buffer[], int const_nums)
{
	const PgStromDeviceInfo *dev_info = pgstrom_get_device_info(dev_index);
	PgStromDevContext  *dev_cxt;
	CUresult	ret;
	int			n_threads;

	/*
	 * Allocate PgStromDevCtx for each GPU devices
	 */
	dev_cxt = palloc0(sizeof(PgStromDevContext));
	dev_cxt->dev_index = dev_index;

	/*
	 * Switch the current GPU device
	 */
	pgstrom_set_device_context(dev_index);

	/*
	 * Load the module to GPU device
	 */
	ret = cuModuleLoadData(&dev_cxt->dev_module, image);
	if (ret != CUDA_SUCCESS)
	{
		pfree(dev_cxt);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to load device executable module: %s",
						cuda_error_to_string(ret))));
	}

	/*
	 * Lookup function to execute the supplied qualifer
	 */
	ret = cuModuleGetFunction(&dev_cxt->dev_function,
							  dev_cxt->dev_module,
							  "pgstrom_qual");
	if (ret != CUDA_SUCCESS)
	{
		cuModuleUnload(dev_cxt->dev_module);
		pfree(dev_cxt);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to get device function \"%s\": %s",
						"pgstrom_qual", cuda_error_to_string(ret))));
	}

	/*
	 * Switch L1 cache config -- Because qualifier does not use shared
	 * memory anyway, so larger L1 cache will give us performance gain.
	 */
	ret = cuFuncSetCacheConfig(dev_cxt->dev_function,
							   CU_FUNC_CACHE_PREFER_L1);
	if (ret != CUDA_SUCCESS)
	{
		cuModuleUnload(dev_cxt->dev_module);
		pfree(dev_cxt);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to switch L1 cache configuration: %s",
						cuda_error_to_string(ret))));
	}

	/*
	 * Set up constant values, if exists
	 */
	if (const_nums > 0)
	{
		CUdeviceptr	const_ptr;
		size_t		const_size;

		ret = cuModuleGetGlobal(&const_ptr, &const_size,
								dev_cxt->dev_module, "constval");
		if (ret != CUDA_SUCCESS)
		{
			cuModuleUnload(dev_cxt->dev_module);
			pfree(dev_cxt);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to lookup constant value symbol: %s",
							cuda_error_to_string(ret))));
		}
		if (const_size != sizeof(int64) * const_nums)
		{
			cuModuleUnload(dev_cxt->dev_module);
			pfree(dev_cxt);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Unexpected constant size %lu, but %lu expected",
							const_size, sizeof(int64) * const_nums)));
		}
		ret = cuMemcpyHtoD(const_ptr, const_buffer, const_size);
		if (ret != CUDA_SUCCESS)
		{
			cuModuleUnload(dev_cxt->dev_module);
			pfree(dev_cxt);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Failed to copy constant values: %s",
							cuda_error_to_string(ret))));
		}
	}

	/*
	 * Compute an optimized number of threads and blocks
	 */
	ret = cuFuncGetAttribute(&n_threads,
							 CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
							 dev_cxt->dev_function);
	if (ret != CUDA_SUCCESS)
	{
		cuModuleUnload(dev_cxt->dev_module);
		pfree(dev_cxt);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to reference number of registers: %s",
						cuda_error_to_string(ret))));
	}

	/*
	 * Larger number of threads within a particular block is better
	 * strategy as long as it can be executable; from the perspective
	 * that increase occupacy of streaming processor.
	 */
	n_threads = (n_threads - (n_threads % dev_info->dev_proc_warp_sz));

	/*
	 * However, it is not desirable the number of threads are too large
	 * not to utilize all the streaming processors concurrently.
	 * In this case, we adjust number of threads to appropriate level.
	 */
	if (PGSTROM_CHUNK_SIZE / n_threads < dev_info->dev_proc_nums)
	{
		n_threads = PGSTROM_CHUNK_SIZE / dev_info->dev_proc_nums;
		n_threads -= (n_threads % dev_info->dev_proc_warp_sz);
	}
	dev_cxt->dev_nthreads = n_threads;

	/*
	 * TODO: maximun number of concurrent chunks also shoukd be modified
	 * to avoid over-consumption of device memory.
	 */

	return dev_cxt;
}

static PgStromExecState *
pgstrom_init_exec_state(ForeignScanState *fss)
{
	ForeignScan		   *fscan = (ForeignScan *) fss->ss.ps.plan;
	PgStromExecState   *sestate;
	ListCell		   *cell;
	AttrNumber			nattrs;
	MemoryContext		newcxt;
	MemoryContext		oldcxt;
	bool				nevermatch = false;
	char			   *kernel_source = NULL;
	List			   *const_values = NIL;
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
		else if (strcmp(defel->defname, "const_values") == 0)
		{
			const_values = (List *)defel->arg;
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
	 * Create per-scan memory context
	 */
	newcxt = AllocSetContextCreate(pgstrom_scan_memcxt,
                                   "PG-Strom Exec Status",
                                   ALLOCSET_DEFAULT_MINSIZE,
                                   ALLOCSET_DEFAULT_INITSIZE,
                                   ALLOCSET_DEFAULT_MAXSIZE);
	PG_TRY();
	{
		sestate = MemoryContextAllocZero(newcxt, sizeof(PgStromExecState));
		sestate->es_owner = CurrentResourceOwner;
		sestate->es_relation = fss->ss.ss_currentRelation;
		sestate->es_snapshot = fss->ss.ps.state->es_snapshot;
		sestate->es_memcxt = newcxt;

		/*
		 * The new sestate should be linked to the pgstrom_exec_state_list
		 * at first; to prevent unexpected memory leak on errors.
		 */
		oldcxt = MemoryContextSwitchTo(pgstrom_scan_memcxt);
		pgstrom_exec_state_list
			= lappend(pgstrom_exec_state_list, sestate);
		MemoryContextSwitchTo(oldcxt);
	}
	PG_CATCH();
	{
		MemoryContextDelete(newcxt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/*
	 * Initialization of rest of fields
	 */
	oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);

	sestate->nevermatch = nevermatch;
	sestate->required_cols = bms_copy(required_cols);
	sestate->clause_cols = bms_copy(clause_cols);

	sestate->cs_rels = palloc0(sizeof(Relation) * nattrs);
	sestate->cs_idxs = palloc0(sizeof(Relation) * nattrs);

	sestate->cs_scan = palloc0(sizeof(IndexScanDesc) * nattrs);
	sestate->cs_cur_isnull = palloc0(sizeof(bytea *) * nattrs);
	sestate->cs_cur_values = palloc0(sizeof(bytea *) * nattrs);
	sestate->cs_cur_rowid_min = palloc0(sizeof(int64) * nattrs);
	sestate->cs_cur_rowid_max = palloc0(sizeof(int64) * nattrs);

	sestate->chunk_exec_list = NIL;
	sestate->chunk_ready_list = NIL;
	sestate->curr_chunk = NULL;
	sestate->curr_index = 0;

	MemoryContextSwitchTo(oldcxt);

	if (!sestate->nevermatch && kernel_source != NULL)
	{
		void   *image;
		int64  *const_buffer = NULL;
		int		const_nums = 0;
		int		dev_index;
		int		dev_nums;
		struct timeval tv1, tv2;

		/*
		 * Build the kernel source code
		 */
		if (pgstrom_exec_profile)
			gettimeofday(&tv1, NULL);
		image = pgstrom_nvcc_kernel_build(kernel_source);
		if (pgstrom_exec_profile)
		{
			gettimeofday(&tv2, NULL);
			sestate->pf_jit_compile += TIMEVAL_ELAPSED(&tv1, &tv2);
		}

		/*
		 * Set up constant values, if exist
		 */
		if (const_values != NIL)
		{
			const_buffer = alloca(sizeof(int64) * list_length(const_values));

			foreach (cell, const_values)
			{
				Assert(IsA(lfirst(cell), Integer) ||
					   IsA(lfirst(cell), String));
				if (IsA(lfirst(cell), Integer))
					const_buffer[const_nums++] = intVal(lfirst(cell));
				else
				{
					Datum	temp = CStringGetDatum(strVal(lfirst(cell)));
					const_buffer[const_nums++] =
						DatumGetInt64(DirectFunctionCall1(int8in, temp));
				}
			}
		}

		/*
		 * Set up device execution schedule
		 */
		if (pgstrom_exec_profile)
			gettimeofday(&tv1, NULL);

		dev_nums = pgstrom_get_num_devices();

		oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
		for (dev_index=0; dev_index < dev_nums; dev_index++)
		{
			PgStromDevContext *dev_cxt
				= pgstrom_init_exec_device(image, dev_index,
										   const_buffer, const_nums);
			sestate->dev_list = lappend(sestate->dev_list, dev_cxt);
		}
		MemoryContextSwitchTo(oldcxt);

		if (pgstrom_exec_profile)
		{
			gettimeofday(&tv2, NULL);
            sestate->pf_device_init += TIMEVAL_ELAPSED(&tv1, &tv2);
        }
		sestate->dev_curr = NULL;
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
	struct timeval		tv1, tv2;

	/*
	 * Do nothing for EXPLAIN or ANALYZE cases
	 */
	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;

	if (pgstrom_exec_profile)
		gettimeofday(&tv1, NULL);

	sestate = pgstrom_init_exec_state(fss);

	/*
	 * Open the required relation, but kept close for unused relations
	 */
	sestate->id_rel = pgstrom_open_shadow_table(base_rel,
												InvalidAttrNumber,
												AccessShareLock);
	tempset = bms_union(sestate->required_cols,
						sestate->clause_cols);
	while ((attnum = bms_first_member(tempset)) > 0)
	{
		sestate->cs_rels[attnum - 1]
			= pgstrom_open_shadow_table(base_rel, attnum, AccessShareLock);
		sestate->cs_idxs[attnum - 1]
			= pgstrom_open_shadow_index(base_rel, attnum, AccessShareLock);
	}
	bms_free(tempset);

	/*
	 * Begin the scan
	 */
	sestate->id_scan = heap_beginscan(sestate->id_rel,
									  sestate->es_snapshot,
									  0, NULL);

	tempset = bms_union(sestate->required_cols,
						sestate->clause_cols);
	while ((attnum = bms_first_member(tempset)) > 0)
	{
		/*
		 * Columns used in qualifier-clause needs two keys on index-scan;
		 * both lower and higher limit of rowid.
		 * Columns only referenced to upper layer needs one key on
		 * its index-scan.
		 */
		if (bms_is_member(attnum, sestate->clause_cols))
		{
			sestate->cs_scan[attnum - 1]
				= index_beginscan(sestate->cs_rels[attnum - 1],
								  sestate->cs_idxs[attnum - 1],
								  sestate->es_snapshot, 2, 0);
		}
		else
		{
			sestate->cs_scan[attnum - 1]
				= index_beginscan(sestate->cs_rels[attnum - 1],
								  sestate->cs_idxs[attnum - 1],
								  sestate->es_snapshot, 1, 0);
		}
	}
	bms_free(tempset);

	fss->fdw_state = sestate;

	if (pgstrom_exec_profile)
	{
		gettimeofday(&tv2, NULL);
		sestate->pf_pgstrom_total += TIMEVAL_ELAPSED(&tv1, &tv2);
	}
}

TupleTableSlot*
pgstrom_iterate_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	TupleTableSlot	   *slot = fss->ss.ss_ScanTupleSlot;
	int					num_chunks;
	struct timeval		tv1, tv2;

	if (pgstrom_exec_profile)
		gettimeofday(&tv1, NULL);

	ExecClearTuple(slot);
	if (sestate->nevermatch)
		goto out;

	/* Is it the first call? */
	if (sestate->curr_chunk == NULL)
	{
		num_chunks = (pgstrom_max_async_chunks -
					  list_length(sestate->chunk_exec_list));

		if (pgstrom_load_chunk_buffer(sestate, num_chunks) < 1)
			goto out;

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

		if (list_length(sestate->chunk_ready_list) +
			list_length(sestate->chunk_exec_list) < pgstrom_max_async_chunks)
		{
			num_chunks = (pgstrom_max_async_chunks -
						  list_length(sestate->chunk_exec_list));
			num_chunks = pgstrom_load_chunk_buffer(sestate, num_chunks);

			if (sestate->chunk_ready_list == NIL)
			{
				if (num_chunks < 1)
					goto out;
				else
					pgstrom_sync_kernel_exec(sestate);
			}
		}
		Assert(sestate->chunk_ready_list != NIL);
		sestate->curr_chunk = list_head(sestate->chunk_ready_list);
		sestate->curr_index = 0;
		goto retry;
	}
out:
	if (pgstrom_exec_profile)
	{
		gettimeofday(&tv2, NULL);
		sestate->pf_pgstrom_total += TIMEVAL_ELAPSED(&tv1, &tv2);
	}
	return slot;
}

void
pgstrom_rescan_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	ListCell		   *cell;
	struct timeval		tv1, tv2;

	if (pgstrom_exec_profile)
		gettimeofday(&tv1, NULL);

	/*
	 * Rewind rowid scan
	 */
	if (sestate->id_scan != NULL)
		heap_endscan(sestate->id_scan);
	sestate->id_scan = heap_beginscan(sestate->id_rel,
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

	if (pgstrom_exec_profile)
	{
		gettimeofday(&tv2, NULL);
		sestate->pf_pgstrom_total += TIMEVAL_ELAPSED(&tv1, &tv2);
	}
}

void
pgstrom_end_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	int					nattrs, i;
	struct timeval		tv1, tv2;

	/* if sestate is NULL, we are in EXPLAIN; nothing to do */
	if (!sestate)
		return;

	if (pgstrom_exec_profile)
		gettimeofday(&tv1, NULL);

	/*
	 * End the rowid scan
	 */
	nattrs = RelationGetNumberOfAttributes(fss->ss.ss_currentRelation);
	for (i=0; i < nattrs; i++)
	{
		if (sestate->cs_scan[i])
			index_endscan(sestate->cs_scan[i]);
	}
	if (sestate->id_scan != NULL)
		heap_endscan(sestate->id_scan);

	for (i=0; i < nattrs; i++)
	{
		if (sestate->cs_rels[i])
			relation_close(sestate->cs_rels[i], AccessShareLock);
		if (sestate->cs_idxs[i])
			relation_close(sestate->cs_idxs[i], AccessShareLock);
	}
	relation_close(sestate->id_rel, AccessShareLock);

	if (pgstrom_exec_profile)
	{
		gettimeofday(&tv2, NULL);
		sestate->pf_pgstrom_total += TIMEVAL_ELAPSED(&tv1, &tv2);

		elog(INFO, "PG-Strom Exec Profile on \"%s\"",
			 RelationGetRelationName(sestate->es_relation));
		elog(INFO, "Total PG-Strom consumed time: %.3f ms",
			 ((double)sestate->pf_pgstrom_total) / 1000.0);
		elog(INFO, "Time to JIT GPU comple:       %.3f ms",
			 ((double)sestate->pf_jit_compile) / 1000.0);
		elog(INFO, "Time to initialize devices:   %.3f ms",
			 ((double)sestate->pf_device_init) / 1000.0);
		elog(INFO, "Time of Async memcpy:         %.3f ms",
			 ((double)sestate->pf_async_memcpy) / 1000.0);
		elog(INFO, "Time of Async kernel exec:    %.3f ms",
			 ((double)sestate->pf_async_kernel) / 1000.0);
		elog(INFO, "Time of GPU Synchronization:  %.3f ms",
			 ((double)sestate->pf_synchronization) / 1000.0);
		elog(INFO, "Time of Load column-stores:   %.3f ms",
			 ((double)sestate->pf_load_chunk) / 1000.0);
	}
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
	DefineCustomBoolVariable("pg_strom.exec_profile",
							 "print execution profile information",
							 NULL,
							 &pgstrom_exec_profile,
							 false,
							 PGC_USERSET,
							 0,
							 NULL, NULL, NULL);
}
