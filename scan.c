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
static cl_context	pgstrom_device_context = NULL;
static int			pgstrom_max_async_chunks;
static int			pgstrom_work_group_size;

typedef struct {
	int			nattrs;
	int64		rowid;
	VarBit	   *rowmap;
	bits8	  **cs_nulls;
	void	  **cs_values;
	cl_event	dev_event;
} PgStromChunkBuf;

typedef struct {
	RelationSet		relset;

	/* parameters come from planner */
	int				predictable;	/* is the result set predictable? */
	Bitmapset	   *required_cols;	/* columns being returned to executor */
	Bitmapset	   *clause_cols;	/* columns being copied to device */
	const char	   *device_kernel;	/* kernel part of device code */

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
	List		   *chunk_exec_pending_list; /* chunk being pending to exec */
	List		   *chunk_exec_list;	/* chunks under kernel execution */
	List		   *chunk_ready_list;	/* chunks being ready to 2nd scan */
	ListCell	   *curr_chunk;
	int				curr_index;

	/* opencl related stuff */
	//size_t				dev_global_mem_required;
	cl_context			dev_context;
	cl_program			dev_program;
	cl_int				dev_command_queue_index;
	cl_command_queue	dev_command_queue[0];
} PgStromExecState;

static void
pgstrom_cleanup_exec_state(PgStromExecState *sestate)
{
	elog(NOTICE, "pgstrom_release_exec_state called: %p", sestate);

	if (sestate->dev_program)
	{
		int		i;

		for (i=0; i < pgstrom_num_devices; i++)
		{
			if (sestate->dev_command_queue[i])
				clReleaseCommandQueue(sestate->dev_command_queue[i]);
		}
		clReleaseProgram(sestate->dev_program);
	}
}

static void
pgstrom_load_column_store(PgStromExecState *sestate,
						  PgStromChunkBuf *chunk, AttrNumber attnum)
{
	Form_pg_attribute	attr;
	IndexScanDesc	iscan;
	ScanKeyData		skeys[2];
	HeapTuple		tup;
	int				csidx = attnum - 1;

	/*
	 * XXX - Because this column shall be copied to device to execute
	 * kernel function, variable length value should not be appeared
	 * in this stage.
	 */
	attr = RelationGetDescr(sestate->relset->base_rel)->attrs[csidx];
	Assert(attr->attlen > 0);

	chunk->cs_values[csidx]
		= MemoryContextAllocZero(sestate->es_memcxt,
								 PGSTROM_CHUNK_SIZE * attr->attlen);

	/*
	 * null-bitmap shall be initialized as if all the values are NULL.
	 */
	chunk->cs_nulls[csidx]
		= MemoryContextAlloc(sestate->es_memcxt,
							 PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
	memset(chunk->cs_nulls[csidx], -1,
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
		Assert(ARR_ELEMTYPE(cur_array) == attr->atttypid);

		nitems = ARR_DIMS(cur_array)[0];
		memcpy(((char *)chunk->cs_values[csidx]) + offset * attr->attlen,
			   ARR_DATA_PTR(cur_array),
			   nitems * attr->attlen);
		nullbitmap = ARR_NULLBITMAP(cur_array);
		if (nullbitmap)
		{
			// XXX - nitems also should be multiple-number of 8
			memcpy(chunk->cs_nulls[csidx] + offset / BITS_PER_BYTE,
				   nullbitmap,
				   (nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		}
		else
		{
			/* clear nullbitmap, if all items are not null */
			memset(chunk->cs_nulls[csidx] + offset / BITS_PER_BYTE,
				   0,
				   (nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		}
	}
	index_endscan(iscan);
}

static bool
pgstrom_exec_kernel_qual(PgStromExecState *sestate, PgStromChunkBuf *chunk)
{
	Form_pg_attribute	attr;
	MemoryContext		oldcxt;
	cl_buffer_region	region;
	cl_kernel	dev_kernel;
	size_t		dbuf_global_size;
	cl_mem		dbuf_global_mem;
	cl_mem		sbuf_rowmap;
	cl_mem	   *sbuf_values = alloca(sizeof(cl_mem) * chunk->nattrs);
	cl_mem	   *sbuf_nulls = alloca(sizeof(cl_mem) * chunk->nattrs);
	cl_mem		sbuf_null_buffer = NULL;
	cl_event	ev_exec_kernel;
	cl_event   *ev_copy_to_dev
		= alloca(sizeof(cl_event) * (2 * chunk->nattrs + 1));
	cl_int		dev_index;
	cl_int		n_events = 0;
	cl_int		n_args = 0;
	cl_int		ret;
	size_t		global_work_size;
	size_t		local_work_size;
	int			csidx;

	/*
	 * XXX - currently, we use all the GPU devices with round-robin
	 * storategy. However, it should be configurable in the future.
	 */
	dev_index = sestate->dev_command_queue_index++ % pgstrom_num_devices;

	/*
	 * Create kernel object
	 */
	dev_kernel = clCreateKernel(sestate->dev_program,
								"pgstrom_qual",
								&ret);
	Assert(ret == CL_SUCCESS);

	/*
	 * Estimate required global memory size
	 */
	dbuf_global_size = PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;
	for (csidx=0; csidx < chunk->nattrs; csidx++)
	{
		attr = RelationGetDescr(sestate->es_relation)->attrs[csidx];

		if (chunk->cs_values[csidx])
			dbuf_global_size += PGSTROM_CHUNK_SIZE * attr->attlen;
		if (chunk->cs_nulls[csidx])
			dbuf_global_size += PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;
	}

	/*
	 * Allocate global device memory
	 */
	dbuf_global_mem = clCreateBuffer(pgstrom_device_context,
									 CL_MEM_READ_WRITE,
									 dbuf_global_size,
									 NULL,
									 &ret);
	Assert(ret == CL_SUCCESS);

	/*
	 * Divide the global device memory into sub-buffers, set up argument
	 * of the kernel, and enqueue task to copy.
	 */
	region.origin = 0;
	region.size = PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;
	sbuf_rowmap = clCreateSubBuffer(dbuf_global_mem,
									CL_MEM_READ_WRITE,
									CL_BUFFER_CREATE_TYPE_REGION,
									&region,
									&ret);
	Assert(ret == CL_SUCCESS);

	ret = clSetKernelArg(dev_kernel, n_args++,
						 sizeof(cl_mem), &sbuf_rowmap);
	Assert(ret == CL_SUCCESS);

	ret = clEnqueueWriteBuffer(sestate->dev_command_queue[dev_index],
							   sbuf_rowmap,
							   CL_FALSE,
							   0, (VARBITLEN(chunk->rowmap) + 7) / 8,
							   VARBITS(chunk->rowmap),
                               0, NULL,
							   &ev_copy_to_dev[n_events++]);
	Assert(ret == CL_SUCCESS);

	memset(sbuf_values, 0, sizeof(cl_mem) * chunk->nattrs);
	memset(sbuf_nulls, 0, sizeof(cl_mem) * chunk->nattrs);

	for (csidx=0; csidx < chunk->nattrs; csidx++)
    {
		attr = RelationGetDescr(sestate->es_relation)->attrs[csidx];

		if (!chunk->cs_values[csidx])
			continue;

		region.origin += region.size;
		region.size = PGSTROM_CHUNK_SIZE * attr->attlen;
		sbuf_values[csidx] = clCreateSubBuffer(dbuf_global_mem,
											   CL_MEM_READ_WRITE,
											   CL_BUFFER_CREATE_TYPE_REGION,
											   &region,
											   &ret);
		Assert(ret == CL_SUCCESS);

		ret = clSetKernelArg(dev_kernel, n_args++,
							 sizeof(cl_mem), &sbuf_values[csidx]);
		Assert(ret == CL_SUCCESS);

		ret = clEnqueueWriteBuffer(sestate->dev_command_queue[dev_index],
								   sbuf_values[csidx],
								   CL_FALSE,
								   0, region.size,
								   chunk->cs_values[csidx],
								   0, NULL,
								   &ev_copy_to_dev[n_events++]);
		Assert(ret == CL_SUCCESS);

		Assert(chunk->cs_nulls[csidx] != NULL);
		region.origin += region.size;
		region.size = PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;
		sbuf_nulls[csidx] = clCreateSubBuffer(dbuf_global_mem,
											  CL_MEM_READ_WRITE,
											  CL_BUFFER_CREATE_TYPE_REGION,
											  &region,
											  &ret);
		Assert(ret == CL_SUCCESS);

		ret = clSetKernelArg(dev_kernel, n_args++,
							 sizeof(cl_mem), &sbuf_nulls[csidx]);
		Assert(ret == CL_SUCCESS);

		ret = clEnqueueWriteBuffer(sestate->dev_command_queue[dev_index],
								   sbuf_nulls[csidx],
								   CL_FALSE,
								   0, region.size,
								   chunk->cs_nulls[csidx],
								   0, NULL,
								   &ev_copy_to_dev[n_events++]);
		Assert(ret == CL_SUCCESS);
	}
	/*
	 * Enqueue async-kernel execution
	 */
	global_work_size = PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;
	local_work_size = 30;
	ret = clEnqueueNDRangeKernel(sestate->dev_command_queue[dev_index],
								 dev_kernel,
								 1,
								 NULL,
								 &global_work_size,
                                 &local_work_size,
								 n_events, ev_copy_to_dev,
								 &ev_exec_kernel);
	Assert(ret == CL_SUCCESS);

	/*
	 * Enqueue copy back to the host
	 */
	ret = clEnqueueReadBuffer(sestate->dev_command_queue[dev_index],
							  sbuf_rowmap,
							  CL_FALSE,
							  0, (VARBITLEN(chunk->rowmap) + 7) / 8,
							  VARBITS(chunk->rowmap),
							  1, &ev_exec_kernel,
							  &chunk->dev_event);
	Assert(ret == CL_SUCCESS);

	/*
	 * Decrement reference counter of event/memory objects
	 */
	clReleaseMemObject(sbuf_rowmap);
	for (csidx=0; csidx < chunk->nattrs; csidx++)
	{
		if (sbuf_values[csidx])
			clReleaseMemObject(sbuf_values[csidx]);
		if (sbuf_nulls[csidx])
			clReleaseMemObject(sbuf_nulls[csidx]);
	}
	clReleaseMemObject(dbuf_global_mem);

	while (n_events > 0)
		clReleaseEvent(ev_copy_to_dev[--n_events]);
	clReleaseEvent(ev_exec_kernel);

	clReleaseKernel(dev_kernel);

	/*
	 * Append this chunk to chunk_exec_list
	 */
	oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
	sestate->chunk_exec_list
		= lappend(sestate->chunk_exec_list, chunk);
	MemoryContextSwitchTo(oldcxt);

	return true;
}

/*
 * pgstrom_sync_kernel_qual
 *
 * This routine waits for one of the asynchronous events completed.
 * If true was returned, it means a chunk is in chunk_ready_list.
 */
static void
pgstrom_sync_kernel_qual(PgStromExecState *sestate)
{
	int	num_got_ready = 0;

	while (num_got_ready == 0)
	{
		ListCell   *cell;
		ListCell   *prev;
		ListCell   *next;
		cl_event	event = NULL;

		if (sestate->chunk_exec_list == NIL)
			return;

		prev = NULL;
		for (cell = list_head(sestate->chunk_exec_list);
			 cell != NULL;
			 cell = next)
		{
			PgStromChunkBuf	*chunk = lfirst(cell);
			MemoryContext	 oldcxt;
			cl_int	ev_status;
			cl_int	ret;

			next = lnext(cell);

			ret = clGetEventInfo(chunk->dev_event,
								 CL_EVENT_COMMAND_EXECUTION_STATUS,
								 sizeof(ev_status), &ev_status, NULL);
			Assert(ret == CL_SUCCESS);

			if (ev_status == CL_COMPLETE)
			{
				oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);

				sestate->chunk_exec_list 
					= list_delete_cell(sestate->chunk_exec_list, cell, prev);

				sestate->chunk_ready_list
					= lappend(sestate->chunk_ready_list, chunk);

				num_got_ready++;

				MemoryContextSwitchTo(oldcxt);

				/* Decrement event object's refcount */
				clReleaseEvent(chunk->dev_event);
			}
			else if (ev_status < 0)
			{
				/* Decrement event object's refcount */
				clReleaseEvent(chunk->dev_event);

				elog(ERROR, "OpenCL events return error : %s",
					 opencl_error_to_string(ev_status));
			}
			else
			{
				if (event == NULL)
					event = chunk->dev_event;
				prev = cell;
			}
		}
		/*
		 * XXX - we temtatively assume first-event will complete first.
		 */
		if (num_got_ready == 0 && event != NULL)
			clWaitForEvents(1, &event);
	}
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

		oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);

		chunk = palloc0(sizeof(PgStromChunkBuf));
		chunk->rowid = DatumGetInt64(values[0]);
		chunk->rowmap = DatumGetVarBitPCopy(values[1]);
		chunk->nattrs = RelationGetNumberOfAttributes(sestate->es_relation);
		chunk->cs_nulls = palloc0(sizeof(bool *) * chunk->nattrs);
		chunk->cs_values = palloc0(sizeof(void *) * chunk->nattrs);
		MemoryContextSwitchTo(oldcxt);

		if (!sestate->predictable)
		{
			Bitmapset  *temp;
			AttrNumber	attnum;

			/*
			 * Load necessary column store
			 */
			temp = bms_copy(sestate->clause_cols);
			while ((attnum = bms_first_member(temp)) > 0)
				pgstrom_load_column_store(sestate, chunk, attnum);
			bms_free(temp);

			/*
			 * Exec kernel on this chunk asynchronously
			 */
			pgstrom_exec_kernel_qual(sestate, chunk);
		}
		else
		{
			/*
			 * In predictable query, chunks are ready to scan using rowid.
			 */
			oldcxt = MemoryContextSwitchTo(sestate->es_memcxt);
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

	for (index = sestate->curr_index;
		 index < VARBITLEN(chunk->rowmap);
		 index++)
	{
		int		index_h = (index / BITS_PER_BYTE);
		int		index_l = (index & (BITS_PER_BYTE - 1));
		int		csidx;
		int64	rowid;

		if ((VARBITS(chunk->rowmap)[index_h] & (1 << index_l)) == 0)
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
			if (chunk->cs_values[csidx])
			{
				if (!chunk->cs_nulls[csidx] ||
					(chunk->cs_nulls[csidx][index_h] & (1<<index_l)) == 0)
				{
					Form_pg_attribute	attr
						= slot->tts_tupleDescriptor->attrs[csidx];
					slot->tts_isnull[csidx] = false;
					slot->tts_values[csidx] =
						fetchatt(attr, ((char *)chunk->cs_values[csidx] +
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
	cl_int				i, ret;

#if 0
	if (!pgstrom_device_context)
	{
		pgstrom_device_context = clCreateContext(NULL,
												 pgstrom_num_devices,
												 pgstrom_device_id,
												 NULL, NULL, &ret);
		elog(NOTICE, "clCreateContext : %s", opencl_error_to_string(ret));
		Assert(ret == CL_SUCCESS);
	}
#endif

	nattrs = RelationGetNumberOfAttributes(fss->ss.ss_currentRelation);
	sestate = palloc0(sizeof(PgStromExecState) +
					  sizeof(cl_command_queue) * pgstrom_num_devices);
	sestate->cs_scan = palloc0(sizeof(IndexScanDesc) * nattrs);
	sestate->cs_cur_values = palloc0(sizeof(ArrayType *) * nattrs);
	sestate->cs_cur_rowid_min = palloc0(sizeof(int64) * nattrs);
	sestate->cs_cur_rowid_max = palloc0(sizeof(int64) * nattrs);

	//sestate->dev_global_mem_required = PGSTROM_CHUNK_SIZE / BITS_PER_BYTE;
	sestate->es_relation = fss->ss.ss_currentRelation;
    sestate->es_snapshot = fss->ss.ps.state->es_snapshot;
	sestate->es_memcxt = fss->ss.ps.ps_ExprContext->ecxt_per_query_memory;

	foreach (l, fscan->fdwplan->fdw_private)
	{
		DefElem	   *defel = (DefElem *)lfirst(l);

		if (strcmp(defel->defname, "predictable") == 0)
		{
			if (intVal(defel->arg) == TRUE)
				sestate->predictable = 1;	/* all the tuples are visible */
			else
				sestate->predictable = -1;	/* all the tuples are invisible */
		}
		else if (strcmp(defel->defname, "device_kernel") == 0)
		{
			sestate->device_kernel = strVal(defel->arg);
		}
		else if (strcmp(defel->defname, "clause_cols") == 0)
		{
			//Form_pg_attribute	attr;
			int		csidx = (intVal(defel->arg));

			Assert(csidx > 0);

			sestate->clause_cols
				= bms_add_member(sestate->clause_cols, csidx);
			//attr = RelationGetDescr(sestate->es_relation)->attrs[csidx];
			//sestate->dev_global_mem_required
			//	+= (PGSTROM_CHUNK_SIZE / BITS_PER_BYTE +
			//		PGSTROM_CHUNK_SIZE * attr->attlen);
		}
		else if (strcmp(defel->defname, "required_cols") == 0)
		{
			int		csidx = (intVal(defel->arg));

			if (csidx < 1)
			{
				Assert(fscan->fsSystemCol);
				continue;
			}
			sestate->required_cols
				= bms_add_member(sestate->required_cols, csidx);
		}
		else
			elog(ERROR, "pg_strom: unexpected private plan information: %s",
				 defel->defname);
	}

	/*
	 * Skip stuff related to OpenCL, if the query is predictable
	 */
	if (sestate->predictable)
		goto skip_opencl;



		pgstrom_device_context = clCreateContext(NULL,
												 pgstrom_num_devices,
												 pgstrom_device_id,
												 NULL, NULL, &ret);
		elog(NOTICE, "clCreateContext : %s", opencl_error_to_string(ret));
		Assert(ret == CL_SUCCESS);


	/*
	 * Build kernel function to binary representation
	 */
	Assert(pgstrom_device_context != NULL);
	sestate->dev_program
		= clCreateProgramWithSource(pgstrom_device_context,
									1, &sestate->device_kernel,
									NULL, &ret);
	if (ret != CL_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL failed to create program with source: %s",
						opencl_error_to_string(ret))));

	ret = clBuildProgram(sestate->dev_program,
						 0, NULL,		/* for all the devices */
						 NULL,			/* no build options */
						 NULL, NULL);	/* no callback, so synchronous build */
	if (ret != CL_SUCCESS)
	{
		cl_build_status	status;
		char			logbuf[4096];

		for (i=0; i < pgstrom_num_devices; i++)
		{
			clGetProgramBuildInfo(sestate->dev_program,
								  pgstrom_device_id[i],
								  CL_PROGRAM_BUILD_STATUS,
								  sizeof(status),
								  &status,
								  NULL);
			if (status != CL_BUILD_ERROR)
				continue;

			clGetProgramBuildInfo(sestate->dev_program,
								  pgstrom_device_id[i],
								  CL_PROGRAM_BUILD_LOG,
								  sizeof(logbuf),
								  logbuf,
								  NULL);
			elog(NOTICE, "%s", logbuf);
		}
		clReleaseProgram(sestate->dev_program);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL failed to build program: %s",
						opencl_error_to_string(ret))));
	}

	/*
	 * Create command queues for each devices
	 */
	for (i=0; i < pgstrom_num_devices; i++)
	{
		sestate->dev_command_queue[i]
			= clCreateCommandQueue(pgstrom_device_context,
								   pgstrom_device_id[i],
								   0,	/* no out-of-order, no profiling */
								   &ret);
		if (ret != CL_SUCCESS)
		{
			while (i > 0)
				clReleaseCommandQueue(sestate->dev_command_queue[--i]);
			clReleaseProgram(sestate->dev_program);

			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("OpenCL failed to create command queue: %s",
							opencl_error_to_string(ret))));
		}
	}

skip_opencl:
	//sestate->es_errcxt.callback = pgstrom_release_exec_state;
	//sestate->es_errcxt.arg = (void *) sestate;
	//sestate->es_errcxt.previous = error_context_stack;
	//error_context_stack = &sestate->es_errcxt;

	sestate->chunk_exec_pending_list = NIL;
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
	if (sestate->predictable < 0)
		return slot;

	/* Is it the first call? */
	if (sestate->curr_chunk == NULL)
	{
		num_chunks = (pgstrom_max_async_chunks -
					  list_length(sestate->chunk_exec_list));
		if (pgstrom_load_chunk_buffer(sestate, num_chunks) < 1)
			return slot;

		pgstrom_sync_kernel_qual(sestate);
		Assert(sestate->chunk_ready_list != NIL);
		sestate->curr_chunk = list_head(sestate->chunk_ready_list);
		sestate->curr_index = 0;
	}
retry:
	if (!pgstrom_scan_chunk_buffer(sestate, slot))
	{
		sestate->chunk_ready_list
			= list_delete(sestate->chunk_ready_list,
						  lfirst(sestate->curr_chunk));

		pgstrom_sync_kernel_qual(sestate);

		num_chunks = (pgstrom_max_async_chunks -
					  list_length(sestate->chunk_exec_list));
		num_chunks = pgstrom_load_chunk_buffer(sestate, num_chunks);
		if (sestate->chunk_ready_list == NIL)
			pgstrom_sync_kernel_qual(sestate);

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
	while (sestate->chunk_exec_list != NIL)
		pgstrom_sync_kernel_qual(sestate);
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
