/*
 * datastore.c
 *
 * Routines to manage data store; row-store, column-store, toast-buffer,
 * and param-buffer.
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#include "postgres.h"
#include "access/relscan.h"
#include "catalog/catalog.h"
#include "catalog/pg_tablespace.h"
#include "catalog/pg_type.h"
#include "optimizer/cost.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "storage/predicate.h"
#include "utils/builtins.h"
#include "utils/bytea.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/tqual.h"
#include "pg_strom.h"
#include "cuda_numeric.h"
#include <float.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

/*
 * GUC variables
 */
static int		pgstrom_chunk_size_kb;
static int		pgstrom_chunk_limit_kb = INT_MAX;

/*
 * pgstrom_chunk_size - configured chunk size
 */
Size
pgstrom_chunk_size(void)
{
	return ((Size)pgstrom_chunk_size_kb) << 10;
}

static bool
check_guc_chunk_size(int *newval, void **extra, GucSource source)
{
	if (*newval > pgstrom_chunk_limit_kb)
	{
		GUC_check_errdetail("pg_strom.chunk_size = %d, is larger than "
							"pg_strom.chunk_limit = %d",
							*newval, pgstrom_chunk_limit_kb);
		return false;
	}
	return true;
}

/*
 * pgstrom_chunk_size_limit
 */
Size
pgstrom_chunk_size_limit(void)
{
	return ((Size)pgstrom_chunk_limit_kb) << 10;
}

static bool
check_guc_chunk_limit(int *newval, void **extra, GucSource source)
{
	if (*newval < pgstrom_chunk_size_kb)
	{
		GUC_check_errdetail("pg_strom.chunk_limit = %d, is less than "
							"pg_strom.chunk_size = %d",
							*newval, pgstrom_chunk_size_kb);
	}
	return true;
}

/*
 * pgstrom_bulk_exec_supported - returns true, if supplied planstate
 * supports bulk execution mode.
 */
bool
pgstrom_bulk_exec_supported(const PlanState *planstate)
{
	if (pgstrom_plan_is_gpuscan(planstate->plan) ||
        pgstrom_plan_is_gpujoin(planstate->plan) ||
        pgstrom_plan_is_gpupreagg(planstate->plan) ||
        pgstrom_plan_is_gpusort(planstate->plan))
	{
		GpuTaskState   *gts = (GpuTaskState *) planstate;

		if (gts->cb_bulk_exec != NULL)
			return true;
	}
	return false;
}

/*
 * estimate_num_chunks
 *
 * it estimates number of chunks to be fetched from the supplied Path
 */
cl_uint
estimate_num_chunks(Path *pathnode)
{
	RelOptInfo *rel = pathnode->parent;
	int			ncols = list_length(rel->reltargetlist);
    Size        htup_size;
	cl_uint		num_chunks;

	htup_size = MAXALIGN(offsetof(HeapTupleHeaderData,
								  t_bits[BITMAPLEN(ncols)]));
	if (rel->reloptkind != RELOPT_BASEREL)
		htup_size += MAXALIGN(rel->width);
	else
	{
		double      heap_size = (double)
			(BLCKSZ - SizeOfPageHeaderData) * rel->pages;

		htup_size += MAXALIGN(heap_size / Max(rel->tuples, 1.0) -
							  sizeof(ItemIdData) - SizeofHeapTupleHeader);
	}
	num_chunks = (cl_uint)
		((double)(htup_size + sizeof(cl_int)) * pathnode->rows /
		 (double)(pgstrom_chunk_size() -
				  STROMALIGN(offsetof(kern_data_store, colmeta[ncols]))));
	num_chunks = Max(num_chunks, 1);

	return num_chunks;
}

/*
 * pgstrom_temp_dirpath - makes a temporary file according to the system
 * setting. Note that we never gueran
 */
static int
pgstrom_open_tempfile(const char **p_tempfilepath)
{
	static long	tempFileCounter = 0;
	static char	tempfilepath[MAXPGPATH];
	char		tempdirpath[MAXPGPATH];
	int			file_desc;
	int			file_flags;
	Oid			tablespace_oid = GetNextTempTableSpace();

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = (OidIsValid(MyDatabaseTableSpace)
						  ? MyDatabaseTableSpace :
						  DEFAULTTABLESPACE_OID);

	if (tablespace_oid == DEFAULTTABLESPACE_OID ||
		tablespace_oid == GLOBALTABLESPACE_OID)
	{
		/* The default tablespace is {datadir}/base */
		snprintf(tempdirpath, sizeof(tempdirpath),
				 "base/%s", PG_TEMP_FILES_DIR);
	}
	else
	{
		/* All other tablespaces are accessed via symlinks */
		snprintf(tempdirpath, sizeof(tempdirpath),
				 "pg_tblspc/%u/%s/%s",
				 tablespace_oid,
				 TABLESPACE_VERSION_DIRECTORY,
				 PG_TEMP_FILES_DIR);
	}

	/*
	 * Generate a tempfile name that should be unique within the current
	 * database instance.
	 */
	snprintf(tempfilepath, sizeof(tempfilepath),
			 "%s/%s_strom_%d.%ld.map",
			 tempdirpath, PG_TEMP_FILE_PREFIX,
			 MyProcPid, tempFileCounter++);

	file_flags = O_RDWR | O_CREAT | O_TRUNC | PG_BINARY;
	file_desc = OpenTransientFile(tempfilepath, file_flags, 0600);
	if (file_desc < 0)
	{
		/*
		 * We might need to create the tablespace's tempfile directory,
		 * if no one has yet done so. However, no error check needed,
		 * because concurrent mkdir() can happen and OpenTransientFile
		 * below eventually raise an error.
		 */
		mkdir(tempdirpath, S_IRWXU);

		file_desc = OpenTransientFile(tempfilepath, file_flags, 0600);
		if (file_desc < 0)
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not create temporary file \"%s\": %m",
							tempfilepath)));
	}
	if (p_tempfilepath)
		*p_tempfilepath = tempfilepath;
	return file_desc;
}

/*
 * BulkExecProcNode
 *
 * It runs the underlying sub-plan managed by PG-Strom in bulk-execution
 * mode. Caller can expect the data-store shall be filled up by the rows
 * read from the sub-plan.
 */
pgstrom_data_store *
BulkExecProcNode(GpuTaskState *gts, size_t chunk_size)
{
	PlanState		   *plannode = &gts->css.ss.ps;
	pgstrom_data_store *pds;

	CHECK_FOR_INTERRUPTS();

	if (plannode->chgParam != NULL)			/* If something changed, */
		ExecReScan(&gts->css.ss.ps);		/* let ReScan handle this */

	Assert(IsA(gts, CustomScanState));		/* rough checks */
	if (gts->cb_bulk_exec)
	{
		/* must provide our own instrumentation support */
		if (plannode->instrument)
			InstrStartNode(plannode->instrument);
		/* execution per chunk */
		pds = gts->cb_bulk_exec(gts, chunk_size);

		/* must provide our own instrumentation support */
		if (plannode->instrument)
			InstrStopNode(plannode->instrument,
						  !pds ? 0.0 : (double)pds->kds->nitems);
		Assert(!pds || pds->kds->nitems > 0);
		return pds;
	}
	elog(ERROR, "Bug? exec_chunk callback was not implemented");
}

bool
kern_fetch_data_store(TupleTableSlot *slot,
					  kern_data_store *kds,
					  size_t row_index,
					  HeapTuple tuple)
{
	if (row_index >= kds->nitems)
		return false;	/* out of range */

	/* in case of KDS_FORMAT_ROW */
	if (kds->format == KDS_FORMAT_ROW)
	{
		kern_tupitem   *tup_item = KERN_DATA_STORE_TUPITEM(kds, row_index);

		ExecClearTuple(slot);
		tuple->t_len = tup_item->t_len;
		tuple->t_self = tup_item->t_self;
		//tuple->t_tableOid = InvalidOid;
		tuple->t_data = &tup_item->htup;

		ExecStoreTuple(tuple, slot, InvalidBuffer, false);

		return true;
	}
	/* in case of KDS_FORMAT_SLOT */
	if (kds->format == KDS_FORMAT_SLOT)
	{
		Datum  *tts_values = (Datum *)KERN_DATA_STORE_VALUES(kds, row_index);
		bool   *tts_isnull = (bool *)KERN_DATA_STORE_ISNULL(kds, row_index);
		int		natts = slot->tts_tupleDescriptor->natts;

		memcpy(slot->tts_values, tts_values, sizeof(Datum) * natts);
		memcpy(slot->tts_isnull, tts_isnull, sizeof(bool) * natts);
#ifdef NOT_USED
		/*
		 * XXX - pointer reference is better than memcpy from performance
		 * perspectives, however, we need to ensure tts_values/tts_isnull
		 * shall be restored when pgstrom-data-store is released.
		 * It will be cause of complicated / invisible bugs.
		 */
		slot->tts_values = tts_values;
		slot->tts_isnull = tts_isnull;
#endif
		ExecStoreVirtualTuple(slot);
		return true;
	}
	elog(ERROR, "Bug? unexpected data-store format: %d", kds->format);
	return false;
}

bool
pgstrom_fetch_data_store(TupleTableSlot *slot,
						 pgstrom_data_store *pds,
						 size_t row_index,
						 HeapTuple tuple)
{
	return kern_fetch_data_store(slot, pds->kds, row_index, tuple);
}

pgstrom_data_store *
pgstrom_acquire_data_store(pgstrom_data_store *pds)
{
	Assert(pds->refcnt > 0);

	pds->refcnt++;

	return pds;
}

void
pgstrom_release_data_store(pgstrom_data_store *pds)
{
	Assert(pds->refcnt > 0);
	/* acquired by multiple owners? */
	if (--pds->refcnt > 0)
		return;

	/* detach from the GpuContext */
	if (pds->pds_chain.prev && pds->pds_chain.next)
	{
		dlist_delete(&pds->pds_chain);
		memset(&pds->pds_chain, 0, sizeof(dlist_node));
	}

	/* release relevant toast store, if any */
	if (pds->ptoast)
		pgstrom_release_data_store(pds->ptoast);
	/* release body of the data store */
	if (!pds->kds_fname)
		pfree(pds->kds);
	else
	{
		size_t		mmap_length = TYPEALIGN(BLCKSZ, pds->kds_length);
		CUresult	rc;

		rc = cuMemHostUnregister(pds->kds);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemHostUnregister: %s", errorText(rc));

		if (munmap(pds->kds, mmap_length) != 0)
			ereport(WARNING,
					(errcode_for_file_access(),
					 errmsg("could not unmap file \"%s\" from %p-%p: %m",
							pds->kds_fname,
							(char *)pds->kds,
							(char *)pds->kds + mmap_length - 1)));
		if (!pds->ptoast)
		{
			/* Also unlink the backend file, if responsible */
			if (unlink(pds->kds_fname) != 0)
				elog(WARNING, "failed on unlink(\"%s\") : %m", pds->kds_fname);
		}
		pfree(pds->kds_fname);
	}
	pfree(pds);
}

void
init_kernel_data_store(kern_data_store *kds,
					   TupleDesc tupdesc,
					   Size length,
					   int format,
					   uint nrooms)
{
	int		i, attcacheoff;

	memset(kds, 0, offsetof(kern_data_store, colmeta));
	kds->hostptr = (hostptr_t) &kds->hostptr;
	kds->length = length;
	kds->usage = 0;
	kds->ncols = tupdesc->natts;
	kds->nitems = 0;
	kds->nrooms = nrooms;
	kds->format = format;
	kds->tdhasoid = tupdesc->tdhasoid;
	kds->tdtypeid = tupdesc->tdtypeid;
	kds->tdtypmod = tupdesc->tdtypmod;
	kds->table_oid = InvalidOid;	/* caller shall set */
	kds->nslots = 0;				/* caller shall set, if any */
	kds->hash_min = 0;
	kds->hash_max = UINT_MAX;

	attcacheoff = offsetof(HeapTupleHeaderData, t_bits);
	if (tupdesc->tdhasoid)
		attcacheoff += sizeof(Oid);
	attcacheoff = MAXALIGN(attcacheoff);

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];
		int		attalign = typealign_get_width(attr->attalign);

		if (!attr->attbyval)
			kds->has_notbyval = true;
		if (attr->atttypid == NUMERICOID)
			kds->has_numeric = true;

		if (attcacheoff > 0)
		{
			if (attr->attlen > 0)
				attcacheoff = TYPEALIGN(attalign, attcacheoff);
			else
				attcacheoff = -1;	/* no more shortcut any more */
		}
		kds->colmeta[i].attbyval = attr->attbyval;
		kds->colmeta[i].attalign = attalign;
		kds->colmeta[i].attlen = attr->attlen;
		kds->colmeta[i].attnum = attr->attnum;
		kds->colmeta[i].attcacheoff = attcacheoff;
		kds->colmeta[i].atttypid = (cl_uint)attr->atttypid;
		kds->colmeta[i].atttypmod = (cl_int)attr->atttypmod;
		if (attcacheoff >= 0)
			attcacheoff += attr->attlen;
	}
}

void
PDS_expand_size(GpuContext *gcontext,
				pgstrom_data_store *pds,
				Size kds_length_new)
{
	kern_data_store	   *kds_old = pds->kds;
	kern_data_store	   *kds_new;
	Size				kds_length_old = kds_old->length;
	Size				kds_usage = kds_old->usage;
	cl_uint				i, nitems = kds_old->nitems;

	/* sanity checks */
	Assert(pds->kds_offset == 0);
	Assert(pds->kds_length == kds_length_old);
	Assert(pds->ptoast == NULL);
	Assert(kds_old->format == KDS_FORMAT_ROW ||
		   kds_old->format == KDS_FORMAT_HASH);
	Assert(kds_old->nslots == 0);

	/* no need to expand? */
	if (kds_length_old >= kds_length_new)
		return;

	/* file mapped data-store is no longer supported */
	Assert(!pds->kds_fname);

	kds_new = MemoryContextAllocHuge(gcontext->memcxt,
									 kds_length_new);
	memcpy(kds_new, kds_old, KERN_DATA_STORE_HEAD_LENGTH(kds_old));
	kds_new->hostptr = (hostptr_t)&kds_new->hostptr;
	kds_new->length = kds_length_new;

	/*
	 * Move the contents to new buffer from the old one
	 */
	if (kds_new->format == KDS_FORMAT_ROW ||
		kds_new->format == KDS_FORMAT_HASH)
	{
		cl_uint	   *row_index_old = KERN_DATA_STORE_ROWINDEX(kds_old);
		cl_uint	   *row_index_new = KERN_DATA_STORE_ROWINDEX(kds_new);
		size_t		shift = STROMALIGN_DOWN(kds_length_new - kds_length_old);
		size_t		offset = kds_length_old - kds_usage;

		/*
		 * If supplied new nslots is too big, larger than the expanded,
		 * it does not make sense to expand the buffer.
		 */
		if ((kds_new->format == KDS_FORMAT_HASH
			 ? KDS_CALCULATE_HASH_LENGTH(kds_new->ncols,
										 kds_new->nitems,
										 kds_new->usage)
			 : KDS_CALCULATE_ROW_LENGTH(kds_new->ncols,
										kds_new->nitems,
										kds_new->usage)) >= kds_new->length)
			elog(ERROR, "New nslots consumed larger than expanded");

		memcpy((char *)kds_new + offset + shift,
			   (char *)kds_old + offset,
			   kds_length_old - offset);
		for (i = 0; i < nitems; i++)
			row_index_new[i] = row_index_old[i] + shift;
	}
	else if (kds_new->format == KDS_FORMAT_SLOT)
	{
		/*
		 * We cannot expand KDS_FORMAT_SLOT with extra area because we don't
		 * know the way to fix pointers that reference the extra area.
		 */
		if (kds_new->usage > 0)
			elog(ERROR, "cannot expand KDS_FORMAT_SLOT with extra area");
		/* copy the values/isnull pair */
		memcpy(KERN_DATA_STORE_BODY(kds_new),
			   KERN_DATA_STORE_BODY(kds_old),
			   KERN_DATA_STORE_SLOT_LENGTH(kds_old, kds_old->nitems));
	}
	else
		elog(ERROR, "unexpected KDS format: %d", kds_new->format);
	/* old KDS is no longer referenced */
	pfree(kds_old);

	/* update length and kernel buffer */
	pds->kds_length = kds_length_new;
	pds->kds = kds_new;
}

void
PDS_shrink_size(pgstrom_data_store *pds)
{
	kern_data_store	   *kds = pds->kds;
	size_t				new_length;

	if (kds->format == KDS_FORMAT_ROW ||
		kds->format == KDS_FORMAT_HASH)
	{
		cl_uint	   *hash_slot = KERN_DATA_STORE_HASHSLOT(kds);
		cl_uint	   *row_index = KERN_DATA_STORE_ROWINDEX(kds);
		cl_uint		i, nslots = kds->nslots;
		size_t		shift;
		char	   *baseptr;

		/* small shift has less advantage than CPU cycle consumption */
		shift = kds->length - (kds->format == KDS_FORMAT_HASH
							   ? KDS_CALCULATE_HASH_LENGTH(kds->ncols,
														   kds->nitems,
														   kds->usage)
							   : KDS_CALCULATE_ROW_LENGTH(kds->ncols,
														  kds->nitems,
														  kds->usage));
		shift = STROMALIGN_DOWN(shift);

		if (shift < BLCKSZ || shift < sizeof(Datum) * kds->nitems)
			return;

		/* move the kern_tupitem / kern_hashitem */
		baseptr = (char *)kds + (kds->format == KDS_FORMAT_HASH
								 ? KDS_CALCULATE_HASH_FRONTLEN(kds->ncols,
															   kds->nitems)
								 : KDS_CALCULATE_ROW_FRONTLEN(kds->ncols,
															  kds->nitems));
		memmove(baseptr, baseptr + shift, kds->length - shift);

		/* clear the hash slot once */
		if (nslots > 0)
		{
			Assert(kds->format == KDS_FORMAT_HASH);
			memset(hash_slot, 0, sizeof(cl_uint) * nslots);
		}

		/* adjust row_index and hash_slot */
		for (i=0; i < kds->nitems; i++)
		{
			row_index[i] -= shift;
			if (nslots > 0)
			{
				kern_hashitem  *khitem = KERN_DATA_STORE_HASHITEM(kds, i);
				cl_uint			khindex;

				Assert(khitem->rowid == i);
				khindex = khitem->hash % nslots;
                khitem->next = hash_slot[khindex];
                hash_slot[khindex] = (uintptr_t)khitem - (uintptr_t)kds;
			}
		}
		new_length = kds->length - shift;
	}
	else if (kds->format == KDS_FORMAT_SLOT)
	{
		new_length = KERN_DATA_STORE_SLOT_LENGTH(kds, kds->nitems);

		/*
		 * We cannot know which datum references the extra area with
		 * reasonable cost. So, prohibit it simply. We don't use SLOT
		 * format for data source, so usually no matter.
		 */
		if (kds->usage > 0)
			elog(ERROR, "cannot shirink KDS_SLOT with extra region");
	}
	else
		elog(ERROR, "Bug? unexpected PDS to be shrinked");

	Assert(new_length <= kds->length);
	kds->length = new_length;
	pds->kds_length = new_length;
}

pgstrom_data_store *
pgstrom_create_data_store_row(GpuContext *gcontext,
							  TupleDesc tupdesc, Size length,
							  bool file_mapped)
{
	pgstrom_data_store *pds;
	MemoryContext	gmcxt = gcontext->memcxt;

	/* allocation of pds */
	pds = MemoryContextAllocZero(gmcxt, sizeof(pgstrom_data_store));
	pds->refcnt = 1;	/* owned by the caller at least */

	/* allocation of kds */
	pds->kds_length = STROMALIGN_DOWN(length);
	pds->kds_offset = 0;

	if (!file_mapped)
		pds->kds = MemoryContextAllocHuge(gmcxt, pds->kds_length);
	else
	{
		const char *kds_fname;
		int			kds_fdesc;
		size_t		mmap_length;
		CUresult	rc;

		kds_fdesc = pgstrom_open_tempfile(&kds_fname);
		pds->kds_fname = MemoryContextStrdup(gmcxt, kds_fname);

		mmap_length = TYPEALIGN(BLCKSZ, pds->kds_length);
		if (ftruncate(kds_fdesc, mmap_length) != 0)
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not truncate file \"%s\" to %zu: %m",
							pds->kds_fname, pds->kds_length)));

		pds->kds = mmap(NULL, mmap_length,
						PROT_READ | PROT_WRITE,
#ifdef MAP_POPULATE
						MAP_POPULATE |
#endif
						MAP_SHARED,
						kds_fdesc, 0);
		if (pds->kds == MAP_FAILED)
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not mmap \"%s\" with length=%zu: %m",
							pds->kds_fname, pds->kds_length)));

		/* kds is still mapped - not to manage file desc by ourself */
		CloseTransientFile(kds_fdesc);

		/*
		 * Registers the file-mapped KDS as page-locked region,
		 * for asynchronous DMA transfer
		 *
		 * Unless PDS is not tracked by GpuContext, mapped file is
		 * never unmapped, so we have to care about the case when
		 * cuMemHostRegister() got failed.
		 */
		PG_TRY();
		{
			rc = cuCtxPushCurrent(gcontext->gpu[0].cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

			rc = cuMemHostRegister(pds->kds, mmap_length,
								   CU_MEMHOSTREGISTER_PORTABLE);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemHostRegister: %s", errorText(rc));

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		}
		PG_CATCH();
		{
			if (munmap(pds->kds, mmap_length) != 0)
				elog(WARNING, "failed to unmap \"%s\" %p-%p",
					 pds->kds_fname,
					 (char *)pds->kds,
					 (char *)pds->kds + mmap_length - 1);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	pds->ptoast = NULL;	/* never used */

	/*
	 * initialize common part of kds. Note that row-format cannot
	 * determine 'nrooms' preliminary, so INT_MAX instead.
	 */
	init_kernel_data_store(pds->kds, tupdesc, pds->kds_length,
						   KDS_FORMAT_ROW, INT_MAX);

	/* OK, it is now tracked by GpuContext */
	dlist_push_tail(&gcontext->pds_list, &pds->pds_chain);

	return pds;
}

pgstrom_data_store *
pgstrom_create_data_store_slot(GpuContext *gcontext,
							   TupleDesc tupdesc, cl_uint nrooms,
							   Size extra_length,
							   pgstrom_data_store *ptoast)
{
	pgstrom_data_store *pds;
	size_t			kds_length;
	MemoryContext	gmcxt = gcontext->memcxt;

	/* allocation of pds */
	pds = MemoryContextAllocZero(gmcxt, sizeof(pgstrom_data_store));
	pds->refcnt = 1;	/* owned by the caller at least */

	/* allocation of kds */
	kds_length = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[tupdesc->natts])) +
				  STROMALIGN(LONGALIGN((sizeof(Datum) + sizeof(char)) *
									   tupdesc->natts) * nrooms));
	kds_length += STROMALIGN(extra_length);

	pds->kds_length = kds_length;

	if (!ptoast || !ptoast->kds_fname)
		pds->kds = MemoryContextAllocHuge(gmcxt, pds->kds_length);
	else
	{
		int			kds_fdesc;
		size_t		file_length;
		size_t		mmap_length;
		CUresult	rc;

		/* append KDS after the ktoast file */
		pds->kds_offset = TYPEALIGN(BLCKSZ, ptoast->kds_length);

		pds->kds_fname = MemoryContextStrdup(gmcxt, ptoast->kds_fname);
		kds_fdesc = OpenTransientFile(pds->kds_fname,
									  O_RDWR | PG_BINARY, 0600);
		if (kds_fdesc < 0)
			ereport(ERROR,
                    (errcode_for_file_access(),
					 errmsg("could not open file-mapped data store \"%s\"",
							pds->kds_fname)));

		mmap_length = TYPEALIGN(BLCKSZ, pds->kds_length);
		file_length = pds->kds_offset + mmap_length;
		if (ftruncate(kds_fdesc, file_length) != 0)
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not truncate file \"%s\" to %zu: %m",
							pds->kds_fname, file_length)));

		pds->kds = mmap(NULL, mmap_length,
						PROT_READ | PROT_WRITE,
						MAP_SHARED | MAP_POPULATE,
						kds_fdesc, pds->kds_offset);
		if (pds->kds == MAP_FAILED)
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not mmap \"%s\" with len/ofs=%zu/%zu: %m",
							pds->kds_fname,
							pds->kds_length, pds->kds_offset)));

		/* kds is still mapped - not to manage file desc by ourself */
		CloseTransientFile(kds_fdesc);

		/*
		 * registers this KDS as page-locked area
		 * Unless PDS is not tracked by GpuContext, mapped file is
		 * never unmapped, so we have to care about the case when
		 * cuMemHostRegister() got failed.
		 */
		PG_TRY();
		{
			rc = cuCtxPushCurrent(gcontext->gpu[0].cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

			rc = cuMemHostRegister(pds->kds, mmap_length,
								   CU_MEMHOSTREGISTER_PORTABLE);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemHostRegister: %s", errorText(rc));

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		}
		PG_CATCH();
		{
			if (munmap(pds->kds, mmap_length) != 0)
				elog(WARNING, "failed to unmap \"%s\" %p-%p",
					 pds->kds_fname,
					 (char *)pds->kds,
					 (char *)pds->kds + mmap_length - 1);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	/*
	 * toast buffer shall be released with main data-store together,
	 * so we don't need to track it individually.
	 */
	if (ptoast)
	{
		dlist_delete(&ptoast->pds_chain);
		memset(&ptoast->pds_chain, 0, sizeof(dlist_node));
		pds->ptoast = ptoast;
	}
	init_kernel_data_store(pds->kds, tupdesc, pds->kds_length,
						   KDS_FORMAT_SLOT, nrooms);

	/* OK, now it is tracked by GpuContext */
	dlist_push_tail(&gcontext->pds_list, &pds->pds_chain);

	return pds;
}

pgstrom_data_store *
pgstrom_create_data_store_hash(GpuContext *gcontext,
							   TupleDesc tupdesc,
							   Size length,
							   bool file_mapped)
{
	pgstrom_data_store *pds;

	if (KDS_CALCULATE_HEAD_LENGTH(tupdesc->natts) > length)
		elog(ERROR, "Required length for KDS-Hash is too short");

	/*
	 * KDS_FORMAT_HASH has almost same initialization to KDS_FORMAT_ROW,
	 * so we once create it as _row format, then fixup the pds/kds.
	 */
	pds = pgstrom_create_data_store_row(gcontext, tupdesc,
										length, file_mapped);
	pds->kds->format = KDS_FORMAT_HASH;
	Assert(pds->kds->nslots == 0);	/* to be set later */

	return pds;
}

/*
 * pgstrom_file_mmap_data_store
 *
 * This routine assume that dynamic background worker maps shared-file
 * to process it. So, here is no GpuContext.
 */
void
pgstrom_file_mmap_data_store(FileName kds_fname,
                             Size kds_offset, Size kds_length,
							 kern_data_store **p_kds,
							 kern_data_store **p_ktoast)
{
	size_t		mmap_length = TYPEALIGN(BLCKSZ, kds_length);
	size_t		mmap_offset = TYPEALIGN(BLCKSZ, kds_offset);
	void	   *mmap_addr;
	int			kds_fdesc;

	kds_fdesc = OpenTransientFile(kds_fname,
								  O_RDWR | PG_BINARY, 0600);
	if (kds_fdesc < 0)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open file-mapped data store \"%s\"",
						kds_fname)));

	if (ftruncate(kds_fdesc, kds_offset + kds_length) != 0)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not truncate file \"%s\" to %zu: %m",
						kds_fname, kds_offset + kds_length)));
	/*
	 * Toast buffer of file-mapped data store is located on the header
	 * portion of the same file. So, offset will be truncated to 0,
	 * and length is expanded.
	 */
	if (p_ktoast)
	{
		mmap_length += mmap_offset;
		mmap_offset = 0;
	}
	mmap_addr = mmap(NULL, mmap_length,
					 PROT_READ | PROT_WRITE,
#ifdef MAP_POPULATE
					 MAP_POPULATE |
#endif
					 MAP_SHARED,
					 kds_fdesc, mmap_offset);
	if (mmap_addr == MAP_FAILED)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not mmap \"%s\" with len/ofs=%zu/%zu: %m",
						kds_fname, mmap_length, mmap_offset)));
	if (!p_ktoast)
		*p_kds = (kern_data_store *)mmap_addr;
	else
	{
		*p_kds = (kern_data_store *)((char *)mmap_addr + kds_offset);
		*p_ktoast = (kern_data_store *)mmap_addr;
	}
	/* kds is still mapped - not to manage file desc by ourself */
	CloseTransientFile(kds_fdesc);
}

int
pgstrom_data_store_insert_block(pgstrom_data_store *pds,
								Relation rel, BlockNumber blknum,
								Snapshot snapshot, bool page_prune)
{
	kern_data_store	*kds = pds->kds;
	Buffer			buffer;
	Page			page;
	int				lines;
	int				ntup;
	OffsetNumber	lineoff;
	ItemId			lpp;
	uint		   *tup_index;
	kern_tupitem   *tup_item;
	bool			all_visible;
	Size			max_consume;

	/* only row-store can block read */
	Assert(kds->format == KDS_FORMAT_ROW && kds->nslots == 0);

	CHECK_FOR_INTERRUPTS();

	/* Load the target buffer */
	buffer = ReadBuffer(rel, blknum);

	/* Just like heapgetpage(), however, jobs we focus on is OLAP
	 * workload, so it's uncertain whether we should vacuum the page
	 * here.
	 */
	if (page_prune)
		heap_page_prune_opt(rel, buffer);

	/* we will check tuple's visibility under the shared lock */
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	page = (Page) BufferGetPage(buffer);
	lines = PageGetMaxOffsetNumber(page);
	ntup = 0;

	/*
	 * Check whether we have enough rooms to store expected number of
	 * tuples on the remaining space. If it is hopeless to load all
	 * the items in a block, we inform the caller this block shall be
	 * loaded on the next data store.
	 */
	max_consume = KDS_CALCULATE_HASH_LENGTH(kds->ncols,
											kds->nitems + lines,
											offsetof(kern_tupitem,
													 htup) * lines +
											BLCKSZ + kds->usage);
	if (max_consume > kds->length)
	{
		UnlockReleaseBuffer(buffer);
		return -1;
	}

	/*
	 * Logic is almost same as heapgetpage() doing.
	 */
	all_visible = PageIsAllVisible(page) && !snapshot->takenDuringRecovery;

	/* TODO: make SerializationNeededForRead() an external function
	 * on the core side. It kills necessity of setting up HeapTupleData
	 * when all_visible and non-serialized transaction.
	 */
	tup_index = KERN_DATA_STORE_ROWINDEX(kds) + kds->nitems;
	for (lineoff = FirstOffsetNumber, lpp = PageGetItemId(page, lineoff);
		 lineoff <= lines;
		 lineoff++, lpp++)
	{
		HeapTupleData	tup;
		bool			valid;

		if (!ItemIdIsNormal(lpp))
			continue;

		tup.t_tableOid = RelationGetRelid(rel);
		tup.t_data = (HeapTupleHeader) PageGetItem((Page) page, lpp);
		tup.t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&tup.t_self, blknum, lineoff);

		if (all_visible)
			valid = true;
		else
			valid = HeapTupleSatisfiesVisibility(&tup, snapshot, buffer);

		CheckForSerializableConflictOut(valid, rel, &tup, buffer, snapshot);
		if (!valid)
			continue;

		/* put tuple */
		kds->usage += LONGALIGN(offsetof(kern_tupitem, htup) + tup.t_len);
		tup_item = (kern_tupitem *)((char *)kds + kds->length - kds->usage);
		tup_index[ntup] = (uintptr_t)tup_item - (uintptr_t)kds;
		tup_item->t_len = tup.t_len;
		tup_item->t_self = tup.t_self;
		memcpy(&tup_item->htup, tup.t_data, tup.t_len);

		ntup++;
	}
	UnlockReleaseBuffer(buffer);
	Assert(ntup <= MaxHeapTuplesPerPage);
	Assert(kds->nitems + ntup <= kds->nrooms);
	kds->nitems += ntup;

	return ntup;
}

/*
 * pgstrom_data_store_insert_tuple
 *
 * It inserts a tuple on the data store. Unlike block read mode, we can use
 * this interface for both of row and column data store.
 */
bool
pgstrom_data_store_insert_tuple(pgstrom_data_store *pds,
								TupleTableSlot *slot)
{
	kern_data_store	   *kds = pds->kds;
	size_t				required;
	HeapTuple			tuple;
	cl_uint			   *tup_index;
	kern_tupitem	   *tup_item;

	Assert(pds->kds_length == kds->length);

	/* No room to store a new kern_rowitem? */
	if (kds->nitems >= kds->nrooms)
		return false;
	Assert(kds->ncols == slot->tts_tupleDescriptor->natts);

	if (kds->format != KDS_FORMAT_ROW)
		elog(ERROR, "Bug? unexpected data-store format: %d", kds->format);

	/* OK, put a record */
	tup_index = KERN_DATA_STORE_ROWINDEX(kds);

	/* reference a HeapTuple in TupleTableSlot */
	tuple = ExecFetchSlotTuple(slot);

	/* check whether we have room for this tuple */
	required = LONGALIGN(offsetof(kern_tupitem, htup) + tuple->t_len);
	if (KDS_CALCULATE_ROW_LENGTH(kds->ncols,
								 kds->nitems + 1,
								 required + kds->usage) > kds->length)
		return false;

	kds->usage += required;
	tup_item = (kern_tupitem *)((char *)kds + kds->length - kds->usage);
	tup_item->t_len = tuple->t_len;
	tup_item->t_self = tuple->t_self;
	memcpy(&tup_item->htup, tuple->t_data, tuple->t_len);
	tup_index[kds->nitems++] = (uintptr_t)tup_item - (uintptr_t)kds;

	return true;
}


/*
 * pgstrom_data_store_insert_hashitem
 *
 * It inserts a tuple to the data store of hash format.
 */
bool
pgstrom_data_store_insert_hashitem(pgstrom_data_store *pds,
								   TupleTableSlot *slot,
								   cl_uint hash_value)
{
	kern_data_store	   *kds = pds->kds;
	cl_uint			   *row_index = KERN_DATA_STORE_ROWINDEX(kds);
	Size				required;
	HeapTuple			tuple;
	kern_hashitem	   *khitem;

	Assert(pds->kds_length == kds->length);

	/* No room to store a new kern_hashitem? */
	if (kds->nitems >= kds->nrooms)
		return false;
	Assert(kds->ncols == slot->tts_tupleDescriptor->natts);

	/* KDS has to be KDS_FORMAT_HASH */
	if (kds->format != KDS_FORMAT_HASH)
		elog(ERROR, "Bug? unexpected data-store format: %d", kds->format);

	/* compute required length */
	tuple = ExecFetchSlotTuple(slot);
	required = MAXALIGN(offsetof(kern_hashitem, t.htup) + tuple->t_len);

	Assert(kds->usage == MAXALIGN(kds->usage));
	if (KDS_CALCULATE_HASH_LENGTH(kds->ncols,
								  kds->nitems + 1,
								  required + kds->usage) > pds->kds_length)
		return false;	/* no more space to put */

	/* OK, put a tuple */
	Assert(kds->usage == MAXALIGN(kds->usage));
	khitem = (kern_hashitem *)((char *)kds + kds->length
							   - (kds->usage + required));
	kds->usage += required;
	khitem->hash = hash_value;
	khitem->next = 0x7f7f7f7f;	/* to be set later */
	khitem->rowid = kds->nitems++;
	khitem->t.t_len = tuple->t_len;
	khitem->t.t_self = tuple->t_self;
	memcpy(&khitem->t.htup, tuple->t_data, tuple->t_len);

	row_index[khitem->rowid] = (cl_uint)((uintptr_t)&khitem->t.t_len -
										 (uintptr_t)kds);
	return true;
}

/*
 * PDS_build_hashtable
 *
 * construct hash table according to the current contents
 */
void
PDS_build_hashtable(pgstrom_data_store *pds)
{
	kern_data_store *kds = pds->kds;
	cl_uint		   *row_index = KERN_DATA_STORE_ROWINDEX(kds);
	cl_uint		   *hash_slot = KERN_DATA_STORE_HASHSLOT(kds);
	cl_uint			i, j, nslots = __KDS_NSLOTS(kds->nitems);

	if (kds->format != KDS_FORMAT_HASH)
		elog(ERROR, "Bug? Only KDS_FORMAT_HASH can build a hash table");
	if (kds->nslots > 0)
		elog(ERROR, "Bug? hash table is already built");

	memset(hash_slot, 0, sizeof(cl_uint) * nslots);
	for (i = 0; i < kds->nitems; i++)
	{
		kern_hashitem  *khitem = (kern_hashitem *)
			((char *)kds + row_index[i] - offsetof(kern_hashitem, t));

		Assert(khitem->rowid == i);
		j = khitem->hash % nslots;
		khitem->next = hash_slot[j];
		hash_slot[j] = (uintptr_t)khitem - (uintptr_t)kds;
	}
	kds->nslots = nslots;
}


#ifdef NOT_USED
/*
 * pgstrom_dump_data_store
 *
 * A utility routine that dumps properties of data store
 */
static inline void
__dump_datum(StringInfo buf, kern_colmeta *cmeta, char *datum)
{
	int		i;

	switch (cmeta->attlen)
	{
		case sizeof(char):
			appendStringInfo(buf, "%02x", *((char *)datum));
			break;
		case sizeof(short):
			appendStringInfo(buf, "%04x", *((short *)datum));
			break;
		case sizeof(int):
			appendStringInfo(buf, "%08x", *((int *)datum));
			break;
		case sizeof(long):
			appendStringInfo(buf, "%016lx", *((long *)datum));
			break;
		default:
			if (cmeta->attlen >= 0)
			{
				for (i=0; i < cmeta->attlen; i++)
					appendStringInfo(buf, "%02x", datum[i]);
			}
			else
			{
				Datum	vl_txt = DirectFunctionCall1(byteaout,
													 PointerGetDatum(datum));
				appendStringInfo(buf, "%s", DatumGetCString(vl_txt));
			}
	}
}

void
pgstrom_dump_data_store(pgstrom_data_store *pds)
{
	kern_data_store	   *kds = pds->kds;
	StringInfoData		buf;
	int					i, j;

	elog(INFO,
		 "pds {kds_fname=%s kds_offset=%zu kds_length=%zu kds=%p ktoast=%p}",
		 pds->kds_fname, pds->kds_offset, pds->kds_length,
		 pds->kds, pds->ptoast);
	elog(INFO,
		 "kds {hostptr=%lu length=%u usage=%u ncols=%u nitems=%u nrooms=%u"
		 " format=%s tdhasoid=%s tdtypeid=%u tdtypmod=%d}",
		 kds->hostptr,
		 kds->length, kds->usage, kds->ncols, kds->nitems, kds->nrooms,
		 kds->format == KDS_FORMAT_ROW ? "row" :
		 kds->format == KDS_FORMAT_SLOT ? "slot" : "unknown",
		 kds->tdhasoid ? "true" : "false", kds->tdtypeid, kds->tdtypmod);
	for (i=0; i < kds->ncols; i++)
	{
		elog(INFO, "column[%d] "
			 "{attbyval=%d attalign=%d attlen=%d attnum=%d attcacheoff=%d}",
			 i,
			 kds->colmeta[i].attbyval,
			 kds->colmeta[i].attalign,
			 kds->colmeta[i].attlen,
			 kds->colmeta[i].attnum,
			 kds->colmeta[i].attcacheoff);
	}

	if (kds->format == KDS_FORMAT_ROW)
	{
		initStringInfo(&buf);
		for (i=0; i < kds->nitems; i++)
		{
			kern_tupitem *tup_item = KERN_DATA_STORE_TUPITEM(kds, i);
			HeapTupleHeaderData *htup = &tup_item->htup;
			size_t		offset = (uintptr_t)tup_item - (uintptr_t)kds;
			cl_int		natts = (htup->t_infomask2 & HEAP_NATTS_MASK);
			cl_int		curr = htup->t_hoff;
			char	   *datum;

			resetStringInfo(&buf);
			appendStringInfo(&buf, "htup[%d] @%zu natts=%u {",
							 i, offset, natts);
			for (j=0; j < kds->ncols; j++)
			{
				if (j > 0)
					appendStringInfo(&buf, ", ");
				if ((htup->t_infomask & HEAP_HASNULL) != 0 &&
					att_isnull(j, htup->t_bits))
					appendStringInfo(&buf, "null");
				else if (kds->colmeta[j].attlen > 0)
				{
					curr = TYPEALIGN(kds->colmeta[j].attalign, curr);
					datum = (char *)htup + curr;

					__dump_datum(&buf, &kds->colmeta[j], datum);
				}
				else
				{
					if (!VARATT_NOT_PAD_BYTE((char *)htup + curr))
						curr = TYPEALIGN(kds->colmeta[j].attalign, curr);
					datum = (char *)htup + curr;
					__dump_datum(&buf, &kds->colmeta[j], datum);
					curr += VARSIZE_ANY(datum);
				}
			}
			appendStringInfo(&buf, "}");
			elog(INFO, "%s", buf.data);
		}
		pfree(buf.data);
	}
}
#endif

void
pgstrom_init_datastore(void)
{
	DefineCustomIntVariable("pg_strom.chunk_size",
							"default size of pgstrom_data_store",
							NULL,
							&pgstrom_chunk_size_kb,
							15872,
							4096,
							MAX_KILOBYTES,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							check_guc_chunk_size, NULL, NULL);
	DefineCustomIntVariable("pg_strom.chunk_limit",
							"limit size of pgstrom_data_store",
							NULL,
							&pgstrom_chunk_limit_kb,
							5 * pgstrom_chunk_size_kb,
							4096,
							MAX_KILOBYTES,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							check_guc_chunk_limit, NULL, NULL);
}
