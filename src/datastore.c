/*
 * datastore.c
 *
 * Routines to manage data store; row-store, column-store, toast-buffer,
 * and param-buffer.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "access/sysattr.h"
#include "catalog/catalog.h"
#include "catalog/pg_tablespace.h"
#include "catalog/pg_type.h"
#include "commands/tablespace.h"
#include "miscadmin.h"
#include "port.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "storage/predicate.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/tqual.h"
#include "pg_strom.h"
#include "opencl_numeric.h"
#include <sys/mman.h>

/*
 * GUC variables
 */
static int		pgstrom_chunk_size_kb;
static char	   *pgstrom_temp_tablespace;

/*
 * pgstrom_chunk_size - configured chunk size
 */
Size
pgstrom_chunk_size(void)
{
	return ((Size)pgstrom_chunk_size_kb) << 10;
}

/*
 * pgstrom_create_param_buffer
 *
 * It construct a param-buffer on the shared memory segment, according to
 * the supplied Const/Param list. Its initial reference counter is 1, so
 * this buffer can be released using pgstrom_put_param_buffer().
 */
kern_parambuf *
pgstrom_create_kern_parambuf(List *used_params,
							 ExprContext *econtext)
{
	StringInfoData	str;
	kern_parambuf  *kpbuf;
	char		padding[STROMALIGN_LEN];
	ListCell   *cell;
	Size		offset;
	int			index = 0;
	int			nparams = list_length(used_params);

	/* seek to the head of variable length field */
	offset = STROMALIGN(offsetof(kern_parambuf, poffset[nparams]));
	initStringInfo(&str);
	enlargeStringInfo(&str, offset);
	memset(str.data, 0, offset);
	str.len = offset;
	/* walks on the Para/Const list */
	foreach (cell, used_params)
	{
		Node   *node = lfirst(cell);

		if (IsA(node, Const))
		{
			Const  *con = (Const *) node;

			kpbuf = (kern_parambuf *)str.data;
			if (con->constisnull)
				kpbuf->poffset[index] = 0;	/* null */
			else
			{
				kpbuf->poffset[index] = str.len;
				if (con->constlen > 0)
					appendBinaryStringInfo(&str,
										   (char *)&con->constvalue,
										   con->constlen);
				else
					appendBinaryStringInfo(&str,
										   DatumGetPointer(con->constvalue),
										   VARSIZE(con->constvalue));
			}
		}
		else if (IsA(node, Param))
		{
			ParamListInfo param_info = econtext->ecxt_param_list_info;
			Param  *param = (Param *) node;

			if (param_info &&
				param->paramid > 0 && param->paramid <= param_info->numParams)
			{
				ParamExternData	*prm = &param_info->params[param->paramid - 1];

				/* give hook a chance in case parameter is dynamic */
				if (!OidIsValid(prm->ptype) && param_info->paramFetch != NULL)
					(*param_info->paramFetch) (param_info, param->paramid);

				kpbuf = (kern_parambuf *)str.data;
				if (!OidIsValid(prm->ptype))
				{
					elog(INFO, "debug: Param has no particular data type");
					kpbuf->poffset[index++] = 0;	/* null */
					continue;
				}
				/* safety check in case hook did something unexpected */
				if (prm->ptype != param->paramtype)
					ereport(ERROR,
							(errcode(ERRCODE_DATATYPE_MISMATCH),
							 errmsg("type of parameter %d (%s) does not match that when preparing the plan (%s)",
									param->paramid,
									format_type_be(prm->ptype),
									format_type_be(param->paramtype))));
				if (prm->isnull)
					kpbuf->poffset[index] = 0;	/* null */
				else
				{
					int		typlen = get_typlen(prm->ptype);

					if (typlen == 0)
						elog(ERROR, "cache lookup failed for type %u",
							 prm->ptype);
					if (typlen > 0)
						appendBinaryStringInfo(&str,
											   (char *)&prm->value,
											   typlen);
					else
						appendBinaryStringInfo(&str,
											   DatumGetPointer(prm->value),
											   VARSIZE(prm->value));
				}
			}
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(node));

		/* alignment */
		if (STROMALIGN(str.len) != str.len)
			appendBinaryStringInfo(&str, padding,
								   STROMALIGN(str.len) - str.len);
		index++;
	}
	Assert(STROMALIGN(str.len) == str.len);
	kpbuf = (kern_parambuf *)str.data;
	kpbuf->length = str.len;
	kpbuf->nparams = nparams;

	return kpbuf;
}

Datum
pgstrom_fixup_kernel_numeric(Datum datum)
{
	cl_ulong	numeric_value = (cl_ulong) datum;
	bool		sign = PG_NUMERIC_SIGN(numeric_value);
	int			expo = PG_NUMERIC_EXPONENT(numeric_value);
	cl_ulong	mantissa = PG_NUMERIC_MANTISSA(numeric_value);
	char		temp[100];

	/* more effective implementation in the future :-) */
	snprintf(temp, sizeof(temp), "%c%lue%d",
			 sign ? '-' : '+', mantissa, expo);
	//elog(INFO, "numeric %016lx -> %s", numeric_value, temp);
	return DirectFunctionCall3(numeric_in,
							   CStringGetDatum(temp),
							   Int32GetDatum(0),
							   Int32GetDatum(-1));
}

bool
kern_fetch_data_store(TupleTableSlot *slot,
					  kern_data_store *kds,
					  size_t row_index,
					  HeapTuple tuple)
{
	if (row_index >= kds->nitems)
		return false;	/* out of range */

	/* make clear the result tuple-slot */
	ExecClearTuple(slot);

	/* in case of row-store */
	if (kds->format == KDS_FORMAT_ROW)
	{
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, row_index);
		kern_blkitem   *bitem;
		BlockNumber		blknum;
		ItemId			lpp;

		Assert(ritem->blk_index < kds->nblocks);
		bitem = KERN_DATA_STORE_BLKITEM(kds, ritem->blk_index);
		lpp = PageGetItemId(bitem->page, ritem->item_offset);
		Assert(ItemIdIsNormal(lpp));
		blknum = BufferGetBlockNumber(bitem->buffer);

		tuple->t_data = (HeapTupleHeader) PageGetItem(bitem->page, lpp);
		tuple->t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&tuple->t_self, blknum, ritem->item_offset);

		ExecStoreTuple(tuple, slot, bitem->buffer, false);

		return true;
	}
	/* in case of row-flat-store */
	if (kds->format == KDS_FORMAT_ROW_FLAT ||
		kds->format == KDS_FORMAT_ROW_FMAP)
	{
		TupleDesc		tupdesc = slot->tts_tupleDescriptor;
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, row_index);
		HeapTupleHeader	htup;

		Assert(ritem->htup_offset < kds->length);
		htup = (HeapTupleHeader)((char *)kds + ritem->htup_offset);

		memset(tuple, 0, sizeof(HeapTupleData));
		tuple->t_data = htup;
		heap_deform_tuple(tuple, tupdesc,
						  slot->tts_values,
						  slot->tts_isnull);
		ExecStoreVirtualTuple(slot);

		return true;
	}
	/* in case of tuple-slot format */
	if (kds->format == KDS_FORMAT_TUPSLOT)
	{
		Datum  *tts_values = (Datum *)KERN_DATA_STORE_VALUES(kds, row_index);
		bool   *tts_isnull = (bool *)KERN_DATA_STORE_ISNULL(kds, row_index);

		ExecClearTuple(slot);
		slot->tts_values = tts_values;
		slot->tts_isnull = tts_isnull;
		ExecStoreVirtualTuple(slot);

		/*
		 * MEMO: Is it really needed to take memcpy() here? If we can 
		 * do same job with pointer opertion, it makes more sense from
		 * performance standpoint.
		 * (NOTE: hash-join materialization is a hot point)
		 */
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

void
pgstrom_release_data_store(pgstrom_data_store *pds)
{
	ResourceOwner		saved_owner;
	int					i, rc;

	saved_owner = CurrentResourceOwner;
	PG_TRY();
	{
		kern_data_store	   *kds = pds->kds;

		CurrentResourceOwner = pds->resowner;

		/*
		 * NOTE: A page buffer is assigned on the PG-Strom's shared memory,
		 * if referenced table is local / temporary relation (because its
		 * buffer is originally assigned on the private memory; invisible
		 * from OpenCL server process).
		 *
		 * Also note that, we don't need to call ReleaseBuffer() in case
		 * when the current context is OpenCL server or cleanup callback
		 * of resource-tracker.
		 * If pgstrom_release_data_store() is called on the OpenCL server
		 * context, it means the source transaction was already aborted
		 * thus the shared-buffers being mapped were already released.
		 * (It leads probability to send invalid region by DMA; so
		 * kern_get_datum_rs() takes careful data validation.)
		 * If pgstrom_release_data_store() is called under the cleanup
		 * callback context of resource-tracker, it means shared-buffers
		 * being pinned by the current transaction were already released
		 * by the built-in code prior to PG-Strom's cleanup. So, no need
		 * to do anything by ourselves.
		 */
		if (kds)
		{
			Assert(kds->nblocks <= kds->maxblocks);
			for (i = kds->nblocks - 1; i >= 0; i--)
			{
				kern_blkitem   *bitem = KERN_DATA_STORE_BLKITEM(kds, i);

				if (BufferIsLocal(bitem->buffer) ||
					BufferIsInvalid(bitem->buffer))
					continue;
				if (!pgstrom_i_am_clserv &&
					!pgstrom_restrack_cleanup_context())
					ReleaseBuffer(bitem->buffer);
			}
			if (kds->format != KDS_FORMAT_ROW_FMAP)
				pgstrom_shmem_free(kds);
			else
			{
				Assert(pds->kds_fname != NULL);
				rc = munmap(kds, kds->length);
				if (rc != 0)
					elog(LOG, "Bug? failed to unmap kds:%p of \"%s\" (%s)",
						 pds->kds, pds->kds_fname, strerror(errno));
				CloseTransientFile(pds->kds_fdesc);
				unlink(pds->kds_fname);
			}
		}
	}
	PG_CATCH();
	{
		CurrentResourceOwner = saved_owner;
		PG_RE_THROW();
	}
	PG_END_TRY();
	CurrentResourceOwner = saved_owner;
	if (pds->resowner &&
		!pgstrom_i_am_clserv &&
		!pgstrom_restrack_cleanup_context())
		ResourceOwnerDelete(pds->resowner);
	if (pds->ktoast)
		pgstrom_release_data_store(pds->ktoast);
	if (pds->local_pages)
		pgstrom_shmem_free(pds->local_pages);
	pgstrom_shmem_free(pds);
}

static void
init_kern_data_store(kern_data_store *kds,
					 TupleDesc tupdesc,
					 Size length,
					 int format,
					 cl_uint maxblocks,
					 cl_uint nrooms,
					 bool internal_format)
{
	int		i, attcacheoff;

	kds->hostptr = (hostptr_t) &kds->hostptr;
	kds->length = length;
	kds->usage = 0;
	kds->ncols = tupdesc->natts;
	kds->nitems = 0;
	kds->nrooms = nrooms;
	kds->nblocks = 0;
	kds->maxblocks = maxblocks;
	kds->format = format;
	kds->tdhasoid = tupdesc->tdhasoid;
	kds->tdtypeid = tupdesc->tdtypeid;
	kds->tdtypmod = tupdesc->tdtypmod;

	attcacheoff = offsetof(HeapTupleHeaderData, t_bits);
	if (tupdesc->tdhasoid)
		attcacheoff += sizeof(Oid);
	attcacheoff = MAXALIGN(attcacheoff);

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];
		bool	attbyval = attr->attbyval;
		int		attalign = typealign_get_width(attr->attalign);
		int		attlen   = attr->attlen;
		int		attnum   = attr->attnum;

		/*
		 * If variable is expected to have special internal format
		 * different from the host representation, we need to fixup
		 * colmeta catalog also. Right now, only NUMERIC can have
		 * special internal format.
		 */
		if (internal_format)
		{
			if (attr->atttypid == NUMERICOID)
			{
				attbyval = true;
				attalign = sizeof(cl_ulong);
				attlen   = sizeof(cl_ulong);
			}
		}

		if (attcacheoff > 0)
		{
			if (attlen > 0)
				attcacheoff = TYPEALIGN(attalign, attcacheoff);
			else
				attcacheoff = -1;	/* no more shortcut any more */
		}
		kds->colmeta[i].attbyval = attbyval;
		kds->colmeta[i].attalign = attalign;
		kds->colmeta[i].attlen = attlen;
		kds->colmeta[i].attnum = attnum;
		kds->colmeta[i].attcacheoff = attcacheoff;
		if (attcacheoff >= 0)
			attcacheoff += attlen;
	}
}

pgstrom_data_store *
__pgstrom_create_data_store_row(const char *filename, int lineno,
								TupleDesc tupdesc,
								Size pds_length,
								Size tup_width)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	Size		required;
	cl_uint		maxblocks;
	cl_uint		nrooms;

	/* size of data-store has to be aligned to BLCKSZ */
	pds_length = TYPEALIGN(BLCKSZ, pds_length);
	maxblocks = pds_length / BLCKSZ;
	nrooms = (cl_uint)((double)BLCKSZ *
					   (double)maxblocks * 1.25 /
					   (double)tup_width);

	/* allocation of kern_data_store */
	required = (STROMALIGN(offsetof(kern_data_store,
									colmeta[tupdesc->natts])) +
				STROMALIGN(sizeof(kern_blkitem) * maxblocks) +
				STROMALIGN(sizeof(kern_rowitem) * nrooms));
	kds = __pgstrom_shmem_alloc(filename,lineno,
								required);
	if (!kds)
		elog(ERROR, "out of shared memory");
	init_kern_data_store(kds, tupdesc, required,
						 KDS_FORMAT_ROW, maxblocks, nrooms, false);
	/* allocation of pgstrom_data_store */
	pds = __pgstrom_shmem_alloc(filename,lineno,
								sizeof(pgstrom_data_store));
	if (!pds)
	{
		pgstrom_shmem_free(kds);
		elog(ERROR, "out of shared memory");
	}
	/* ResourceOwnerCreate() may raise an error */
	PG_TRY();
	{
		pds->sobj.stag = StromTag_DataStore;
		SpinLockInit(&pds->lock);
		pds->refcnt = 1;
		pds->kds = kds;
		pds->kds_length = kds->length;
		pds->kds_fdesc = -1;	/* never used */
		pds->ktoast = NULL;		/* never used */
		pds->resowner = ResourceOwnerCreate(CurrentResourceOwner,
											"pgstrom_data_store");
		pds->local_pages = NULL;	/* allocation on demand */
	}
	PG_CATCH();
	{
		pgstrom_shmem_free(kds);
		pgstrom_shmem_free(pds);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return pds;
}

pgstrom_data_store *
__pgstrom_create_data_store_row_flat(const char *filename, int lineno,
									 TupleDesc tupdesc, Size length)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	Size		allocated;
	cl_uint		nrooms;

	/* allocation of kern_data_store */
	kds = __pgstrom_shmem_alloc_alap(filename, lineno,
									 length, &allocated);
	if (!kds)
		elog(ERROR, "out of shared memory");
	/*
	 * NOTE: length of row-flat format has to be strictly aligned
	 * because location of heaptuple is calculated using offset from
	 * the buffer tail!
	 */
	allocated = STROMALIGN_DOWN(allocated);

	/* max number of rooms according to the allocated buffer length */
	nrooms = (STROMALIGN_DOWN(length) -
			  STROMALIGN(offsetof(kern_data_store,
								  colmeta[tupdesc->natts])))
		/ sizeof(kern_rowitem);
	init_kern_data_store(kds, tupdesc, allocated,
						 KDS_FORMAT_ROW_FLAT, 0, nrooms, false);
	/* allocation of pgstrom_data_store */
	pds = __pgstrom_shmem_alloc(filename, lineno,
								sizeof(pgstrom_data_store));
	if (!pds)
	{
		pgstrom_shmem_free(kds);
		elog(ERROR, "out of shared memory");
	}
	pds->sobj.stag = StromTag_DataStore;
	SpinLockInit(&pds->lock);
	pds->refcnt = 1;
	pds->kds = kds;
	pds->kds_length = kds->length;
	pds->kds_fdesc = -1;	/* never used */
	pds->ktoast = NULL;		/* never used */
	pds->resowner = NULL;	/* never used */
	pds->local_pages = NULL;/* never used */

	return pds;
}

/*
 * get_pgstrom_temp_filename - logic is almost same as OpenTemporaryFile,
 * but it returns cstring of filename, for OpenTransientFile and mmap(2)
 */
static char *
get_pgstrom_temp_filename(void)
{
	char	tempdirpath[MAXPGPATH];
	char	tempfilepath[MAXPGPATH];
	Oid		tablespace_oid = InvalidOid;
	static long tempFileCounter = 0;

	if (pgstrom_temp_tablespace != NULL)
		tablespace_oid = get_tablespace_oid(pgstrom_temp_tablespace, false);

	if (!OidIsValid(tablespace_oid) ||
		tablespace_oid == DEFAULTTABLESPACE_OID ||
		tablespace_oid == GLOBALTABLESPACE_OID)
	{
		/* The default tablespace is {datadir}/base */
		snprintf(tempdirpath, sizeof(tempdirpath), "base/%s",
				 PG_TEMP_FILES_DIR);
	}
	else
	{
		/* All other tablespaces are accessed via symlinks */
		snprintf(tempdirpath, sizeof(tempdirpath), "pg_tblspc/%u/%s/%s",
				 tablespace_oid,
				 TABLESPACE_VERSION_DIRECTORY,
				 PG_TEMP_FILES_DIR);
    }

	/*
	 * Generate a tempfile name that should be unique within the current
	 * database instance.
	 */
	snprintf(tempfilepath, sizeof(tempfilepath), "%s/strom_tmp%d.%ld",
			 tempdirpath, MyProcPid, tempFileCounter++);

	return pstrdup(tempfilepath);
}

pgstrom_data_store *
__pgstrom_create_data_store_row_fmap(const char *filename, int lineno,
									 TupleDesc tupdesc, Size length)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	char	   *kds_fname;
	int			kds_fdesc;
	Size		kds_length = STROMALIGN(length);
	cl_uint		nrooms;

	kds_fname = get_pgstrom_temp_filename();
	kds_fdesc = OpenTransientFile(kds_fname,
								  O_RDWR | O_CREAT | O_TRUNC | PG_BINARY,
								  0600);
	if (kds_fdesc < 0)
		ereport(ERROR,
                (errcode_for_file_access(),
				 errmsg("could not create file-mapped data store \"%s\"",
						kds_fname)));

	if (ftruncate(kds_fdesc, kds_length) != 0)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not truncate file \"%s\" to %zu: %m",
						kds_fname, kds_length)));

	kds = mmap(NULL, kds_length,
			   PROT_READ | PROT_WRITE,
			   MAP_SHARED | MAP_POPULATE,
			   kds_fdesc, 0);
	if (kds == MAP_FAILED)
		elog(ERROR, "failed to map file-mapped data store \"%s\"", kds_fname);

	/* OK, let's initialize the file-mapped kern_data_store */
	nrooms = (STROMALIGN_DOWN(kds_length) -
			  STROMALIGN(offsetof(kern_data_store,
								  colmeta[tupdesc->natts])))
		/ sizeof(kern_rowitem);
	init_kern_data_store(kds, tupdesc, kds_length,
						 KDS_FORMAT_ROW_FMAP, 0, nrooms, false);
	/* Also, allocation of pgstrom_data_store */
	pds = __pgstrom_shmem_alloc(filename, lineno,
								sizeof(pgstrom_data_store) +
								strlen(kds_fname) + 1);
	if (!pds)
	{
		pgstrom_shmem_free(kds);
		elog(ERROR, "out of shared memory");
	}
	pds->sobj.stag = StromTag_DataStore;
	SpinLockInit(&pds->lock);
	pds->refcnt = 1;
	pds->kds = kds;
	pds->kds_length = kds->length;
	pds->kds_fname = (char *)(pds + 1);
	strcpy(pds->kds_fname, kds_fname);
	pds->kds_fdesc = kds_fdesc;
	pds->ktoast = NULL;		/* never used */
	pds->resowner = NULL;	/* never used */
	pds->local_pages = NULL;/* never used */

	return pds;
}

kern_data_store *
filemap_kern_data_store(const char *kds_fname, size_t kds_length, int *p_fdesc)
{
	kern_data_store	   *kds;
	int					kds_fdesc;

	kds_fdesc = open(kds_fname, O_RDWR, 0);
	if (kds_fdesc < 0)
	{
		clserv_log("failed to open \"%s\" (%s)", kds_fname, strerror(errno));
		return NULL;
	}

	kds = mmap(NULL, kds_length,
			   PROT_READ | PROT_WRITE,
			   MAP_SHARED | MAP_POPULATE,
			   kds_fdesc, 0);
	if (kds == MAP_FAILED)
	{
		clserv_log("failed to map \"%s\" (%s)", kds_fname, strerror(errno));
		close(kds_fdesc);
		return NULL;
	}
	Assert(kds->format == KDS_FORMAT_ROW_FMAP);
	if (p_fdesc)
		*p_fdesc = kds_fdesc;
	return kds;
}

void
fileunmap_kern_data_store(kern_data_store *kds, int fdesc)
{
	int		rc;

	rc = munmap(kds, kds->length);
	if (rc != 0)
		clserv_log("failed to unmap kds:%p (%s)", kds, strerror(errno));
	rc = close(fdesc);
	if (rc != 0)
		clserv_log("failed to close file:%d (%s)", fdesc, strerror(errno));
}

pgstrom_data_store *
__pgstrom_create_data_store_tupslot(const char *filename, int lineno,
									TupleDesc tupdesc, cl_uint nrooms,
									bool internal_format)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	Size				required;

	/* kern_data_store */
	required = (STROMALIGN(offsetof(kern_data_store,
									colmeta[tupdesc->natts])) +
				(LONGALIGN(sizeof(bool) * tupdesc->natts) +
				 LONGALIGN(sizeof(Datum) * tupdesc->natts)) * nrooms);
	kds = __pgstrom_shmem_alloc(filename, lineno,
								STROMALIGN(required));
	if (!kds)
		elog(ERROR, "out of shared memory");
	init_kern_data_store(kds, tupdesc, required,
						 KDS_FORMAT_TUPSLOT, 0, nrooms,
						 internal_format);

	/* pgstrom_data_store */
	pds = __pgstrom_shmem_alloc(filename, lineno,
								sizeof(pgstrom_data_store));
	if (!pds)
	{
		pgstrom_shmem_free(kds);
		elog(ERROR, "out of shared memory");
	}
	pds->sobj.stag = StromTag_DataStore;
	SpinLockInit(&pds->lock);
	pds->refcnt = 1;
	pds->kds = kds;
	pds->ktoast = NULL;		/* assigned on demand */
	pds->resowner = NULL;	/* never used for tuple-slot */
	pds->local_pages = NULL;/* never used for tuple-slot */

	return pds;
}

pgstrom_data_store *
pgstrom_get_data_store(pgstrom_data_store *pds)
{
	SpinLockAcquire(&pds->lock);
	Assert(pds->refcnt > 0);
	pds->refcnt++;
	SpinLockRelease(&pds->lock);

	return pds;
}

void
pgstrom_put_data_store(pgstrom_data_store *pds)
{
	bool	do_release = false;

	SpinLockAcquire(&pds->lock);
    Assert(pds->refcnt > 0);
	if (--pds->refcnt == 0)
		do_release = true;
	SpinLockRelease(&pds->lock);
	if (do_release)
		pgstrom_release_data_store(pds);
}

int
pgstrom_data_store_insert_block(pgstrom_data_store *pds,
								Relation rel, BlockNumber blknum,
								Snapshot snapshot, bool page_prune)
{
	kern_data_store	*kds = pds->kds;
	kern_rowitem   *ritem;
	kern_blkitem   *bitem;
	Buffer			buffer;
	Page			page;
	int				lines;
	int				ntup;
	OffsetNumber	lineoff;
	ItemId			lpp;
	bool			all_visible;
	ResourceOwner	saved_owner;

	/* only row-store can block read */
	Assert(kds->format == KDS_FORMAT_ROW);
	/* we never use all the block slots */
	Assert(kds->nblocks < kds->maxblocks);
	/* we need a resource owner to track shared buffer */
	Assert(pds->resowner != NULL);

	CHECK_FOR_INTERRUPTS();

	saved_owner = CurrentResourceOwner;
	PG_TRY();
	{
		CurrentResourceOwner = pds->resowner;

		/* load the target buffer */
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

		/* Check whether we have enough rooms to store expected rowitems
		 * and shared buffer pages, any mode.
		 * If not, we have to inform the caller this block shall be loaded
		 * on the next data-store.
		 */
		if (kds->nitems + lines > kds->nrooms ||
			STROMALIGN(offsetof(kern_data_store, colmeta[kds->ncols])) +
			STROMALIGN(sizeof(kern_blkitem) * (kds->maxblocks)) +
			STROMALIGN(sizeof(kern_rowitem) * (kds->nitems + lines)) +
			BLCKSZ * kds->nblocks >= BLCKSZ * kds->maxblocks)
		{
			UnlockReleaseBuffer(buffer);
			ntup = -1;
			goto out;		/* must restore exception stack */
		}

		/*
		 * Logic is almost same as heapgetpage() doing.
		 */
		all_visible = PageIsAllVisible(page) && !snapshot->takenDuringRecovery;

		/* TODO: make SerializationNeededForRead() an external function
		 * on the core side. It kills necessity of setting up HeapTupleData
		 * when all_visible and non-serialized transaction.
		 */
		ritem = KERN_DATA_STORE_ROWITEM(kds, kds->nitems);
		bitem = KERN_DATA_STORE_BLKITEM(kds, kds->nblocks);
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

			CheckForSerializableConflictOut(valid, rel, &tup,
											buffer, snapshot);
			if (!valid)
				continue;

			ritem->blk_index = kds->nblocks;
			ritem->item_offset = lineoff;
			ritem++;
			ntup++;
		}
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
		Assert(ntup <= MaxHeapTuplesPerPage);
		Assert(kds->nitems + ntup <= kds->nrooms);
		kds->nitems += ntup;

		/*
		 * NOTE: Local buffers are allocated on the private address space,
		 * it is not visible to opencl server. so, we make a duplication
		 * instead. Shared buffer can be referenced with zero-copy.
		 */
		bitem->buffer = buffer;
		if (!BufferIsLocal(buffer))
			bitem->page = page;
		else
		{
			Page	dup_page;

			/*
			 * NOTE: We expect seldom cases requires to mix shared buffers
			 * and private buffers in a single data-chunk. So, we allocate
			 * all the expected local-pages at once. It has two another
			 * benefit; 1. duplicated pages tend to have continuous address
			 * that may reduce number of DMA call, 2. memory allocation
			 * request to BLCKSZ actually consumes 2*BLCKSZ because of
			 * additional fields that does not fit 2^N manner.
			 */
			if (!pds->local_pages)
			{
				Size	required = BLCKSZ * kds->maxblocks;

				pds->local_pages = pgstrom_shmem_alloc(required);
				if (!pds->local_pages)
					elog(ERROR, "out of memory");
			}
			dup_page = (Page)(pds->local_pages + BLCKSZ * kds->nblocks);
			memcpy(dup_page, page, BLCKSZ);
			bitem->page = dup_page;
			ReleaseBuffer(buffer);
		}
		kds->nblocks++;
	out:
		;
	}
	PG_CATCH();
	{
		CurrentResourceOwner = saved_owner;
		PG_RE_THROW();
	}
	PG_END_TRY();
	CurrentResourceOwner = saved_owner;

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

	/* No room to store a new kern_rowitem? */
	if (kds->nitems >= kds->nrooms)
		return false;
	Assert(kds->ncols == slot->tts_tupleDescriptor->natts);

	if (kds->format == KDS_FORMAT_ROW)
	{
		HeapTuple		tuple;
		OffsetNumber	offnum;
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, kds->nitems);
		kern_blkitem   *bitem;
		Page			page = NULL;

		/* reference a HeapTuple in TupleTableSlot */
		tuple = ExecFetchSlotTuple(slot);

		if (kds->nblocks == 0)
			bitem = NULL;
		else
		{
			bitem = KERN_DATA_STORE_BLKITEM(kds, kds->nblocks - 1);
			page = bitem->page;
		}

		if (!bitem ||
			!BufferIsInvalid(bitem->buffer) ||
			PageGetFreeSpace(bitem->page) < MAXALIGN(tuple->t_len))
		{
			/*
			 * Expand blocks if the last one is associated with a particular
			 * shared or private buffer, or free space is not available to
			 * put this new item.
			 */

			/* No rooms to expand blocks? */
			if (kds->nblocks >= kds->maxblocks)
				return false;

			/*
			 * Allocation of anonymous pages at once. See the comments at
			 * pgstrom_data_store_insert_block().
			 */
			if (!pds->local_pages)
			{
				Size	required = kds->maxblocks * BLCKSZ;

				pds->local_pages = pgstrom_shmem_alloc(required);
				if (!pds->local_pages)
					elog(ERROR, "out of memory");
			}
			page = (Page)(pds->local_pages + kds->nblocks * BLCKSZ);
			PageInit(page, BLCKSZ, 0);
			if (PageGetFreeSpace(page) < MAXALIGN(tuple->t_len))
			{
				pgstrom_shmem_free(page);
				elog(ERROR, "tuple too large (%zu)",
					 (Size)MAXALIGN(tuple->t_len));
			}
			bitem = KERN_DATA_STORE_BLKITEM(kds, kds->nblocks);
			bitem->buffer = InvalidBuffer;
			bitem->page = page;
			kds->nblocks++;
		}
		Assert(kds->nblocks > 0);
		offnum = PageAddItem(page, (Item) tuple->t_data, tuple->t_len,
							 InvalidOffsetNumber, false, true);
		if (offnum == InvalidOffsetNumber)
			elog(ERROR, "failed to add tuple");

		ritem->blk_index = kds->nblocks - 1;
		ritem->item_offset = offnum;
		kds->nitems++;

		return true;
	}
	else if (kds->format == KDS_FORMAT_ROW_FLAT ||
			 kds->format == KDS_FORMAT_ROW_FMAP)
	{
		HeapTuple		tuple;
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, kds->nitems);
		uintptr_t		usage;
		char		   *dest_addr;

		/* reference a HeapTuple in TupleTableSlot */
		tuple = ExecFetchSlotTuple(slot);

		/* check whether the tuple touches the watermark */
		usage = ((uintptr_t)(ritem + 1) -
				 (uintptr_t)(kds) +			/* from head */
				 kds->usage +				/* from tail */
				 LONGALIGN(tuple->t_len));	/* newly added */
		if (usage > kds->length)
			return false;

		dest_addr = ((char *)kds + kds->length -
					 kds->usage - LONGALIGN(tuple->t_len));
		memcpy(dest_addr, tuple->t_data, tuple->t_len);
		ritem->htup_offset = (hostptr_t)((char *)dest_addr - (char *)kds);
		kds->usage += LONGALIGN(tuple->t_len);
		kds->nitems++;

		return true;
	}
	elog(ERROR, "Bug? data-store with format %d is not expected",
		 kds->format);
	return false;
}

/*
 * clserv_dmasend_data_store
 *
 * It enqueues DMA send request of the supplied data-store.
 * Note that we expect this routine is called under the OpenCL server
 * context.
 */
cl_int
clserv_dmasend_data_store(pgstrom_data_store *pds,
						  cl_command_queue kcmdq,
						  cl_mem kds_buffer,
						  cl_mem ktoast_buffer,
						  cl_uint num_blockers,
						  const cl_event *blockers,
						  cl_uint *ev_index,
						  cl_event *events,
						  pgstrom_perfmon *pfm)
{
	kern_data_store	   *kds = pds->kds;
	kern_blkitem	   *bitem;
	size_t				length;
	size_t				offset;
	cl_int				i, n, rc;

#ifdef USE_ASSERT_CHECKING
	Assert(pgstrom_i_am_clserv);
	rc = clGetMemObjectInfo(kds_buffer,
							CL_MEM_SIZE,
							sizeof(length),
							&length,
							NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clGetMemObjectInfo (%s)",
				   opencl_strerror(rc));
		return rc;
	}
	Assert(length >= KERN_DATA_STORE_LENGTH(kds));
#endif
	if (kds->format == KDS_FORMAT_ROW_FLAT ||
		kds->format == KDS_FORMAT_TUPSLOT)
	{
		rc = clEnqueueWriteBuffer(kcmdq,
								  kds_buffer,
								  CL_FALSE,
								  0,
								  kds->length,
								  kds,
								  num_blockers,
								  blockers,
								  events + *ev_index);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			return rc;
		}
		(*ev_index)++;
		pfm->bytes_dma_send += kds->length;
		pfm->num_dma_send++;

		if (pds->ktoast)
		{
			pgstrom_data_store	*ktoast = pds->ktoast;

			Assert(!ktoast->ktoast);

			rc = clserv_dmasend_data_store(ktoast,
										   kcmdq,
										   ktoast_buffer,
										   NULL,
										   num_blockers,
										   blockers,
										   ev_index,
										   events,
										   pfm);
		}
		return rc;
	}
	Assert(kds->format == KDS_FORMAT_ROW);
	length = ((uintptr_t)KERN_DATA_STORE_ROWITEM(kds, kds->nitems) -
			  (uintptr_t)(kds));
	rc = clEnqueueWriteBuffer(kcmdq,
							  kds_buffer,
							  CL_FALSE,
							  0,
							  length,
							  kds,
							  num_blockers,
							  blockers,
							  events + (*ev_index));
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s",
				   opencl_strerror(rc));
		return rc;
	}
	(*ev_index)++;
	pfm->bytes_dma_send += kds->length;
	pfm->num_dma_send++;

	offset = ((uintptr_t)KERN_DATA_STORE_ROWBLOCK(kds, 0) -
			  (uintptr_t)(kds));
	length = BLCKSZ;
	bitem = KERN_DATA_STORE_BLKITEM(kds, 0);
	for (i=0, n=0; i < kds->nblocks; i++)
	{
		/* simple sanity check */
		Assert(bitem[i].buffer <= NBuffers);

		/*
		 * NOTE: A micro optimization; if next page is located
		 * on the continuous region, the upcoming DMA request
		 * can be merged to reduce DMA management cost
		 */
		if (i+1 < kds->nblocks &&
			(uintptr_t)bitem[i].page + BLCKSZ == (uintptr_t)bitem[i+1].page)
		{
			n++;
			continue;
		}
		rc = clEnqueueWriteBuffer(kcmdq,
								  kds_buffer,
								  CL_FALSE,
								  offset,
								  BLCKSZ * (n+1),
								  bitem[i-n].page,
								  num_blockers,
								  blockers,
								  events + (*ev_index));
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			return rc;
		}
		(*ev_index)++;
		pfm->bytes_dma_send += BLCKSZ * (n+1);
		pfm->num_dma_send++;
		offset += BLCKSZ * (n+1);
		n = 0;
	}
	return CL_SUCCESS;
}

/*
 * pgstrom_dump_data_store
 *
 * A utility routine that dumps properties of data store
 */
void
pgstrom_dump_data_store(pgstrom_data_store *pds)
{
	kern_data_store	   *kds = pds->kds;
	StringInfoData		buf;
	int					i, j, k;

#define PDS_DUMP(...)							\
	do {										\
		if (pgstrom_i_am_clserv)				\
			clserv_log(__VA_ARGS__);			\
		else									\
			elog(INFO, __VA_ARGS__);			\
	} while (0)

	initStringInfo(&buf);
	PDS_DUMP("pds {refcnt=%d kds=%p ktoast=%p}",
			 pds->refcnt, pds->kds, pds->ktoast);
	PDS_DUMP("kds (%s) {length=%u ncols=%u nitems=%u nrooms=%u "
			 "nblocks=%u maxblocks=%u}",
			 kds->format == KDS_FORMAT_ROW ? "row-store" :
			 kds->format == KDS_FORMAT_ROW_FLAT ? "row-flat" :
			 kds->format == KDS_FORMAT_TUPSLOT ? "tuple-slot" : "unknown",
			 kds->length, kds->ncols, kds->nitems, kds->nrooms,
			 kds->nblocks, kds->maxblocks);
	for (i=0; i < kds->ncols; i++)
	{
		PDS_DUMP("attr[%d] {attbyval=%d attalign=%d attlen=%d "
				 "attnum=%d attcacheoff=%d}",
				 i,
				 kds->colmeta[i].attbyval,
				 kds->colmeta[i].attalign,
				 kds->colmeta[i].attlen,
				 kds->colmeta[i].attnum,
				 kds->colmeta[i].attcacheoff);
	}

	if (kds->format == KDS_FORMAT_ROW_FLAT)
	{
		for (i=0; i < kds->nitems; i++)
		{
			kern_rowitem *ritem = KERN_DATA_STORE_ROWITEM(kds, i);
			HeapTupleHeaderData *htup =
				(HeapTupleHeaderData *)((char *)kds + ritem->htup_offset);
			cl_int		natts = (htup->t_infomask2 & HEAP_NATTS_MASK);
			cl_int		curr = htup->t_hoff;
			cl_int		vl_len;
			char	   *datum;

			resetStringInfo(&buf);
			appendStringInfo(&buf, "htup[%d] @%u natts=%u {",
							 i, ritem->htup_offset, natts);
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

					if (kds->colmeta[j].attlen == sizeof(cl_char))
						appendStringInfo(&buf, "%02x", *((cl_char *)datum));
					else if (kds->colmeta[j].attlen == sizeof(cl_short))
						appendStringInfo(&buf, "%04x", *((cl_short *)datum));
					else if (kds->colmeta[j].attlen == sizeof(cl_int))
						appendStringInfo(&buf, "%08x", *((cl_int *)datum));
					else if (kds->colmeta[j].attlen == sizeof(cl_long))
						appendStringInfo(&buf, "%016lx", *((cl_long *)datum));
					else
					{
						for (k=0; k < kds->colmeta[j].attlen; k++)
						{
							if (k > 0)
								appendStringInfoChar(&buf, ' ');
							appendStringInfo(&buf, "%02x", datum[k]);
						}
					}
					curr += kds->colmeta[j].attlen;
				}
				else
				{
					if (!VARATT_NOT_PAD_BYTE((char *)htup + curr))
						curr = TYPEALIGN(kds->colmeta[j].attalign, curr);
					datum = (char *)htup + curr;
					vl_len = VARSIZE_ANY_EXHDR(datum);
					appendBinaryStringInfo(&buf, VARDATA_ANY(datum), vl_len);
					curr += vl_len;
				}
			}
			appendStringInfo(&buf, "}");
			PDS_DUMP("%s", buf.data);
		}
		pfree(buf.data);
	}

	if (false)
	{
		kern_blkitem	   *bitem = KERN_DATA_STORE_BLKITEM(kds, 0);

		for (i=0; i < kds->nblocks; i++)
		{
			PDS_DUMP("block[%d] {buffer=%u page=%p}",
					 i, bitem[i].buffer, bitem[i].page);
		}
	}
#undef PDS_DUMP
}

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
							NULL, NULL, NULL);
	DefineCustomStringVariable("pg_strom.temp_tablespace",
							   "tablespace of file mapped data store",
							   NULL,
							   &pgstrom_temp_tablespace,
							   NULL,
							   PGC_USERSET,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
}
