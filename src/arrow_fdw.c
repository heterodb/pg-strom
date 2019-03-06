/*
 * arrow_fdw.c
 *
 * Routines to map Apache Arrow files as PG's Foreign-Table.
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#include "pg_strom.h"
#include "arrow_defs.h"
#include "cuda_numeric.h"

/*
 * RecordBatchState
 */
typedef struct
{
	ArrowType	arrow_type;
	ArrowTypeOptions attopts;
	int64		nitems;				/* usually, same with rb_nitems */
	int64		null_count;
	off_t		nullmap_offset;
	size_t		nullmap_length;
	off_t		values_offset;
	size_t		values_length;
	off_t		extra_offset;
	size_t		extra_length;
} RecordBatchFieldState;

typedef struct
{
	const char *fname;
	File		fdesc;
	struct stat	stat_buf;
	int			rb_index;	/* index number in a file */
	off_t		rb_offset;	/* offset from the head */
	size_t		rb_length;	/* length of the entire RecordBatch */
	int64		rb_nitems;	/* number of items */
	/* per column information */
	int			ncols;
	RecordBatchFieldState columns[FLEXIBLE_ARRAY_MEMBER];
} RecordBatchState;

/*
 * ArrowFdwState
 */
typedef struct ArrowFdwState
{
	List	   *filesList;
	List	   *fdescList;
	Bitmapset  *referenced;
	pg_atomic_uint32   *rbatch_index;
	pg_atomic_uint32	__rbatch_index;	/* in case of single process */

	pgstrom_data_store *curr_pds;	/* current focused buffer */
	cl_ulong	curr_index;			/* current index to row on KDS */
	/* state of RecordBatches */
	uint32		num_rbatches;
	RecordBatchState *rbatches[FLEXIBLE_ARRAY_MEMBER];
} ArrowFdwState;

/*
 * metadata cache (on shared memory)
 */
typedef struct
{
	dlist_node	chain;
	dlist_node	lru_chain;
	dlist_head	siblings;	/* if two or more record batches per file */
	/* key of RecordBatch metadata cache */
	struct stat	stat_buf;
	uint32		hash;
	/* fields from RecordBatchState */
	int			rb_index;	/* index of the RecordBatch */
	off_t		rb_offset;	/* offset from the head */
    size_t		rb_length;	/* length of the entire RecordBatch */
    int64		rb_nitems;	/* number of items */
	int			ncols;
	RecordBatchFieldState fstate[FLEXIBLE_ARRAY_MEMBER];
} arrowMetadataCache;

typedef struct
{
	LWLock		lock;
	cl_uint		nitems;
	cl_uint		nslots;
	dlist_head	lru_list;
	dlist_head	free_list;
	dlist_head	hash_slots[FLEXIBLE_ARRAY_MEMBER];
} arrowMetadataState;

/* ---------- static variables ---------- */
static FdwRoutine		pgstrom_arrow_fdw_routine;
static shmem_startup_hook_type shmem_startup_next = NULL;
static arrowMetadataState *arrow_metadata_state = NULL;
static int				arrow_metadata_cache_size_kb;	/* GUC */
static int				arrow_metadata_entry_size;		/* GUC */
#define arrowMetadataCacheSize								\
	MAXALIGN(offsetof(arrowMetadataCache,					\
					  fstate[arrow_metadata_entry_size]))
/* ---------- static functions ---------- */
static Oid		arrowTypeToPgSQLTypeOid(ArrowType *ftype);
static ssize_t	arrowTypeValuesLength(ArrowType *type, int64 nitems);
static bool		arrowSchemaCompatibilityCheck(TupleDesc tupdesc,
											  RecordBatchState *rb_state);
static List			   *arrowFdwExtractFilesList(List *options_list);
static RecordBatchState *makeRecordBatchState(ArrowSchema *schema,
											  ArrowBlock *block,
											  ArrowRecordBatch *rbatch);
static List	   *arrowLookupMetadataCache(const char *fname, File fdesc);
static void		arrowUpdateMetadataCache(List *rbstateList);

Datum	pgstrom_arrow_fdw_handler(PG_FUNCTION_ARGS);
Datum	pgstrom_arrow_fdw_validator(PG_FUNCTION_ARGS);

/*
 * ArrowGetForeignRelSize
 */
static void
ArrowGetForeignRelSize(PlannerInfo *root,
					   RelOptInfo *baserel,
					   Oid foreigntableid)
{
	ForeignTable   *ft = GetForeignTable(foreigntableid);
	List		   *filesList = arrowFdwExtractFilesList(ft->options);
	BlockNumber		npages = 0;
	double			ntuples = 0.0;
	ListCell	   *lc;

	foreach (lc, filesList)
	{
		const char *fname = strVal(lfirst(lc));
		struct stat	stat_buf;

		if (stat(fname, &stat_buf) != 0)
			elog(ERROR,
				 "failed on stat('%s') on behalf of foreign table %s: %m",
				 fname, get_rel_name(foreigntableid));
		npages += BLCKALIGN(stat_buf.st_size);
	}
	baserel->fdw_private = NULL;
	baserel->pages = npages;
	baserel->tuples = ntuples;
	baserel->rows = ntuples *
		clauselist_selectivity(root,
							   baserel->baserestrictinfo,
							   0,
							   JOIN_INNER,
							   NULL);
}

static void
cost_arrow_fdw_seqscan(Path *path,
					   PlannerInfo *root,
					   RelOptInfo *baserel,
					   ParamPathInfo *param_info,
					   int num_workers)
{
	Cost		startup_cost = 0.0;
	Cost		disk_run_cost = 0.0;
	Cost		cpu_run_cost = 0.0;
	QualCost	qcost;
	double		nrows;
	double		spc_seq_page_cost;

	if (param_info)
		nrows = param_info->ppi_rows;
	else
		nrows = baserel->rows;

	/*
	 * Storage costs
	 *
	 * XXX - smaller number of columns to read shall have less disk cost
	 * because of columnar format. Right now, we don't discount cost for
	 * the pages not to be read.
	 */
	get_tablespace_page_costs(baserel->reltablespace,
							  NULL,
							  &spc_seq_page_cost);
	disk_run_cost = spc_seq_page_cost * baserel->pages;

	/* CPU costs */
	if (param_info)
	{
		cost_qual_eval(&qcost, param_info->ppi_clauses, root);
		qcost.startup += baserel->baserestrictcost.startup;
        qcost.per_tuple += baserel->baserestrictcost.per_tuple;
	}
	else
		qcost = baserel->baserestrictcost;
	startup_cost += qcost.startup;
	cpu_run_cost = (cpu_tuple_cost + qcost.per_tuple) * baserel->tuples;

	/* tlist evaluation costs */
	startup_cost += path->pathtarget->cost.startup;
	cpu_run_cost += path->pathtarget->cost.per_tuple * path->rows;

	/* adjust cost for CPU parallelism */
	if (num_workers > 0)
	{
		double		leader_contribution;
		double		parallel_divisor = (double) num_workers;

		/* see get_parallel_divisor() */
		leader_contribution = 1.0 - (0.3 * (double)num_workers);
		parallel_divisor += Max(leader_contribution, 0.0);

		/* The CPU cost is divided among all the workers. */
		cpu_run_cost /= parallel_divisor;

		/* Estimated row count per background worker process */
		nrows = clamp_row_est(nrows / parallel_divisor);
	}
	path->rows = nrows;
	path->startup_cost = startup_cost;
	path->total_cost = startup_cost + cpu_run_cost + disk_run_cost;
	//XXXX
	if (num_workers > 0)
		path->total_cost = startup_cost;
	path->parallel_workers = num_workers;
}

/*
 * ArrowGetForeignPaths
 */
static void
ArrowGetForeignPaths(PlannerInfo *root,
					 RelOptInfo *baserel,
					 Oid foreigntableid)
{
	ForeignPath	   *fpath;
	ParamPathInfo  *param_info;
	Relids			required_outer = baserel->lateral_relids;

	param_info = get_baserel_parampathinfo(root, baserel, required_outer);

	fpath = create_foreignscan_path(root, baserel,
									NULL,	/* default pathtarget */
									-1,		/* dummy */
									-1.0,	/* dummy */
									-1.0,	/* dummy */
									NIL,	/* no pathkeys */
									required_outer,
									NULL,	/* no extra plan */
									NIL);	/* no particular private */
	cost_arrow_fdw_seqscan(&fpath->path, root, baserel, param_info, 0);
	add_path(baserel, (Path *)fpath);

	if (baserel->consider_parallel)
	{
		int		num_workers =
			compute_parallel_worker(baserel,
									baserel->pages, -1.0,
									max_parallel_workers_per_gather);
		if (num_workers == 0)
			return;

		fpath = create_foreignscan_path(root,
										baserel,
										NULL,	/* default pathtarget */
										-1,		/* dummy */
										-1.0,	/* dummy */
										-1.0,	/* dummy */
										NIL,	/* no pathkeys */
										required_outer,
										NULL,	/* no extra plan */
										NIL);	/* no particular private */
		fpath->path.parallel_aware = true;

		cost_arrow_fdw_seqscan(&fpath->path, root, baserel, param_info,
							   num_workers);
		add_partial_path(baserel, (Path *)fpath);
	}
}

/*
 * ArrowGetForeignPlan
 */
static ForeignScan *
ArrowGetForeignPlan(PlannerInfo *root,
					RelOptInfo *baserel,
					Oid foreigntableid,
					ForeignPath *best_path,
					List *tlist,
					List *scan_clauses,
					Plan *outer_plan)
{
	List   *referenced = NIL;
	int		i, j;

	Assert(IS_SIMPLE_REL(baserel));
	for (i=baserel->min_attr, j=0; i <= baserel->max_attr; i++, j++)
	{
		if (i < 0)
			continue;	/* system columns */
		else if (baserel->attr_needed[j])
		{
			if (i == 0)
			{
				/* all the columns must be fetched */
				referenced = NIL;
				break;
			}
			else
			{
				referenced = lappend_int(referenced, i);
			}
		}
	}

	return make_foreignscan(tlist,
							extract_actual_clauses(scan_clauses, false),
							baserel->relid,
							NIL,	/* no expressions to evaluate */
							referenced, /* list of referenced attnum */
							NIL,	/* no custom tlist */
							NIL,	/* no remote quals */
							outer_plan);
}

static RecordBatchState *
makeRecordBatchState(ArrowSchema *schema,
					 ArrowBlock *block,
					 ArrowRecordBatch *rbatch)
{
	RecordBatchState *result;
	ArrowBuffer	   *buffer_curr = rbatch->buffers;
	ArrowBuffer	   *buffer_tail = rbatch->buffers + rbatch->_num_buffers;
	int				j, ncols = schema->_num_fields;

	if (rbatch->_num_nodes != ncols)
		elog(ERROR, "arrow_fdw: RecordBatch may have corruption.");

	result = palloc0(offsetof(RecordBatchState, columns[ncols]));
	result->ncols = ncols;
	result->rb_offset = block->offset + block->metaDataLength;
	result->rb_length = block->bodyLength;
	result->rb_nitems = rbatch->length;

	for (j=0; j < ncols; j++)
	{
		ArrowField	   *field = &schema->fields[j];
		ArrowFieldNode *fnode = &rbatch->nodes[j];
		RecordBatchFieldState *c = &result->columns[j];

		c->arrow_type = field->type;
		c->nitems     = fnode->length;
		c->null_count = fnode->null_count;

		switch (field->type.node.tag)
		{
			case ArrowNodeTag__Int:
			case ArrowNodeTag__FloatingPoint:
			case ArrowNodeTag__Bool:
			case ArrowNodeTag__Decimal:
			case ArrowNodeTag__Date:
			case ArrowNodeTag__Time:
			case ArrowNodeTag__Timestamp:
			case ArrowNodeTag__Interval:
				/* fixed-length values */
				if (buffer_curr + 1 >= buffer_tail)
					elog(ERROR, "RecordBatch has less buffers than expected");
				if (c->null_count > 0)
				{
					c->nullmap_offset = buffer_curr->offset;
					c->nullmap_length = buffer_curr->length;
					if (c->nullmap_length < BITMAPLEN(c->null_count))
						elog(ERROR, "nullmap length is smaller than expected");
					if ((c->nullmap_offset & (MAXIMUM_ALIGNOF - 1)) != 0 ||
						(c->nullmap_length & (MAXIMUM_ALIGNOF - 1)) != 0)
						elog(ERROR, "nullmap is not aligned well");
				}
				buffer_curr++;
				c->values_offset = buffer_curr->offset;
				c->values_length = buffer_curr->length;
				if (c->values_length < arrowTypeValuesLength(&field->type,
															 c->nitems))
					elog(ERROR, "values array is smaller than expected");
				if ((c->values_offset & (MAXIMUM_ALIGNOF - 1)) != 0 ||
					(c->values_length & (MAXIMUM_ALIGNOF - 1)) != 0)
					elog(ERROR, "values array is not aligned well");
				buffer_curr++;
				break;

			case ArrowNodeTag__Utf8:
			case ArrowNodeTag__Binary:
				/* variable length values */
				if (buffer_curr + 2 >= buffer_tail)
					elog(ERROR, "RecordBatch has less buffers than expected");
				if (c->null_count > 0)
				{
					c->nullmap_offset = buffer_curr->offset;
					c->nullmap_length = buffer_curr->length;
					if (c->nullmap_length < BITMAPLEN(c->null_count))
						elog(ERROR, "nullmap length is smaller than expected");
					if ((c->nullmap_offset & (MAXIMUM_ALIGNOF - 1)) != 0 ||
						(c->nullmap_length & (MAXIMUM_ALIGNOF - 1)) != 0)
						elog(ERROR, "nullmap is not aligned well");
				}
				buffer_curr++;

				c->values_offset = buffer_curr->offset;
				c->values_length = buffer_curr->length;
				if (c->values_length < arrowTypeValuesLength(&field->type,
															 c->nitems))
					elog(ERROR, "offset array is smaller than expected");
				if ((c->values_offset & (MAXIMUM_ALIGNOF - 1)) != 0 ||
					(c->values_length & (MAXIMUM_ALIGNOF - 1)) != 0)
					elog(ERROR, "offset array is not aligned well");
				buffer_curr++;

				c->extra_offset = buffer_curr->offset;
				c->extra_length = buffer_curr->length;
				if ((c->extra_offset & (MAXIMUM_ALIGNOF - 1)) != 0 ||
					(c->extra_length & (MAXIMUM_ALIGNOF - 1)) != 0)
					elog(ERROR, "extra buffer is not aligned well");
				buffer_curr++;
				break;

			case ArrowNodeTag__List:
				//TODO: Add support of fixed-length array.
			default:
				elog(ERROR, "Bug? ArrowSchema contains unsupported types");
		}

		/* Extra attributes (precision, unitsz, ...) */
		switch (field->type.node.tag)
        {
            case ArrowNodeTag__Decimal:
				if (field->type.Decimal.precision < SHRT_MIN ||
					field->type.Decimal.precision > SHRT_MAX)
					elog(ERROR, "Decimal precision is out of range");
				if (field->type.Decimal.scale < SHRT_MIN ||
					field->type.Decimal.scale > SHRT_MAX)
					elog(ERROR, "Decimal scale is out of range");
				c->attopts.decimal.precision = field->type.Decimal.precision;
				c->attopts.decimal.scale     = field->type.Decimal.scale;
				break;
            case ArrowNodeTag__Date:
				switch (field->type.Date.unit)
				{
					case ArrowDateUnit__Day:
					case ArrowDateUnit__MilliSecond:
						break;
					default:
						elog(ERROR, "unknown unit of Date");
				}
				c->attopts.date.unit = field->type.Date.unit;
				break;
            case ArrowNodeTag__Time:
				switch (field->type.Time.unit)
				{
					case ArrowTimeUnit__Second:
					case ArrowTimeUnit__MilliSecond:
					case ArrowTimeUnit__MicroSecond:
					case ArrowTimeUnit__NanoSecond:
						break;
					default:
						elog(ERROR, "unknown unit of Time");
				}
				c->attopts.time.unit = field->type.Time.unit;
				break;
            case ArrowNodeTag__Timestamp:
				switch (field->type.Timestamp.unit)
				{
					case ArrowTimeUnit__Second:
					case ArrowTimeUnit__MilliSecond:
					case ArrowTimeUnit__MicroSecond:
					case ArrowTimeUnit__NanoSecond:
						break;
					default:
						elog(ERROR, "unknown unit of Time");
				}
				c->attopts.time.unit = field->type.Timestamp.unit;
				break;
            case ArrowNodeTag__Interval:
				switch (field->type.Interval.unit)
				{
					case ArrowIntervalUnit__Year_Month:
					case ArrowIntervalUnit__Day_Time:
						break;
					default:
						elog(ERROR, "unknown unit of Interval");
				}
				c->attopts.interval.unit = field->type.Interval.unit;
				break;
			default:
				/* no extra attributes */
				break;
		}
	}
	return result;
}

/*
 * ArrowBeginForeignScan
 */
static void
ArrowBeginForeignScan(ForeignScanState *node, int eflags)
{
	Relation		relation = node->ss.ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(relation);
	ForeignScan	   *fscan = (ForeignScan *) node->ss.ps.plan;
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(relation));
	List		   *filesList = arrowFdwExtractFilesList(ft->options);
	List		   *fdescList = NIL;
	List		   *rb_state_list = NIL;
	ListCell	   *lc;
	Bitmapset	   *referenced = NULL;
	ArrowFdwState  *af_state;
	int				i, num_rbatches;

	foreach (lc, fscan->fdw_private)
	{
		AttrNumber	anum = lfirst_int(lc);

		Assert(anum > 0 && anum <= RelationGetNumberOfAttributes(relation));
		referenced = bms_add_member(referenced, anum -
									FirstLowInvalidHeapAttributeNumber);
	}

	foreach (lc, filesList)
	{
		char	   *fname = strVal(lfirst(lc));
		File		fdesc;
		List	   *rb_cached = NIL;
		ListCell   *cell;

		fdesc = PathNameOpenFile(fname, O_RDONLY | PG_BINARY);
		if (fdesc < 0)
			elog(ERROR, "failed to open '%s'", fname);
		fdescList = lappend_int(fdescList, fdesc);

		rb_cached = arrowLookupMetadataCache(fname, fdesc);
		if (rb_cached == NIL)
		{
			ArrowFileInfo af_info;
			int			index = 0;

			readArrowFileDesc(fdesc, &af_info);
			if (af_info.dictionaries != NIL)
				elog(ERROR, "DictionaryBatch is not supported");
			Assert(af_info.footer._num_dictionaries == 0);

			foreach (cell, af_info.recordBatches)
			{
				ArrowBlock		 *block = &af_info.footer.recordBatches[index];
				ArrowRecordBatch *rbatch = lfirst(cell);
				RecordBatchState *rb_state;

				rb_state = makeRecordBatchState(&af_info.footer.schema,
												block, rbatch);
				rb_state->fname = fname;
				rb_state->fdesc = fdesc;
				if (fstat(FileGetRawDesc(fdesc), &rb_state->stat_buf) != 0)
					elog(ERROR, "failed on stat('%s'): %m", fname);
				rb_state->rb_index = index;
				rb_cached = lappend(rb_cached, rb_state);
				index++;
			}
			if (rb_cached == NIL)
				elog(ERROR, "arrow file '%s' contains no RecordBatch", fname);
			arrowUpdateMetadataCache(rb_cached);
		}
		/* check schema compatibility */
		foreach (cell, rb_cached)
		{
			RecordBatchState   *rb_state = lfirst(cell);

			if (!arrowSchemaCompatibilityCheck(tupdesc, rb_state))
				elog(ERROR, "arrow file '%s' on behalf of foreign table '%s' has incompatible schema definition",
					 fname, RelationGetRelationName(relation));
		}
		rb_state_list = list_concat(rb_state_list, rb_cached);
	}
	num_rbatches = list_length(rb_state_list);
	af_state = palloc0(offsetof(ArrowFdwState, rbatches[num_rbatches]));
	af_state->filesList = filesList;
	af_state->fdescList = fdescList;
	af_state->referenced = referenced;
	af_state->rbatch_index = &af_state->__rbatch_index;
	i = 0;
	foreach (lc, rb_state_list)
		af_state->rbatches[i++] = (RecordBatchState *)lfirst(lc);
	af_state->num_rbatches = num_rbatches;

	node->fdw_state = af_state;
}


typedef struct
{
	cl_uint			block_sz;
	cl_uint			nchunks;
	struct {
		cl_ulong	m_offset;	/* copy destination (aligned to block_sz) */
		cl_ulong	f_pos;		/* copy source (aligned to block_sz) */
		cl_int		nblocks;	/* number of blocks to copy */
		cl_int		status;		/* status of P2P DMA in this chunk */
	} io[FLEXIBLE_ARRAY_MEMBER];
} strom_io_vector;

/*
 * arrowFdwSetupIOvector
 */
static strom_io_vector *
arrowFdwSetupIOvector(kern_data_store *kds,
					  RecordBatchState *rb_state,
					  Bitmapset *referenced)
{
	int			page_sz = sysconf(_SC_PAGESIZE);
	off_t		rb_offset = rb_state->rb_offset;
	off_t		f_offset = ~0UL;
	cl_ulong	m_offset;
	int			io_index = -1;
	int			j, ncols = kds->ncols;
	strom_io_vector *iovec;

	iovec = palloc0(offsetof(strom_io_vector, io[3 * ncols]));
	iovec->block_sz = page_sz;
	Assert((page_sz & (page_sz - 1)) == 0);

	m_offset = TYPEALIGN(page_sz, KERN_DATA_STORE_HEAD_LENGTH(kds));
	for (j=0; j < ncols; j++)
	{
		RecordBatchFieldState *fstate = &rb_state->columns[j];
		kern_colmeta   *cmeta = &kds->colmeta[j];
		off_t			f_base, f_pos;
		int				attidx = j + 1 - FirstLowInvalidHeapAttributeNumber;

		if (referenced && !bms_is_member(attidx, referenced))
			continue;

		if (fstate->nullmap_length > 0)
		{
			Assert(fstate->null_count > 0);
			f_pos = rb_offset + fstate->nullmap_offset;
			f_base = TYPEALIGN_DOWN(page_sz, f_pos);
			if (f_pos == f_offset)
			{
				/* good, buffer is continuous */
				cmeta->nullmap_offset = __kds_packed(m_offset);
				cmeta->nullmap_length = __kds_packed(fstate->nullmap_length);

				m_offset += fstate->nullmap_length;
				f_offset += fstate->nullmap_length;
			}
			else
			{
				off_t	shift = f_pos - f_base;

				if (io_index < 0)
					io_index = 0;	/* here is no previous chunk */
				else
				{
					size_t	len = (TYPEALIGN(page_sz, f_offset) -
								   iovec->io[io_index].f_pos);
					Assert((len & (page_sz-1)) == 0);
					iovec->io[io_index++].nblocks = len / page_sz;

					m_offset = TYPEALIGN(page_sz, m_offset);
				}
				iovec->io[io_index].m_offset = m_offset;
				iovec->io[io_index].f_pos    = f_base;

				cmeta->nullmap_offset = __kds_packed(m_offset + shift);
				cmeta->nullmap_length = __kds_packed(fstate->nullmap_length);

				m_offset += shift + fstate->nullmap_length;
				f_offset =  f_pos + fstate->nullmap_length;
			}
		}

		if (fstate->values_length > 0)
		{
			f_pos = rb_offset + fstate->values_offset;
			f_base = TYPEALIGN_DOWN(page_sz, f_pos);
			if (f_pos == f_offset)
			{
				/* good, buffer is continuous */
				cmeta->values_offset = __kds_packed(m_offset);
				cmeta->values_length = __kds_packed(fstate->values_length);

				m_offset += fstate->values_length;
				f_offset += fstate->values_length;
			}
			else
			{
				off_t	shift = f_pos - f_base;

				if (io_index < 0)
					io_index = 0;	/* here is no previous chunk */
				else
				{
					size_t	len = (TYPEALIGN(page_sz, f_offset) -
								   iovec->io[io_index].f_pos);
					Assert((len & (page_sz-1)) == 0);
					iovec->io[io_index++].nblocks = len / page_sz;

					m_offset = TYPEALIGN(page_sz, m_offset);
				}
				iovec->io[io_index].m_offset = m_offset;
				iovec->io[io_index].f_pos    = f_base;

				cmeta->values_offset = __kds_packed(m_offset + shift);
				cmeta->values_length = __kds_packed(fstate->values_length);

				m_offset += shift + fstate->values_length;
				f_offset =  f_pos + fstate->values_length;
			}
		}

		if (fstate->extra_length > 0)
		{
			f_pos = rb_offset + fstate->extra_offset;
			f_base = TYPEALIGN_DOWN(page_sz, f_pos);
			if (f_pos == f_offset)
			{
				/* good, buffer is continuous */
				cmeta->extra_offset = __kds_packed(m_offset);
				cmeta->extra_length = __kds_packed(fstate->extra_length);

				m_offset += fstate->extra_length;
				f_offset += fstate->extra_length;
			}
			else
			{
				off_t	shift = f_pos - f_base;

				if (io_index < 0)
					io_index = 0;	/* here is no previous chunk */
				else
				{
					off_t	len = (TYPEALIGN(page_sz, f_offset) -
								   iovec->io[io_index].f_pos);
					Assert((len & (page_sz-1)) == 0);
					iovec->io[io_index++].nblocks = len / page_sz;

					m_offset = TYPEALIGN(page_sz, m_offset);
				}
				iovec->io[io_index].m_offset = m_offset;
				iovec->io[io_index].f_pos    = f_base;

				cmeta->extra_offset = __kds_packed(m_offset + shift);
				cmeta->extra_length = __kds_packed(fstate->values_length);

				m_offset += shift + fstate->extra_length;
				f_offset =  f_pos + fstate->extra_length;
			}
		}
	}

	/* close the last i/o chunk */
	if (io_index < 0)
		iovec->nchunks = 0;
	else
	{
		size_t	len = (TYPEALIGN(page_sz, f_offset) -
					   iovec->io[io_index].f_pos);
		Assert((len & (page_sz-1)) == 0);
		iovec->io[io_index++].nblocks = len / page_sz;
		iovec->nchunks = io_index;
	}
	kds->length = TYPEALIGN(page_sz, m_offset);

	return iovec;
}

/*
 * arrowFdwLoadRecordBatch - Loads the required RecordBatch to PDS
 */
static pgstrom_data_store *
arrowFdwLoadRecordBatch(Relation relation,
						MemoryContext memcxt,
						RecordBatchState *rb_state,
						Bitmapset *referenced)
{
	TupleDesc			tupdesc = RelationGetDescr(relation);
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	strom_io_vector	   *iovec;
	int					j, ncols = rb_state->ncols;
	int					fdesc;

	kds = alloca(offsetof(kern_data_store, colmeta[ncols]));
	init_kernel_data_store(kds, tupdesc, 0, KDS_FORMAT_ARROW, 0, false);
	kds->nitems = rb_state->rb_nitems;
	kds->nrooms = rb_state->rb_nitems;
	kds->table_oid = RelationGetRelid(relation);
	for (j=0; j < ncols; j++)
		kds->colmeta[j].attopts = rb_state->columns[j].attopts;
	iovec = arrowFdwSetupIOvector(kds, rb_state, referenced);
#if 0
	elog(INFO, "nchunks = %d", iovec->nchunks);
	for (j=0; j < iovec->nchunks; j++)
	{
		elog(INFO, "io[%d] [ m_offset=%lu, f_pos=%lu, nblocks=%u, status=%d}",
			 j,
			 iovec->io[j].m_offset,
			 iovec->io[j].f_pos,
			 iovec->io[j].nblocks,
			 iovec->io[j].status);
	}

	elog(INFO, "kds {length=%zu nitems=%u typeid=%u typmod=%u table_oid=%u}",
		 kds->length, kds->nitems,
		 kds->tdtypeid, kds->tdtypmod, kds->table_oid);
	for (j=0; j < kds->ncols; j++)
	{
		kern_colmeta *cmeta = &kds->colmeta[j];

		elog(INFO, "col[%d] nullmap=%lu,%lu values=%lu,%lu extra=%lu,%lu", j,
			 __kds_unpack(cmeta->nullmap_offset),
			 __kds_unpack(cmeta->nullmap_length),
			 __kds_unpack(cmeta->values_offset),
			 __kds_unpack(cmeta->values_length),
			 __kds_unpack(cmeta->extra_offset),
			 __kds_unpack(cmeta->extra_length));

	}
#endif
	/*
	 * PDS creation and load from file
	 */
	pds = MemoryContextAllocHuge(memcxt, offsetof(pgstrom_data_store,
												  kds) + kds->length);
	memset(pds, 0, offsetof(pgstrom_data_store, kds));
	pg_atomic_init_u32(&pds->refcnt, 1);
	memcpy(&pds->kds, kds, offsetof(kern_data_store, colmeta[ncols]));

	fdesc = FileGetRawDesc(rb_state->fdesc);
	for (j=0; j < iovec->nchunks; j++)
	{
		size_t		page_sz = iovec->block_sz;
		char	   *dest = (char *)&pds->kds + iovec->io[j].m_offset;
		size_t		len = page_sz * (size_t)iovec->io[j].nblocks;
		off_t		f_pos = iovec->io[j].f_pos;
		ssize_t		sz;

		while (len > 0 && f_pos < rb_state->stat_buf.st_size)
		{
			CHECK_FOR_INTERRUPTS();

			sz = pread(fdesc, dest, len, f_pos);
			if (sz > 0)
			{
				Assert(sz <= len);
				dest += sz;
				f_pos += sz;
				len -= sz;
			}
			else if (sz == 0)
			{
				elog(ERROR, "file '%s' is shorter than schema definition",
					rb_state->fname);
			}
			else if (errno != EINTR)
			{
				elog(ERROR, "failed on pread('%s'): %m", rb_state->fname);
			}
		}
		/*
		 * NOTE: Due to the page_sz alignment, we may try to read the file
		 * over the its tail. So, above loop may terminate with non-zero
		 * remaining length.
		 */
		if (len > 0)
		{
			Assert(len < page_sz);
			memset(dest, 0, len);
		}
	}
	return pds;
}

/*
 * ArrowIterateForeignScan
 */
static TupleTableSlot *
ArrowIterateForeignScan(ForeignScanState *node)
{
	ArrowFdwState  *af_state = node->fdw_state;
	Relation		relation = node->ss.ss_currentRelation;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	pgstrom_data_store *pds;

	while ((pds = af_state->curr_pds) == NULL ||
		   af_state->curr_index >= pds->kds.nitems)
	{
		EState	   *estate = node->ss.ps.state;
		RecordBatchState *rb_state;
		uint32		rb_index;

		/* unload the previous RecordBatch, if any */
		if (pds)
			PDS_release(pds);
		af_state->curr_pds = NULL;
		af_state->curr_index = 0;

		/* load the next RecordBatch */
		rb_index = pg_atomic_fetch_add_u32(af_state->rbatch_index, 1);
		if (rb_index >= af_state->num_rbatches)
			return NULL;	/* no more items to read */
		rb_state = af_state->rbatches[rb_index];
		af_state->curr_pds = arrowFdwLoadRecordBatch(relation,
													 estate->es_query_cxt,
													 rb_state,
													 af_state->referenced);
	}
	Assert(pds && af_state->curr_index < pds->kds.nitems);
	if (KDS_fetch_tuple_arrow(slot, &pds->kds, af_state->curr_index++))
		return slot;
	return NULL;
}

/*
 * ArrowReScanForeignScan
 */
static void
ArrowReScanForeignScan(ForeignScanState *node)
{
	ArrowFdwState  *af_state = node->fdw_state;

	/* rewind the current scan state */
	pg_atomic_write_u32(af_state->rbatch_index, 0);
	if (af_state->curr_pds)
		PDS_release(af_state->curr_pds);
	af_state->curr_pds = NULL;
	af_state->curr_index = 0;
}

/*
 * ArrowEndForeignScan
 */
static void
ArrowEndForeignScan(ForeignScanState *node)
{
	ArrowFdwState  *af_state = node->fdw_state;
	ListCell	   *lc;

	foreach (lc, af_state->fdescList)
		FileClose((File)lfirst_int(lc));
}

/*
 * ArrowExplainForeignScan 
 */
static void
ArrowExplainForeignScan(ForeignScanState *node, ExplainState *es)
{
	ArrowFdwState  *af_state = node->fdw_state;
	ListCell	   *lc1, *lc2;
	int				fcount = 0;
	int				rbcount = 0;
	char			label[64];
	char			temp[1024];

	forboth (lc1, af_state->filesList,
			 lc2, af_state->fdescList)
	{
		const char *fname = strVal(lfirst(lc1));
		File		fdesc = (File)lfirst_int(lc2);

		snprintf(label, sizeof(label), "files%d", fcount++);
		ExplainPropertyText(label, fname, es);
		if (!es->verbose)
			continue;

		/* below only verbose mode */
		while (rbcount < af_state->num_rbatches)
		{
			RecordBatchState *rb_state = af_state->rbatches[rbcount];

			if (rb_state->fdesc != fdesc)
				break;
			if (es->format == EXPLAIN_FORMAT_TEXT)
			{
				snprintf(label, sizeof(label),
						 "  record-batch%d", rbcount);
				snprintf(temp, sizeof(temp),
						 "offset=%lu, length=%zu, nitems=%ld",
						 rb_state->rb_offset,
						 rb_state->rb_length,
						 rb_state->rb_nitems);
				ExplainPropertyText(label, temp, es);
			}
			else
			{
				snprintf(label, sizeof(label),
						 "file%d-ecord-batch%d-offset", fcount, rbcount);
				ExplainPropertyInteger(label, NULL, rb_state->rb_offset, es);
				snprintf(label, sizeof(label),
						 "file%d-ecord-batch%d-length", fcount, rbcount);
				ExplainPropertyInteger(label, NULL, rb_state->rb_length, es);
				snprintf(label, sizeof(label),
						 "file%d-ecord-batch%d-nitems", fcount, rbcount);
				ExplainPropertyInteger(label, NULL, rb_state->rb_nitems, es);
			}
			rbcount++;
		}
	}
}

/*
 * ArrowImportForeignSchema
 */
static List *
ArrowImportForeignSchema(ImportForeignSchemaStmt *stmt, Oid serverOid)
{
	ArrowSchema		schema;
	List		   *filesList;
	ListCell	   *lc;
	int				j;
	StringInfoData	cmd;

	/* sanity checks */
	switch (stmt->list_type)
	{
		case FDW_IMPORT_SCHEMA_ALL:
			break;
		case FDW_IMPORT_SCHEMA_LIMIT_TO:
			elog(ERROR, "arrow_fdw does not support LIMIT TO clause");
		case FDW_IMPORT_SCHEMA_EXCEPT:
			elog(ERROR, "arrow_fdw does not support EXCEPT clause");
		default:
			elog(ERROR, "arrow_fdw: Bug? unknown list-type");
	}
	filesList = arrowFdwExtractFilesList(stmt->options);
	if (filesList == NIL)
		ereport(ERROR,
				(errmsg("No valid apache arrow files are specified"),
				 errhint("Use 'file' or 'dir' option to specify apache arrow files on behalf of the foreign table")));

	/* read the schema */
	memset(&schema, 0, sizeof(ArrowSchema));
	foreach (lc, filesList)
	{
		char	   *fname = strVal(lfirst(lc));
		ArrowFileInfo af_info;

		readArrowFile(fname, &af_info);
		if (lc == linitial(filesList))
		{
			memcpy(&schema, &af_info.footer.schema, sizeof(ArrowSchema));
		}
		else
		{
			/* compatibility checks */
			ArrowSchema	   *stemp = &af_info.footer.schema;

			if (schema.endianness != stemp->endianness ||
				schema._num_fields != stemp->_num_fields)
				elog(ERROR, "file '%s' has incompatible schema definition",
					 fname);
			for (j=0; j < schema._num_fields; j++)
			{
				ArrowField *field = &schema.fields[j];
				ArrowField *ftemp = &stemp->fields[j];

				if (arrowTypeToPgSQLTypeOid(&field->type) !=
					arrowTypeToPgSQLTypeOid(&ftemp->type))
					elog(ERROR, "file '%s' has incompatible schema definition",
						 fname);
			}
		}
	}

	/* makes a command to define foreign table */
	initStringInfo(&cmd);
	appendStringInfo(&cmd, "CREATE FOREIGN TABLE %s (\n",
					 quote_identifier(stmt->remote_schema));
	for (j=0; j < schema._num_fields; j++)
	{
		ArrowField	   *field = &schema.fields[j];
		ArrowType	   *t = &field->type;
		const char	   *type_name;

		switch (t->node.tag)
		{
			case ArrowNodeTag__Int:
				if (!t->Int.is_signed)
					elog(NOTICE, "Uint of ArrowType is mapped to signed-integer at PostgreSQL");
				switch (t->Int.bitWidth)
				{
					case 16: type_name = "pg_catalog.int2"; break;
					case 32: type_name = "pg_catalog.int4"; break;
					case 64: type_name = "pg_catalog.int8"; break;
					default:
						elog(ERROR, "Int%d is not supported",
							 t->Int.bitWidth);
				}
				break;
			case ArrowNodeTag__FloatingPoint:
				switch (t->FloatingPoint.precision)
				{
					case ArrowPrecision__Half:
						type_name = "pg_catalog.float2";
						break;
					case ArrowPrecision__Single:
						type_name = "pg_catalog.float4";
						break;
					case ArrowPrecision__Double:
						type_name = "pg_catalog.float8";
						break;
					default:
						elog(ERROR, "Unknown precision of floating-point");
				}
				break;
			case ArrowNodeTag__Utf8:
				type_name = "pg_catalog.text";
				break;
			case ArrowNodeTag__Binary:
				type_name = "pg_catalog.bytea";
				break;
			case ArrowNodeTag__Bool:
				type_name = "pg_catalog.bool";
				break;
			case ArrowNodeTag__Decimal:
				type_name = "pg_catalog.numeric";
				break;
			case ArrowNodeTag__Date:
				type_name = "pg_catalog.date";
				break;
			case ArrowNodeTag__Time:
				type_name = "pg_catalog.time";
				break;
			case ArrowNodeTag__Timestamp:
				type_name = "pg_catalog.timestamp";
				break;
			case ArrowNodeTag__Interval:
				type_name = "pg_catalog.interval";
				break;
			case ArrowNodeTag__Null:
			case ArrowNodeTag__List:
			case ArrowNodeTag__Struct:
			case ArrowNodeTag__Union:
			case ArrowNodeTag__FixedSizeBinary:
			case ArrowNodeTag__FixedSizeList:
			case ArrowNodeTag__Map:
				elog(ERROR, "ArrowType '%s' is not supported, right now",
					 t->node.tagName);
			default:
				elog(ERROR, "Unknown ArrowType: %s", t->node.tagName);
		}
		if (j > 0)
			appendStringInfo(&cmd, ",\n");
		if (!field->name || field->_name_len == 0)
		{
			elog(NOTICE, "field %d has no name, so \"__col%02d\" is used",
				 j+1, j+1);
			appendStringInfo(&cmd, "  __col%02d  %s", j+1, type_name);
		}
		else
			appendStringInfo(&cmd, "  %s %s",
							 quote_identifier(field->name), type_name);
	}
	appendStringInfo(&cmd,
					 "\n"
					 ") SERVER %s\n"
					 "  OPTIONS (", stmt->server_name);
	foreach (lc, stmt->options)
	{
		DefElem	   *defel = lfirst(lc);

		if (lc != list_head(stmt->options))
			appendStringInfo(&cmd, ",\n           ");
		appendStringInfo(&cmd, "%s '%s'",
						 defel->defname,
						 strVal(defel->arg));
	}
	appendStringInfo(&cmd, ")");

	return list_make1(cmd.data);
}

/*
 * ArrowIsForeignScanParallelSafe
 */
static bool
ArrowIsForeignScanParallelSafe(PlannerInfo *root,
							   RelOptInfo *rel,
							   RangeTblEntry *rte)
{
	/* we have no special restrictions for parallel execution */
	return true;
}

/*
 * ArrowEstimateDSMForeignScan 
 */
static Size
ArrowEstimateDSMForeignScan(ForeignScanState *node,
							ParallelContext *pcxt)
{
	//elog(INFO, "pid=%u ArrowEstimateDSMForeignScan", getpid());
	return MAXALIGN(sizeof(pg_atomic_uint32));
}

/*
 * ArrowInitializeDSMForeignScan
 */
static void
ArrowInitializeDSMForeignScan(ForeignScanState *node,
							  ParallelContext *pcxt,
							  void *coordinate)
{
	ArrowFdwState	   *af_state = node->fdw_state;
	pg_atomic_uint32   *rbatch_index = coordinate;

	//elog(INFO, "pid=%u ArrowInitializeDSMForeignScan", getpid());
	pg_atomic_init_u32(rbatch_index, 0);
	af_state->rbatch_index = rbatch_index;
}

/*
 * ArrowReInitializeDSMForeignScan
 */
static void
ArrowReInitializeDSMForeignScan(ForeignScanState *node,
								ParallelContext *pcxt,
								void *coordinate)
{
	ArrowFdwState	   *af_state = node->fdw_state;

	pg_atomic_write_u32(af_state->rbatch_index, 0);
}

/*
 * ArrowInitializeWorkerForeignScan
 */
static void
ArrowInitializeWorkerForeignScan(ForeignScanState *node,
								 shm_toc *toc,
								 void *coordinate)
{
	ArrowFdwState	   *af_state = node->fdw_state;

	//elog(INFO, "pid=%u ArrowInitializeWorkerForeignScan", getpid());
	af_state->rbatch_index = (pg_atomic_uint32 *) coordinate;
}

/*
 * ArrowShutdownForeignScan
 */
static void
ArrowShutdownForeignScan(ForeignScanState *node)
{
	//elog(INFO, "pid=%u ArrowShutdownForeignScan", getpid());
}

/*
 * handler of Arrow_Fdw
 */
Datum
pgstrom_arrow_fdw_handler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&pgstrom_arrow_fdw_routine);
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_handler);

/*
 * arrowTypeToPgSQLTypeOid
 */
static Oid
arrowTypeToPgSQLTypeOid(ArrowType *ftype)
{
	Oid		type_oid = InvalidOid;

	switch (ftype->node.tag)
	{
		case ArrowNodeTag__Int:
			switch (ftype->Int.bitWidth)
			{
				case 16:
					type_oid = INT2OID;
					break;
				case 32:
					type_oid = INT4OID;
					break;
				case 64:
					type_oid = INT8OID;
					break;
				default:
					elog(ERROR, "%s%d of Arrow is not supported",
						 ftype->Int.is_signed ? "Int" : "Uint",
						 ftype->Int.bitWidth);
					break;
			}
			break;
		case ArrowNodeTag__FloatingPoint:
			switch (ftype->FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					type_oid = FLOAT2OID;
					break;
				case ArrowPrecision__Single:
					type_oid = FLOAT4OID;
					break;
				case ArrowPrecision__Double:
					type_oid = FLOAT8OID;
					break;
				default:
					elog(ERROR, "unknown floating-point precision");
					break;
			}
			break;
		case ArrowNodeTag__Utf8:
			type_oid = TEXTOID;
			break;
		case ArrowNodeTag__Binary:
			type_oid = BYTEAOID;
			break;
		case ArrowNodeTag__Bool:
			type_oid = BOOLOID;
			break;
		case ArrowNodeTag__Decimal:
			type_oid = NUMERICOID;
			break;
		case ArrowNodeTag__Date:
			type_oid = DATEOID;
			break;
		case ArrowNodeTag__Time:
			type_oid = TIMEOID;
			break;
		case ArrowNodeTag__Timestamp:
			if (ftype->Timestamp.timezone)
				elog(ERROR, "Timestamp with timezone of Arrow is not supported, right now");
			type_oid = TIMESTAMPOID;
			break;
		case ArrowNodeTag__Interval:
			type_oid = INTERVALOID;
			break;
		default:
			elog(ERROR, "Not a supported Apache Arrow type");
			break;
	}
	return type_oid;
}

/*
 * arrowTypeValuesLength - calculates minimum required length of the values
 */
static ssize_t
arrowTypeValuesLength(ArrowType *type, int64 nitems)
{
	ssize_t	length = -1;

	switch (type->node.tag)
	{
		case ArrowNodeTag__Int:
			switch (type->Int.bitWidth)
			{
				case sizeof(cl_short) * BITS_PER_BYTE:
					length = sizeof(cl_short) * nitems;
					break;
				case sizeof(cl_int) * BITS_PER_BYTE:
					length = sizeof(cl_int) * nitems;
					break;
				case sizeof(cl_long) * BITS_PER_BYTE:
					length = sizeof(cl_long) * nitems;
					break;
				default:
					elog(ERROR, "Not a supported Int width: %d",
						 type->Int.bitWidth);
			}
			break;
		case ArrowNodeTag__FloatingPoint:
			switch (type->FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					length = sizeof(cl_short) * nitems;
					break;
				case ArrowPrecision__Single:
					length = sizeof(cl_float) * nitems;
					break;
				case ArrowPrecision__Double:
					length = sizeof(cl_double) * nitems;
					break;
				default:
					elog(ERROR, "Not a supported FloatingPoint precision");
			}
			break;
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__Binary:
			length = sizeof(cl_uint) * (nitems + 1);
			break;
		case ArrowNodeTag__Bool:
			length = BITMAPLEN(nitems);
			break;
		case ArrowNodeTag__Decimal:
			length = sizeof(int128) * nitems;
			break;
		case ArrowNodeTag__Date:
			switch (type->Date.unit)
			{
				case ArrowDateUnit__Day:
					length = sizeof(cl_int) * nitems;
					break;
				case ArrowDateUnit__MilliSecond:
					length = sizeof(cl_long) * nitems;
					break;
				default:
					elog(ERROR, "Not a supported Date unit");
			}
			break;
		case ArrowNodeTag__Time:
			switch (type->Time.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
					length = sizeof(cl_int) * nitems;
					break;
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					length = sizeof(cl_long) * nitems;
					break;
				default:
					elog(ERROR, "Not a supported Time unit");
			}
			break;
		case ArrowNodeTag__Timestamp:
			length = sizeof(cl_long) * nitems;
			break;
		case ArrowNodeTag__Interval:
			switch (type->Interval.unit)
			{
				case ArrowIntervalUnit__Year_Month:
					length = sizeof(cl_uint) * nitems;
					break;
				case ArrowIntervalUnit__Day_Time:
					length = sizeof(cl_long) * nitems;
					break;
				default:
					elog(ERROR, "Not a supported Interval unit");
			}
			break;
		case ArrowNodeTag__List:	//to be supported later
		case ArrowNodeTag__Struct:	//to be supported later
		default:
			elog(ERROR, "Arrow Type '%s' is not supported now",
				 type->node.tagName);
			break;
	}
	return length;
}

/*
 * arrowSchemaCompatibilityCheck
 */
static bool
arrowSchemaCompatibilityCheck(TupleDesc tupdesc, RecordBatchState *rb_state)
{
	int		j;

	if (tupdesc->natts != rb_state->ncols)
		return false;
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
		RecordBatchFieldState *field = &rb_state->columns[j];

		if (attr->atttypid != arrowTypeToPgSQLTypeOid(&field->arrow_type))
			return false;
	}
	return true;
}

/*
 * pg_XXX_arrow_ref
 */
static Datum
pg_int2_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	cl_short   *values = (cl_short *)
		((char *)kds + __kds_unpack(cmeta->values_offset));
	return values[index];
}

static Datum
pg_int4_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	cl_int	   *values = (cl_int *)
		((char *)kds + __kds_unpack(cmeta->values_offset));
	return values[index];
}

static Datum
pg_int8_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	cl_long	   *values = (cl_long *)
		((char *)kds + __kds_unpack(cmeta->values_offset));
	return values[index];
}

static Datum
pg_float2_arrow_ref(kern_data_store *kds,
					kern_colmeta *cmeta, size_t index)
{
	return pg_int2_arrow_ref(kds, cmeta, index);
}

static Datum
pg_float4_arrow_ref(kern_data_store *kds,
					kern_colmeta *cmeta, size_t index)
{
	return pg_int4_arrow_ref(kds, cmeta, index);
}

static Datum
pg_float8_arrow_ref(kern_data_store *kds,
					kern_colmeta *cmeta, size_t index)
{
	return pg_int8_arrow_ref(kds, cmeta, index);
}

static Datum
pg_text_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	cl_uint	   *offset = (cl_uint *)((char *)kds +
									 __kds_unpack(cmeta->values_offset));
	char	   *extra = (char *)kds + __kds_unpack(cmeta->extra_offset);
	size_t		extra_len = __kds_unpack(cmeta->extra_offset);
	cl_uint		len = offset[index+1] - offset[index];
	text	   *res;

	if (extra_len <= offset[index] ||
		extra_len <= offset[index + 1])
		elog(ERROR, "corrupted arrow file? offset points out of extra buffer");

	res = palloc(VARHDRSZ + len);
	SET_VARSIZE(res, VARHDRSZ + len);
	memcpy(VARDATA(res), extra + offset[index], len);

	return PointerGetDatum(res);
}

static Datum
pg_bytea_arrow_ref(kern_data_store *kds,
				   kern_colmeta *cmeta, size_t index)
{
	return pg_text_arrow_ref(kds, cmeta, index);
}

static Datum
pg_bool_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char   *bitmap = (char *)kds + __kds_unpack(cmeta->values_offset);

	return BoolGetDatum(att_isnull(index, bitmap));
}

static Datum
pg_numeric_arrow_ref(kern_data_store *kds,
					 kern_colmeta *cmeta, size_t index)
{
	char	   *result = palloc0(sizeof(struct NumericData));
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	Int128_t	decimal;

	decimal.ival = ((int128 *)base)[index];
	pg_numeric_to_varlena(result, cmeta->attopts.decimal.precision, decimal);

	return PointerGetDatum(result);
}

static Datum
pg_date_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	DateADT		dt;

	switch (cmeta->attopts.date.unit)
	{
		case ArrowDateUnit__Day:
			dt = ((cl_uint *)base)[index]
				+ (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
			break;
		case ArrowDateUnit__MilliSecond:
			dt = ((cl_ulong *)base)[index] / 1000
				+ (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Date type");
	}
	return DateADTGetDatum(dt);
}

static Datum
pg_time_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	TimeADT		tm;

	switch (cmeta->attopts.time.unit)
	{
		case ArrowTimeUnit__Second:
			tm = ((cl_long)((cl_int *)base)[index]) * 1000000L;
			break;
		case ArrowTimeUnit__MilliSecond:
			tm = ((cl_long)((cl_int *)base)[index]) * 1000L;
			break;
		case ArrowTimeUnit__MicroSecond:
			tm = ((cl_long *)base)[index];
			break;
		case ArrowTimeUnit__NanoSecond:
			tm = ((cl_ulong *)base)[index] / 1000L;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Time type");
	}
	return TimeADTGetDatum(tm);
}

static Datum
pg_timestamp_arrow_ref(kern_data_store *kds,
					   kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	Timestamp	ts;

	switch (cmeta->attopts.time.unit)
	{
		case ArrowTimeUnit__Second:
			ts = ((cl_ulong *)base)[index] * 1000000UL;
			break;
		case ArrowTimeUnit__MilliSecond:
			ts = ((cl_ulong *)base)[index] * 1000UL;
			break;
		case ArrowTimeUnit__MicroSecond:
			ts = ((cl_ulong *)base)[index];
			break;
		case ArrowTimeUnit__NanoSecond:
			ts = ((cl_ulong *)base)[index] / 1000UL;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Timestamp type");
	}
	return TimestampGetDatum(ts);
}

static Datum
pg_interval_arrow_ref(kern_data_store *kds,
					  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	Interval   *iv = palloc0(sizeof(Interval));

	switch (cmeta->attopts.interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			/* 32bit: number of months */
			iv->month = ((cl_uint *)base)[index];
			break;
		case ArrowIntervalUnit__Day_Time:
			/* 32bit+32bit: number of days and milliseconds */
			iv->day = (int32)(((cl_uint *)base)[2 * index]);
			iv->time = (TimeOffset)(((cl_uint *)base)[2 * index + 1]) * 1000;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Interval type");
	}
	return PointerGetDatum(iv);
}

/*
 * KDS_fetch_tuple_arrow
 */
bool
KDS_fetch_tuple_arrow(TupleTableSlot *slot,
					  kern_data_store *kds,
					  size_t index)
{
	Datum  *values = slot->tts_values;
	bool   *isnull = slot->tts_isnull;
	int		j;

	if (index >= kds->nitems)
		return false;
	ExecStoreAllNullTuple(slot);
	for (j=0; j < kds->ncols; j++)
	{
		kern_colmeta   *cmeta = &kds->colmeta[j];
		Datum			datum;

		if (cmeta->values_offset == 0)
			continue;	/* not loaded, always null */

		if (cmeta->nullmap_offset != 0)
		{
			size_t		nullmap_offset = __kds_unpack(cmeta->nullmap_offset);
			uint8	   *nullmap = (uint8 *)kds + nullmap_offset;

			if (att_isnull(index, nullmap))
				continue;
		}

		switch (cmeta->atttypid)
		{
			case INT2OID:
				datum = pg_int2_arrow_ref(kds, cmeta, index);
				break;
			case INT4OID:
				datum = pg_int4_arrow_ref(kds, cmeta, index);
				break;
			case INT8OID:
				datum = pg_int8_arrow_ref(kds, cmeta, index);
				break;
			case FLOAT4OID:
				datum = pg_float4_arrow_ref(kds, cmeta, index);
				break;
			case FLOAT8OID:
				datum = pg_float8_arrow_ref(kds, cmeta, index);
				break;
			case TEXTOID:
				datum = pg_text_arrow_ref(kds, cmeta, index);
				break;
			case BYTEAOID:
				datum = pg_bytea_arrow_ref(kds, cmeta, index);
				break;
			case BOOLOID:
				datum = pg_bool_arrow_ref(kds, cmeta, index);
				break;
			case NUMERICOID:
				datum = pg_numeric_arrow_ref(kds, cmeta, index);
				break;
			case DATEOID:
				datum = pg_date_arrow_ref(kds, cmeta, index);
				break;
			case TIMEOID:
				datum = pg_time_arrow_ref(kds, cmeta, index);
				break;
			case TIMESTAMPOID:
				datum = pg_timestamp_arrow_ref(kds, cmeta, index);
				break;
			case INTERVALOID:
				datum = pg_interval_arrow_ref(kds, cmeta, index);
				break;
			default:
				if (cmeta->atttypid == FLOAT2OID)
					datum = pg_float2_arrow_ref(kds, cmeta, index);
				else
					elog(ERROR, "Bug? unexpected datum type");
				break;
		}
		isnull[j] = false;
		values[j] = datum;
	}
	return true;
}

/*
 * arrowFdwExtractFilesList
 */
static List *
arrowFdwExtractFilesList(List *options_list)
{
	ListCell   *lc;
	List	   *filesList = NIL;
	char	   *dir_path = NULL;
	char	   *dir_suffix = NULL;

	foreach (lc, options_list)
	{
		DefElem	   *defel = lfirst(lc);

		Assert(IsA(defel->arg, String));
		if (strcmp(defel->defname, "file") == 0)
		{
			char   *temp = strVal(defel->arg);
			filesList = lappend(filesList, makeString(pstrdup(temp)));
		}
		else if (strcmp(defel->defname, "files") == 0)
		{
			char   *temp = pstrdup(strVal(defel->arg));
			char   *saveptr;
			char   *tok, *pos;

			while ((tok = strtok_r(temp, ",", &saveptr)) != NULL)
			{
				while (isspace(*tok))
					tok++;
				pos = tok + strlen(tok) - 1;
				while (pos >= tok && isspace(*pos))
					*pos-- = '\0';

				filesList = lappend(filesList, makeString(pstrdup(tok)));

				temp = NULL;
			}
		}
		else if (strcmp(defel->defname, "dir") == 0)
		{
			dir_path = strVal(defel->arg);
		}
		else if (strcmp(defel->defname, "suffix") == 0)
		{
			dir_suffix = strVal(defel->arg);
		}
	}
	if (dir_suffix && !dir_path)
		elog(ERROR, "arrow: cannot use 'suffix' option without 'dir'");

	if (dir_path)
	{
		struct dirent *dentry;
		DIR	   *dir;
		char   *temp;

		dir = AllocateDir(dir_path);
		while ((dentry = ReadDir(dir, dir_path)) != NULL)
		{
			if (strcmp(dentry->d_name, ".") == 0 ||
				strcmp(dentry->d_name, "..") == 0)
				continue;
			if (dir_suffix)
			{
				int		dlen = strlen(dentry->d_name);
				int		slen = strlen(dir_suffix);
				int		diff;

				if (dlen < 2 + slen)
					continue;
				diff = dlen - slen;
				if (dentry->d_name[diff-1] != '.' ||
					strcmp(dentry->d_name + diff, dir_suffix) != 0)
					continue;
			}
			temp = psprintf("%s/%s", dir_path, dentry->d_name);
			filesList = lappend(filesList, makeString(temp));
		}
		FreeDir(dir);
	}

	if (filesList == NIL)
		elog(ERROR, "arrow: no files are on behalf of the foreign table");
	foreach (lc, filesList)
	{
		Value  *val = lfirst(lc);

		if (access(strVal(val), F_OK | R_OK) != 0)
			elog(ERROR, "arrow: permission error at '%s': %m", strVal(val));
	}
	return filesList;
}

/*
 * validator of Arrow_Fdw
 */
Datum
pgstrom_arrow_fdw_validator(PG_FUNCTION_ARGS)
{
	List	   *options_list = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid			catalog = PG_GETARG_OID(1);

	if (catalog == ForeignTableRelationId)
	{
		List	   *filesList = arrowFdwExtractFilesList(options_list);
		ListCell   *lc;

		foreach (lc, filesList)
		{
			ArrowFileInfo	af_info;
			char		   *fname = strVal(lfirst(lc));

			readArrowFile(fname, &af_info);
			elog(DEBUG2, "%s", dumpArrowNode((ArrowNode *)&af_info.footer));
			//TODO: make cache item here
		}
	}
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_validator);

/*
 * Routines for Arrow metadata cache
 */
typedef struct
{
	dev_t		st_dev;
	ino_t		st_ino;
} MetadataCacheKey;

static void
arrowInvalidateMetadataCache(struct stat *stat_buf)
{
	MetadataCacheKey key;
	dlist_head	   *slot;
	dlist_node	   *dnode;
	dlist_iter		iter;
	uint32			hash, index;

	memset(&key, 0, sizeof(key));
	key.st_dev      = stat_buf->st_dev;
	key.st_ino      = stat_buf->st_ino;
	hash = hash_any((unsigned char *)&key, sizeof(key));
	index = hash % arrow_metadata_state->nslots;
	slot = &arrow_metadata_state->hash_slots[index];

	dlist_foreach(iter, slot)
	{
		arrowMetadataCache *mcache, *mtemp;

		mcache = dlist_container(arrowMetadataCache, chain, iter.cur);
		if (mcache->hash == hash &&
			mcache->stat_buf.st_dev == stat_buf->st_dev &&
			mcache->stat_buf.st_ino == stat_buf->st_ino)
		{
			while (!dlist_is_empty(&mcache->siblings))
			{
				dnode = dlist_pop_head_node(&mcache->siblings);
				mtemp = dlist_container(arrowMetadataCache, chain, dnode);
				Assert(dlist_is_empty(&mtemp->siblings) &&
					   mtemp->lru_chain.prev == NULL &&
					   mtemp->lru_chain.next == NULL);
				memset(mtemp, 0, arrowMetadataCacheSize);
				dlist_push_head(&arrow_metadata_state->free_list,
								&mtemp->chain);
			}
			dlist_delete(&mcache->chain);
			dlist_delete(&mcache->lru_chain);
			Assert(dlist_is_empty(&mcache->siblings));
			dlist_push_head(&arrow_metadata_state->free_list,
							&mcache->chain);
			break;
		}
	}
}

/*
 * arrowLookupMetadataCache
 */
static List *
arrowLookupMetadataCache(const char *fname, File fdesc)
{
	MetadataCacheKey key;
	uint32			hash, index;
	dlist_iter		iter;
	dlist_head	   *slot;
	struct stat		stat_buf;
	List		   *result = NIL;

	if (fstat(FileGetRawDesc(fdesc), &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", fname);

	memset(&key, 0, sizeof(key));
	key.st_dev		= stat_buf.st_dev;
	key.st_ino		= stat_buf.st_ino;
	hash = hash_any((unsigned char *)&key, sizeof(key));
	index = hash % arrow_metadata_state->nslots;
	slot = &arrow_metadata_state->hash_slots[index];
	LWLockAcquire(&arrow_metadata_state->lock, LW_EXCLUSIVE);
	dlist_foreach (iter, slot)
	{
		arrowMetadataCache *mcache
			= dlist_container(arrowMetadataCache, chain, iter.cur);

		if (mcache->hash == hash &&
			mcache->stat_buf.st_dev == stat_buf.st_dev &&
			mcache->stat_buf.st_ino == stat_buf.st_ino)
		{
			RecordBatchState *rbstate;
			int		j, ncols = mcache->ncols;

			if (mcache->stat_buf.st_mtime < stat_buf.st_mtime ||
				mcache->stat_buf.st_ctime < stat_buf.st_ctime)
			{
				char	buf1[50], buf2[50], buf3[50], buf4[50];

				arrowInvalidateMetadataCache(&stat_buf);
				elog(DEBUG2, "arrow_fdw: metadata cache for '%s' (m=%s c=%s) is older than the file (m=%s c=%s), so invalidated",
					 fname,
					 ctime_r(&mcache->stat_buf.st_mtime, buf1),
					 ctime_r(&mcache->stat_buf.st_ctime, buf2),
					 ctime_r(&stat_buf.st_mtime, buf3),
					 ctime_r(&stat_buf.st_ctime, buf4));
				goto out;
			}
			/* setup RecordBatchState from arrowMetadataCache */
			rbstate = palloc0(offsetof(RecordBatchState, columns[ncols]));

			rbstate->fname = fname;
			rbstate->fdesc = fdesc;
			memcpy(&rbstate->stat_buf, &mcache->stat_buf, sizeof(struct stat));
			rbstate->rb_index = mcache->rb_index;
			rbstate->rb_offset = mcache->rb_offset;
			rbstate->rb_length = mcache->rb_length;
			rbstate->rb_nitems = mcache->rb_nitems;
			rbstate->ncols = mcache->ncols;
			rbstate->ncols = mcache->ncols;
			for (j=0; j < mcache->ncols; j++)
				rbstate->columns[j] = mcache->fstate[j];
			result = lappend(result, rbstate);

			dlist_move_head(slot, &mcache->lru_chain);
			break;
		}
	}
	elog(DEBUG2, "arrow_fdw: metadata cache for '%s'%s found", fname,
		 result == NIL ? " not" : "");
out:
	LWLockRelease(&arrow_metadata_state->lock);
	return result;
}

static void
arrowUpdateMetadataCache(List *rbstateList)
{
	MetadataCacheKey key;
	arrowMetadataCache *mcache;
	dlist_head		free_list;
	dlist_head	   *slot;
	dlist_node	   *dnode;
	dlist_iter		iter;
	ListCell	   *lc;
	const char	   *fname;
	uint32			hash, index;
	int				j, nitems;
	char			buf1[50], buf2[50];
	RecordBatchState *rbstate;

	LWLockAcquire(&arrow_metadata_state->lock, LW_EXCLUSIVE);
	/* check duplicated entry */
	rbstate = linitial(rbstateList);
	fname = rbstate->fname;
	memset(&key, 0, sizeof(key));
	key.st_dev = rbstate->stat_buf.st_dev;
	key.st_ino = rbstate->stat_buf.st_ino;
	hash = hash_any((unsigned char *)&key, sizeof(key));
	index = hash % arrow_metadata_state->nslots;
	slot = &arrow_metadata_state->hash_slots[index];
	dlist_foreach (iter, slot)
	{
		mcache = dlist_container(arrowMetadataCache, chain, iter.cur);

		if (mcache->hash == hash &&
			mcache->stat_buf.st_dev == rbstate->stat_buf.st_dev &&
			mcache->stat_buf.st_ino == rbstate->stat_buf.st_ino)
		{
			if (mcache->stat_buf.st_mtime >= rbstate->stat_buf.st_mtime &&
				mcache->stat_buf.st_ctime >= rbstate->stat_buf.st_ctime)
			{
				elog(DEBUG2, "arrow_fdw: metadata cache for '%s' is already built at (m=%s c=%s)",
					 fname,
					 ctime_r(&mcache->stat_buf.st_mtime, buf1),
					 ctime_r(&mcache->stat_buf.st_ctime, buf2));
				goto out;	/* already built */
			}
			/* invalidation of old entries, prior to build new ones */
			arrowInvalidateMetadataCache(&rbstate->stat_buf);
			break;
		}
	}
	/*
	 * reclaim free entries
	 */
	nitems = list_length(rbstateList);
	if (nitems > arrow_metadata_state->nitems / 4)
	{
		elog(DEBUG2, "arrow_fdw: number of RecordBatch in '%s' is too large (%u of total %u cache items)",
			 fname, nitems, arrow_metadata_state->nitems);
		goto out;	/* it will consume too large entries */
	}
	/* # of columns too large, so unable to cache metadata */
    if (rbstate->ncols > arrow_metadata_entry_size)
	{
		elog(DEBUG2, "arrow_fdw: file '%s' contains too much columns larger than unit size of metadata cache entry size (%d)",
			 fname, arrow_metadata_entry_size);
		goto out;
	}

	dlist_init(&free_list);
	while (nitems > 0)
	{
		if (!dlist_is_empty(&arrow_metadata_state->free_list))
		{
			/* pick up a free entry */
			dnode = dlist_pop_head_node(&arrow_metadata_state->free_list);
			dlist_push_tail(&free_list, dnode);
			nitems--;
		}
		else
		{
			/* we have no free entry, so reclaim one by LRU */
			dnode = dlist_tail_node(&arrow_metadata_state->lru_list);
			dlist_delete(dnode);
			mcache = dlist_container(arrowMetadataCache, lru_chain, dnode);
			dlist_delete(&mcache->chain);

			while (!dlist_is_empty(&mcache->siblings))
			{
				arrowMetadataCache *mtemp;

				dnode = dlist_pop_head_node(&mcache->siblings);
				mtemp = dlist_container(arrowMetadataCache, chain, dnode);
				Assert(dlist_is_empty(&mtemp->siblings));
				Assert(mtemp->lru_chain.prev == NULL &&
					   mtemp->lru_chain.next == NULL);
				memset(mtemp, 0, arrowMetadataCacheSize);
				dlist_push_tail(&arrow_metadata_state->free_list,
								&mtemp->chain);
			}
			memset(mcache, 0, arrowMetadataCacheSize);
			dlist_push_tail(&arrow_metadata_state->free_list,
							&mcache->chain);
		}
	}
	/* copy the RecordBatchState to metadata cache */
	mcache = NULL;
	foreach (lc, rbstateList)
	{
		RecordBatchState   *rbstate = lfirst(lc);
		arrowMetadataCache *mtemp;

		dnode = dlist_pop_head_node(&free_list);
		mtemp = dlist_container(arrowMetadataCache, chain, dnode);

		memset(mtemp, 0, offsetof(arrowMetadataCache, fstate));
		dlist_init(&mtemp->siblings);
		memcpy(&mtemp->stat_buf, &rbstate->stat_buf, sizeof(struct stat));
		mtemp->hash      = hash;
		mtemp->rb_index  = rbstate->rb_index;
		mtemp->rb_offset = rbstate->rb_offset;
		mtemp->rb_length = rbstate->rb_length;
		mtemp->rb_nitems = rbstate->rb_nitems;
		mtemp->ncols     = rbstate->ncols;
		for (j=0; j < rbstate->ncols; j++)
			mtemp->fstate[j] = rbstate->columns[j];

		if (!mcache)
			mcache = mtemp;
		else
			dlist_push_tail(&mcache->siblings, &mtemp->chain);
	}
	dlist_push_head(slot, &mcache->chain);
	dlist_push_head(&arrow_metadata_state->lru_list,
					&mcache->lru_chain);
	elog(DEBUG2, "arrow_fdw: metadata cache for '%s' is built (dev=%u:%u, ino=%lu, m=%s, c=%s)",
		 fname,
		 major(mcache->stat_buf.st_dev),
		 minor(mcache->stat_buf.st_dev),
		 (cl_ulong)mcache->stat_buf.st_ino,
		 ctime_r(&mcache->stat_buf.st_mtime, buf1),
		 ctime_r(&mcache->stat_buf.st_ctime, buf2));
out:
	LWLockRelease(&arrow_metadata_state->lock);
}

/*
 * pgstrom_startup_arrow_fdw
 */
static void
pgstrom_startup_arrow_fdw(void)
{
	size_t		metadata_cache_sz = (size_t)arrow_metadata_cache_size_kb << 10;
	bool		found;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	arrow_metadata_state = ShmemInitStruct("arrow_metadata_state",
										   metadata_cache_sz,
										   &found);
	if (!found)
	{
		arrowMetadataCache *mcache;
		size_t		i, nitems;

		nitems = (metadata_cache_sz - offsetof(arrowMetadataState, hash_slots))
			/ (arrowMetadataCacheSize + 4 * sizeof(dlist_head));
		if (nitems < 100)
			elog(ERROR, "pg_strom.arrow_metadata_cache_size is too small, or pg_strom.arrow_metadata_cache_num_columns is too large");

		LWLockInitialize(&arrow_metadata_state->lock, -1);
		arrow_metadata_state->nslots = nitems;
		dlist_init(&arrow_metadata_state->free_list);
		for (i=0; i < nitems; i++)
			dlist_init(&arrow_metadata_state->hash_slots[i]);

		/* adjust nitems again */
		nitems = MAXALIGN(metadata_cache_sz - offsetof(arrowMetadataState,
													   hash_slots[nitems]))
			/ arrowMetadataCacheSize;
		arrow_metadata_state->nitems = nitems;

		mcache = (arrowMetadataCache *)
			(arrow_metadata_state->hash_slots + nitems);
		for (i=0; i < nitems; i++)
		{
			memset(mcache, 0, sizeof(arrowMetadataCache));
			dlist_push_tail(&arrow_metadata_state->free_list,
							&mcache->chain);
			mcache = (arrowMetadataCache *)
				((char *)mcache + arrowMetadataCacheSize);
		}
	}
}

/*
 * pgstrom_init_arrow_fdw
 */
void
pgstrom_init_arrow_fdw(void)
{
	FdwRoutine *r = &pgstrom_arrow_fdw_routine;

	memset(r, 0, sizeof(FdwRoutine));
	NodeSetTag(r, T_FdwRoutine);
	/* SCAN support */
	r->GetForeignRelSize			= ArrowGetForeignRelSize;
	r->GetForeignPaths				= ArrowGetForeignPaths;
	r->GetForeignPlan				= ArrowGetForeignPlan;
	r->BeginForeignScan				= ArrowBeginForeignScan;
	r->IterateForeignScan			= ArrowIterateForeignScan;
	r->ReScanForeignScan			= ArrowReScanForeignScan;
	r->EndForeignScan				= ArrowEndForeignScan;
	/* EXPLAIN support */
	r->ExplainForeignScan			= ArrowExplainForeignScan;
	/* IMPORT FOREIGN SCHEMA support */
	r->ImportForeignSchema			= ArrowImportForeignSchema;
	/* CPU Parallel support (not yet) */
	r->IsForeignScanParallelSafe	= ArrowIsForeignScanParallelSafe;
	r->EstimateDSMForeignScan		= ArrowEstimateDSMForeignScan;
	r->InitializeDSMForeignScan		= ArrowInitializeDSMForeignScan;
	r->ReInitializeDSMForeignScan	= ArrowReInitializeDSMForeignScan;
	r->InitializeWorkerForeignScan	= ArrowInitializeWorkerForeignScan;
	r->ShutdownForeignScan			= ArrowShutdownForeignScan;

	/*
	 * Configurations for arrow_fdw metadata cache
	 */
	DefineCustomIntVariable("pg_strom.arrow_metadata_cache_size",
							"size of shared metadata cache for arrow files",
							NULL,
							&arrow_metadata_cache_size_kb,
							32768,		/* 32MB */
							1024,		/* 1MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);

	DefineCustomIntVariable("pg_strom.arrow_metadata_cache_num_columns",
							"size of metadata cache entry for arrow files",
							NULL,
							&arrow_metadata_entry_size,
							80,		/* up to 80 columns */
							10,
							SHRT_MAX,
							PGC_POSTMASTER,
                            GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
                            NULL, NULL, NULL);
	RequestAddinShmemSpace(MAXALIGN(sizeof(arrowMetadataState)) +
						   ((size_t)arrow_metadata_cache_size_kb << 10));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_arrow_fdw;
}
