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

/*
 * RecordBatchState
 */
typedef struct
{
	union {
		struct {
			cl_int		precision;	/* Decimal.precision */
			cl_int		scale;		/* Decimal.scale */
		} decimal;
		struct {
			cl_int		unit;		/* unit of time related types */
		} time;
		struct {
			cl_int		unit;
		} date;
		struct {
			cl_int		unit;
		} timestamp;
		struct {
			cl_int		unit;
		} interval;
	} xattr;
	int64		nitems;				/* usually, same with rb_nitems */
	int64		null_count;
	off_t		nullmap_ofs;
	size_t		nullmap_len;
	off_t		values_ofs;
	size_t		values_len;
	off_t		extra_ofs;
	size_t		extra_len;
} RecordBatchFieldState;

typedef struct
{
	const char *fname;
	File		fdesc;
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

	pgstrom_data_store *pds_curr;	/* current focused buffer */
	cl_ulong	curr_index;		/* current index to row on KDS */
	/* state of RecordBatches */
	uint32		num_rbatches;
	RecordBatchState *rbatches[FLEXIBLE_ARRAY_MEMBER];
} ArrowFdwState;

/* ---------- static variables ---------- */
static FdwRoutine		pgstrom_arrow_fdw_routine;
/* ---------- static functions ---------- */
static ssize_t			arrowTypeValuesLength(ArrowType *type, int64 nitems);
static kern_tupdesc	   *arrowSchemaToKernTupdesc(ArrowSchema *schema);
static List				*arrowFdwExtractFilesList(List *options_list);
static RecordBatchState *makeRecordBatchState(ArrowSchema *schema,
											  ArrowBlock *block,
											  ArrowRecordBatch *rbatch);
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
	kern_tupdesc   *ktdesc = NULL;
	BlockNumber		npages = 0;
	double			ntuples = 0.0;
	ListCell	   *lc;

	foreach (lc, filesList)
	{
		ArrowFileInfo	af_info;
		char		   *fname = strVal(lfirst(lc));
		kern_tupdesc   *kttemp;
		struct stat		st_buf;

		if (stat(fname, &st_buf) != 0)
			elog(ERROR, "failed on stat('%s') on behalf of foreign table: %s",
				 fname, get_rel_name(foreigntableid));
		npages += (st_buf.st_size + BLCKSZ - 1) / BLCKSZ;

		readArrowFile(fname, &af_info);
		kttemp = arrowSchemaToKernTupdesc(&af_info.footer.schema);
		if (!ktdesc)
			ktdesc = kttemp;
		else if (!kern_tupdesc_equal(ktdesc, kttemp))
			elog(ERROR, "file '%s' has incompatible schema definition from other files on behalf of foreign table: %s",
				 fname, get_rel_name(foreigntableid));
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
					   int parallel_nworkers)
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
	if (parallel_nworkers > 0)
	{
		double		leader_contribution;
		double		parallel_divisor = (double) parallel_nworkers;

		/* see get_parallel_divisor() */
		leader_contribution = 1.0 - (0.3 * (double)parallel_nworkers);
		parallel_divisor += Max(leader_contribution, 0.0);

		/* The CPU cost is divided among all the workers. */
		cpu_run_cost /= parallel_divisor;

		/* Estimated row count per background worker process */
		nrows = clamp_row_est(nrows / parallel_divisor);
	}
	path->rows = nrows;
	path->startup_cost = startup_cost;
	path->total_cost = startup_cost + cpu_run_cost + disk_run_cost;
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

	fpath = create_foreignscan_path(root, baserel,
									NULL,	/* default pathtarget */
									-1,		/* dummy */
									-1.0,	/* dummy */
									-1.0,	/* dummy */
									NIL,	/* no pathkeys */
									required_outer,
									NULL,	/* no extra plan */
									NIL);	/* no particular private */
	param_info = get_baserel_parampathinfo(root, baserel, required_outer);
	/* update nrows, startup_cost and total_cost */
	cost_arrow_fdw_seqscan(&fpath->path, root, baserel, param_info, 0);

	add_path(baserel, (Path *)fpath);
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
	int		i;

	Assert(IS_SIMPLE_REL(baserel));
	for (i=baserel->min_attr; i <= baserel->max_attr; i++)
	{
		if (i < 0)
			continue;	/* system columns */
		else if (i == 0)
		{
			/* all the columns must be fetched */
			referenced = NIL;
			break;
		}
		else if (baserel->attr_needed[i])
			referenced = lappend_int(referenced, i);
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
					elog(ERROR, "arrow_fdw: RecordBatch contains less Buffer than expected");
				if (c->null_count > 0)
				{
					c->nullmap_ofs = buffer_curr->offset;
					c->nullmap_len = buffer_curr->length;
					if (c->nullmap_len < BITMAPLEN(c->null_count))
						elog(ERROR, "arrow_fdw: nullmap length is smaller than expected");
				}
				buffer_curr++;
				c->values_ofs = buffer_curr->offset;
				c->values_len = buffer_curr->length;
				if (c->values_len < arrowTypeValuesLength(&field->type,
														  c->nitems))
					elog(ERROR, "arrow_fdw: values length is smaller than expected");
				buffer_curr++;
				break;

			case ArrowNodeTag__Utf8:
			case ArrowNodeTag__Binary:
				/* variable length values */
				if (buffer_curr + 2 >= buffer_tail)
					elog(ERROR, "arrow_fdw: RecordBatch contains less Buffer than expected");
				if (c->null_count > 0)
				{
					c->nullmap_ofs = buffer_curr->offset;
					c->nullmap_len = buffer_curr->length;
					if (c->nullmap_len < BITMAPLEN(c->null_count))
						elog(ERROR, "arrow_fdw: nullmap length is smaller than expected");
				}
				buffer_curr++;
				c->values_ofs = buffer_curr->offset;
				c->values_len = buffer_curr->length;
				if (c->values_len < arrowTypeValuesLength(&field->type,
														  c->nitems))
					elog(ERROR, "arrow_fdw: values length is smaller than expected");
				buffer_curr++;
				c->extra_ofs = buffer_curr->offset;
				c->extra_len = buffer_curr->length;
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
				c->xattr.decimal.precision = field->type.Decimal.precision;
				c->xattr.decimal.scale     = field->type.Decimal.scale;
				break;
            case ArrowNodeTag__Date:
				c->xattr.date.unit = field->type.Date.unit;
				break;
            case ArrowNodeTag__Time:
				c->xattr.time.unit = field->type.Time.unit;
				break;
            case ArrowNodeTag__Timestamp:
				c->xattr.timestamp.unit = field->type.Timestamp.unit;
				break;
            case ArrowNodeTag__Interval:
				c->xattr.interval.unit = field->type.Interval.unit;
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
	ForeignScan	   *fscan = (ForeignScan *) node->ss.ps.plan;
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(relation));
	List		   *filesList = arrowFdwExtractFilesList(ft->options);
	List		   *fdescList = NIL;
	List		   *rb_state_list = NIL;
	ListCell	   *lc;
	Bitmapset	   *referenced = NULL;
	kern_tupdesc   *ktdesc;
	kern_tupdesc   *kttemp;
	ArrowFdwState  *af_state;
	int				i, num_rbatches;

	foreach (lc, fscan->fdw_private)
		referenced = bms_add_member(referenced, lfirst_int(lc));

	ktdesc = kern_tupdesc_create(RelationGetDescr(relation));
	foreach (lc, filesList)
	{
		char	   *fname = strVal(lfirst(lc));
		File		fdesc;
		ListCell   *cell;
		int			rb_count = 0;
		ArrowFileInfo af_info;

		fdesc = PathNameOpenFile(fname, O_RDONLY | PG_BINARY);
		if (fdesc < 0)
			elog(ERROR, "failed to open '%s'", fname);
		fdescList = lappend_int(fdescList, fdesc);

		memset(&af_info, 0, sizeof(ArrowFileInfo));
		//XXX: check cache instead of the file read
		readArrowFileDesc(fdesc, &af_info);
		kttemp = arrowSchemaToKernTupdesc(&af_info.footer.schema);
		if (!kern_tupdesc_equal(ktdesc, kttemp))
			elog(ERROR, "arrow_fdw: incompatible file '%s' on behalf of the foreign table '%s'.",
				 fname, RelationGetRelationName(relation));
		pfree(kttemp);

		Assert(af_info.footer._num_dictionaries ==
			   list_length(af_info.dictionaries));
		if (af_info.dictionaries != NIL)
			elog(ERROR, "arrow_fdw: does not support DictionaryBatch");
		Assert(af_info.footer._num_recordBatches ==
			   list_length(af_info.recordBatches));
		foreach (cell, af_info.recordBatches)
		{
			ArrowBlock		 *block = &af_info.footer.recordBatches[rb_count];
			ArrowRecordBatch *rbatch = lfirst(cell);
			RecordBatchState *rb_state;

			rb_state = makeRecordBatchState(&af_info.footer.schema,
											block, rbatch);
			rb_state->fname = fname;
			rb_state->fdesc = fdesc;
			rb_state_list = lappend(rb_state_list, rb_state);
			rb_count++;
		}
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

/*
 * ArrowIterateForeignScan
 */
static TupleTableSlot *
ArrowIterateForeignScan(ForeignScanState *node)
{
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
	if (af_state->pds_curr)
		PDS_release(af_state->pds_curr);
	af_state->pds_curr = NULL;
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
	return NIL;
}

/*
 * ArrowIsForeignScanParallelSafe
 */
static bool
ArrowIsForeignScanParallelSafe(PlannerInfo *root,
							   RelOptInfo *rel,
							   RangeTblEntry *rte)
{
	return false;
}

/*
 * ArrowEstimateDSMForeignScan 
 */
static Size
ArrowEstimateDSMForeignScan(ForeignScanState *node,
							ParallelContext *pcxt)
{
	return 0;
}

/*
 * ArrowInitializeDSMForeignScan
 */
static void
ArrowInitializeDSMForeignScan(ForeignScanState *node,
							  ParallelContext *pcxt,
							  void *coordinate)
{}

/*
 * ArrowReInitializeDSMForeignScan
 */
static void
ArrowReInitializeDSMForeignScan(ForeignScanState *node,
								ParallelContext *pcxt,
								void *coordinate)
{}

/*
 * ArrowInitializeWorkerForeignScan
 */
static void
ArrowInitializeWorkerForeignScan(ForeignScanState *node,
								 shm_toc *toc,
								 void *coordinate)
{}

/*
 * ArrowShutdownForeignScan
 */
static void
ArrowShutdownForeignScan(ForeignScanState *node)
{}

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
			length = sizeof(cl_long) * nitems;
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

static kern_tupdesc *
arrowSchemaToKernTupdesc(ArrowSchema *schema)
{
	kern_tupdesc   *result = NULL;
	int				i, ncols = schema->_num_fields;
	int				attcacheoff = 0;

	result = palloc0(offsetof(kern_tupdesc, colmeta[ncols]));
	result->ncols = ncols;
	for (i=0; i < ncols; i++)
	{
		ArrowField	   *field = &schema->fields[i];
		kern_colmeta   *cmeta = &result->colmeta[i];
		Oid				type_oid = InvalidOid;
		int32			type_mod = -1;
		Oid				nsp_id;
		int16			typlen;
		bool			typbyval;
		char			typalign;

		switch (field->type.node.tag)
		{
			case ArrowNodeTag__Int:
				switch (field->type.Int.bitWidth)
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
						elog(ERROR, "cannot map %s%d of Apache Arrow on PostgreSQL type",
							 field->type.Int.is_signed ? "Int" : "Uint",
							 field->type.Int.bitWidth);
						break;
				}
				break;
			case ArrowNodeTag__FloatingPoint:
				switch (field->type.FloatingPoint.precision)
				{
					case ArrowPrecision__Half:
						nsp_id = get_namespace_oid(PGSTROM_SCHEMA_NAME, false);
						type_oid = GetSysCacheOid2(TYPENAMENSP,
												   PointerGetDatum("float2"),
												   ObjectIdGetDatum(nsp_id));
						if (!OidIsValid(type_oid))
							elog(ERROR, "float2 is not defined at PostgreSQL");
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
				if (field->type.Timestamp.timezone)
					elog(ERROR, "Timestamp with timezone of Apache Arrow is not supported, right now");
				type_oid = TIMESTAMPOID;
				break;
			case ArrowNodeTag__Interval:
				type_oid = INTERVALOID;
				break;
			default:
				elog(ERROR, "Not a supported Apache Arrow type");
				break;
		}
		get_typlenbyvalalign(type_oid, &typlen, &typbyval, &typalign);
		if (attcacheoff > 0)
		{
			if (typlen > 0)
				attcacheoff = att_align_nominal(attcacheoff, typalign);
			else
				attcacheoff = -1;	/* no more shortcut any more */
		}
		cmeta->attbyval		= typbyval;
		cmeta->attalign		= typealign_get_width(typalign);
		cmeta->attlen		= typlen;
		cmeta->attnum		= i+1;
		cmeta->attcacheoff	= attcacheoff;
		cmeta->atttypid		= type_oid;
		cmeta->atttypmod	= type_mod;
		if (attcacheoff >= 0)
			attcacheoff += typlen;
	}
	return result;
}




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
			elog(INFO, "%s", dumpArrowNode((ArrowNode *)&af_info.footer));
			//dump...
			//TODO: make cache item here
		}
	}
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_validator);



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
}
