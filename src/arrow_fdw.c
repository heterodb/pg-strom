/*
 * arrow_fdw.c
 *
 * Routines to map Apache Arrow files as PG's Foreign-Table.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "arrow_defs.h"
#include "arrow_ipc.h"
#include "cuda_numeric.cu"

/*
 * RecordBatchState
 */
typedef struct RecordBatchFieldState
{
	Oid			atttypid;
	int			atttypmod;
	ArrowTypeOptions attopts;
	int64		nitems;				/* usually, same with rb_nitems */
	int64		null_count;
	off_t		nullmap_offset;
	size_t		nullmap_length;
	off_t		values_offset;
	size_t		values_length;
	off_t		extra_offset;
	size_t		extra_length;
	/* min/max statistics */
	SQLstat__datum stat_min;
	SQLstat__datum stat_max;
	bool		stat_isnull;
	/* sub-fields if any */
	int			num_children;
	struct RecordBatchFieldState *children;
} RecordBatchFieldState;

typedef struct RecordBatchState
{
	File		fdesc;
	GPUDirectFileDesc *dfile;
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
 * metadata cache (on shared memory)
 */
typedef struct
{
	dev_t		st_dev;
	ino_t		st_ino;
	uint32		hash;
} MetadataCacheKey;

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
	int			nfields;	/* length of fstate[] array */
	RecordBatchFieldState fstate[FLEXIBLE_ARRAY_MEMBER];
} arrowMetadataCache;

#define ARROW_METADATA_HASH_NSLOTS		2048
typedef struct
{
	slock_t		lru_lock;
	dlist_head	lru_list;
	pg_atomic_uint64 consumed;

	LWLock		lock_slots[ARROW_METADATA_HASH_NSLOTS];
	dlist_head	hash_slots[ARROW_METADATA_HASH_NSLOTS];
	dlist_head	mvcc_slots[ARROW_METADATA_HASH_NSLOTS];
} arrowMetadataState;

/* setup of MetadataCacheKey */
static inline int
initMetadataCacheKey(MetadataCacheKey *mkey, struct stat *stat_buf)
{
	memset(mkey, 0, sizeof(MetadataCacheKey));
	mkey->st_dev	= stat_buf->st_dev;
	mkey->st_ino	= stat_buf->st_ino;
	mkey->hash		= hash_any((unsigned char *)mkey,
							   offsetof(MetadataCacheKey, hash));
	return mkey->hash % ARROW_METADATA_HASH_NSLOTS;
}

/*
 * executor hint by min/max statistics per record batch
 */
typedef struct
{
	List		   *orig_quals;
	List		   *eval_quals;
	ExprState	   *eval_state;
	Bitmapset	   *stat_attrs;
	Bitmapset	   *load_attrs;
	ExprContext	   *econtext;
} arrowStatsHint;

/*
 * MVCC state for the pending writes
 */
typedef struct
{
	dlist_node	chain;
	MetadataCacheKey key;
	TransactionId xid;
	CommandId	cid;
	uint32		record_batch;
} arrowWriteMVCCLog;

/*
 * REDO Log for INSERT/TRUNCATE
 */
typedef struct
{
	dlist_node	chain;
	MetadataCacheKey key;
	TransactionId xid;
	CommandId	cid;
	char	   *pathname;
	bool		is_truncate;
	/* for TRUNCATE */
	uint32		suffix;
	/* for INSERT */
	loff_t		footer_offset;
	size_t		footer_length;
	char		footer_backup[FLEXIBLE_ARRAY_MEMBER];
} arrowWriteRedoLog;

/*
 * arrowWriteState
 */
typedef struct
{
	MemoryContext memcxt;
	File		file;
	MetadataCacheKey key;
	uint32		hash;
	SQLtable	sql_table;
} arrowWriteState;

/*
 * ArrowFdwState
 */
struct ArrowFdwState
{
	GpuContext *gcontext;			/* valid if owned by GpuXXX plan */
	List	   *gpuDirectFileDescList;	/* list of GPUDirectFileDesc */
	List	   *fdescList;				/* list of File (buffered i/o) */
	Bitmapset  *referenced;
	arrowStatsHint *stats_hint;
	pg_atomic_uint32   *rbatch_index;
	pg_atomic_uint32	__rbatch_index_local;	/* if single process */
	pg_atomic_uint32   *rbatch_nload;
	pg_atomic_uint32	__rbatch_nload_local;	/* if single process */
	pg_atomic_uint32   *rbatch_nskip;
	pg_atomic_uint32	__rbatch_nskip_local;	/* if single process */
	pgstrom_data_store *curr_pds;	/* current focused buffer */
	cl_ulong	curr_index;			/* current index to row on KDS */
	/* state of RecordBatches */
	uint32		num_rbatches;
	RecordBatchState *rbatches[FLEXIBLE_ARRAY_MEMBER];
};

/* ---------- static variables ---------- */
static FdwRoutine		pgstrom_arrow_fdw_routine;
static shmem_request_hook_type shmem_request_next = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static arrowMetadataState *arrow_metadata_state = NULL;
static dlist_head		arrow_write_redo_list;
static bool				arrow_fdw_enabled;				/* GUC */
static bool				arrow_fdw_stats_hint_enabled;	/* GUC */
static int				arrow_metadata_cache_size_kb;	/* GUC */
static size_t			arrow_metadata_cache_size;
static int				arrow_record_batch_size_kb;		/* GUC */

/* ---------- static functions ---------- */
static Oid		arrowTypeToPGTypeOid(ArrowField *field, int *typmod);
static const char *arrowTypeToPGTypeName(ArrowField *field);
static size_t	arrowFieldLength(ArrowField *field, int64 nitems);
static bool		arrowSchemaCompatibilityCheck(TupleDesc tupdesc,
											  RecordBatchState *rb_state);
static List	   *__arrowFdwExtractFilesList(List *options_list,
										   int *p_parallel_nworkers,
										   bool *p_writable);
static List	   *arrowFdwExtractFilesList(List *options_list);
static List	   *arrowLookupOrBuildMetadataCache(File fdesc, Bitmapset **p_stat_attrs);
static void		pg_datum_arrow_ref(kern_data_store *kds,
								   kern_colmeta *cmeta,
								   size_t index,
								   Datum *p_datum,
								   bool *p_isnull);
/* routines for writable arrow_fdw foreign tables */
static void	setupArrowSQLbufferSchema(SQLtable *table, TupleDesc tupdesc,
									  ArrowFileInfo *af_info);
static void setupArrowSQLbufferBatches(SQLtable *table,
									   ArrowFileInfo *af_info);
static loff_t createArrowWriteRedoLog(File filp, bool is_newfile);
static void writeOutArrowRecordBatch(arrowWriteState *aw_state,
									 bool with_footer);

Datum	pgstrom_arrow_fdw_handler(PG_FUNCTION_ARGS);
Datum	pgstrom_arrow_fdw_validator(PG_FUNCTION_ARGS);
Datum	pgstrom_arrow_fdw_precheck_schema(PG_FUNCTION_ARGS);
Datum	pgstrom_arrow_fdw_truncate(PG_FUNCTION_ARGS);
Datum	pgstrom_arrow_fdw_import_file(PG_FUNCTION_ARGS);

/*
 * timespec_comp - compare timespec values
 */
static inline int
timespec_comp(struct timespec *tv1, struct timespec *tv2)
{
	if (tv1->tv_sec < tv2->tv_sec)
		return -1;
	if (tv1->tv_sec > tv2->tv_sec)
		return 1;
	if (tv1->tv_nsec < tv2->tv_nsec)
		return -1;
	if (tv1->tv_nsec > tv2->tv_nsec)
		return 1;
	return 0;
}

/*
 * baseRelIsArrowFdw
 */
bool
baseRelIsArrowFdw(RelOptInfo *baserel)
{
	if ((baserel->reloptkind == RELOPT_BASEREL ||
		 baserel->reloptkind == RELOPT_OTHER_MEMBER_REL) &&
		baserel->rtekind == RTE_RELATION &&
		OidIsValid(baserel->serverid) &&
		baserel->fdwroutine &&
		memcmp(baserel->fdwroutine,
			   &pgstrom_arrow_fdw_routine,
			   sizeof(FdwRoutine)) == 0)
		return true;

	return false;
}

/*
 * RelationIsArrowFdw
 */
bool
RelationIsArrowFdw(Relation frel)
{
	if (RelationGetForm(frel)->relkind == RELKIND_FOREIGN_TABLE)
	{
		FdwRoutine *routine = GetFdwRoutineForRelation(frel, false);

		if (memcmp(routine, &pgstrom_arrow_fdw_routine,
				   sizeof(FdwRoutine)) == 0)
			return true;
	}
	return false;
}

/*
 * RecordBatchFieldCount
 */
static int
__RecordBatchFieldCount(RecordBatchFieldState *fstate)
{
	int		j, count = 1;

	for (j=0; j < fstate->num_children; j++)
		count += __RecordBatchFieldCount(&fstate->children[j]);

	return count;
}

static int
RecordBatchFieldCount(RecordBatchState *rbstate)
{
	int		j, count = 0;

	for (j=0; j < rbstate->ncols; j++)
		count += __RecordBatchFieldCount(&rbstate->columns[j]);

	return count;
}

/*
 * RecordBatchFieldLength
 */
static size_t
RecordBatchFieldLength(RecordBatchFieldState *fstate)
{
	size_t	len;
	int		j;

	len = BLCKALIGN(fstate->nullmap_length +
					fstate->values_length +
					fstate->extra_length);
	for (j=0; j < fstate->num_children; j++)
		len += RecordBatchFieldLength(&fstate->children[j]);
	return len;
}

/*
 * ArrowGetForeignRelSize
 */
static void
ArrowGetForeignRelSize(PlannerInfo *root,
					   RelOptInfo *baserel,
					   Oid foreigntableid)
{
	ForeignTable   *ft = GetForeignTable(foreigntableid);
	List		   *filesList;
	Size			filesSizeTotal = 0;
	Bitmapset	   *referenced = NULL;
	BlockNumber		npages = 0;
	double			ntuples = 0.0;
	ListCell	   *lc;
	int				parallel_nworkers;
	bool			writable;
	Bitmapset	   *optimal_gpus = (void *)(~0UL);
	int				j, k;

	/* columns to be fetched */
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(lc);

		pull_varattnos((Node *)rinfo->clause, baserel->relid, &referenced);
	}
	referenced = pgstrom_pullup_outer_refs(root, baserel, referenced);

	filesList = __arrowFdwExtractFilesList(ft->options,
										   &parallel_nworkers,
										   &writable);
	foreach (lc, filesList)
	{
		char	   *fname = strVal(lfirst(lc));
		File		fdesc;
		List	   *rb_cached;
		ListCell   *cell;
		Bitmapset  *__gpus;
		size_t		len = 0;

		fdesc = PathNameOpenFile(fname, O_RDONLY | PG_BINARY);
		if (fdesc < 0)
		{
			if (writable && errno == ENOENT)
				continue;
			elog(ERROR, "failed to open file '%s' on behalf of '%s'",
				 fname, get_rel_name(foreigntableid));
		}
		/* lookup optimal GPUs */
		__gpus = extraSysfsLookupOptimalGpus(fdesc);
		if (optimal_gpus == (void *)(~0UL))
			optimal_gpus = __gpus;
		else
			optimal_gpus = bms_intersect(optimal_gpus, __gpus);
		/* lookup or build metadata cache */
		rb_cached = arrowLookupOrBuildMetadataCache(fdesc, NULL);
		foreach (cell, rb_cached)
		{
			RecordBatchState   *rb_state = lfirst(cell);

			if (cell == list_head(rb_cached))
				filesSizeTotal += BLCKALIGN(rb_state->stat_buf.st_size);

			if (bms_is_member(-FirstLowInvalidHeapAttributeNumber, referenced))
			{
				for (j=0; j < rb_state->ncols; j++)
					len += RecordBatchFieldLength(&rb_state->columns[j]);
			}
			else
			{
				for (k = bms_next_member(referenced, -1);
					 k >= 0;
					 k = bms_next_member(referenced, k))
				{
					j = k + FirstLowInvalidHeapAttributeNumber;
					if (j < 0 || j >= rb_state->ncols)
						continue;
					len += RecordBatchFieldLength(&rb_state->columns[j]);
				}
			}
			ntuples += rb_state->rb_nitems;
		}
		npages = len / BLCKSZ;
		FileClose(fdesc);
	}
	bms_free(referenced);

	if (optimal_gpus == (void *)(~0UL) ||
		filesSizeTotal < pgstrom_gpudirect_threshold())
		optimal_gpus = NULL;

	baserel->rel_parallel_workers = parallel_nworkers;
	baserel->fdw_private = list_make1(optimal_gpus);
	baserel->pages = npages;
	baserel->tuples = ntuples;
	baserel->rows = ntuples *
		clauselist_selectivity(root,
							   baserel->baserestrictinfo,
							   0,
							   JOIN_INNER,
							   NULL);
}

/*
 * GetOptimalGpusForArrowFdw
 *
 * optimal GPUs bitmap is saved at baserel->fdw_private
 */
Bitmapset *
GetOptimalGpusForArrowFdw(PlannerInfo *root, RelOptInfo *baserel)
{
	if (baserel->fdw_private == NIL)
	{
		RangeTblEntry *rte = root->simple_rte_array[baserel->relid];

		ArrowGetForeignRelSize(root, baserel, rte->relid);
	}
	return linitial(baserel->fdw_private);
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

	/* arrow_fdw.enabled */
	if (!arrow_fdw_enabled)
		startup_cost += disable_cost;

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
	Bitmapset  *referenced = NULL;
	List	   *ref_list = NIL;
	ListCell   *lc;
	int			j, k;

	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(lc);

		pull_varattnos((Node *)rinfo->clause, baserel->relid, &referenced);
	}
	referenced = pgstrom_pullup_outer_refs(root, baserel, referenced);

	for (k = bms_next_member(referenced, -1);
		 k >= 0;
		 k = bms_next_member(referenced, k))
	{
		j = k + FirstLowInvalidHeapAttributeNumber;
		ref_list = lappend_int(ref_list, j);
	}
	bms_free(referenced);

	return make_foreignscan(tlist,
							extract_actual_clauses(scan_clauses, false),
							baserel->relid,
							NIL,	/* no expressions to evaluate */
							ref_list, /* list of referenced attnums */
							NIL,	/* no custom tlist */
							NIL,	/* no remote quals */
							outer_plan);
}

/* ----------------------------------------------------------------
 *
 * Routines related to min/max statistics and scan hint
 *
 * If mapped Apache Arrow files have custome-metadata of "min_values" and
 * "max_values" at the Field, arrow_fdw deals with this comma separated
 * integer values as min/max value for each field, if any.
 * Once we can know min/max value of the field, we can skip record batches
 * that shall not match with WHERE-clause.
 *
 * This min/max array is expected to have as many integer elements or nulls
 * as there are record-batches.
 * ----------------------------------------------------------------
 */

/*
 * buildArrowStatsBinary
 *
 * It reconstruct binary min/max statistics per record-batch
 * from the custom-metadata of ArrowField.
 */
typedef struct arrowFieldStatsBinary
{
	uint32	nrooms;		/* number of record-batches */
	int		unitsz;		/* unit size of min/max statistics */
	bool   *isnull;
	char   *min_values;
	char   *max_values;
	int		nfields;	/* if List/Struct data type */
	struct arrowFieldStatsBinary *subfields;
} arrowFieldStatsBinary;

typedef struct
{
	int		nitems;		/* number of record-batches */
	int		ncols;
	arrowFieldStatsBinary columns[FLEXIBLE_ARRAY_MEMBER];
} arrowStatsBinary;

static void
__releaseArrowFieldStatsBinary(arrowFieldStatsBinary *bstats)
{
	int			j;

	if (bstats->subfields)
	{
		for (j=0; j < bstats->nfields; j++)
			__releaseArrowFieldStatsBinary(&bstats->subfields[j]);
		pfree(bstats->subfields);
	}
	if (bstats->isnull)
		pfree(bstats->isnull);
	if (bstats->min_values)
		pfree(bstats->min_values);
	if (bstats->max_values)
		pfree(bstats->max_values);
}

static void
releaseArrowStatsBinary(arrowStatsBinary *arrow_bstats)
{
	int			j;

	if (arrow_bstats)
	{
		for (j=0; j < arrow_bstats->ncols; j++)
			__releaseArrowFieldStatsBinary(&arrow_bstats->columns[j]);
		pfree(arrow_bstats);
	}
}

static int128_t
__atoi128(const char *tok, bool *p_isnull)
{
	int128_t	ival = 0;
	bool		is_minus = false;

	if (*tok == '-')
	{
		is_minus = true;
		tok++;
	}
	while (isdigit(*tok))
	{
		ival = 10 * ival + (*tok - '0');
		tok++;
	}

	if (*tok != '\0')
		*p_isnull = true;
	if (is_minus)
	{
		if (ival == 0)
			*p_isnull = true;
		ival = -ival;
	}
	return ival;
}

static bool
__parseArrowFieldStatsBinary(arrowFieldStatsBinary *bstats,
							 ArrowField *field,
							 const char *min_tokens,
							 const char *max_tokens)
{
	int			unitsz = -1;
	char	   *min_buffer;
	char	   *max_buffer;
	char	   *min_values = NULL;
	char	   *max_values = NULL;
	bool	   *isnull = NULL;
	char	   *tok1, *pos1;
	char	   *tok2, *pos2;
	uint32		index;

	/* determine the unitsz of datum */
	switch (field->type.node.tag)
	{
		case ArrowNodeTag__Int:
			switch (field->type.Int.bitWidth)
			{
				case 8:
					unitsz = sizeof(uint8_t);
					break;
				case 16:
					unitsz = sizeof(uint16_t);
					break;
				case 32:
					unitsz = sizeof(uint32_t);
					break;
				case 64:
					unitsz = sizeof(uint64_t);
					break;
				default:
					return false;
			}
			break;

		case ArrowNodeTag__FloatingPoint:
			switch (field->type.FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					unitsz = sizeof(uint16_t);
					break;
				case ArrowPrecision__Single:
					unitsz = sizeof(uint32_t);
					break;
				case ArrowPrecision__Double:
					unitsz = sizeof(uint64_t);
					break;
				default:
					return false;
			}
			break;

		case ArrowNodeTag__Decimal:
			unitsz = sizeof(int128_t);
			break;

		case ArrowNodeTag__Date:
			switch (field->type.Date.unit)
			{
				case ArrowDateUnit__Day:
					unitsz = sizeof(uint32_t);
					break;
				case ArrowDateUnit__MilliSecond:
					unitsz = sizeof(uint64_t);
					break;
				default:
					return false;
			}
			break;

		case ArrowNodeTag__Time:
			switch (field->type.Time.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
					unitsz = sizeof(uint32_t);
					break;
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					unitsz = sizeof(uint64_t);
					break;
				default:
					return false;
			}
			break;

		case ArrowNodeTag__Timestamp:
			switch (field->type.Timestamp.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					unitsz = sizeof(uint64_t);
					break;
				default:
					return false;
			}
			break;
		default:
			return false;
	}
	Assert(unitsz > 0);
	/* parse the min_tokens/max_tokens */
	min_buffer = alloca(strlen(min_tokens) + 1);
	max_buffer = alloca(strlen(max_tokens) + 1);
	strcpy(min_buffer, min_tokens);
	strcpy(max_buffer, max_tokens);

	min_values = palloc0(unitsz * bstats->nrooms);
	max_values = palloc0(unitsz * bstats->nrooms);
	isnull     = palloc0(sizeof(bool) * bstats->nrooms);
	for (tok1 = strtok_r(min_buffer, ",", &pos1),
		 tok2 = strtok_r(max_buffer, ",", &pos2), index = 0;
		 tok1 != NULL && tok2 != NULL && index < bstats->nrooms;
		 tok1 = strtok_r(NULL, ",", &pos1),
		 tok2 = strtok_r(NULL, ",", &pos2), index++)
	{
		bool		__isnull = false;
		int128_t	__min = __atoi128(__trim(tok1), &__isnull);
		int128_t	__max = __atoi128(__trim(tok2), &__isnull);

		if (__isnull)
			isnull[index] = true;
		else
		{
			memcpy(min_values + unitsz * index, &__min, unitsz);
			memcpy(max_values + unitsz * index, &__max, unitsz);
		}
	}
	/* sanity checks */
	if (!tok1 && !tok2 && index == bstats->nrooms)
	{
		bstats->unitsz = unitsz;
		bstats->isnull = isnull;
		bstats->min_values = min_values;
		bstats->max_values = max_values;
		return true;
	}
	/* elsewhere, something wrong */
	pfree(min_values);
	pfree(max_values);
	pfree(isnull);
	return false;
}

static bool
__buildArrowFieldStatsBinary(arrowFieldStatsBinary *bstats,
							 ArrowField *field,
							 uint32 numRecordBatches)
{
	const char *min_tokens = NULL;
	const char *max_tokens = NULL;
	int			j, k;
	bool		retval = false;

	for (k=0; k < field->_num_custom_metadata; k++)
	{
		ArrowKeyValue *kv = &field->custom_metadata[k];

		if (strcmp(kv->key, "min_values") == 0)
			min_tokens = kv->value;
		else if (strcmp(kv->key, "max_values") == 0)
			max_tokens = kv->value;
	}

	bstats->nrooms = numRecordBatches;
	bstats->unitsz = -1;
	if (min_tokens && max_tokens)
	{
		if (__parseArrowFieldStatsBinary(bstats, field,
										 min_tokens,
										 max_tokens))
		{
			retval = true;
		}
		else
		{
			/* parse error, ignore the stat */
			if (bstats->isnull)
				pfree(bstats->isnull);
			if (bstats->min_values)
				pfree(bstats->min_values);
			if (bstats->max_values)
				pfree(bstats->max_values);
			bstats->unitsz     = -1;
			bstats->isnull     = NULL;
			bstats->min_values = NULL;
			bstats->max_values = NULL;
		}
	}

	if (field->_num_children > 0)
	{
		bstats->nfields = field->_num_children;
		bstats->subfields = palloc0(sizeof(arrowFieldStatsBinary) * bstats->nfields);
		for (j=0; j < bstats->nfields; j++)
		{
			if (__buildArrowFieldStatsBinary(&bstats->subfields[j],
											 &field->children[j],
											 numRecordBatches))
				retval = true;
		}
	}
	return retval;
}

static arrowStatsBinary *
buildArrowStatsBinary(const ArrowFooter *footer, Bitmapset **p_stat_attrs)
{
	arrowStatsBinary *arrow_bstats;
	int		j, ncols = footer->schema._num_fields;
	bool	found = false;

	arrow_bstats = palloc0(offsetof(arrowStatsBinary,
									columns[ncols]));
	arrow_bstats->nitems = footer->_num_recordBatches;
	arrow_bstats->ncols = ncols;
	for (j=0; j < ncols; j++)
	{
		if (__buildArrowFieldStatsBinary(&arrow_bstats->columns[j],
										 &footer->schema.fields[j],
										 footer->_num_recordBatches))
		{
			if (p_stat_attrs)
				*p_stat_attrs = bms_add_member(*p_stat_attrs, j+1);
			found = true;
		}
	}
	if (!found)
	{
		releaseArrowStatsBinary(arrow_bstats);
		return NULL;
	}
	return arrow_bstats;
}

/*
 * applyArrowStatsBinary
 *
 * It applies the fetched min/max values on the cached record-batch metadata
 */
static void
__applyArrowFieldStatsBinary(RecordBatchFieldState *fstate,
							 arrowFieldStatsBinary *bstats,
							 int rb_index)
{
	int		j;

	if (bstats->unitsz > 0 &&
		bstats->isnull != NULL &&
		bstats->min_values != NULL &&
		bstats->max_values != NULL)
	{
		size_t	off = bstats->unitsz * rb_index;

		memcpy(&fstate->stat_min,
			   bstats->min_values + off, bstats->unitsz);
		memcpy(&fstate->stat_max,
			   bstats->max_values + off, bstats->unitsz);
		fstate->stat_isnull = false;
	}
	else
	{
		memset(&fstate->stat_min, 0, sizeof(SQLstat__datum));
		memset(&fstate->stat_max, 0, sizeof(SQLstat__datum));
		fstate->stat_isnull = true;
	}
	
	Assert(fstate->num_children == bstats->nfields);
	for (j=0; j < fstate->num_children; j++)
	{
		RecordBatchFieldState  *__fstate = &fstate->children[j];
		arrowFieldStatsBinary  *__bstats = &bstats->subfields[j];

		__applyArrowFieldStatsBinary(__fstate, __bstats, rb_index);
	}
}

static void
applyArrowStatsBinary(RecordBatchState *rb_state, arrowStatsBinary *arrow_bstats)
{
	int		j, ncols = rb_state->ncols;

	Assert(rb_state->ncols == arrow_bstats->ncols &&
		   rb_state->rb_index < arrow_bstats->nitems);
	for (j=0; j < ncols; j++)
	{
		RecordBatchFieldState  *fstate = &rb_state->columns[j];
		arrowFieldStatsBinary  *bstats = &arrow_bstats->columns[j];

		__applyArrowFieldStatsBinary(fstate, bstats, rb_state->rb_index);
	}
}

static SQLstat *
__buildArrowFieldStatsList(ArrowField *field, uint32 numRecordBatches)
{
	const char *min_tokens = NULL;
	const char *max_tokens = NULL;
	char	   *min_buffer;
	char	   *max_buffer;
	char	   *tok1, *pos1;
	char	   *tok2, *pos2;
	SQLstat	   *results = NULL;
	int			k, index;

	for (k=0; k < field->_num_custom_metadata; k++)
	{
		ArrowKeyValue *kv = &field->custom_metadata[k];

		if (strcmp(kv->key, "min_values") == 0)
			min_tokens = kv->value;
		else if (strcmp(kv->key, "max_values") == 0)
			max_tokens = kv->value;
	}
	if (!min_tokens || !max_tokens)
		return NULL;
	min_buffer = alloca(strlen(min_tokens) + 1);
	max_buffer = alloca(strlen(max_tokens) + 1);
	strcpy(min_buffer, min_tokens);
	strcpy(max_buffer, max_tokens);

	for (tok1 = strtok_r(min_buffer, ",", &pos1),
		 tok2 = strtok_r(max_buffer, ",", &pos2), index = 0;
		 tok1 && tok2;
		 tok1 = strtok_r(NULL, ",", &pos1),
		 tok2 = strtok_r(NULL, ",", &pos2), index++)
	{
		bool		__isnull = false;
		int128_t	__min = __atoi128(__trim(tok1), &__isnull);
		int128_t	__max = __atoi128(__trim(tok2), &__isnull);

		if (!__isnull)
		{
			SQLstat *item = palloc0(sizeof(SQLstat));

			item->next = results;
			item->rb_index = index;
			item->is_valid = true;
			item->min.i128 = __min;
			item->max.i128 = __max;
			results = item;
		}
	}
	/* sanity checks */
	if (!tok1 && !tok2 && index == numRecordBatches)
		return results;
	/* ah, error... */
	while (results)
	{
		SQLstat *next = results->next;

		pfree(results);
		results = next;
	}
	return NULL;
}

/*
 * execInitArrowStatsHint / execCheckArrowStatsHint / execEndArrowStatsHint
 *
 * ... are executor routines for min/max statistics.
 */
static bool
__buildArrowStatsOper(arrowStatsHint *arange,
					  ScanState *ss,
					  OpExpr *op,
					  bool reverse)
{
	Index		scanrelid = ((Scan *)ss->ps.plan)->scanrelid;
	Oid			opcode;
	Var		   *var;
	Node	   *arg;
	Expr	   *expr;
	Oid			opfamily = InvalidOid;
	StrategyNumber strategy = InvalidStrategy;
	CatCList   *catlist;
	int			i;

	if (!reverse)
	{
		opcode = op->opno;
		var = linitial(op->args);
		arg = lsecond(op->args);
	}
	else
	{
		opcode = get_commutator(op->opno);
		var = lsecond(op->args);
		arg = linitial(op->args);
	}
	/* Is it VAR <OPER> ARG form? */
	if (!IsA(var, Var) || var->varno != scanrelid)
		return false;
	if (!bms_is_member(var->varattno, arange->stat_attrs))
		return false;
	if (contain_var_clause(arg) ||
		contain_volatile_functions(arg))
		return false;

	catlist = SearchSysCacheList1(AMOPOPID, ObjectIdGetDatum(opcode));
	for (i=0; i < catlist->n_members; i++)
	{
		HeapTuple	tuple = &catlist->members[i]->tuple;
		Form_pg_amop amop = (Form_pg_amop) GETSTRUCT(tuple);

		if (amop->amopmethod == BRIN_AM_OID)
		{
			opfamily = amop->amopfamily;
			strategy = amop->amopstrategy;
			break;
		}
	}
	ReleaseSysCacheList(catlist);

	if (strategy == BTLessStrategyNumber ||
		strategy == BTLessEqualStrategyNumber)
	{
		/* (VAR < ARG) --> (Min < ARG) */
		/* (VAR <= ARG) --> (Min <= ARG) */
		arange->load_attrs = bms_add_member(arange->load_attrs,
											var->varattno);
		expr = make_opclause(opcode,
							 op->opresulttype,
							 op->opretset,
							 (Expr *)makeVar(INNER_VAR,
											 var->varattno,
											 var->vartype,
											 var->vartypmod,
											 var->varcollid,
											 0),
							 (Expr *)copyObject(arg),
							 op->opcollid,
							 op->inputcollid);
		set_opfuncid((OpExpr *)expr);
		arange->eval_quals = lappend(arange->eval_quals, expr);
	}
	else if (strategy == BTGreaterEqualStrategyNumber ||
			 strategy == BTGreaterStrategyNumber)
	{
		/* (VAR >= ARG) --> (Max >= ARG) */
		/* (VAR > ARG) --> (Max > ARG) */
		arange->load_attrs = bms_add_member(arange->load_attrs,
											var->varattno);
		expr = make_opclause(opcode,
							 op->opresulttype,
							 op->opretset,
							 (Expr *)makeVar(OUTER_VAR,
											 var->varattno,
											 var->vartype,
											 var->vartypmod,
											 var->varcollid,
											 0),
							 (Expr *)copyObject(arg),
							 op->opcollid,
							 op->inputcollid);
		set_opfuncid((OpExpr *)expr);
		arange->eval_quals = lappend(arange->eval_quals, expr);
	}
	else if (strategy == BTEqualStrategyNumber)
	{
		/* (VAR = ARG) --> (Max >= ARG && Min <= ARG) */
		opcode = get_opfamily_member(opfamily, var->vartype,
									 exprType((Node *)arg),
									 BTGreaterEqualStrategyNumber);
		expr = make_opclause(opcode,
							 op->opresulttype,
							 op->opretset,
							 (Expr *)makeVar(OUTER_VAR,
											 var->varattno,
											 var->vartype,
											 var->vartypmod,
											 var->varcollid,
											 0),
							 (Expr *)copyObject(arg),
							 op->opcollid,
							 op->inputcollid);
		set_opfuncid((OpExpr *)expr);
		arange->eval_quals = lappend(arange->eval_quals, expr);

		opcode = get_opfamily_member(opfamily, var->vartype,
									 exprType((Node *)arg),
									 BTLessEqualStrategyNumber);
		expr = make_opclause(opcode,
							 op->opresulttype,
							 op->opretset,
							 (Expr *)makeVar(INNER_VAR,
											 var->varattno,
											 var->vartype,
											 var->vartypmod,
											 var->varcollid,
											 0),
							 (Expr *)copyObject(arg),
							 op->opcollid,
							 op->inputcollid);
		set_opfuncid((OpExpr *)expr);
		arange->eval_quals = lappend(arange->eval_quals, expr);
	}
	else
	{
		return false;
	}
	arange->load_attrs = bms_add_member(arange->load_attrs,
										var->varattno);
	return true;
}

static arrowStatsHint *
execInitArrowStatsHint(ScanState *ss,
					   Bitmapset *stat_attrs,
					   List *outer_quals)
{
	Relation		relation = ss->ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(relation);
	ExprContext	   *econtext;
	arrowStatsHint *result, temp;
	Expr		   *eval_expr;
	ListCell	   *lc;

	memset(&temp, 0, sizeof(arrowStatsHint));
	temp.stat_attrs = stat_attrs;
	foreach (lc, outer_quals)
	{
		OpExpr *op = lfirst(lc);

		if (IsA(op, OpExpr) && list_length(op->args) == 2 &&
			(__buildArrowStatsOper(&temp, ss, op, false) ||
			 __buildArrowStatsOper(&temp, ss, op, true)))
		{
			temp.orig_quals = lappend(temp.orig_quals, copyObject(op));
		}
	}
	if (!temp.orig_quals)
		return NULL;

	Assert(list_length(temp.eval_quals) > 0);
	if (list_length(temp.eval_quals) == 1)
		eval_expr = linitial(temp.eval_quals);
	else
		eval_expr = make_andclause(temp.eval_quals);

	econtext = CreateExprContext(ss->ps.state);
	econtext->ecxt_innertuple = MakeSingleTupleTableSlot(tupdesc, &TTSOpsVirtual);
	econtext->ecxt_outertuple = MakeSingleTupleTableSlot(tupdesc, &TTSOpsVirtual);

	result = palloc0(sizeof(arrowStatsHint));
	result->orig_quals = temp.orig_quals;
	result->eval_quals = temp.eval_quals;
	result->eval_state = ExecInitExpr(eval_expr, &ss->ps);
	result->stat_attrs = bms_copy(stat_attrs);
	result->load_attrs = temp.load_attrs;
	result->econtext   = econtext;

	return result;
}

static bool
__fetchArrowStatsDatum(RecordBatchFieldState *fstate,
					   SQLstat__datum *sval,
					   Datum *p_datum, bool *p_isnull)
{
	Datum		datum;
	int64		shift;

	switch (fstate->atttypid)
	{
		case INT1OID:
			datum = Int8GetDatum(sval->i8);
			break;
		case INT2OID:
		case FLOAT2OID:
			datum = Int16GetDatum(sval->i16);
			break;
		case INT4OID:
		case FLOAT4OID:
			datum = Int32GetDatum(sval->i32);
			break;
		case INT8OID:
		case FLOAT8OID:
			datum = Int64GetDatum(sval->i64);
			break;
		case NUMERICOID:
			{
				Int128_t	decimal;
				int			dscale = fstate->attopts.decimal.scale;
				char	   *result = palloc0(sizeof(struct NumericData));

				decimal.ival = sval->i128;
				while (dscale > 0 && decimal.ival % 10 == 0)
				{
					decimal.ival /= 10;
					dscale--;
				}
				pg_numeric_to_varlena(result, dscale, decimal);

				datum = PointerGetDatum(result);
			}
			break;
		case DATEOID:
			shift = POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE;
			switch (fstate->attopts.date.unit)
			{
				case ArrowDateUnit__Day:
					datum = DateADTGetDatum((DateADT)sval->i32 - shift);
					break;
				case ArrowDateUnit__MilliSecond:
					datum = DateADTGetDatum((DateADT)sval->i64 / 1000L - shift);
					break;
				default:
					return false;
			}
			break;

		case TIMEOID:
			switch (fstate->attopts.time.unit)
			{
				case ArrowTimeUnit__Second:
					datum = TimeADTGetDatum((TimeADT)sval->u32 * 1000000L);
					break;
				case ArrowTimeUnit__MilliSecond:
					datum = TimeADTGetDatum((TimeADT)sval->u32 * 1000L);
					break;
				case ArrowTimeUnit__MicroSecond:
					datum = TimeADTGetDatum((TimeADT)sval->u64);
					break;
				case ArrowTimeUnit__NanoSecond:
					datum = TimeADTGetDatum((TimeADT)sval->u64 / 1000L);
					break;
				default:
					return false;
			}
			break;
		case TIMESTAMPOID:
		case TIMESTAMPTZOID:
			shift = (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * USECS_PER_DAY;
			switch (fstate->attopts.timestamp.unit)
			{
				case ArrowTimeUnit__Second:
					datum = TimestampGetDatum((Timestamp)sval->i64 * 1000000L - shift);
					break;
				case ArrowTimeUnit__MilliSecond:
					datum = TimestampGetDatum((Timestamp)sval->i64 * 1000L - shift);
					break;
				case ArrowTimeUnit__MicroSecond:
					datum = TimestampGetDatum((Timestamp)sval->i64 - shift);
					break;
				case ArrowTimeUnit__NanoSecond:
					datum = TimestampGetDatum((Timestamp)sval->i64 / 1000L - shift);
					break;
				default:
					return false;
			}
			break;
		default:
			return false;
	}
	*p_datum = datum;
	*p_isnull = false;
	return true;
}

static bool
execCheckArrowStatsHint(arrowStatsHint *stats_hint,
						RecordBatchState *rb_state)
{
	ExprContext	   *econtext = stats_hint->econtext;
	TupleTableSlot *min_values = econtext->ecxt_innertuple;
	TupleTableSlot *max_values = econtext->ecxt_outertuple;
	int				anum;
	Datum			datum;
	bool			isnull;

	/* load the min/max statistics */
	ExecStoreAllNullTuple(min_values);
	ExecStoreAllNullTuple(max_values);
	for (anum = bms_next_member(stats_hint->load_attrs, -1);
		 anum >= 0;
		 anum = bms_next_member(stats_hint->load_attrs, anum))
	{
		RecordBatchFieldState *fstate = &rb_state->columns[anum-1];

		Assert(anum > 0 && anum <= rb_state->ncols);
		/*
		 * In case when min/max statistics are missing, we cannot determine
		 * whether we can skip the current record-batch.
		 */
		if (fstate->stat_isnull)
			return false;

		if (!__fetchArrowStatsDatum(fstate, &fstate->stat_min,
									&min_values->tts_values[anum-1],
									&min_values->tts_isnull[anum-1]))
			return false;

		if (!__fetchArrowStatsDatum(fstate, &fstate->stat_max,
									&max_values->tts_values[anum-1],
									&max_values->tts_isnull[anum-1]))
			return false;
	}
	datum = ExecEvalExprSwitchContext(stats_hint->eval_state, econtext, &isnull);

//	elog(INFO, "file [%s] rb_index=%u datum=%lu isnull=%d",
//		 FilePathName(rb_state->fdesc), rb_state->rb_index, datum, (int)isnull);
	if (!isnull && DatumGetBool(datum))
		return true;
	return false;
}

static void
execEndArrowStatsHint(arrowStatsHint *stats_hint)
{
	ExprContext	   *econtext = stats_hint->econtext;

	ExecDropSingleTupleTableSlot(econtext->ecxt_innertuple);
	ExecDropSingleTupleTableSlot(econtext->ecxt_outertuple);
	econtext->ecxt_innertuple = NULL;
	econtext->ecxt_outertuple = NULL;

	FreeExprContext(econtext, true);
}

/*
 * Routines to setup record-batches
 */
typedef struct
{
	ArrowBuffer    *buffer_curr;
	ArrowBuffer    *buffer_tail;
	ArrowFieldNode *fnode_curr;
	ArrowFieldNode *fnode_tail;
} setupRecordBatchContext;

static void
assignArrowTypeOptions(ArrowTypeOptions *attopts, const ArrowType *atype)
{
	memset(attopts, 0, sizeof(ArrowTypeOptions));
	switch (atype->node.tag)
	{
		case ArrowNodeTag__Decimal:
			if (atype->Decimal.precision < SHRT_MIN ||
				atype->Decimal.precision > SHRT_MAX)
				elog(ERROR, "Decimal precision is out of range");
			if (atype->Decimal.scale < SHRT_MIN ||
				atype->Decimal.scale > SHRT_MAX)
				elog(ERROR, "Decimal scale is out of range");
			attopts->decimal.precision = atype->Decimal.precision;
			attopts->decimal.scale     = atype->Decimal.scale;
			break;
		case ArrowNodeTag__Date:
			if (atype->Date.unit == ArrowDateUnit__Day ||
				atype->Date.unit == ArrowDateUnit__MilliSecond)
				attopts->date.unit = atype->Date.unit;
			else
				elog(ERROR, "unknown unit of Date");
			break;
		case ArrowNodeTag__Time:
			if (atype->Time.unit == ArrowTimeUnit__Second ||
				atype->Time.unit == ArrowTimeUnit__MilliSecond ||
				atype->Time.unit == ArrowTimeUnit__MicroSecond ||
				atype->Time.unit == ArrowTimeUnit__NanoSecond)
				attopts->time.unit = atype->Time.unit;
			else
				elog(ERROR, "unknown unit of Time");
			break;
		case ArrowNodeTag__Timestamp:
			if (atype->Timestamp.unit == ArrowTimeUnit__Second ||
				atype->Timestamp.unit == ArrowTimeUnit__MilliSecond ||
				atype->Timestamp.unit == ArrowTimeUnit__MicroSecond ||
				atype->Timestamp.unit == ArrowTimeUnit__NanoSecond)
				attopts->timestamp.unit = atype->Timestamp.unit;
			else
				elog(ERROR, "unknown unit of Timestamp");
			break;
		case ArrowNodeTag__Interval:
			if (atype->Interval.unit == ArrowIntervalUnit__Year_Month ||
				atype->Interval.unit == ArrowIntervalUnit__Day_Time)
				attopts->interval.unit = atype->Interval.unit;
			else
				elog(ERROR, "unknown unit of Interval");
			break;
		case ArrowNodeTag__FixedSizeBinary:
			attopts->fixed_size_binary.byteWidth = atype->FixedSizeBinary.byteWidth;
			break;
		default:
			/* no extra attributes */
			break;
	}
}

static void
setupRecordBatchField(setupRecordBatchContext *con,
					  RecordBatchFieldState *fstate,
					  ArrowField  *field,
					  int depth)
{
	ArrowBuffer	   *buffer_curr;
	ArrowFieldNode *fnode;

	if (con->fnode_curr >= con->fnode_tail)
		elog(ERROR, "RecordBatch has less ArrowFieldNode than expected");
	fnode = con->fnode_curr++;
	fstate->atttypid   = arrowTypeToPGTypeOid(field, &fstate->atttypmod);
	fstate->nitems     = fnode->length;
	fstate->null_count = fnode->null_count;
	fstate->stat_isnull = true;

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
		case ArrowNodeTag__FixedSizeBinary:
			/* fixed length values */
			if (con->buffer_curr + 2 > con->buffer_tail)
				elog(ERROR, "RecordBatch has less buffers than expected");
			buffer_curr = con->buffer_curr++;
			if (fstate->null_count > 0)
			{
				fstate->nullmap_offset = buffer_curr->offset;
				fstate->nullmap_length = buffer_curr->length;
				if (fstate->nullmap_length < BITMAPLEN(fstate->nitems))
					elog(ERROR, "nullmap length is smaller than expected");
				if ((fstate->nullmap_offset & (MAXIMUM_ALIGNOF - 1)) != 0)
					elog(ERROR, "nullmap is not aligned well");
			}
			buffer_curr = con->buffer_curr++;
			fstate->values_offset = buffer_curr->offset;
			fstate->values_length = buffer_curr->length;
			if (fstate->values_length < arrowFieldLength(field,fstate->nitems))
				elog(ERROR, "values array is smaller than expected");
			if ((fstate->values_offset & (MAXIMUM_ALIGNOF - 1)) != 0)
				elog(ERROR, "values array is not aligned well");
			break;

		case ArrowNodeTag__List:
			if (field->_num_children != 1)
				elog(ERROR, "Bug? List of arrow type is corrupted");
			if (depth > 0)
				elog(ERROR, "nested array type is not supported");
			/* nullmap */
			if (con->buffer_curr + 1 > con->buffer_tail)
				elog(ERROR, "RecordBatch has less buffers than expected");
			buffer_curr = con->buffer_curr++;
			if (fstate->null_count > 0)
			{
				fstate->nullmap_offset = buffer_curr->offset;
				fstate->nullmap_length = buffer_curr->length;
				if (fstate->nullmap_length < BITMAPLEN(fstate->nitems))
					elog(ERROR, "nullmap length is smaller than expected");
				if ((fstate->nullmap_offset & (MAXIMUM_ALIGNOF - 1)) != 0)
					elog(ERROR, "nullmap is not aligned well");
			}
			/* offset values */
			buffer_curr = con->buffer_curr++;
			fstate->values_offset = buffer_curr->offset;
			fstate->values_length = buffer_curr->length;
			if (fstate->values_length < arrowFieldLength(field,fstate->nitems))
				elog(ERROR, "offset array is smaller than expected");
			if ((fstate->values_offset & (MAXIMUM_ALIGNOF - 1)) != 0)
				elog(ERROR, "offset array is not aligned well");
			/* setup array element */
			fstate->children = palloc0(sizeof(RecordBatchFieldState));
			setupRecordBatchField(con,
								  &fstate->children[0],
								  &field->children[0],
								  depth+1);
			fstate->num_children = 1;
			break;

		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__Binary:
			/* variable length values */
			if (con->buffer_curr + 3 > con->buffer_tail)
				elog(ERROR, "RecordBatch has less buffers than expected");
			buffer_curr = con->buffer_curr++;
			if (fstate->null_count > 0)
			{
				fstate->nullmap_offset = buffer_curr->offset;
				fstate->nullmap_length = buffer_curr->length;
				if (fstate->nullmap_length < BITMAPLEN(fstate->nitems))
					elog(ERROR, "nullmap length is smaller than expected");
				if ((fstate->nullmap_offset & (MAXIMUM_ALIGNOF - 1)) != 0)
					elog(ERROR, "nullmap is not aligned well");
			}

			buffer_curr = con->buffer_curr++;
			fstate->values_offset = buffer_curr->offset;
			fstate->values_length = buffer_curr->length;
			if (fstate->values_length < arrowFieldLength(field,fstate->nitems))
				elog(ERROR, "offset array is smaller than expected");
			if ((fstate->values_offset & (MAXIMUM_ALIGNOF - 1)) != 0)
				elog(ERROR, "offset array is not aligned well (%lu %lu)", fstate->values_offset, fstate->values_length);

			buffer_curr = con->buffer_curr++;
			fstate->extra_offset = buffer_curr->offset;
			fstate->extra_length = buffer_curr->length;
			if ((fstate->extra_offset & (MAXIMUM_ALIGNOF - 1)) != 0)
				elog(ERROR, "extra buffer is not aligned well");
			break;

		case ArrowNodeTag__Struct:
			if (depth > 0)
				elog(ERROR, "nested composite type is not supported");
			/* only nullmap */
			if (con->buffer_curr + 1 > con->buffer_tail)
				elog(ERROR, "RecordBatch has less buffers than expected");
			buffer_curr = con->buffer_curr++;
			if (fstate->null_count > 0)
			{
				fstate->nullmap_offset = buffer_curr->offset;
				fstate->nullmap_length = buffer_curr->length;
				if (fstate->nullmap_length < BITMAPLEN(fstate->nitems))
					elog(ERROR, "nullmap length is smaller than expected");
				if ((fstate->nullmap_offset & (MAXIMUM_ALIGNOF - 1)) != 0)
					elog(ERROR, "nullmap is not aligned well");
			}

			if (field->_num_children > 0)
			{
				int		i;

				fstate->children = palloc0(sizeof(RecordBatchFieldState) *
									  field->_num_children);
				for (i=0; i < field->_num_children; i++)
				{
					setupRecordBatchField(con,
										  &fstate->children[i],
										  &field->children[i],
										  depth+1);
				}
			}
			fstate->num_children = field->_num_children;
			break;
		default:
			elog(ERROR, "Bug? ArrowSchema contains unsupported types");
	}
	/* assign extra attributes (precision, unitsz, ...) */
	assignArrowTypeOptions(&fstate->attopts, &field->type);
}

static RecordBatchState *
makeRecordBatchState(ArrowSchema *schema,
					 ArrowBlock *block,
					 ArrowRecordBatch *rbatch)
{
	setupRecordBatchContext con;
	RecordBatchState *result;
	int			j, ncols = schema->_num_fields;

	/*
	 * Right now, we have no support for compressed RecordBatches
	 */
	if (rbatch->compression)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("arrow_fdw: compressed record-batches are not supported")));
	
	result = palloc0(offsetof(RecordBatchState, columns[ncols]));
	result->ncols = ncols;
	result->rb_offset = block->offset + block->metaDataLength;
	result->rb_length = block->bodyLength;
	result->rb_nitems = rbatch->length;

	memset(&con, 0, sizeof(setupRecordBatchContext));
	con.buffer_curr = rbatch->buffers;
	con.buffer_tail = rbatch->buffers + rbatch->_num_buffers;
	con.fnode_curr  = rbatch->nodes;
	con.fnode_tail  = rbatch->nodes + rbatch->_num_nodes;

	for (j=0; j < ncols; j++)
	{
		RecordBatchFieldState *fstate = &result->columns[j];
		ArrowField	   *field = &schema->fields[j];

		setupRecordBatchField(&con, fstate, field, 0);
	}
	if (con.buffer_curr != con.buffer_tail ||
		con.fnode_curr  != con.fnode_tail)
		elog(ERROR, "arrow_fdw: RecordBatch may have corruption.");

	return result;
}

/*
 * ExecInitArrowFdw
 */
ArrowFdwState *
ExecInitArrowFdw(ScanState *ss,
				 GpuContext *gcontext,
				 List *outer_quals,
				 Bitmapset *outer_refs)
{
	Relation		relation = ss->ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(relation);
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(relation));
	List		   *filesList = NIL;
	List		   *fdescList = NIL;
	List		   *gpuDirectFileDescList = NIL;
	Bitmapset	   *referenced = NULL;
	Bitmapset	   *stat_attrs = NULL;
	bool			whole_row_ref = false;
	ArrowFdwState  *af_state;
	List		   *rb_state_list = NIL;
	ListCell	   *lc;
	bool			writable;
	int				i, num_rbatches;

	Assert(RelationGetForm(relation)->relkind == RELKIND_FOREIGN_TABLE &&
		   memcmp(GetFdwRoutineForRelation(relation, false),
				  &pgstrom_arrow_fdw_routine, sizeof(FdwRoutine)) == 0);
	/* expand 'referenced' if it has whole-row reference */
	if (bms_is_member(-FirstLowInvalidHeapAttributeNumber, outer_refs))
		whole_row_ref = true;
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, i);
		int		k = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		if (attr->attisdropped)
			continue;
		if (whole_row_ref || bms_is_member(k, outer_refs))
			referenced = bms_add_member(referenced, k);
	}

	filesList = __arrowFdwExtractFilesList(ft->options,
										   NULL,
										   &writable);
	foreach (lc, filesList)
	{
		char	   *fname = strVal(lfirst(lc));
		File		fdesc;
		List	   *rb_cached = NIL;
		ListCell   *cell;
		GPUDirectFileDesc *dfile = NULL;

		fdesc = PathNameOpenFile(fname, O_RDONLY | PG_BINARY);
		if (fdesc < 0)
		{
			if (writable && errno == ENOENT)
				continue;
			elog(ERROR, "failed to open '%s' on behalf of '%s'",
				 fname, RelationGetRelationName(relation));
		}
		fdescList = lappend_int(fdescList, fdesc);

		/*
		 * Open file for GPUDirect I/O
		 */
		if (gcontext)
		{
			dfile = palloc0(sizeof(GPUDirectFileDesc));

			gpuDirectFileDescOpen(dfile, fdesc);
			if (!trackRawFileDesc(gcontext, dfile, __FILE__, __LINE__))
			{
				gpuDirectFileDescClose(dfile);
				elog(ERROR, "out of memory");
			}
			gpuDirectFileDescList = lappend(gpuDirectFileDescList, dfile);
		}

		rb_cached = arrowLookupOrBuildMetadataCache(fdesc, &stat_attrs);
		/* check schema compatibility */
		foreach (cell, rb_cached)
		{
			RecordBatchState   *rb_state = lfirst(cell);

			if (!arrowSchemaCompatibilityCheck(tupdesc, rb_state))
				elog(ERROR, "arrow file '%s' on behalf of foreign table '%s' has incompatible schema definition",
					 fname, RelationGetRelationName(relation));
			/* GPUDirect I/O state, if any */
			rb_state->dfile = dfile;
		}
		rb_state_list = list_concat(rb_state_list, rb_cached);
	}
	num_rbatches = list_length(rb_state_list);
	af_state = palloc0(offsetof(ArrowFdwState, rbatches[num_rbatches]));
	af_state->gcontext = gcontext;
	af_state->gpuDirectFileDescList = gpuDirectFileDescList;
	af_state->fdescList = fdescList;
	af_state->referenced = referenced;
	if (arrow_fdw_stats_hint_enabled)
		af_state->stats_hint = execInitArrowStatsHint(ss, stat_attrs,
													  outer_quals);
	af_state->rbatch_index = &af_state->__rbatch_index_local;
	af_state->rbatch_nload = &af_state->__rbatch_nload_local;
	af_state->rbatch_nskip = &af_state->__rbatch_nskip_local;
	i = 0;
	foreach (lc, rb_state_list)
		af_state->rbatches[i++] = (RecordBatchState *)lfirst(lc);
	af_state->num_rbatches = num_rbatches;

	return af_state;
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
	ListCell	   *lc;
	Bitmapset	   *referenced = NULL;

	foreach (lc, fscan->fdw_private)
	{
		int		j = lfirst_int(lc);

		if (j >= 0 && j <= tupdesc->natts)
			referenced = bms_add_member(referenced, j -
										FirstLowInvalidHeapAttributeNumber);
	}
	node->fdw_state = ExecInitArrowFdw(&node->ss,
									   NULL,
									   fscan->scan.plan.qual,
									   referenced);
}

typedef struct
{
	off_t		rb_offset;
	off_t		f_offset;
	off_t		m_offset;
	cl_int		io_index;
	cl_int      depth;
	strom_io_chunk ioc[FLEXIBLE_ARRAY_MEMBER];
} arrowFdwSetupIOContext;

/*
 * arrowFdwSetupIOvectorField
 */
static void
__setupIOvectorField(arrowFdwSetupIOContext *con,
					 off_t chunk_offset,
					 size_t chunk_length,
					 cl_uint *p_cmeta_offset,
					 cl_uint *p_cmeta_length)
{
	off_t		f_pos = con->rb_offset + chunk_offset;
	size_t		__length = MAXALIGN(chunk_length);

	Assert((con->m_offset & (MAXIMUM_ALIGNOF - 1)) == 0);

	if (f_pos == con->f_offset)
	{
		/* good, buffer is fully continuous */
		*p_cmeta_offset = __kds_packed(con->m_offset);
		*p_cmeta_length = __kds_packed(__length);

		con->m_offset += __length;
		con->f_offset += __length;
	}
	else if (f_pos > con->f_offset &&
			 (f_pos & ~PAGE_MASK) == (con->f_offset & ~PAGE_MASK) &&
			 ((f_pos - con->f_offset) & (MAXIMUM_ALIGNOF-1)) == 0)
	{
		/*
		 * we can also consolidate the i/o of two chunks, if file position
		 * of the next chunk (f_pos) and the current file tail position
		 * (con->f_offset) locate within the same file page, and if gap bytes
		 * on the file does not break alignment.
		 */
		size_t	__gap = (f_pos - con->f_offset);

		/* put gap bytes */
		Assert(__gap < PAGE_SIZE);
		con->m_offset += __gap;
		con->f_offset += __gap;

		*p_cmeta_offset = __kds_packed(con->m_offset);
		*p_cmeta_length = __kds_packed(__length);

		con->m_offset += __length;
		con->f_offset += __length;
	}
	else
	{
		/*
		 * Elsewhere, we have no chance to consolidate this chunk to
		 * the previous i/o-chunk. So, make a new i/o-chunk.
		 */
		off_t		f_base = TYPEALIGN_DOWN(PAGE_SIZE, f_pos);
		off_t		f_tail;
		off_t		shift = f_pos - f_base;
		strom_io_chunk *ioc;

		if (con->io_index < 0)
			con->io_index = 0;	/* no previous i/o chunks */
		else
		{
			ioc = &con->ioc[con->io_index++];

			f_tail = TYPEALIGN(PAGE_SIZE, con->f_offset);
			ioc->nr_pages = f_tail / PAGE_SIZE - ioc->fchunk_id;
			con->m_offset += (f_tail - con->f_offset); //safety margin;
		}
		ioc = &con->ioc[con->io_index];
		/* adjust position if con->m_offset is not aligned well */
		if (con->m_offset + shift != MAXALIGN(con->m_offset + shift))
			con->m_offset = MAXALIGN(con->m_offset + shift) - shift;
		ioc->m_offset   = con->m_offset;
		ioc->fchunk_id  = f_base / PAGE_SIZE;

		*p_cmeta_offset = __kds_packed(con->m_offset + shift);
		*p_cmeta_length = __kds_packed(__length);

		con->m_offset  += shift + __length;
		con->f_offset   = f_pos + __length;
	}
}

static void
arrowFdwSetupIOvectorField(arrowFdwSetupIOContext *con,
						   RecordBatchFieldState *fstate,
						   kern_data_store *kds,
						   kern_colmeta *cmeta)
{
	//int		index = cmeta - kds->colmeta;

	if (fstate->nullmap_length > 0)
	{
		Assert(fstate->null_count > 0);
		__setupIOvectorField(con,
							 fstate->nullmap_offset,
							 fstate->nullmap_length,
							 &cmeta->nullmap_offset,
							 &cmeta->nullmap_length);
		//elog(INFO, "D%d att[%d] nullmap=%lu,%lu m_offset=%lu f_offset=%lu", con->depth, index, fstate->nullmap_offset, fstate->nullmap_length, con->m_offset, con->f_offset);
	}
	if (fstate->values_length > 0)
	{
		__setupIOvectorField(con,
							 fstate->values_offset,
							 fstate->values_length,
							 &cmeta->values_offset,
							 &cmeta->values_length);
		//elog(INFO, "D%d att[%d] values=%lu,%lu m_offset=%lu f_offset=%lu", con->depth, index, fstate->values_offset, fstate->values_length, con->m_offset, con->f_offset);
	}
	if (fstate->extra_length > 0)
	{
		__setupIOvectorField(con,
							 fstate->extra_offset,
							 fstate->extra_length,
							 &cmeta->extra_offset,
							 &cmeta->extra_length);
		//elog(INFO, "D%d att[%d] extra=%lu,%lu m_offset=%lu f_offset=%lu", con->depth, index, fstate->extra_offset, fstate->extra_length, con->m_offset, con->f_offset);
	}

	/* nested sub-fields if composite types */
	if (cmeta->atttypkind == TYPE_KIND__ARRAY ||
		cmeta->atttypkind == TYPE_KIND__COMPOSITE)
	{
		kern_colmeta *subattr;
		int		j;

		Assert(fstate->num_children == cmeta->num_subattrs);
		con->depth++;
		for (j=0, subattr = &kds->colmeta[cmeta->idx_subattrs];
			 j < cmeta->num_subattrs;
			 j++, subattr++)
		{
			RecordBatchFieldState *child = &fstate->children[j];

			arrowFdwSetupIOvectorField(con, child, kds, subattr);
		}
		con->depth--;
	}
}

/*
 * arrowFdwSetupIOvector
 */
static strom_io_vector *
arrowFdwSetupIOvector(kern_data_store *kds,
					  RecordBatchState *rb_state,
					  Bitmapset *referenced)
{
	arrowFdwSetupIOContext *con;
	strom_io_vector *iovec = NULL;
	int			j, nr_chunks = 0;

	Assert(kds->nr_colmeta >= kds->ncols);
	con = alloca(offsetof(arrowFdwSetupIOContext,
						  ioc[3 * kds->nr_colmeta]));
	con->rb_offset = rb_state->rb_offset;
	con->f_offset  = ~0UL;	/* invalid offset */
	con->m_offset  = TYPEALIGN(PAGE_SIZE, KERN_DATA_STORE_HEAD_LENGTH(kds));
	con->io_index  = -1;
	for (j=0; j < kds->ncols; j++)
	{
		RecordBatchFieldState *fstate = &rb_state->columns[j];
		kern_colmeta *cmeta = &kds->colmeta[j];
		int			attidx = j + 1 - FirstLowInvalidHeapAttributeNumber;

		if (referenced && bms_is_member(attidx, referenced))
			arrowFdwSetupIOvectorField(con, fstate, kds, cmeta);
		else
			cmeta->atttypkind = TYPE_KIND__NULL;	/* unreferenced */
	}
	if (con->io_index >= 0)
	{
		/* close the last I/O chunks */
		strom_io_chunk *ioc = &con->ioc[con->io_index];

		ioc->nr_pages = (TYPEALIGN(PAGE_SIZE, con->f_offset) / PAGE_SIZE -
						 ioc->fchunk_id);
		con->m_offset = ioc->m_offset + PAGE_SIZE * ioc->nr_pages;
		nr_chunks = con->io_index + 1;
	}
	kds->length = con->m_offset;

	iovec = palloc0(offsetof(strom_io_vector, ioc[nr_chunks]));
	iovec->nr_chunks = nr_chunks;
	if (nr_chunks > 0)
		memcpy(iovec->ioc, con->ioc, sizeof(strom_io_chunk) * nr_chunks);
	return iovec;
}

/*
 * __dump_kds_and_iovec - just for debug
 */
static inline void
__dump_kds_and_iovec(kern_data_store *kds, strom_io_vector *iovec)
{
#if 0
	int		j;

	elog(INFO, "nchunks = %d", iovec->nr_chunks);
	for (j=0; j < iovec->nr_chunks; j++)
	{
		strom_io_chunk *ioc = &iovec->ioc[j];

		elog(INFO, "io[%d] [ m_offset=%lu, f_read=%lu...%lu, nr_pages=%u}",
			 j,
			 ioc->m_offset,
			 ioc->fchunk_id * PAGE_SIZE,
			 (ioc->fchunk_id + ioc->nr_pages) * PAGE_SIZE,
			 ioc->nr_pages);
	}

	elog(INFO, "kds {length=%zu nitems=%u typeid=%u typmod=%u table_oid=%u}",
		 kds->length, kds->nitems,
		 kds->tdtypeid, kds->tdtypmod, kds->table_oid);
	for (j=0; j < kds->nr_colmeta; j++)
	{
		kern_colmeta *cmeta = &kds->colmeta[j];

		elog(INFO, "%ccol[%d] nullmap=%lu,%lu values=%lu,%lu extra=%lu,%lu",
			 j < kds->ncols ? ' ' : '*', j,
			 __kds_unpack(cmeta->nullmap_offset),
			 __kds_unpack(cmeta->nullmap_length),
			 __kds_unpack(cmeta->values_offset),
			 __kds_unpack(cmeta->values_length),
			 __kds_unpack(cmeta->extra_offset),
			 __kds_unpack(cmeta->extra_length));

	}
#endif
}

/*
 * arrowFdwLoadRecordBatch
 */
static void
__arrowFdwAssignTypeOptions(kern_data_store *kds,
							int base, int ncols,
							RecordBatchFieldState *rb_fstate)
{
	int		i;

	for (i=0; i < ncols; i++)
	{
		kern_colmeta   *cmeta = &kds->colmeta[base+i];

		cmeta->attopts = rb_fstate[i].attopts;
		if (cmeta->atttypkind == TYPE_KIND__ARRAY)
		{
			Assert(cmeta->idx_subattrs >= kds->ncols &&
				   cmeta->num_subattrs == 1 &&
				   cmeta->idx_subattrs + cmeta->num_subattrs <= kds->nr_colmeta);
			Assert(rb_fstate[i].num_children == 1);
			__arrowFdwAssignTypeOptions(kds,
										cmeta->idx_subattrs,
										cmeta->num_subattrs,
										rb_fstate[i].children);
		}
		else if (cmeta->atttypkind == TYPE_KIND__COMPOSITE)
		{
			Assert(cmeta->idx_subattrs >= kds->ncols &&
				   cmeta->idx_subattrs + cmeta->num_subattrs <= kds->nr_colmeta);
			Assert(rb_fstate[i].num_children == cmeta->num_subattrs);
			__arrowFdwAssignTypeOptions(kds,
										cmeta->idx_subattrs,
										cmeta->num_subattrs,
										rb_fstate[i].children);
		}
	}
}

static pgstrom_data_store *
__arrowFdwLoadRecordBatch(RecordBatchState *rb_state,
						  Relation relation,
						  Bitmapset *referenced,
						  GpuContext *gcontext,
						  MemoryContext mcontext,
						  const Bitmapset *optimal_gpus)
{
	TupleDesc			tupdesc = RelationGetDescr(relation);
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	strom_io_vector	   *iovec;
	size_t				head_sz;
	CUresult			rc;

	/* setup KDS and I/O-vector */
	head_sz = KDS_calculateHeadSize(tupdesc);
	kds = alloca(head_sz);
	init_kernel_data_store(kds, tupdesc, 0, KDS_FORMAT_ARROW, 0);
	kds->nitems = rb_state->rb_nitems;
	kds->nrooms = rb_state->rb_nitems;
	kds->table_oid = RelationGetRelid(relation);
	Assert(head_sz == KERN_DATA_STORE_HEAD_LENGTH(kds));
	Assert(kds->ncols == rb_state->ncols);
	__arrowFdwAssignTypeOptions(kds, 0, kds->ncols, rb_state->columns);
	iovec = arrowFdwSetupIOvector(kds, rb_state, referenced);
	__dump_kds_and_iovec(kds, iovec);

	/*
	 * If SSD-to-GPU Direct SQL is available on the arrow file, setup a small
	 * PDS on host-pinned memory, with strom_io_vector.
	 */
	if (gcontext &&
		bms_is_member(gcontext->cuda_dindex, optimal_gpus) &&
		iovec->nr_chunks > 0 &&
		kds->length <= gpuMemAllocIOMapMaxLength() &&
		rb_state->dfile != NULL)
	{
		size_t	iovec_sz = offsetof(strom_io_vector, ioc[iovec->nr_chunks]);

		rc = gpuMemAllocHost(gcontext, (void **)&pds,
							 offsetof(pgstrom_data_store, kds) +
							 head_sz + iovec_sz);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAllocHost: %s", errorText(rc));

		pds->gcontext = gcontext;
		pg_atomic_init_u32(&pds->refcnt, 1);
		pds->nblocks_uncached = 0;
		memcpy(&pds->filedesc, rb_state->dfile, sizeof(GPUDirectFileDesc));
		pds->iovec = (strom_io_vector *)((char *)&pds->kds + head_sz);
		memcpy(&pds->kds, kds, head_sz);
		memcpy(pds->iovec, iovec, iovec_sz);
	}
	else
	{
		/* Elsewhere, load RecordBatch by filesystem */
		int		fdesc = FileGetRawDesc(rb_state->fdesc);

		if (gcontext)
		{
			rc = gpuMemAllocManaged(gcontext,
									(CUdeviceptr *)&pds,
									offsetof(pgstrom_data_store,
											 kds) + kds->length,
									CU_MEM_ATTACH_GLOBAL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));
		}
		else
		{
			pds = MemoryContextAllocHuge(mcontext,
										 offsetof(pgstrom_data_store,
												  kds) + kds->length);
		}
		__PDS_fillup_arrow(pds, gcontext, kds, fdesc, iovec);
	}
	pfree(iovec);
	return pds;
}

static pgstrom_data_store *
arrowFdwLoadRecordBatch(ArrowFdwState *af_state,
						Relation relation,
						EState *estate,
						GpuContext *gcontext,
						const Bitmapset *optimal_gpus)
{
	RecordBatchState *rb_state;
	uint32		rb_index;

retry:
	/* fetch next RecordBatch */
	rb_index = pg_atomic_fetch_add_u32(af_state->rbatch_index, 1);
	if (rb_index >= af_state->num_rbatches)
		return NULL;	/* no more RecordBatch to read */
	rb_state = af_state->rbatches[rb_index];

	if (af_state->stats_hint)
	{
		if (execCheckArrowStatsHint(af_state->stats_hint, rb_state))
			pg_atomic_fetch_add_u32(af_state->rbatch_nload, 1);
		else
		{
			pg_atomic_fetch_add_u32(af_state->rbatch_nskip, 1);
			goto retry;
		}
	}
	return __arrowFdwLoadRecordBatch(rb_state,
									 relation,
									 af_state->referenced,
									 gcontext,
									 estate->es_query_cxt,
									 optimal_gpus);
}

/*
 * ExecScanChunkArrowFdw
 */
pgstrom_data_store *
ExecScanChunkArrowFdw(GpuTaskState *gts)
{
	pgstrom_data_store *pds;

	InstrStartNode(&gts->outer_instrument);
	pds = arrowFdwLoadRecordBatch(gts->af_state,
								  gts->css.ss.ss_currentRelation,
								  gts->css.ss.ps.state,
								  gts->gcontext,
								  gts->optimal_gpus);
	InstrStopNode(&gts->outer_instrument,
				  !pds ? 0.0 : (double)pds->kds.nitems);
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

		/* unload the previous RecordBatch, if any */
		if (pds)
			PDS_release(pds);
		af_state->curr_index = 0;
		af_state->curr_pds = arrowFdwLoadRecordBatch(af_state,
													 relation,
													 estate,
													 NULL,
													 NULL);
		if (!af_state->curr_pds)
			return NULL;
	}
	Assert(pds && af_state->curr_index < pds->kds.nitems);
	if (KDS_fetch_tuple_arrow(slot, &pds->kds, af_state->curr_index++))
		return slot;
	return NULL;
}

/*
 * ArrowReScanForeignScan
 */
void
ExecReScanArrowFdw(ArrowFdwState *af_state)
{
	/* rewind the current scan state */
	pg_atomic_write_u32(af_state->rbatch_index, 0);
	if (af_state->curr_pds)
		PDS_release(af_state->curr_pds);
	af_state->curr_pds = NULL;
	af_state->curr_index = 0;
}

static void
ArrowReScanForeignScan(ForeignScanState *node)
{
	ExecReScanArrowFdw((ArrowFdwState *)node->fdw_state);
}

/*
 * ArrowEndForeignScan
 */
void
ExecEndArrowFdw(ArrowFdwState *af_state)
{
	ListCell   *lc;

	foreach (lc, af_state->fdescList)
		FileClose((File)lfirst_int(lc));
	foreach (lc, af_state->gpuDirectFileDescList)
	{
		GPUDirectFileDesc *dfile = lfirst(lc);

		untrackRawFileDesc(af_state->gcontext, dfile);
		gpuDirectFileDescClose(dfile);
	}
	if (af_state->stats_hint)
		execEndArrowStatsHint(af_state->stats_hint);
}

static void
ArrowEndForeignScan(ForeignScanState *node)
{
	ExecEndArrowFdw((ArrowFdwState *)node->fdw_state);
}

/*
 * ArrowExplainForeignScan 
 */
void
ExplainArrowFdw(ArrowFdwState *af_state,
				Relation frel,
				ExplainState *es,
				List *dcontext)
{
	TupleDesc	tupdesc = RelationGetDescr(frel);
	ListCell   *lc;
	int			fcount = 0;
	char		label[80];
	size_t	   *chunk_sz = alloca(sizeof(size_t) * tupdesc->natts);
	int			i, j, k;
	StringInfoData	buf;

	/* shows referenced columns */
	initStringInfo(&buf);
	for (k = bms_next_member(af_state->referenced, -1);
		 k >= 0;
		 k = bms_next_member(af_state->referenced, k))
	{
		j = k + FirstLowInvalidHeapAttributeNumber - 1;

		if (j >= 0)
		{
			Form_pg_attribute	attr = tupleDescAttr(tupdesc, j);
			const char		   *attName = NameStr(attr->attname);
			if (buf.len > 0)
				appendStringInfoString(&buf, ", ");
			appendStringInfoString(&buf, quote_identifier(attName));
		}
	}
	ExplainPropertyText("referenced", buf.data, es);

	/* shows stats hint if any */
	if (af_state->stats_hint)
	{
		arrowStatsHint *stats_hint = af_state->stats_hint;

		resetStringInfo(&buf);

		if (dcontext == NIL)
		{
			int		anum;

			for (anum = bms_next_member(stats_hint->load_attrs, -1);
				 anum >= 0;
				 anum = bms_next_member(stats_hint->load_attrs, anum))
			{
				Form_pg_attribute attr = tupleDescAttr(tupdesc, anum-1);
				const char *attName = NameStr(attr->attname);

				if (buf.len > 0)
					appendStringInfoString(&buf, ", ");
				appendStringInfoString(&buf, quote_identifier(attName));
			}
		}
		else
		{
			ListCell   *lc;

			foreach (lc, stats_hint->orig_quals)
			{
				Node   *qual = lfirst(lc);
				char   *temp;

				temp = deparse_expression(qual, dcontext, es->verbose, false);
				if (buf.len > 0)
					appendStringInfoString(&buf, ", ");
				appendStringInfoString(&buf, temp);
				pfree(temp);
			}
		}
		if (es->analyze)
			appendStringInfo(&buf, "  [loaded: %u, skipped: %u]",
							 pg_atomic_read_u32(af_state->rbatch_nload),
							 pg_atomic_read_u32(af_state->rbatch_nskip));
		ExplainPropertyText("Stats-Hint", buf.data, es);
	}

	/* shows files on behalf of the foreign table */
	foreach (lc, af_state->fdescList)
	{
		File		fdesc = (File)lfirst_int(lc);
		const char *fname = FilePathName(fdesc);
		int			rbcount = 0;
		size_t		read_sz = 0;
		char	   *pos = label;
		struct stat	st_buf;

		pos += snprintf(label, sizeof(label), "files%d", fcount++);
		if (fstat(FileGetRawDesc(fdesc), &st_buf) != 0)
			memset(&st_buf, 0, sizeof(struct stat));

		/* size count per chunk */
		memset(chunk_sz, 0, sizeof(size_t) * tupdesc->natts);
		for (i=0; i < af_state->num_rbatches; i++)
		{
			RecordBatchState *rb_state = af_state->rbatches[i];
			size_t		sz;

			if (rb_state->fdesc != fdesc)
				continue;

			for (k = bms_next_member(af_state->referenced, -1);
				 k >= 0;
				 k = bms_next_member(af_state->referenced, k))
			{
				j = k + FirstLowInvalidHeapAttributeNumber - 1;
				if (j < 0 || j >= tupdesc->natts)
					continue;
				sz = RecordBatchFieldLength(&rb_state->columns[j]);
				read_sz += sz;
				chunk_sz[j] += sz;
			}
			rbcount++;
		}

		/* file size and read size */
		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			resetStringInfo(&buf);
			if (st_buf.st_size == 0)
				appendStringInfoString(&buf, fname);
			else
				appendStringInfo(&buf, "%s (read: %s, size: %s)",
								 fname,
								 format_bytesz(read_sz),
								 format_bytesz(st_buf.st_size));
			ExplainPropertyText(label, buf.data, es);
		}
		else
		{
			ExplainPropertyText(label, fname, es);

			sprintf(pos, "-size");
			ExplainPropertyText(label, format_bytesz(st_buf.st_size), es);

			sprintf(pos, "-read");
			ExplainPropertyText(label, format_bytesz(read_sz), es);
		}

		/* read-size per column (verbose mode only)  */
		if (es->verbose && rbcount >= 0)
		{
			for (k = bms_next_member(af_state->referenced, -1);
                 k >= 0;
                 k = bms_next_member(af_state->referenced, k))
            {
				Form_pg_attribute attr;

				j = k + FirstLowInvalidHeapAttributeNumber - 1;
				if (j < 0 || j >= tupdesc->natts)
					continue;
				attr = tupleDescAttr(tupdesc, j);
				snprintf(label, sizeof(label),
						 "  %s", NameStr(attr->attname));
				ExplainPropertyText(label, format_bytesz(chunk_sz[j]), es);
			}
		}
	}
	pfree(buf.data);
}

static void
ArrowExplainForeignScan(ForeignScanState *node, ExplainState *es)
{
	Relation	frel = node->ss.ss_currentRelation;

	ExplainArrowFdw((ArrowFdwState *)node->fdw_state, frel, es, NIL);
}

/*
 * readArrowFile
 */
static bool
readArrowFile(const char *pathname, ArrowFileInfo *af_info, bool missing_ok)
{
    File	filp = PathNameOpenFile(pathname, O_RDONLY | PG_BINARY);

	if (filp < 0)
	{
		if (missing_ok && errno == ENOENT)
			return false;
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open file \"%s\": %m", pathname)));
	}
	readArrowFileDesc(FileGetRawDesc(filp), af_info);
	FileClose(filp);
	return true;
}

/*
 * RecordBatchAcquireSampleRows - random sampling
 */
static int
RecordBatchAcquireSampleRows(Relation relation,
							 RecordBatchState *rb_state,
							 HeapTuple *rows,
							 int nsamples)
{
	TupleDesc		tupdesc = RelationGetDescr(relation);
	pgstrom_data_store *pds;
	Bitmapset	   *referenced = NULL;
	Datum		   *values;
	bool		   *isnull;
	int				count;
	int				i, j, nwords;

	/* ANALYZE needs to fetch all the attributes */
	nwords = (tupdesc->natts - FirstLowInvalidHeapAttributeNumber +
			  BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	referenced = alloca(offsetof(Bitmapset, words[nwords]));
	referenced->nwords = nwords;
	memset(referenced->words, -1, sizeof(bitmapword) * nwords);
	
	pds = __arrowFdwLoadRecordBatch(rb_state,
									relation,
									referenced,
									NULL,
									CurrentMemoryContext,
									NULL);
	values = alloca(sizeof(Datum) * tupdesc->natts);
	isnull = alloca(sizeof(bool)  * tupdesc->natts);
	for (count = 0; count < nsamples; count++)
	{
		/* fetch a row randomly */
		i = (double)pds->kds.nitems * drand48();
		Assert(i < pds->kds.nitems);

		for (j=0; j < pds->kds.ncols; j++)
		{
			kern_colmeta   *cmeta = &pds->kds.colmeta[j];
			
			pg_datum_arrow_ref(&pds->kds,
							   cmeta,
							   i,
							   values + j,
							   isnull + j);
		}
		rows[count] = heap_form_tuple(tupdesc, values, isnull);
	}
	PDS_release(pds);

	return count;
}

/*
 * ArrowAcquireSampleRows
 */
static int
ArrowAcquireSampleRows(Relation relation,
					   int elevel,
					   HeapTuple *rows,
					   int nrooms,
					   double *p_totalrows,
					   double *p_totaldeadrows)
{
	TupleDesc		tupdesc = RelationGetDescr(relation);
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(relation));
	List		   *filesList = NIL;
	List		   *fdescList = NIL;
	List		   *rb_state_list = NIL;
	ListCell	   *lc;
	bool			writable;
	int64			total_nrows = 0;
	int64			count_nrows = 0;
	int				nsamples_min = nrooms / 100;
	int				nitems = 0;

	filesList = __arrowFdwExtractFilesList(ft->options,
										   NULL,
										   &writable);
	foreach (lc, filesList)
	{
		char	   *fname = strVal(lfirst(lc));
		File		fdesc;
		List	   *rb_cached;
		ListCell   *cell;

		fdesc = PathNameOpenFile(fname, O_RDONLY | PG_BINARY);
        if (fdesc < 0)
		{
			if (writable && errno == ENOENT)
				continue;
			elog(ERROR, "failed to open file '%s' on behalf of '%s'",
				 fname, RelationGetRelationName(relation));
		}
		fdescList = lappend_int(fdescList, fdesc);
		
		rb_cached = arrowLookupOrBuildMetadataCache(fdesc, NULL);
		foreach (cell, rb_cached)
		{
			RecordBatchState *rb_state = lfirst(cell);

			if (!arrowSchemaCompatibilityCheck(tupdesc, rb_state))
				elog(ERROR, "arrow file '%s' on behalf of foreign table '%s' has incompatible schema definition",
					 fname, RelationGetRelationName(relation));
			if (rb_state->rb_nitems == 0)
				continue;	/* not reasonable to sample, skipped */
			total_nrows += rb_state->rb_nitems;

			rb_state_list = lappend(rb_state_list, rb_state);
		}
	}
	nrooms = Min(nrooms, total_nrows);

	/* fetch samples for each record-batch */
	foreach (lc, rb_state_list)
	{
		RecordBatchState *rb_state = lfirst(lc);
		int			nsamples;

		count_nrows += rb_state->rb_nitems;
		nsamples = (double)nrooms * ((double)count_nrows /
									 (double)total_nrows) - nitems;
		if (nitems + nsamples > nrooms)
			nsamples = nrooms - nitems;
		if (nsamples > nsamples_min)
			nitems += RecordBatchAcquireSampleRows(relation,
												   rb_state,
												   rows + nitems,
												   nsamples);
	}
	foreach (lc, fdescList)
		FileClose((File)lfirst_int(lc));

	*p_totalrows = total_nrows;
	*p_totaldeadrows = 0.0;

	return nitems;
}

/*
 * ArrowAnalyzeForeignTable
 */
static bool
ArrowAnalyzeForeignTable(Relation frel,
						 AcquireSampleRowsFunc *p_sample_rows_func,
						 BlockNumber *p_totalpages)
{
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(frel));
	List		   *filesList = arrowFdwExtractFilesList(ft->options);
	ListCell	   *lc;
	Size			totalpages = 0;

	foreach (lc, filesList)
	{
		const char *fname = strVal(lfirst(lc));
		struct stat	statbuf;

		if (stat(fname, &statbuf) != 0)
		{
			elog(NOTICE, "failed on stat('%s') on behalf of '%s', skipped",
				 fname, get_rel_name(ft->relid));
			continue;
		}
		totalpages += (statbuf.st_size + BLCKSZ - 1) / BLCKSZ;
	}

	if (totalpages > MaxBlockNumber)
		totalpages = MaxBlockNumber;

	*p_sample_rows_func = ArrowAcquireSampleRows;
	*p_totalpages = totalpages;

	return true;
}

/*
 * ArrowImportForeignSchema
 */
static List *
ArrowImportForeignSchema(ImportForeignSchemaStmt *stmt, Oid serverOid)
{
	ArrowSchema	schema;
	List	   *filesList;
	ListCell   *lc;
	int			j;
	StringInfoData	cmd;

	/* sanity checks */
	switch (stmt->list_type)
	{
		case FDW_IMPORT_SCHEMA_ALL:
			break;
		case FDW_IMPORT_SCHEMA_LIMIT_TO:
			elog(ERROR, "arrow_fdw does not support LIMIT TO clause");
			break;
		case FDW_IMPORT_SCHEMA_EXCEPT:
			elog(ERROR, "arrow_fdw does not support EXCEPT clause");
			break;
		default:
			elog(ERROR, "arrow_fdw: Bug? unknown list-type");
			break;
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
		const char   *fname = strVal(lfirst(lc));
		ArrowFileInfo af_info;

		readArrowFile(fname, &af_info, false);
		if (lc == list_head(filesList))
		{
			copyArrowNode(&schema.node, &af_info.footer.schema.node);
		}
		else
		{
			/* compatibility checks */
			ArrowSchema	   *stemp = &af_info.footer.schema;

			if (schema.endianness != stemp->endianness ||
				schema._num_fields != stemp->_num_fields)
				elog(ERROR, "file '%s' has incompatible schema definition", fname);
			for (j=0; j < schema._num_fields; j++)
			{
				if (arrowFieldTypeIsEqual(&schema.fields[j],
										  &stemp->fields[j]))
					elog(ERROR, "file '%s' has incompatible schema definition", fname);
			}
		}
	}

	/* makes a command to define foreign table */
	initStringInfo(&cmd);
	appendStringInfo(&cmd, "CREATE FOREIGN TABLE %s (\n",
					 quote_identifier(stmt->remote_schema));
	for (j=0; j < schema._num_fields; j++)
	{
		ArrowField *field = &schema.fields[j];
		const char *type_name = arrowTypeToPGTypeName(field);

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
 * pgstrom_arrow_fdw_import_file
 *
 * NOTE: Due to historical reason, PostgreSQL does not allow to define
 * columns more than MaxHeapAttributeNumber (1600) for foreign-tables also,
 * not only heap-tables. This restriction comes from NULL-bitmap length
 * in HeapTupleHeaderData and width of t_hoff.
 * However, it is not a reasonable restriction for foreign-table, because
 * it does not use heap-format internally.
 */
static void
__insertPgAttributeTuple(Relation pg_attr_rel,
						 CatalogIndexState pg_attr_index,
						 Oid ftable_oid,
						 AttrNumber attnum,
						 ArrowField *field)
{
	Oid			type_oid;
	int32		type_mod;
	int16		type_len;
	bool		type_byval;
	char		type_align;
	int32		type_ndims;
	char		type_storage;
	Datum		values[Natts_pg_attribute];
	bool		isnull[Natts_pg_attribute];
	HeapTuple	tup;
	ObjectAddress myself, referenced;

	type_oid = arrowTypeToPGTypeOid(field, &type_mod);
	get_typlenbyvalalign(type_oid,
						 &type_len,
						 &type_byval,
						 &type_align);
	type_ndims = (type_is_array(type_oid) ? 1 : 0);
	type_storage = get_typstorage(type_oid);

	memset(values, 0, sizeof(values));
	memset(isnull, 0, sizeof(isnull));

	values[Anum_pg_attribute_attrelid - 1] = ObjectIdGetDatum(ftable_oid);
	values[Anum_pg_attribute_attname - 1] = CStringGetDatum(field->name);
	values[Anum_pg_attribute_atttypid - 1] = ObjectIdGetDatum(type_oid);
	values[Anum_pg_attribute_attstattarget - 1] = Int32GetDatum(-1);
	values[Anum_pg_attribute_attlen - 1] = Int16GetDatum(type_len);
	values[Anum_pg_attribute_attnum - 1] = Int16GetDatum(attnum);
	values[Anum_pg_attribute_attndims - 1] = Int32GetDatum(type_ndims);
	values[Anum_pg_attribute_attcacheoff - 1] = Int32GetDatum(-1);
	values[Anum_pg_attribute_atttypmod - 1] = Int32GetDatum(type_mod);
	values[Anum_pg_attribute_attbyval - 1] = BoolGetDatum(type_byval);
	values[Anum_pg_attribute_attstorage - 1] = CharGetDatum(type_storage);
	values[Anum_pg_attribute_attalign - 1] = CharGetDatum(type_align);
	values[Anum_pg_attribute_attnotnull - 1] = BoolGetDatum(!field->nullable);
	values[Anum_pg_attribute_attislocal - 1] = BoolGetDatum(true);
	isnull[Anum_pg_attribute_attacl - 1] = true;
	isnull[Anum_pg_attribute_attoptions - 1] = true;
	isnull[Anum_pg_attribute_attfdwoptions - 1] = true;
	isnull[Anum_pg_attribute_attmissingval - 1] = true;

	tup = heap_form_tuple(RelationGetDescr(pg_attr_rel), values, isnull);
	CatalogTupleInsertWithInfo(pg_attr_rel, tup, pg_attr_index);

	/* add dependency */
	myself.classId = RelationRelationId;
	myself.objectId = ftable_oid;
	myself.objectSubId = attnum;
	referenced.classId = TypeRelationId;
	referenced.objectId = type_oid;
	referenced.objectSubId = 0;
	recordDependencyOn(&myself, &referenced, DEPENDENCY_NORMAL);

	heap_freetuple(tup);
}

Datum
pgstrom_arrow_fdw_import_file(PG_FUNCTION_ARGS)
{
	CreateForeignTableStmt stmt;
	ArrowSchema	schema;
	List	   *tableElts = NIL;
	char	   *ftable_name;
	char	   *file_name;
	char	   *namespace_name;
	DefElem	   *defel;
	int			j, nfields;
	Oid			ftable_oid;
	Oid			type_oid;
	int			type_mod;
	ObjectAddress myself;
	ArrowFileInfo af_info;

	/* read schema of the file */
	if (PG_ARGISNULL(0))
		elog(ERROR, "foreign table name is not supplied");
	ftable_name = text_to_cstring(PG_GETARG_TEXT_PP(0));

	if (PG_ARGISNULL(1))
		elog(ERROR, "arrow filename is not supplied");
	file_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
	defel = makeDefElem("file", (Node *)makeString(file_name), -1);

	if (PG_ARGISNULL(2))
		namespace_name = NULL;
	else
		namespace_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

	readArrowFile(file_name, &af_info, false);
	copyArrowNode(&schema.node, &af_info.footer.schema.node);
	if (schema._num_fields > SHRT_MAX)
		Elog("Arrow file '%s' has too much fields: %d",
			 file_name, schema._num_fields);

	/* setup CreateForeignTableStmt */
	memset(&stmt, 0, sizeof(CreateForeignTableStmt));
	NodeSetTag(&stmt, T_CreateForeignTableStmt);
	stmt.base.relation = makeRangeVar(namespace_name, ftable_name, -1);

	nfields = Min(schema._num_fields, 100);
	for (j=0; j < nfields; j++)
	{
		ColumnDef  *cdef;

		type_oid = arrowTypeToPGTypeOid(&schema.fields[j], &type_mod);
		cdef = makeColumnDef(schema.fields[j].name,
							 type_oid,
							 type_mod,
							 InvalidOid);
		tableElts = lappend(tableElts, cdef);
	}
	stmt.base.tableElts = tableElts;
	stmt.base.oncommit = ONCOMMIT_NOOP;
	stmt.servername = "arrow_fdw";
	stmt.options = list_make1(defel);

	myself = DefineRelation(&stmt.base,
							RELKIND_FOREIGN_TABLE,
							InvalidOid,
							NULL,
							__FUNCTION__);
	ftable_oid = myself.objectId;
	CreateForeignTable(&stmt, ftable_oid);

	if (nfields < schema._num_fields)
	{
		Relation	c_rel = table_open(RelationRelationId, RowExclusiveLock);
		Relation	a_rel = table_open(AttributeRelationId, RowExclusiveLock);
		CatalogIndexState c_index = CatalogOpenIndexes(c_rel);
		CatalogIndexState a_index = CatalogOpenIndexes(a_rel);
		HeapTuple	tup;

		tup = SearchSysCacheCopy1(RELOID, ObjectIdGetDatum(ftable_oid));
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "cache lookup failed for relation %u", ftable_oid);
		
		for (j=nfields; j < schema._num_fields; j++)
		{
			__insertPgAttributeTuple(a_rel,
									 a_index,
									 ftable_oid,
									 j+1,
                                     &schema.fields[j]);
		}
		/* update relnatts also */
		((Form_pg_class) GETSTRUCT(tup))->relnatts = schema._num_fields;
		CatalogTupleUpdate(c_rel, &tup->t_self, tup);
		
		CatalogCloseIndexes(a_index);
		CatalogCloseIndexes(c_index);
		table_close(a_rel, RowExclusiveLock);
		table_close(c_rel, RowExclusiveLock);

		CommandCounterIncrement();
	}	
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_import_file);

/*
 * ArrowIsForeignScanParallelSafe
 */
static bool
ArrowIsForeignScanParallelSafe(PlannerInfo *root,
							   RelOptInfo *rel,
							   RangeTblEntry *rte)
{
	return true;
}

/*
 * ArrowEstimateDSMForeignScan 
 */
static Size
ArrowEstimateDSMForeignScan(ForeignScanState *node,
							ParallelContext *pcxt)
{
	return MAXALIGN(sizeof(pg_atomic_uint32) * 3);
}

/*
 * ArrowInitializeDSMForeignScan
 */
static inline void
__ExecInitDSMArrowFdw(ArrowFdwState *af_state,
					  pg_atomic_uint32 *rbatch_index,
					  pg_atomic_uint32 *rbatch_nload,
					  pg_atomic_uint32 *rbatch_nskip)
{
	pg_atomic_init_u32(rbatch_index, 0);
	af_state->rbatch_index = rbatch_index;
	pg_atomic_init_u32(rbatch_nload, 0);
	af_state->rbatch_nload = rbatch_nload;
	pg_atomic_init_u32(rbatch_nskip, 0);
	af_state->rbatch_nskip = rbatch_nskip;
}

void
ExecInitDSMArrowFdw(ArrowFdwState *af_state, GpuTaskSharedState *gtss)
{
	__ExecInitDSMArrowFdw(af_state,
						  &gtss->af_rbatch_index,
						  &gtss->af_rbatch_nload,
						  &gtss->af_rbatch_nskip);
}

static void
ArrowInitializeDSMForeignScan(ForeignScanState *node,
							  ParallelContext *pcxt,
							  void *coordinate)
{
	pg_atomic_uint32 *atomic_buffer = coordinate;

	__ExecInitDSMArrowFdw((ArrowFdwState *)node->fdw_state,
						  atomic_buffer,
						  atomic_buffer + 1,
						  atomic_buffer + 2);
}

/*
 * ArrowReInitializeDSMForeignScan
 */
static void
__ExecReInitDSMArrowFdw(ArrowFdwState *af_state)
{
	pg_atomic_write_u32(af_state->rbatch_index, 0);
}

void
ExecReInitDSMArrowFdw(ArrowFdwState *af_state)
{
	__ExecReInitDSMArrowFdw(af_state);
}


static void
ArrowReInitializeDSMForeignScan(ForeignScanState *node,
								ParallelContext *pcxt,
								void *coordinate)
{
	__ExecReInitDSMArrowFdw((ArrowFdwState *)node->fdw_state);
}

/*
 * ArrowInitializeWorkerForeignScan
 */
static inline void
__ExecInitWorkerArrowFdw(ArrowFdwState *af_state,
						 pg_atomic_uint32 *rbatch_index,
						 pg_atomic_uint32 *rbatch_nload,
						 pg_atomic_uint32 *rbatch_nskip)
{
	af_state->rbatch_index = rbatch_index;
	af_state->rbatch_nload = rbatch_nload;
	af_state->rbatch_nskip = rbatch_nskip;
}

void
ExecInitWorkerArrowFdw(ArrowFdwState *af_state,
					   GpuTaskSharedState *gtss)
{
	__ExecInitWorkerArrowFdw(af_state,
							 &gtss->af_rbatch_index,
							 &gtss->af_rbatch_nload,
							 &gtss->af_rbatch_nskip);
}

static void
ArrowInitializeWorkerForeignScan(ForeignScanState *node,
								 shm_toc *toc,
								 void *coordinate)
{
	pg_atomic_uint32 *atomic_buffer = coordinate;

	__ExecInitWorkerArrowFdw((ArrowFdwState *)node->fdw_state,
							 atomic_buffer,
							 atomic_buffer + 1,
							 atomic_buffer + 2);
}

/*
 * ArrowShutdownForeignScan
 */
static inline void
__ExecShutdownArrowFdw(ArrowFdwState *af_state)
{
	uint32		temp;

	temp = pg_atomic_read_u32(af_state->rbatch_index);
	pg_atomic_write_u32(&af_state->__rbatch_index_local, temp);
	af_state->rbatch_index = &af_state->__rbatch_index_local;

	temp = pg_atomic_read_u32(af_state->rbatch_nload);
	pg_atomic_write_u32(&af_state->__rbatch_nload_local, temp);
	af_state->rbatch_nload = &af_state->__rbatch_nload_local;

	temp = pg_atomic_read_u32(af_state->rbatch_nskip);
	pg_atomic_write_u32(&af_state->__rbatch_nskip_local, temp);
	af_state->rbatch_nskip = &af_state->__rbatch_nskip_local;
}

void
ExecShutdownArrowFdw(ArrowFdwState *af_state)
{
	__ExecShutdownArrowFdw(af_state);
}

static void
ArrowShutdownForeignScan(ForeignScanState *node)
{
	__ExecShutdownArrowFdw((ArrowFdwState *)node->fdw_state);
}

/*
 * ArrowPlanForeignModify
 */
static List *
ArrowPlanForeignModify(PlannerInfo *root,
					   ModifyTable *plan,
					   Index resultRelation,
					   int subplan_index)
{
	RangeTblEntry  *rte = planner_rt_fetch(resultRelation, root);
	ForeignTable   *ft = GetForeignTable(rte->relid);
	List		   *filesList	__attribute__((unused));
	bool			writable;

	if (plan->operation != CMD_INSERT)
		elog(ERROR, "not a supported operation on arrow_fdw foreign tables");

	filesList = __arrowFdwExtractFilesList(ft->options,
										   NULL,
										   &writable);
	if (!writable)
		elog(ERROR, "arrow_fdw: foreign table \"%s\" is not writable",
			 get_rel_name(rte->relid));
	Assert(list_length(filesList) == 1);

	return NIL;
}

/*
 * ArrowBeginForeignModify
 */
static void
__ArrowBeginForeignModify(ResultRelInfo *rrinfo, int eflags)
{
	Relation		frel = rrinfo->ri_RelationDesc;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(frel));
	List		   *filesList = arrowFdwExtractFilesList(ft->options);
	const char	   *fname;
	File			filp;
	struct stat		stat_buf;
	ArrowFileInfo  *af_info = NULL;
	arrowWriteState *aw_state;
	SQLtable	   *table;
	MetadataCacheKey key;
	off_t			f_pos;

	Assert(list_length(filesList) == 1);
	fname = strVal(linitial(filesList));

	LockRelation(frel, ShareRowExclusiveLock);
	filp = PathNameOpenFile(fname, O_RDWR | PG_BINARY);
	if (filp >= 0)
	{
		af_info = alloca(sizeof(ArrowFileInfo));
		readArrowFileDesc(FileGetRawDesc(filp), af_info);
		f_pos = createArrowWriteRedoLog(filp, false);
	}
	else if (errno == ENOENT)
	{
		filp = PathNameOpenFile(fname, O_RDWR | O_CREAT | O_EXCL | PG_BINARY);
		if (filp < 0)
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not open file \"%s\": %m", fname)));
		PG_TRY();
		{
			f_pos = createArrowWriteRedoLog(filp, true);
		}
		PG_CATCH();
		{
			unlink(fname);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	else
	{
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open file \"%s\": %m", fname)));
	}

	if (fstat(FileGetRawDesc(filp), &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", FilePathName(filp));
	initMetadataCacheKey(&key, &stat_buf);

	aw_state = palloc0(offsetof(arrowWriteState,
								sql_table.columns[tupdesc->natts]));
	aw_state->memcxt = CurrentMemoryContext;
	aw_state->file = filp;
	memcpy(&aw_state->key, &key, sizeof(MetadataCacheKey));
	aw_state->hash = key.hash;
	table = &aw_state->sql_table;
	table->filename = FilePathName(filp);
	table->fdesc = FileGetRawDesc(filp);
	table->f_pos = f_pos;
	if (af_info)
		setupArrowSQLbufferBatches(table, af_info);
	setupArrowSQLbufferSchema(table, tupdesc, af_info);

	rrinfo->ri_FdwState = aw_state;
}

static void
ArrowBeginForeignModify(ModifyTableState *mtstate,
						ResultRelInfo *rrinfo,
						List *fdw_private,
						int subplan_index,
						int eflags)
{
	__ArrowBeginForeignModify(rrinfo, eflags);
}

/*
 * ArrowExecForeignInsert
 */
static TupleTableSlot *
ArrowExecForeignInsert(EState *estate,
					   ResultRelInfo *rrinfo,
					   TupleTableSlot *slot,
					   TupleTableSlot *planSlot)
{
	Relation		frel = rrinfo->ri_RelationDesc;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	arrowWriteState *aw_state = rrinfo->ri_FdwState;
	SQLtable	   *table = &aw_state->sql_table;
	MemoryContext	oldcxt;
	size_t			usage = 0;
	int				j;

	slot_getallattrs(slot);
	oldcxt = MemoryContextSwitchTo(aw_state->memcxt);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
		SQLfield   *column = &table->columns[j];
		Datum		datum = slot->tts_values[j];
		bool		isnull = slot->tts_isnull[j];

		if (isnull)
		{
			usage += sql_field_put_value(column, NULL, 0);
		}
		else if (attr->attbyval)
		{
			Assert(column->sql_type.pgsql.typbyval);
			usage += sql_field_put_value(column, (char *)&datum, attr->attlen);
		}
		else if (attr->attlen == -1)
		{
			int		vl_len = VARSIZE_ANY_EXHDR(datum);
			char   *vl_ptr = VARDATA_ANY(datum);

			Assert(column->sql_type.pgsql.typlen == -1);
			usage += sql_field_put_value(column, vl_ptr, vl_len);
		}
		else
		{
			elog(ERROR, "Bug? unsupported type format");
		}
	}
	table->usage = usage;
	table->nitems++;
	MemoryContextSwitchTo(oldcxt);

	/*
	 * If usage exceeds the threshold of record-batch size, make a redo-log
	 * on demand, and write out the buffer.
	 */
	if (usage > table->segment_sz)
		writeOutArrowRecordBatch(aw_state, false);

	return slot;
}

/*
 * ArrowEndForeignModify
 */
static void
ArrowEndForeignModify(EState *estate,
					  ResultRelInfo *rrinfo)
{
	arrowWriteState *aw_state = rrinfo->ri_FdwState;

	writeOutArrowRecordBatch(aw_state, true);
}

#if PG_VERSION_NUM >= 110000
/*
 * MEMO: executor begin/end routine, if arrow_fdw is partitioned-leaf
 * relations. In this case, ArrowBeginForeignModify shall not be called.
 */
static void
ArrowBeginForeignInsert(ModifyTableState *mtstate,
						ResultRelInfo *rrinfo)
{
	__ArrowBeginForeignModify(rrinfo, 0);
}

static void
ArrowEndForeignInsert(EState *estate, ResultRelInfo *rrinfo)
{
	arrowWriteState *aw_state = rrinfo->ri_FdwState;

	writeOutArrowRecordBatch(aw_state, true);
}
#endif

/*
 * ArrowExplainForeignModify
 */
static void
ArrowExplainForeignModify(ModifyTableState *mtstate,
						  ResultRelInfo *rinfo,
						  List *fdw_private,
						  int subplan_index,
						  struct ExplainState *es)
{
	/* print something */
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
 * arrowFieldGetPGTypeHint
 */
static Oid
arrowFieldGetPGTypeHint(ArrowField *field)
{
	Oid		hint_oid = InvalidOid;
	int		i, j;

	for (i=0; i < field->_num_custom_metadata; i++)
	{
		ArrowKeyValue *kv = &field->custom_metadata[i];
		char	   *namebuf, *pos;
		Oid			namespace_oid;
		HeapTuple	tup;

		if (kv->_key_len != 7 || strncmp(kv->key, "pg_type", 7) != 0)
			continue;
		namebuf = alloca(kv->_value_len + 10);
		/* namespace name */
		pos = namebuf;
		for (j=0; j < kv->_value_len; j++)
		{
			int		c = kv->value[j];

			if (c == '.')
				break;
			else if (c == '\\' && ++j < kv->_value_len)
				c = kv->value[j];
			*pos++ = c;
		}
		*pos++ = '\0';

		namespace_oid = get_namespace_oid(namebuf, true);
		if (!OidIsValid(namespace_oid))
			continue;

		/* type name */
		pos = namebuf;
		for (j++; j < kv->_value_len; j++)
		{
			int		c = kv->value[j];

			if (c == '\\' && ++j < kv->_value_len)
				c = kv->value[j];
			*pos++ = c;
		}
		*pos++ = '\0';

		tup = SearchSysCache2(TYPENAMENSP,
							  PointerGetDatum(namebuf),
							  ObjectIdGetDatum(namespace_oid));
		if (!HeapTupleIsValid(tup))
			continue;
		hint_oid = PgTypeTupleGetOid(tup);

		ReleaseSysCache(tup);

		return hint_oid;
	}
	return InvalidOid;
}

static bool
__arrowStructTypeIsCompatible(ArrowField *field, Oid comp_oid)
{
	TupleDesc	tupdesc;
	int			j;
	bool		compatible = false;

	if (pg_type_aclcheck(comp_oid,
						 GetUserId(),
						 ACL_USAGE) != ACLCHECK_OK)
		return false;

	tupdesc = lookup_rowtype_tupdesc_noerror(comp_oid, -1, true);
	if (tupdesc && tupdesc->natts == field->_num_children)
	{
		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
			ArrowField *child = &field->children[j];
			Oid			typoid;
			int			typmod;

			typoid = arrowTypeToPGTypeOid(child, &typmod);
			if (typoid != attr->atttypid)
				break;
		}
		if (j >= tupdesc->natts)
			compatible = true;
	}
	if (tupdesc)
		ReleaseTupleDesc(tupdesc);

	return compatible;
}

static Oid
arrowTypeToPGTypeOid(ArrowField *field, int *p_type_mod)
{
	ArrowType  *t = &field->type;
	Oid			hint_oid;
	int			i;

	hint_oid = arrowFieldGetPGTypeHint(field);

	/* extra module may provide own mapping */
	for (i=0; i < pgstrom_num_users_extra; i++)
	{
		pgstromUsersExtraDescriptor *extra = &pgstrom_users_extra_desc[i];
		Oid		type_oid;

		if (extra->arrow_lookup_pgtype)
		{
			type_oid = extra->arrow_lookup_pgtype(field, hint_oid, p_type_mod);
			if (OidIsValid(type_oid))
				return type_oid;
		}
	}

	*p_type_mod = -1;
	switch (t->node.tag)
	{
		case ArrowNodeTag__Int:
			switch (t->Int.bitWidth)
			{
				case 8:
					return INT1OID;
				case 16:
					return INT2OID;
				case 32:
					return INT4OID;
				case 64:
					return INT8OID;
				default:
					elog(ERROR, "%s is not supported",
						 arrowNodeName(&t->node));
					break;
			}
			break;
		case ArrowNodeTag__FloatingPoint:
			switch (t->FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					return FLOAT2OID;
				case ArrowPrecision__Single:
					return FLOAT4OID;
				case ArrowPrecision__Double:
					return FLOAT8OID;
				default:
					elog(ERROR, "%s is not supported",
						 arrowNodeName(&t->node));
			}
			break;
		case ArrowNodeTag__Utf8:
			return TEXTOID;
		case ArrowNodeTag__Binary:
			return BYTEAOID;
		case ArrowNodeTag__Bool:
			return BOOLOID;
		case ArrowNodeTag__Decimal:
			if (t->Decimal.bitWidth == 128)
				return NUMERICOID;
			break;
		case ArrowNodeTag__Date:
			return DATEOID;
		case ArrowNodeTag__Time:
			return TIMEOID;
		case ArrowNodeTag__Timestamp:
			if (t->Timestamp.timezone)
				return TIMESTAMPTZOID;
			return TIMESTAMPOID;
		case ArrowNodeTag__Interval:
			return INTERVALOID;
		case ArrowNodeTag__List:
			if (field->_num_children != 1)
				elog(ERROR, "arrow_fdw: corrupted List type definition");
			else
			{
				ArrowField *child = &field->children[0];
				Oid		type_oid;
				Oid		elem_oid;
				int		elem_mod;

				elem_oid = arrowTypeToPGTypeOid(child, &elem_mod);
				type_oid = get_array_type(elem_oid);
				if (!OidIsValid(type_oid))
					elog(ERROR, "array of %s type is not defined",
						 arrowNodeName(&t->node));
				return type_oid;
			}
			break;

		case ArrowNodeTag__Struct:
			if (!OidIsValid(hint_oid) ||
				!__arrowStructTypeIsCompatible(field, hint_oid))
			{
				Relation	rel;
				ScanKeyData	skey[2];
				SysScanDesc	sscan;
				HeapTuple	tup;

				/*
				 * lookup composite type definition from pg_class
				 * At least, nattrs == _num_children
				 */
				rel = table_open(RelationRelationId, AccessShareLock);
				ScanKeyInit(&skey[0],
							Anum_pg_class_relkind,
							BTEqualStrategyNumber, F_CHAREQ,
							CharGetDatum(RELKIND_COMPOSITE_TYPE));
				ScanKeyInit(&skey[1],
							Anum_pg_class_relnatts,
							BTEqualStrategyNumber, F_INT2EQ,
							Int16GetDatum(field->_num_children));

				sscan = systable_beginscan(rel, InvalidOid, false,
										   NULL, 2, skey);
				hint_oid = InvalidOid;
				while (!OidIsValid(hint_oid) &&
					   HeapTupleIsValid(tup = systable_getnext(sscan)))
				{
					Oid		reltype = ((Form_pg_class) GETSTRUCT(tup))->reltype;

					if (__arrowStructTypeIsCompatible(field, reltype))
						hint_oid = reltype;
				}
				systable_endscan(sscan);
				table_close(rel, AccessShareLock);

				if (!OidIsValid(hint_oid))
					elog(ERROR, "arrow::%s is not supported",
						 arrowNodeName(&t->node));
			}
			return hint_oid;

		case ArrowNodeTag__FixedSizeBinary:
			if (t->FixedSizeBinary.byteWidth < 1 ||
				t->FixedSizeBinary.byteWidth > BLCKSZ)
				elog(ERROR, "arrow_fdw: %s with byteWidth=%d is not supported",
					 t->node.tagName,
					 t->FixedSizeBinary.byteWidth);
			if (hint_oid == MACADDROID &&
				t->FixedSizeBinary.byteWidth == sizeof(macaddr))
			{
				return MACADDROID;
			}
			else if (hint_oid == INETOID &&
					 (t->FixedSizeBinary.byteWidth == 4 ||
					  t->FixedSizeBinary.byteWidth == 16))
			{
				return INETOID;
			}
			*p_type_mod = t->FixedSizeBinary.byteWidth;
			return BPCHAROID;
		default:
			elog(ERROR, "arrow_fdw: type '%s' is not supported",
				 field->type.node.tagName);
	}
	return InvalidOid;
}

static const char *
arrowTypeToPGTypeName(ArrowField *field)
{
	Oid			typoid;
	int			typmod;
	HeapTuple	tup;
	Form_pg_type type;
	char	   *schema;
	char	   *result;

	typoid = arrowTypeToPGTypeOid(field, &typmod);
	if (!OidIsValid(typoid))
		return NULL;
	tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(typoid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for type %u", typoid);
	type = (Form_pg_type) GETSTRUCT(tup);
	schema = get_namespace_name(type->typnamespace);
	if (typmod < 0)
		result = psprintf("%s.%s",
						  quote_identifier(schema),
						  quote_identifier(NameStr(type->typname)));
	else
		result = psprintf("%s.%s(%d)",
						  quote_identifier(schema),
						  quote_identifier(NameStr(type->typname)),
						  typmod);
	ReleaseSysCache(tup);

	return result;
}

#if 0
//no longer needed?

/*
 * arrowTypeIsConvertible
 */
static bool
arrowTypeIsConvertible(Oid type_oid, int typemod)
{
	HeapTuple		tup;
	Form_pg_type	typeForm;
	bool			retval = false;

	switch (type_oid)
	{
		case INT1OID:		/* Int8 */
		case INT2OID:		/* Int16 */
		case INT4OID:		/* Int32 */
		case INT8OID:		/* Int64 */
		case FLOAT2OID:		/* FP16 */
		case FLOAT4OID:		/* FP32 */
		case FLOAT8OID:		/* FP64 */
		case TEXTOID:		/* Utf8 */
		case BYTEAOID:		/* Binary */
		case BOOLOID:		/* Bool */
		case NUMERICOID:	/* Decimal */
		case DATEOID:		/* Date */
		case TIMEOID:		/* Time */
		case TIMESTAMPOID:	/* Timestamp */
		case TIMESTAMPTZOID:/* TimestampTz */
		case INTERVALOID:	/* Interval */
		case BPCHAROID:		/* FixedSizeBinary */
			return true;
		default:
			tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
			if (!HeapTupleIsValid(tup))
				elog(ERROR, "cache lookup failed for type %u", type_oid);
			typeForm = (Form_pg_type) GETSTRUCT(tup);

			if (OidIsValid(typeForm->typelem) && typeForm->typlen == -1)
			{
				retval = arrowTypeIsConvertible(typeForm->typelem, typemod);
			}
			else if (typeForm->typtype == TYPTYPE_COMPOSITE)
			{
				Relation	rel;
				TupleDesc	tupdesc;
				int			j;

				rel = relation_open(typeForm->typrelid, AccessShareLock);
				tupdesc = RelationGetDescr(rel);
				for (j=0; j < tupdesc->natts; j++)
				{
					Form_pg_attribute	attr = tupleDescAttr(tupdesc, j);

					if (!arrowTypeIsConvertible(attr->atttypid,
												attr->atttypmod))
						break;
				}
				if (j >= tupdesc->natts)
					retval = true;
				relation_close(rel, AccessShareLock);
			}
			ReleaseSysCache(tup);
	}
	return retval;
}
#endif

/*
 * arrowFieldLength
 */
static size_t
arrowFieldLength(ArrowField *field, int64 nitems)
{
	ArrowType  *type = &field->type;
	size_t		length = 0;

	switch (type->node.tag)
	{
		case ArrowNodeTag__Int:
			switch (type->Int.bitWidth)
			{
				case 8:
					length = nitems;
					break;
				case 16:
					length = 2 * nitems;
					break;
				case 32:
					length = 4 * nitems;
					break;
				case 64:
					length = 8 * nitems;
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
		case ArrowNodeTag__List:
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
		case ArrowNodeTag__Struct:	//to be supported later
			length = 0;		/* only nullmap */
			break;
		case ArrowNodeTag__FixedSizeBinary:
			length = (size_t)type->FixedSizeBinary.byteWidth * nitems;
			break;
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
__arrowSchemaCompatibilityCheck(TupleDesc tupdesc,
								RecordBatchFieldState *rb_fstate)
{
	int		j;

	for (j=0; j < tupdesc->natts; j++)
	{
		RecordBatchFieldState *fstate = &rb_fstate[j];
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		if (!fstate->children)
		{
			/* shortcut, it should be a scalar built-in type */
			Assert(fstate->num_children == 0);
			if (attr->atttypid != fstate->atttypid)
				return false;
		}
		else
		{
			Form_pg_type	typ;
			HeapTuple		tup;
			bool			type_is_ok = true;

			tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(attr->atttypid));
			if (!HeapTupleIsValid(tup))
				elog(ERROR, "cache lookup failed for type %u", attr->atttypid);
			typ = (Form_pg_type) GETSTRUCT(tup);
			if (OidIsValid(typ->typelem) && typ->typlen == -1 &&
				fstate->num_children == 1)
			{
				/* Arrow::List */
				RecordBatchFieldState *cstate = &fstate->children[0];

				if (typ->typelem == cstate->atttypid)
				{
					/*
					 * overwrite typoid / typmod because a same arrow file
					 * can be reused, and it may be on behalf of different
					 * user defined data type.
					 */
					fstate->atttypid = attr->atttypid;
					fstate->atttypmod = attr->atttypmod;
				}
				else
				{
					type_is_ok = false;
				}
			}
			else if (typ->typlen == -1 && OidIsValid(typ->typrelid))
			{
				/* Arrow::Struct */
				TupleDesc	sdesc = lookup_rowtype_tupdesc(attr->atttypid,
														   attr->atttypmod);
				if (sdesc->natts == fstate->num_children &&
					__arrowSchemaCompatibilityCheck(sdesc, fstate->children))
				{
					/* see comment above */
					fstate->atttypid = attr->atttypid;
					fstate->atttypmod = attr->atttypmod;
				}
				else
				{
					type_is_ok = false;
				}
				DecrTupleDescRefCount(sdesc);

			}
			else
			{
				/* unknown */
				type_is_ok = false;
			}
			ReleaseSysCache(tup);
			if (!type_is_ok)
				return false;
		}
	}
	return true;
}

static bool
arrowSchemaCompatibilityCheck(TupleDesc tupdesc, RecordBatchState *rb_state)
{
	if (tupdesc->natts != rb_state->ncols)
		return false;
	return __arrowSchemaCompatibilityCheck(tupdesc, rb_state->columns);
}

/*
 * pg_XXX_arrow_ref
 */
static Datum
pg_varlena_arrow_ref(kern_data_store *kds,
					 kern_colmeta *cmeta, size_t index)
{
	cl_uint	   *offset = (cl_uint *)
		((char *)kds + __kds_unpack(cmeta->values_offset));
	char	   *extra = (char *)kds + __kds_unpack(cmeta->extra_offset);
	cl_uint		len;
	struct varlena *res;

	if (sizeof(uint32) * (index+2) > __kds_unpack(cmeta->values_length))
		elog(ERROR, "corruption? varlena index out of range");
	len = offset[index+1] - offset[index];
	if (offset[index] > offset[index+1] ||
		offset[index+1] > __kds_unpack(cmeta->extra_length))
		elog(ERROR, "corruption? varlena points out of extra buffer");

	res = palloc(VARHDRSZ + len);
	SET_VARSIZE(res, VARHDRSZ + len);
	memcpy(VARDATA(res), extra + offset[index], len);

	return PointerGetDatum(res);
}

static Datum
pg_bpchar_arrow_ref(kern_data_store *kds,
					kern_colmeta *cmeta, size_t index)
{
	cl_char	   *values = ((char *)kds + __kds_unpack(cmeta->values_offset));
	size_t		length = __kds_unpack(cmeta->values_length);
	cl_int		unitsz = cmeta->atttypmod - VARHDRSZ;
	struct varlena *res;

	if (unitsz <= 0)
		elog(ERROR, "CHAR(%d) is not expected", unitsz);
	if (unitsz * index >= length)
		elog(ERROR, "corruption? bpchar points out of range");
	res = palloc(VARHDRSZ + unitsz);
	memcpy((char *)res + VARHDRSZ, values + unitsz * index, unitsz);
	SET_VARSIZE(res, VARHDRSZ + unitsz);

	return PointerGetDatum(res);
}

static Datum
pg_bool_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	uint8  *bitmap = (uint8 *)kds + __kds_unpack(cmeta->values_offset);
	size_t	length = __kds_unpack(cmeta->values_length);
	uint8	mask = (1 << (index & 7));

	index >>= 3;
	if (sizeof(uint8) * index >= length)
		elog(ERROR, "corruption? bool points out of range");
	return BoolGetDatum((bitmap[index] & mask) != 0 ? true : false);
}

static Datum
pg_int1_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	int8   *values = (int8 *)((char *)kds + __kds_unpack(cmeta->values_offset));
	size_t	length = __kds_unpack(cmeta->values_length);

	if (sizeof(int8) * index >= length)
		elog(ERROR, "corruption? int8 points out of range");
	return values[index];
}

static Datum
pg_int2_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	int16  *values = (int16 *)((char *)kds + __kds_unpack(cmeta->values_offset));
	size_t	length = __kds_unpack(cmeta->values_length);

	if (sizeof(int16) * index >= length)
		elog(ERROR, "corruption? int16 points out of range");
	return values[index];
}

static Datum
pg_int4_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	int32  *values = (int32 *)((char *)kds + __kds_unpack(cmeta->values_offset));
	size_t  length = __kds_unpack(cmeta->values_length);

	if (sizeof(int32) * index >= length)
		elog(ERROR, "corruption? int32 points out of range");
	return values[index];
}

static Datum
pg_int8_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	int64  *values = (int64 *)((char *)kds + __kds_unpack(cmeta->values_offset));
	size_t	length = __kds_unpack(cmeta->values_length);

	if (sizeof(int64) * index >= length)
		elog(ERROR, "corruption? int64 points out of range");
	return values[index];
}

static Datum
pg_numeric_arrow_ref(kern_data_store *kds,
					 kern_colmeta *cmeta, size_t index)
{
	char	   *result = palloc0(sizeof(struct NumericData));
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	int			dscale = cmeta->attopts.decimal.scale;
	Int128_t	decimal;

	if (sizeof(int128) * index >= length)
		elog(ERROR, "corruption? numeric points out of range");
	decimal.ival = ((int128 *)base)[index];

	while (dscale > 0 && decimal.ival % 10 == 0)
	{
		decimal.ival /= 10;
		dscale--;
	}
	pg_numeric_to_varlena(result, dscale, decimal);

	return PointerGetDatum(result);
}

static Datum
pg_date_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	DateADT		dt;

	switch (cmeta->attopts.date.unit)
	{
		case ArrowDateUnit__Day:
			if (sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Date[day] points out of range");
			dt = ((uint32 *)base)[index];
			break;
		case ArrowDateUnit__MilliSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Date[ms] points out of range");
			dt = ((uint64 *)base)[index] / 1000;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Date type");
	}
	/* convert UNIX epoch to PostgreSQL epoch */
	dt -= (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
	return DateADTGetDatum(dt);
}

static Datum
pg_time_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	TimeADT		tm;

	switch (cmeta->attopts.time.unit)
	{
		case ArrowTimeUnit__Second:
			if (sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Time[sec] points out of range");
			tm = ((uint32 *)base)[index] * 1000000L;
			break;
		case ArrowTimeUnit__MilliSecond:
			if (sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Time[ms] points out of range");
			tm = ((uint32 *)base)[index] * 1000L;
			break;
		case ArrowTimeUnit__MicroSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Time[us] points out of range");
			tm = ((uint64 *)base)[index];
			break;
		case ArrowTimeUnit__NanoSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Time[ns] points out of range");
			tm = ((uint64 *)base)[index] / 1000L;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Time type");
			break;
	}
	return TimeADTGetDatum(tm);
}

static Datum
pg_timestamp_arrow_ref(kern_data_store *kds,
					   kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	Timestamp	ts;

	switch (cmeta->attopts.timestamp.unit)
	{
		case ArrowTimeUnit__Second:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Timestamp[sec] points out of range");
			ts = ((uint64 *)base)[index] * 1000000UL;
			break;
		case ArrowTimeUnit__MilliSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Timestamp[ms] points out of range");
			ts = ((uint64 *)base)[index] * 1000UL;
			break;
		case ArrowTimeUnit__MicroSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Timestamp[us] points out of range");
			ts = ((uint64 *)base)[index];
			break;
		case ArrowTimeUnit__NanoSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Timestamp[ns] points out of range");
			ts = ((uint64 *)base)[index] / 1000UL;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Timestamp type");
			break;
	}
	/* convert UNIX epoch to PostgreSQL epoch */
	ts -= (POSTGRES_EPOCH_JDATE -
		   UNIX_EPOCH_JDATE) * USECS_PER_DAY;
	return TimestampGetDatum(ts);
}

static Datum
pg_interval_arrow_ref(kern_data_store *kds,
					  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	Interval   *iv = palloc0(sizeof(Interval));

	switch (cmeta->attopts.interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			/* 32bit: number of months */
			if (sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Interval[Year/Month] points out of range");
			iv->month = ((uint32 *)base)[index];
			break;
		case ArrowIntervalUnit__Day_Time:
			/* 32bit+32bit: number of days and milliseconds */
			if (2 * sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Interval[Day/Time] points out of range");
			iv->day  = ((int32 *)base)[2 * index];
			iv->time = ((int32 *)base)[2 * index + 1] * 1000;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Interval type");
	}
	return PointerGetDatum(iv);
}

static Datum
pg_macaddr_arrow_ref(kern_data_store *kds,
					 kern_colmeta *cmeta, size_t index)
{
	char   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t	length = __kds_unpack(cmeta->values_length);

	if (cmeta->attopts.fixed_size_binary.byteWidth != sizeof(macaddr))
		elog(ERROR, "Bug? wrong FixedSizeBinary::byteWidth(%d) for macaddr",
			 cmeta->attopts.fixed_size_binary.byteWidth);
	if (sizeof(macaddr) * index >= length)
		elog(ERROR, "corruption? Binary[macaddr] points out of range");

	return PointerGetDatum(base + sizeof(macaddr) * index);
}

static Datum
pg_inet_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t	length = __kds_unpack(cmeta->values_length);
	inet   *ip = palloc(sizeof(inet));

	if (cmeta->attopts.fixed_size_binary.byteWidth == 4)
	{
		if (4 * index >= length)
			elog(ERROR, "corruption? Binary[inet4] points out of range");
		ip->inet_data.family = PGSQL_AF_INET;
		ip->inet_data.bits = 32;
		memcpy(ip->inet_data.ipaddr, base + 4 * index, 4);
	}
	else if (cmeta->attopts.fixed_size_binary.byteWidth == 16)
	{
		if (16 * index >= length)
			elog(ERROR, "corruption? Binary[inet6] points out of range");
		ip->inet_data.family = PGSQL_AF_INET6;
		ip->inet_data.bits = 128;
		memcpy(ip->inet_data.ipaddr, base + 16 * index, 16);
	}
	else
		elog(ERROR, "Bug? wrong FixedSizeBinary::byteWidth(%d) for inet",
			 cmeta->attopts.fixed_size_binary.byteWidth);

	SET_INET_VARSIZE(ip);
	return PointerGetDatum(ip);
}

static Datum
pg_array_arrow_ref(kern_data_store *kds,
				   kern_colmeta *smeta,
				   cl_uint start, cl_uint end)
{
	ArrayType  *res;
	size_t		sz;
	cl_uint		i, nitems = end - start;
	bits8	   *nullmap = NULL;
	size_t		usage, __usage;

	/* sanity checks */
	if (start > end)
		elog(ERROR, "Bug? array index has reversed order [%u..%u]", start, end);

	/* allocation of the result buffer */
	if (smeta->nullmap_offset != 0)
		sz = ARR_OVERHEAD_WITHNULLS(1, nitems);
	else
		sz = ARR_OVERHEAD_NONULLS(1);

	if (smeta->attlen > 0)
	{
		sz += TYPEALIGN(smeta->attalign,
						smeta->attlen) * nitems;
	}
	else if (smeta->attlen == -1)
	{
		sz += 400;		/* tentative allocation */
	}
	else
		elog(ERROR, "Bug? corrupted kernel column metadata");

	res = palloc0(sz);
	res->ndim = 1;
	if (smeta->nullmap_offset != 0)
	{
		res->dataoffset = ARR_OVERHEAD_WITHNULLS(1, nitems);
		nullmap = ARR_NULLBITMAP(res);
	}
	res->elemtype = smeta->atttypid;
	ARR_DIMS(res)[0] = nitems;
	ARR_LBOUND(res)[0] = 1;
	usage = ARR_DATA_OFFSET(res);
	for (i=0; i < nitems; i++)
	{
		Datum	datum;
		bool	isnull;

		pg_datum_arrow_ref(kds, smeta, start+i, &datum, &isnull);
		if (isnull)
		{
			if (!nullmap)
				elog(ERROR, "Bug? element item should not be NULL");
		}
		else if (smeta->attlen > 0)
		{
			if (nullmap)
				nullmap[i>>3] |= (1<<(i&7));
			__usage = TYPEALIGN(smeta->attalign, usage);
			while (__usage + smeta->attlen > sz)
			{
				sz += sz;
				res = repalloc(res, sz);
			}
			if (__usage > usage)
				memset((char *)res + usage, 0, __usage - usage);
			memcpy((char *)res + __usage, &datum, smeta->attlen);
			usage = __usage + smeta->attlen;
		}
		else if (smeta->attlen == -1)
		{
			cl_int		vl_len = VARSIZE(datum);

			if (nullmap)
				nullmap[i>>3] |= (1<<(i&7));
			__usage = TYPEALIGN(smeta->attalign, usage);
			while (__usage + vl_len > sz)
			{
				sz += sz;
				res = repalloc(res, sz);
			}
			if (__usage > usage)
				memset((char *)res + usage, 0, __usage - usage);
			memcpy((char *)res + __usage, DatumGetPointer(datum), vl_len);
			usage = __usage + vl_len;

			pfree(DatumGetPointer(datum));
		}
		else
			elog(ERROR, "Bug? corrupted kernel column metadata");
	}
	SET_VARSIZE(res, usage);

	return PointerGetDatum(res);
}

/*
 * pg_datum_arrow_ref
 */
static void
pg_datum_arrow_ref(kern_data_store *kds,
				   kern_colmeta *cmeta,
				   size_t index,
				   Datum *p_datum,
				   bool *p_isnull)
{
	Datum		datum = 0;
	bool		isnull = true;

	if (cmeta->nullmap_offset != 0)
	{
		size_t	nullmap_offset = __kds_unpack(cmeta->nullmap_offset);
		uint8  *nullmap = (uint8 *)kds + nullmap_offset;

		if (att_isnull(index, nullmap))
			goto out;
	}

	if (cmeta->atttypkind == TYPE_KIND__ARRAY)
	{
		/* array type */
		kern_colmeta   *smeta;
		uint32		   *offset;

		if (cmeta->num_subattrs != 1 ||
			cmeta->idx_subattrs < kds->ncols ||
			cmeta->idx_subattrs >= kds->nr_colmeta)
			elog(ERROR, "Bug? corrupted kernel column metadata");
		if (sizeof(uint32) * (index+2) > __kds_unpack(cmeta->values_length))
			elog(ERROR, "Bug? array index is out of range");
		smeta = &kds->colmeta[cmeta->idx_subattrs];
		offset = (uint32 *)((char *)kds + __kds_unpack(cmeta->values_offset));
		datum = pg_array_arrow_ref(kds, smeta,
								   offset[index],
								   offset[index+1]);
		isnull = false;
	}
	else if (cmeta->atttypkind == TYPE_KIND__COMPOSITE)
	{
		/* composite type */
		TupleDesc	tupdesc = lookup_rowtype_tupdesc(cmeta->atttypid, -1);
		Datum	   *sub_values = alloca(sizeof(Datum) * tupdesc->natts);
		bool	   *sub_isnull = alloca(sizeof(bool)  * tupdesc->natts);
		HeapTuple	htup;
		int			j;

		if (tupdesc->natts != cmeta->num_subattrs)
			elog(ERROR, "Struct definition is conrrupted?");
		if (cmeta->idx_subattrs < kds->ncols ||
			cmeta->idx_subattrs + cmeta->num_subattrs > kds->nr_colmeta)
			elog(ERROR, "Bug? strange kernel column metadata");

		for (j=0; j < tupdesc->natts; j++)
		{
			kern_colmeta *sub_meta = &kds->colmeta[cmeta->idx_subattrs + j];

			pg_datum_arrow_ref(kds, sub_meta, index,
							   sub_values + j,
							   sub_isnull + j);
		}
		htup = heap_form_tuple(tupdesc, sub_values, sub_isnull);

		ReleaseTupleDesc(tupdesc);

		datum = PointerGetDatum(htup->t_data);
		isnull = false;
	}
	else if (cmeta->atttypkind != TYPE_KIND__NULL)
	{
		/* anything else, except for unreferenced column */
		int		i;

		switch (cmeta->atttypid)
		{
			case INT1OID:
				datum = pg_int1_arrow_ref(kds, cmeta, index);
				break;
			case INT2OID:
			case FLOAT2OID:
				datum = pg_int2_arrow_ref(kds, cmeta, index);
				break;
			case INT4OID:
			case FLOAT4OID:
				datum = pg_int4_arrow_ref(kds, cmeta, index);
				break;
			case INT8OID:
			case FLOAT8OID:
				datum = pg_int8_arrow_ref(kds, cmeta, index);
				break;
			case TEXTOID:
			case BYTEAOID:
				datum = pg_varlena_arrow_ref(kds, cmeta, index);
				break;
			case BPCHAROID:
				datum = pg_bpchar_arrow_ref(kds, cmeta, index);
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
			case TIMESTAMPTZOID:
				datum = pg_timestamp_arrow_ref(kds, cmeta, index);
				break;
			case INTERVALOID:
				datum = pg_interval_arrow_ref(kds, cmeta, index);
				break;
			case MACADDROID:
				datum = pg_macaddr_arrow_ref(kds, cmeta, index);
				break;
			case INETOID:
				datum = pg_inet_arrow_ref(kds, cmeta, index);
				break;
			default:
				for (i=0; i < pgstrom_num_users_extra; i++)
				{
					pgstromUsersExtraDescriptor *extra = &pgstrom_users_extra_desc[i];

					if (extra->arrow_datum_ref &&
						extra->arrow_datum_ref(kds, cmeta, index, &datum, &isnull))
					{
						goto out;
					}
				}
				elog(ERROR, "Bug? unexpected datum type: %u", cmeta->atttypid);
				break;
		}
		isnull = false;
	}
out:
	*p_datum  = datum;
	*p_isnull = isnull;
}

/*
 * KDS_fetch_tuple_arrow
 */
bool
KDS_fetch_tuple_arrow(TupleTableSlot *slot,
					  kern_data_store *kds,
					  size_t index)
{
	int		j;

	if (index >= kds->nitems)
		return false;
	ExecStoreAllNullTuple(slot);
	for (j=0; j < kds->ncols; j++)
	{
		kern_colmeta   *cmeta = &kds->colmeta[j];

		pg_datum_arrow_ref(kds, cmeta,
						   index,
						   slot->tts_values + j,
						   slot->tts_isnull + j);
	}
	return true;
}

/*
 * arrowFdwExtractFilesList
 */
static List *
__arrowFdwExtractFilesList(List *options_list,
						   int *p_parallel_nworkers,
						   bool *p_writable)
{
	ListCell   *lc;
	List	   *filesList = NIL;
	char	   *dir_path = NULL;
	char	   *dir_suffix = NULL;
	int			parallel_nworkers = -1;
	bool		writable = false;	/* default: read-only */

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
		else if (strcmp(defel->defname, "parallel_workers") == 0)
		{
			if (parallel_nworkers >= 0)
				elog(ERROR, "'parallel_workers' appeared twice");
			parallel_nworkers = atoi(strVal(defel->arg));
		}
		else if (strcmp(defel->defname, "writable") == 0)
		{
			writable = defGetBoolean(defel);
		}
		else
			elog(ERROR, "arrow: unknown option (%s)", defel->defname);
	}
	if (dir_suffix && !dir_path)
		elog(ERROR, "arrow: cannot use 'suffix' option without 'dir'");

	if (writable)
	{
		if (dir_path)
			elog(ERROR, "arrow: 'dir_path' and 'writable' options are exclusive");
		if (list_length(filesList) == 0)
			elog(ERROR, "arrow: 'writable' needs a backend file specified by 'file' option");
		if (list_length(filesList) > 1)
			elog(ERROR, "arrow: 'writable' cannot use multiple backend files");
	}

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
		elog(ERROR, "no files are configured on behalf of the arrow_fdw foreign table");
	foreach (lc, filesList)
	{
		const char *fname = strVal(lfirst(lc));

		if (!writable)
		{
			if (access(fname, R_OK) != 0)
				elog(ERROR, "unable to read '%s': %m", fname);
		}
		else
		{
			if (access(fname, R_OK | W_OK) != 0)
			{
				if (errno != ENOENT)
					elog(ERROR, "unable to read/write '%s': %m", fname);
				else
				{
					char   *temp = pstrdup(fname);
					char   *dname = dirname(temp);

					if (access(dname, R_OK | W_OK | X_OK) != 0)
						elog(ERROR, "unable to create '%s': %m", fname);
					pfree(temp);
				}
			}
		}
	}
	/* other properties */
	if (p_parallel_nworkers)
		*p_parallel_nworkers = parallel_nworkers;
	if (p_writable)
		*p_writable = writable;

	return filesList;
}

static List *
arrowFdwExtractFilesList(List *options_list)
{
	return __arrowFdwExtractFilesList(options_list, NULL, NULL);
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
		List	   *filesList;
		ListCell   *lc;

		filesList = arrowFdwExtractFilesList(options_list);
		foreach (lc, filesList)
		{
			ArrowFileInfo	af_info;
			const char	   *fname = strVal(lfirst(lc));

			readArrowFile(fname, &af_info, true);
		}
	}
	else if (options_list != NIL)
	{
		const char	   *label;
		char			temp[80];

		switch (catalog)
		{
			case ForeignDataWrapperRelationId:
				label = "FOREIGN DATA WRAPPER";
				break;
			case ForeignServerRelationId:
				label = "SERVER";
				break;
			case UserMappingRelationId:
				label = "USER MAPPING";
				break;
			case AttributeRelationId:
				label = "attribute of FOREIGN TABLE";
				break;
			default:
				snprintf(temp, sizeof(temp),
						 "[unexpected object catalog=%u]", catalog);
				label = temp;
				break;
		}
		elog(ERROR, "Arrow_Fdw does not support any options for %s", label);
	}
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_validator);

/*
 * pgstrom_arrow_fdw_precheck_schema
 */
static void
arrow_fdw_precheck_schema(Relation rel)
{
	TupleDesc		tupdesc = RelationGetDescr(rel);
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(rel));
	List		   *filesList;
	ListCell	   *lc;
	bool			writable;
#if 0
	int				j;

	/* check schema definition is supported by Apache Arrow */
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute	attr = tupleDescAttr(tupdesc, j);

		if (!arrowTypeIsConvertible(attr->atttypid,
									attr->atttypmod))
			elog(ERROR, "column %s of foreign table %s has %s type that is not convertible any supported Apache Arrow types",
				 NameStr(attr->attname),
				 RelationGetRelationName(rel),
				 format_type_be(attr->atttypid));
	}
#endif
	filesList = __arrowFdwExtractFilesList(ft->options,
										   NULL,
										   &writable);
	foreach (lc, filesList)
	{
		const char *fname = strVal(lfirst(lc));
		File		filp;
		List	   *rb_cached = NIL;
		ListCell   *cell;

		filp = PathNameOpenFile(fname, O_RDONLY | PG_BINARY);
		if (filp < 0)
		{
			if (writable && errno == ENOENT)
				continue;
			elog(ERROR, "failed to open '%s' on behalf of '%s': %m",
				 fname, RelationGetRelationName(rel));
		}
		/* check schema compatibility */
		rb_cached = arrowLookupOrBuildMetadataCache(filp, NULL);
		foreach (cell, rb_cached)
		{
			RecordBatchState *rb_state = lfirst(cell);

			if (!arrowSchemaCompatibilityCheck(tupdesc, rb_state))
				elog(ERROR, "arrow file '%s' on behalf of the foreign table '%s' has incompatible schema definition",
					 fname, RelationGetRelationName(rel));
		}
		list_free(rb_cached);
	}
}

Datum
pgstrom_arrow_fdw_precheck_schema(PG_FUNCTION_ARGS)
{
	EventTriggerData   *trigdata;
	
	if (!CALLED_AS_EVENT_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as EventTrigger",
			 __FUNCTION__);
	trigdata = (EventTriggerData *) fcinfo->context;
	if (strcmp(trigdata->event, "ddl_command_end") != 0)
		elog(ERROR, "%s: must be called on ddl_command_end event",
			 __FUNCTION__);
	if (strcmp(GetCommandTagName(trigdata->tag),
			   "CREATE FOREIGN TABLE") == 0)
	{
		CreateStmt	   *stmt = (CreateStmt *)trigdata->parsetree;
		Relation		rel;

		rel = relation_openrv_extended(stmt->relation, AccessShareLock, true);
		if (!rel)
			PG_RETURN_NULL();
		if (rel->rd_rel->relkind == RELKIND_FOREIGN_TABLE &&
			GetFdwRoutineForRelation(rel, false) == &pgstrom_arrow_fdw_routine)
		{
			arrow_fdw_precheck_schema(rel);
		}
		relation_close(rel, AccessShareLock);
	}
	else if (strcmp(GetCommandTagName(trigdata->tag),
					"ALTER FOREIGN TABLE") == 0 &&
			 IsA(trigdata->parsetree, AlterTableStmt))
	{
		AlterTableStmt *stmt = (AlterTableStmt *)trigdata->parsetree;
		Relation		rel;
		ListCell	   *lc;
		bool			has_schema_change = false;

		rel = relation_openrv_extended(stmt->relation, AccessShareLock, true);
		if (!rel)
			PG_RETURN_NULL();
		if (rel->rd_rel->relkind == RELKIND_FOREIGN_TABLE &&
			GetFdwRoutineForRelation(rel, false) == &pgstrom_arrow_fdw_routine)
		{
			foreach (lc, stmt->cmds)
			{
				AlterTableCmd  *cmd = lfirst(lc);

				if (cmd->subtype == AT_AddColumn ||
					cmd->subtype == AT_DropColumn ||
					cmd->subtype == AT_AlterColumnType)
				{
					has_schema_change = true;
					break;
				}
			}
			if (has_schema_change)
				arrow_fdw_precheck_schema(rel);
		}
		relation_close(rel, AccessShareLock);
	}
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_precheck_schema);

/*
 * arrowInvalidateMetadataCache
 *
 * NOTE: caller must have lock_slots[] with EXCLUSIVE mode
 */
static uint64
__arrowInvalidateMetadataCache(arrowMetadataCache *mcache, bool detach_lru)
{
	arrowMetadataCache *mtemp;
	dlist_node	   *dnode;
	uint64			released = 0;

	while (!dlist_is_empty(&mcache->siblings))
	{
		dnode = dlist_pop_head_node(&mcache->siblings);
		mtemp = dlist_container(arrowMetadataCache, chain, dnode);
		Assert(dlist_is_empty(&mtemp->siblings) &&
			   !mtemp->lru_chain.prev && !mtemp->lru_chain.next);
		dlist_delete(&mtemp->chain);
		released += MAXALIGN(offsetof(arrowMetadataCache,
									  fstate[mtemp->nfields]));
		pfree(mtemp);
	}
	released += MAXALIGN(offsetof(arrowMetadataCache,
								  fstate[mcache->nfields]));
	if (detach_lru)
	{
		SpinLockAcquire(&arrow_metadata_state->lru_lock);
		dlist_delete(&mcache->lru_chain);
		SpinLockRelease(&arrow_metadata_state->lru_lock);
	}
	dlist_delete(&mcache->chain);
	pfree(mcache);

	return pg_atomic_sub_fetch_u64(&arrow_metadata_state->consumed, released);
}

static void
arrowInvalidateMetadataCache(MetadataCacheKey *mkey, bool detach_lru)
{
	dlist_mutable_iter miter;
	int		index = mkey->hash % ARROW_METADATA_HASH_NSLOTS;

	dlist_foreach_modify(miter, &arrow_metadata_state->hash_slots[index])
	{
		arrowMetadataCache *mcache
			= dlist_container(arrowMetadataCache, chain, miter.cur);

		if (mcache->stat_buf.st_dev == mkey->st_dev &&
			mcache->stat_buf.st_ino == mkey->st_ino)
		{
			elog(DEBUG2, "arrow_fdw: metadata cache invalidation for the file (st_dev=%lu/st_ino=%lu)",
				 mkey->st_dev, mkey->st_ino);
			__arrowInvalidateMetadataCache(mcache, true);
		}
	}
}

/*
 * copyMetadataFieldCache - copy for nested structure
 */
static int
copyMetadataFieldCache(RecordBatchFieldState *dest_curr,
					   RecordBatchFieldState *dest_tail,
					   int nattrs,
					   RecordBatchFieldState *columns,
					   Bitmapset **p_stat_attrs)
{
	RecordBatchFieldState *dest_next = dest_curr + nattrs;
	int		j, k, nslots = nattrs;

	if (dest_next > dest_tail)
		return -1;

	for (j=0; j < nattrs; j++)
	{
		RecordBatchFieldState *__dest = dest_curr + j;
		RecordBatchFieldState *__orig = columns + j;

		memcpy(__dest, __orig, sizeof(RecordBatchFieldState));
		if (__dest->num_children == 0)
			Assert(__dest->children == NULL);
		else
		{
			__dest->children = dest_next;
			k = copyMetadataFieldCache(dest_next,
									   dest_tail,
									   __orig->num_children,
									   __orig->children,
									   NULL);
			if (k < 0)
				return -1;
			dest_next += k;
			nslots += k;
		}
		if (p_stat_attrs && !__orig->stat_isnull)
			*p_stat_attrs = bms_add_member(*p_stat_attrs, j+1);
	}
	return nslots;
}

/*
 * makeRecordBatchStateFromCache
 *   - setup RecordBatchState from arrowMetadataCache
 */
static RecordBatchState *
makeRecordBatchStateFromCache(arrowMetadataCache *mcache,
							  File fdesc,
							  Bitmapset **p_stat_attrs)
{
	RecordBatchState   *rbstate;

	rbstate = palloc0(offsetof(RecordBatchState,
							   columns[mcache->nfields]));
	rbstate->fdesc = fdesc;
	memcpy(&rbstate->stat_buf, &mcache->stat_buf, sizeof(struct stat));
	rbstate->rb_index  = mcache->rb_index;
	rbstate->rb_offset = mcache->rb_offset;
	rbstate->rb_length = mcache->rb_length;
	rbstate->rb_nitems = mcache->rb_nitems;
	rbstate->ncols = mcache->ncols;
	copyMetadataFieldCache(rbstate->columns,
						   rbstate->columns + mcache->nfields,
						   mcache->ncols,
						   mcache->fstate,
						   p_stat_attrs);
	return rbstate;
}

/*
 * arrowReclaimMetadataCache
 */
static void
arrowReclaimMetadataCache(void)
{
	arrowMetadataCache *mcache;
	LWLock	   *lock = NULL;
	dlist_node *dnode;
	uint32		lru_hash;
	uint32		lru_index;
	uint64		consumed;

	consumed = pg_atomic_read_u64(&arrow_metadata_state->consumed);
	if (consumed <= arrow_metadata_cache_size)
		return;

	SpinLockAcquire(&arrow_metadata_state->lru_lock);
	if (dlist_is_empty(&arrow_metadata_state->lru_list))
	{
		SpinLockRelease(&arrow_metadata_state->lru_lock);
		return;
	}
	dnode = dlist_tail_node(&arrow_metadata_state->lru_list);
	mcache = dlist_container(arrowMetadataCache, lru_chain, dnode);
	lru_hash = mcache->hash;
	SpinLockRelease(&arrow_metadata_state->lru_lock);

	do {
		lru_index = lru_hash % ARROW_METADATA_HASH_NSLOTS;
		lock = &arrow_metadata_state->lock_slots[lru_index];

		LWLockAcquire(lock, LW_EXCLUSIVE);
		SpinLockAcquire(&arrow_metadata_state->lru_lock);
		if (dlist_is_empty(&arrow_metadata_state->lru_list))
		{
			SpinLockRelease(&arrow_metadata_state->lru_lock);
			LWLockRelease(lock);
			break;
		}
		dnode = dlist_tail_node(&arrow_metadata_state->lru_list);
		mcache = dlist_container(arrowMetadataCache, lru_chain, dnode);
		if (mcache->hash == lru_hash)
		{
			dlist_delete(&mcache->lru_chain);
			memset(&mcache->lru_chain, 0, sizeof(dlist_node));
			SpinLockRelease(&arrow_metadata_state->lru_lock);
			consumed = __arrowInvalidateMetadataCache(mcache, false);
		}
		else
		{
			/* LRU-tail was referenced by someone, try again */
			lru_hash = mcache->hash;
            SpinLockRelease(&arrow_metadata_state->lru_lock);
		}
		LWLockRelease(lock);
	} while (consumed > arrow_metadata_cache_size);
}

/*
 * __arrowBuildMetadataCache
 *
 * NOTE: caller must have exclusive lock on arrow_metadata_state->lock_slots[]
 */
static arrowMetadataCache *
__arrowBuildMetadataCache(List *rb_state_list, uint32 hash)
{
	arrowMetadataCache *mcache = NULL;
	arrowMetadataCache *mtemp;
	dlist_node *dnode;
	Size		sz, consumed = 0;
	int			nfields;
	ListCell   *lc;

	foreach (lc, rb_state_list)
	{
		RecordBatchState *rbstate = lfirst(lc);

		if (!mcache)
			nfields = RecordBatchFieldCount(rbstate);
		else
			Assert(nfields == RecordBatchFieldCount(rbstate));

		sz = offsetof(arrowMetadataCache, fstate[nfields]);
		mtemp = MemoryContextAllocZero(TopSharedMemoryContext, sz);
		if (!mtemp)
		{
			/* !!out of memory!! */
			if (mcache)
			{
				while (!dlist_is_empty(&mcache->siblings))
				{
					dnode = dlist_pop_head_node(&mcache->siblings);
					mtemp = dlist_container(arrowMetadataCache,
											chain, dnode);
					pfree(mtemp);
				}
				pfree(mcache);
			}
			return NULL;
		}

		dlist_init(&mtemp->siblings);
		memcpy(&mtemp->stat_buf, &rbstate->stat_buf, sizeof(struct stat));
		mtemp->hash      = hash;
		mtemp->rb_index  = rbstate->rb_index;
        mtemp->rb_offset = rbstate->rb_offset;
        mtemp->rb_length = rbstate->rb_length;
        mtemp->rb_nitems = rbstate->rb_nitems;
        mtemp->ncols     = rbstate->ncols;
		mtemp->nfields   =
			copyMetadataFieldCache(mtemp->fstate,
								   mtemp->fstate + nfields,
								   rbstate->ncols,
								   rbstate->columns,
								   NULL);
		Assert(mtemp->nfields == nfields);

		if (!mcache)
			mcache = mtemp;
		else
			dlist_push_tail(&mcache->siblings, &mtemp->chain);
		consumed += MAXALIGN(sz);
	}
	pg_atomic_add_fetch_u64(&arrow_metadata_state->consumed, consumed);

	return mcache;
}


/*
 * checkArrowRecordBatchIsVisible
 *
 * NOTE: It must be called under shared lock on lock_slots[]
 */
static bool
checkArrowRecordBatchIsVisible(RecordBatchState *rbstate,
							   dlist_head *mvcc_slot)
{
	dlist_iter		iter;

	dlist_foreach(iter, mvcc_slot)
	{
		arrowWriteMVCCLog  *mvcc = dlist_container(arrowWriteMVCCLog,
												   chain, iter.cur);
		if (mvcc->key.st_dev == rbstate->stat_buf.st_dev &&
			mvcc->key.st_ino == rbstate->stat_buf.st_ino &&
			mvcc->record_batch == rbstate->rb_index)
		{
			if (TransactionIdIsCurrentTransactionId(mvcc->xid))
				return true;
			else
				return false;
		}
	}
	return true;
}

/*
 * arrowLookupOrBuildMetadataCache
 */
List *
arrowLookupOrBuildMetadataCache(File fdesc, Bitmapset **p_stat_attrs)
{
	MetadataCacheKey key;
	struct stat	stat_buf;
	uint32		index;
	LWLock	   *lock;
	dlist_head *hash_slot;
	dlist_head *mvcc_slot;
	dlist_iter	iter1, iter2;
	bool		has_exclusive = false;
	List	   *results = NIL;

	if (fstat(FileGetRawDesc(fdesc), &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", FilePathName(fdesc));

	index = initMetadataCacheKey(&key, &stat_buf);
	lock = &arrow_metadata_state->lock_slots[index];
	hash_slot = &arrow_metadata_state->hash_slots[index];
	mvcc_slot = &arrow_metadata_state->mvcc_slots[index];

	LWLockAcquire(lock, LW_SHARED);
retry:
	dlist_foreach(iter1, hash_slot)
	{
	   arrowMetadataCache *mcache
			= dlist_container(arrowMetadataCache, chain, iter1.cur);
		if (mcache->stat_buf.st_dev == stat_buf.st_dev &&
			mcache->stat_buf.st_ino == stat_buf.st_ino)
		{
			RecordBatchState *rbstate;

			Assert(mcache->hash == key.hash);
			if (timespec_comp(&mcache->stat_buf.st_mtim,
							  &stat_buf.st_mtim) < 0 ||
				timespec_comp(&mcache->stat_buf.st_ctim,
							  &stat_buf.st_ctim) < 0)
			{
				char	buf1[80], buf2[80], buf3[80], buf4[80];
				char   *tail;

				if (!has_exclusive)
				{
					LWLockRelease(lock);
					LWLockAcquire(lock, LW_EXCLUSIVE);
					has_exclusive = true;
					goto retry;
				}
				ctime_r(&mcache->stat_buf.st_mtime, buf1);
				ctime_r(&mcache->stat_buf.st_ctime, buf2);
				ctime_r(&stat_buf.st_mtime, buf3);
				ctime_r(&stat_buf.st_ctime, buf4);
				for (tail=buf1+strlen(buf1)-1; isspace(*tail); *tail--='\0');
				for (tail=buf2+strlen(buf2)-1; isspace(*tail); *tail--='\0');
				for (tail=buf3+strlen(buf3)-1; isspace(*tail); *tail--='\0');
				for (tail=buf4+strlen(buf4)-1; isspace(*tail); *tail--='\0');
				elog(DEBUG2, "arrow_fdw: metadata cache for '%s' (m:%s, c:%s) is older than the latest file (m:%s, c:%s), so invalidated",
					 FilePathName(fdesc), buf1, buf2, buf3, buf4);
				__arrowInvalidateMetadataCache(mcache, true);
				break;
			}
			/*
			 * Ok, arrow file metadata cache found and still valid
			 *
			 * NOTE: we currently support min/max statistics on the top-
			 * level variables only, not sub-field of the composite values.
			 */
			rbstate = makeRecordBatchStateFromCache(mcache, fdesc,
													p_stat_attrs);
			if (checkArrowRecordBatchIsVisible(rbstate, mvcc_slot))
				results = list_make1(rbstate);
			dlist_foreach (iter2, &mcache->siblings)
			{
				arrowMetadataCache *__mcache
					= dlist_container(arrowMetadataCache, chain, iter2.cur);
				rbstate = makeRecordBatchStateFromCache(__mcache, fdesc,
														p_stat_attrs);
				if (checkArrowRecordBatchIsVisible(rbstate, mvcc_slot))
					results = lappend(results, rbstate);
			}
			SpinLockAcquire(&arrow_metadata_state->lru_lock);
			dlist_move_head(&arrow_metadata_state->lru_list,
							&mcache->lru_chain);
			SpinLockRelease(&arrow_metadata_state->lru_lock);
			LWLockRelease(lock);

			return results;
		}
	}

	/*
	 * Hmm... no valid metadata cache was not found, so build a new entry
	 * under the exclusive lock on the arrow file.
	 */
	if (!has_exclusive)
	{
		LWLockRelease(lock);
		LWLockAcquire(lock, LW_EXCLUSIVE);
		has_exclusive = true;
		goto retry;
	}
	else
	{
		ArrowFileInfo	af_info;
		arrowMetadataCache *mcache;
		arrowStatsBinary *arrow_bstats;
		List		   *rb_state_any = NIL;

		readArrowFileDesc(FileGetRawDesc(fdesc), &af_info);
		if (af_info.dictionaries != NULL)
			elog(ERROR, "DictionaryBatch is not supported");
		Assert(af_info.footer._num_dictionaries == 0);

		if (af_info.recordBatches == NULL)
			elog(DEBUG2, "arrow file '%s' contains no RecordBatch",
				 FilePathName(fdesc));

		arrow_bstats = buildArrowStatsBinary(&af_info.footer, p_stat_attrs);
		for (index = 0; index < af_info.footer._num_recordBatches; index++)
		{
			RecordBatchState *rb_state;
			ArrowBlock       *block
				= &af_info.footer.recordBatches[index];
			ArrowRecordBatch *rbatch
			   = &af_info.recordBatches[index].body.recordBatch;

			rb_state = makeRecordBatchState(&af_info.footer.schema,
											block, rbatch);
			rb_state->fdesc = fdesc;
			memcpy(&rb_state->stat_buf, &stat_buf, sizeof(struct stat));
			rb_state->rb_index = index;

			if (arrow_bstats)
				applyArrowStatsBinary(rb_state, arrow_bstats);

			if (checkArrowRecordBatchIsVisible(rb_state, mvcc_slot))
				results = lappend(results, rb_state);
			rb_state_any = lappend(rb_state_any, rb_state);
		}
		releaseArrowStatsBinary(arrow_bstats);
		/* try to build a metadata cache for further references */
		mcache = __arrowBuildMetadataCache(rb_state_any, key.hash);
		if (mcache)
		{
			dlist_push_head(hash_slot, &mcache->chain);
			SpinLockAcquire(&arrow_metadata_state->lru_lock);
            dlist_push_head(&arrow_metadata_state->lru_list,
							&mcache->lru_chain);
			SpinLockRelease(&arrow_metadata_state->lru_lock);
		}
	}
	LWLockRelease(lock);
	/*
	 * reclaim unreferenced metadata cache entries based on LRU, if shared-
	 * memory consumption exceeds the configured threshold.
	 */
	arrowReclaimMetadataCache();

	return results;
}

/*
 * lookup_type_extension_info
 */
static void
lookup_type_extension_info(Oid type_oid,
						   const char **p_extname,
						   const char **p_extschema)
{
	Oid		ext_oid;
	char   *extname = NULL;
	char   *extschema = NULL;

	ext_oid = get_object_extension_oid(TypeRelationId,
									   type_oid, 0, true);
	if (OidIsValid(ext_oid))
	{
		Relation	rel;
		SysScanDesc	sscan;
		ScanKeyData	skey;
		HeapTuple	tup;

		rel = table_open(ExtensionRelationId, AccessShareLock);
		ScanKeyInit(&skey,
					Anum_pg_extension_oid,
					BTEqualStrategyNumber, F_OIDEQ,
					ObjectIdGetDatum(ext_oid));
		sscan = systable_beginscan(rel, ExtensionOidIndexId,
								   true, NULL, 1, &skey);
		tup = systable_getnext(sscan);
		if (HeapTupleIsValid(tup))
		{
			Form_pg_extension __ext = (Form_pg_extension) GETSTRUCT(tup);

			extname = pstrdup(NameStr(__ext->extname));
			if (__ext->extrelocatable)
				extschema = get_namespace_name(__ext->extnamespace);
		}
		systable_endscan(sscan);
		table_close(rel, AccessShareLock);
	}
	*p_extname = extname;
	*p_extschema = extschema;
}

/*
 * setupArrowSQLbufferSchema
 */
static void
__setupArrowSQLbufferField(SQLtable *table,
						   SQLfield *column,
						   const char *attname,
						   Oid atttypid,
						   int32 atttypmod,
						   ArrowField *afield)
{
	HeapTuple		tup;
	Form_pg_type	__type;
	const char	   *typname;
	const char	   *typnamespace;
	const char	   *timezone = show_timezone();
	const char	   *extname;
	const char	   *extschema;
	SQLstat		   *stat_list;

	/* walk down to the base type, if domain */
	for (;;)
	{
		tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(atttypid));
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "cache lookup failed for type: %u", atttypid);
		__type = (Form_pg_type) GETSTRUCT(tup);
		if (__type->typtype != TYPTYPE_DOMAIN)
			break;
		atttypid = __type->typbasetype;
		atttypmod = __type->typtypmod;
		ReleaseSysCache(tup);
	}
	typname = NameStr(__type->typname);
	typnamespace = get_namespace_name(__type->typnamespace);
	lookup_type_extension_info(atttypid,
							   &extname,
							   &extschema);
	table->numFieldNodes++;
    table->numBuffers +=
		assignArrowTypePgSQL(column,
							 attname,
							 atttypid,
							 atttypmod,
							 typname,
							 typnamespace,
							 __type->typlen,
							 __type->typbyval,
							 __type->typtype,
							 __type->typalign,
							 __type->typrelid,
							 __type->typelem,
							 timezone,
							 extname,
							 extschema,
							 afield);
	/* assign existing min/max statistics, if any */
	if (afield)
	{
		stat_list = __buildArrowFieldStatsList(afield, table->numRecordBatches);
		if (stat_list)
		{
			column->stat_list = stat_list;
			column->stat_enabled = true;
			table->has_statistics = true;
		}
	}
	
	if (OidIsValid(__type->typelem) && __type->typlen == -1)
	{
		/* array type */
		char		elem_name[NAMEDATALEN+10];
		ArrowField *__afield = NULL;

		snprintf(elem_name, sizeof(elem_name), "_%s[]", attname);
		column->element = palloc0(sizeof(SQLfield));
		if (afield)
		{
			if (afield->_num_children != 1)
				elog(ERROR, "Arrow::Field (%s) is not compatible", afield->name);
			__afield = &afield->children[0];
		}
		__setupArrowSQLbufferField(table,
								   column->element,
								   elem_name,
								   __type->typelem,
								   -1,
								   __afield);
	}
	else if (OidIsValid(__type->typrelid))
	{
		/* composite type */
		TupleDesc	tupdesc = lookup_rowtype_tupdesc(atttypid, atttypmod);
		int			j;

		if (afield && afield->_num_children != tupdesc->natts)
			elog(ERROR, "Arrow::Field (%s) is not compatible", afield->name);

		column->nfields = tupdesc->natts;
		column->subfields = palloc0(sizeof(SQLfield) * tupdesc->natts);
		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute sattr = tupleDescAttr(tupdesc, j);
			ArrowField	   *__afield = NULL;

			if (afield)
				__afield = &afield->children[j];
			__setupArrowSQLbufferField(table,
									   &column->subfields[j],
									   NameStr(sattr->attname),
									   sattr->atttypid,
									   sattr->atttypmod,
									   __afield);
		}
		ReleaseTupleDesc(tupdesc);
	}
	else if (__type->typtype == 'e')
	{
		elog(ERROR, "Enum type is not supported right now");
	}
	ReleaseSysCache(tup);
}

static void
setupArrowSQLbufferSchema(SQLtable *table, TupleDesc tupdesc,
						  ArrowFileInfo *af_info)
{
	int		j;

	Assert(!af_info || af_info->footer.schema._num_fields == tupdesc->natts);
	table->nfields = tupdesc->natts;
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
		ArrowField	   *afield = NULL;

		if (af_info)
			afield = &af_info->footer.schema.fields[j];
		__setupArrowSQLbufferField(table,
								   &table->columns[j],
								   NameStr(attr->attname),
								   attr->atttypid,
								   attr->atttypmod,
								   afield);
	}
	table->segment_sz = (size_t)arrow_record_batch_size_kb << 10;
}

static void
setupArrowSQLbufferBatches(SQLtable *table, ArrowFileInfo *af_info)
{
	loff_t		pos = 0;
	int			i, nitems;

	/* restore DictionaryBatches already in the file */
	nitems = af_info->footer._num_dictionaries;
	table->numDictionaries = nitems;
	if (nitems > 0)
	{
		table->dictionaries = palloc(sizeof(ArrowBlock) * nitems);
		memcpy(table->dictionaries,
			   af_info->footer.dictionaries,
			   sizeof(ArrowBlock) * nitems);
		for (i=0; i < nitems; i++)
		{
			ArrowBlock *block = &table->dictionaries[i];

			pos = Max(pos, ARROWALIGN(block->offset +
									  block->metaDataLength +
									  block->bodyLength));
		}
	}
	else
		table->dictionaries = NULL;

	/* restore RecordBatches already in the file */
	nitems = af_info->footer._num_recordBatches;
	table->numRecordBatches = nitems;
	if (nitems > 0)
	{
		table->recordBatches = palloc(sizeof(ArrowBlock) * nitems);
		memcpy(table->recordBatches,
			   af_info->footer.recordBatches,
			   sizeof(ArrowBlock) * nitems);
		for (i=0; i < nitems; i++)
		{
			ArrowBlock *block = &table->recordBatches[i];

			pos = Max(pos, ARROWALIGN(block->offset +
									  block->metaDataLength +
									  block->bodyLength));
		}
	}
	else
		table->recordBatches = NULL;

	if (lseek(table->fdesc, pos, SEEK_SET) < 0)
		elog(ERROR, "failed on lseek('%s',%lu): %m",
			 table->filename, pos);
	table->f_pos = pos;
}

/*
 * createArrowWriteRedoLog
 */
static loff_t
createArrowWriteRedoLog(File filp, bool is_newfile)
{
	arrowWriteRedoLog *redo;
	int				fdesc = FileGetRawDesc(filp);
	const char	   *fname = FilePathName(filp);
	TransactionId	curr_xid = GetCurrentTransactionId();
	CommandId		curr_cid = GetCurrentCommandId(true);
	dlist_iter		iter;
	MetadataCacheKey key;
	struct stat		stat_buf;
	size_t			main_sz;

	if (fstat(fdesc, &stat_buf) != 0)
		elog(ERROR, "failed on fstat(2): %m");
	initMetadataCacheKey(&key, &stat_buf);

	dlist_foreach(iter, &arrow_write_redo_list)
	{
		redo = dlist_container(arrowWriteRedoLog, chain, iter.cur);

		if (redo->key.st_dev == key.st_dev &&
			redo->key.st_ino == key.st_ino &&
			redo->xid    == curr_xid &&
			redo->cid    <= curr_cid)
		{
			elog(ERROR, "Why? '%s' on behalf of arrow_fdw foreign-table is concurrently opened for update, please confirm the configuration", fname);
		}
	}

	if (is_newfile)
	{
		main_sz = MAXALIGN(offsetof(arrowWriteRedoLog, footer_backup));
		redo = MemoryContextAllocZero(CacheMemoryContext,
									  main_sz + strlen(fname) + 1);
		memcpy(&redo->key, &key, sizeof(MetadataCacheKey));
		redo->xid = curr_xid;
		redo->cid = curr_cid;
		redo->pathname = (char *)redo + main_sz;
		strcpy(redo->pathname, fname);
		redo->is_truncate = false;
		redo->footer_offset = 0;
		redo->footer_length = 0;
	}
	else
	{
		ssize_t			nbytes;
		off_t			offset;
		char			temp[100];

		/* make backup image of the Footer section */
		nbytes = sizeof(int32) + 6;		/* = strlen("ARROW1") */
		offset = stat_buf.st_size - nbytes;
		if (__preadFile(fdesc, temp, nbytes, offset) != nbytes)
			elog(ERROR, "failed on pread(2): %m");
		offset -= *((int32 *)temp);

		nbytes = stat_buf.st_size - offset;
		if (nbytes <= 0)
			elog(ERROR, "strange apache arrow format");
		main_sz = MAXALIGN(offsetof(arrowWriteRedoLog,
									footer_backup[nbytes]));
		redo = MemoryContextAllocZero(CacheMemoryContext,
									  main_sz + strlen(fname) + 1);
		memcpy(&redo->key, &key, sizeof(MetadataCacheKey));
		redo->xid = curr_xid;
		redo->cid = curr_cid;
		redo->pathname = (char *)redo + main_sz;
		strcpy(redo->pathname, fname);
		redo->is_truncate = false;
		PG_TRY();
		{
			if (__preadFile(fdesc, redo->footer_backup, nbytes, offset) != nbytes)
				elog(ERROR, "failed on pread(2): %m");
			if (lseek(fdesc, offset, SEEK_SET) < 0)
				elog(ERROR, "failed on lseek(2): %m");
			redo->footer_offset = offset;
			redo->footer_length = nbytes;
		}
		PG_CATCH();
		{
			pfree(redo);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	elog(DEBUG2, "arrow: redo-log on '%s' (st_dev=%u/st_ino=%u) xid=%u cid=%u offset=%lu length=%zu",
		 redo->pathname, (uint32)redo->key.st_dev, (uint32)redo->key.st_ino,
		 (uint32)redo->xid, (uint32)redo->cid,
		 (uint64)redo->footer_offset,
		 (uint64)redo->footer_length);

	dlist_push_head(&arrow_write_redo_list, &redo->chain);

	return redo->footer_offset;
}

/*
 * writeOutArrowRecordBatch
 */
static void
writeOutArrowRecordBatch(arrowWriteState *aw_state, bool with_footer)
{
	SQLtable   *table = &aw_state->sql_table;
	int			index = aw_state->hash % ARROW_METADATA_HASH_NSLOTS;
	arrowWriteMVCCLog *mvcc = NULL;

	if (table->nitems > 0)
	{
		mvcc = MemoryContextAllocZero(TopSharedMemoryContext,
									  sizeof(arrowWriteMVCCLog));
		memcpy(&mvcc->key, &aw_state->key, sizeof(MetadataCacheKey));
		mvcc->xid = GetCurrentTransactionId();
		mvcc->cid = GetCurrentCommandId(true);
	}

	PG_TRY();
	{
		LWLockAcquire(&arrow_metadata_state->lock_slots[index],
					  LW_EXCLUSIVE);
		/* write out an empty arrow file */
		if (table->f_pos == 0)
		{
			arrowFileWrite(table, "ARROW1\0\0", 8);
			writeArrowSchema(table);
		}
		if (table->nitems > 0)
		{
			mvcc->record_batch = writeArrowRecordBatch(table);
			sql_table_clear(table);
			dlist_push_tail(&arrow_metadata_state->mvcc_slots[index],
							&mvcc->chain);
			elog(DEBUG2,
				 "arrow-write: '%s' (st_dev=%u, st_ino=%u), xid=%u, cid=%u, record_batch=%u nitems=%lu",
				 FilePathName(aw_state->file),
				 (uint32)mvcc->key.st_dev, (uint32)mvcc->key.st_ino,
				 (uint32)mvcc->xid, (uint32)mvcc->cid, mvcc->record_batch,
				 table->nitems);
		}
		if (with_footer)
			writeArrowFooter(table);

		/*
		 * Invalidation of the metadata cache, if any
		 *
		 * NOTE: metadata cache shall be invalidated on the next reference,
		 * if st_mtime of the file is newer than st_mtime of the mcache.
		 * Linux kernel offers nanosecond precision in st_Xtime, but it never
		 * guarantee the st_Xtime is recorded in nanosecond precision...
		 */
		arrowInvalidateMetadataCache(&aw_state->key, true);

		LWLockRelease(&arrow_metadata_state->lock_slots[index]);
	}
	PG_CATCH();
	{
		if (mvcc)
			pfree(mvcc);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * TRUNCATE support
 */
static void
__arrowExecTruncateRelation(Relation frel)
{
	TupleDesc	tupdesc = RelationGetDescr(frel);
	Oid			frel_oid = RelationGetRelid(frel);
	ForeignTable *ft = GetForeignTable(frel_oid);
	arrowWriteRedoLog *redo;
	ArrowFileInfo af_info;
	struct stat	stat_buf;
	MetadataCacheKey key;
	int			index;
	List	   *filesList;
	SQLtable   *table;
	const char *path_name;
	const char *dir_name;
	const char *file_name;
	size_t		main_sz;
	int			fdesc = -1;
	char		backup_path[MAXPGPATH];
	bool		writable;

	filesList = __arrowFdwExtractFilesList(ft->options,
										   NULL,
										   &writable);
	if (!writable)
		elog(ERROR, "arrow_fdw: foreign table \"%s\" is not writable",
			 RelationGetRelationName(frel));
	Assert(list_length(filesList) == 1);
	path_name = strVal(linitial(filesList));
	readArrowFile(path_name, &af_info, false);
	if (stat(path_name, &stat_buf) != 0)
		elog(ERROR, "failed on stat('%s'): %m", path_name);
	/* metadata cache invalidation */
	index = initMetadataCacheKey(&key, &stat_buf);
	LWLockAcquire(&arrow_metadata_state->lock_slots[index], LW_EXCLUSIVE);
	arrowInvalidateMetadataCache(&key, true);
	LWLockRelease(&arrow_metadata_state->lock_slots[index]);

	/* build SQLtable to write out schema */
	table = palloc0(offsetof(SQLtable, columns[tupdesc->natts]));
	setupArrowSQLbufferSchema(table, tupdesc, &af_info);

	/* create REDO log entry */
	main_sz = MAXALIGN(offsetof(arrowWriteRedoLog, footer_backup));
	redo = MemoryContextAllocZero(CacheMemoryContext,
								  main_sz + strlen(path_name) + 1);
	redo->xid    = GetCurrentTransactionId();
	redo->cid    = GetCurrentCommandId(true);
	redo->pathname = (char *)redo + main_sz;
	strcpy(redo->pathname, path_name);
	redo->is_truncate = true;

	PG_TRY();
	{
		/*
		 * move the current arrow file to the backup
		 */
		dir_name = dirname(pstrdup(path_name));
		file_name = basename(pstrdup(path_name));
		for (;;)
		{
			redo->suffix = random();
			snprintf(backup_path, sizeof(backup_path),
					 "%s/%s.%u.backup",
					 dir_name, file_name, redo->suffix);
			if (stat(backup_path, &stat_buf) != 0)
			{
				if (errno == ENOENT)
					break;
				elog(ERROR, "failed on stat('%s'): %m", backup_path);
			}
		}
		if (rename(path_name, backup_path) != 0)
			elog(ERROR, "failed on rename('%s','%s'): %m",
				 path_name, backup_path);

		/*
		 * create an empty arrow file
		 */
		PG_TRY();
		{
			fdesc = open(path_name, O_RDWR | O_CREAT | O_EXCL, 0600);
			if (fdesc < 0)
				elog(ERROR, "failed on open('%s'): %m", path_name);
			if (fstat(fdesc, &stat_buf) != 0)
				elog(ERROR, "failed on fstat('%s'): %m", path_name);
			initMetadataCacheKey(&redo->key, &stat_buf);
			table->filename = path_name;
			table->fdesc = fdesc;
			arrowFileWrite(table, "ARROW1\0\0", 8);
			writeArrowSchema(table);
			writeArrowFooter(table);
		}
		PG_CATCH();
		{
			if (fdesc >= 0)
				close(fdesc);
			if (rename(backup_path, path_name) != 0)
				elog(WARNING, "failed on rename('%s', '%s'): %m",
					 backup_path, path_name);
			PG_RE_THROW();
		}
		PG_END_TRY();
		close(fdesc);
	}
	PG_CATCH();
	{
		pfree(redo);
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* save the REDO log entry */
	dlist_push_head(&arrow_write_redo_list, &redo->chain);
}

#if PG_VERSION_NUM >= 140000
/*
 * TRUNCATE support
 */
static void
ArrowExecForeignTruncate(List *rels, DropBehavior behavior, bool restart_seqs)
{
	ListCell   *lc;

	foreach (lc, rels)
	{
		Relation	frel = lfirst(lc);

		__arrowExecTruncateRelation(frel);
	}
}
#endif

/*
 * pgstrom_arrow_fdw_truncate
 */
Datum
pgstrom_arrow_fdw_truncate(PG_FUNCTION_ARGS)
{
#if PG_VERSION_NUM < 140000
	Oid			frel_oid = PG_GETARG_OID(0);
	Relation	frel;
	FdwRoutine *routine;

	frel = table_open(frel_oid, AccessExclusiveLock);
	if (frel->rd_rel->relkind != RELKIND_FOREIGN_TABLE)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not arrow_fdw foreign table",
						RelationGetRelationName(frel))));
	routine = GetFdwRoutineForRelation(frel, false);
	if (memcmp(routine, &pgstrom_arrow_fdw_routine, sizeof(FdwRoutine)) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not arrow_fdw foreign table",
						RelationGetRelationName(frel))));
	__arrowExecTruncateRelation(frel);

	table_close(frel, NoLock);
#else
	elog(ERROR, "PostgreSQL v14 supports TRUNCATE <foreign table>; use the standard statement instead of the legacy interface");
#endif
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_truncate);

static void
__applyArrowTruncateRedoLog(arrowWriteRedoLog *redo, bool is_commit)
{
	char		backup[MAXPGPATH];

	snprintf(backup, MAXPGPATH, "%s.%u.backup",
			 redo->pathname, redo->suffix);
	if (is_commit)
	{
		elog(DEBUG2, "arrow-redo: unlink [%s]", backup);
		if (unlink(backup) != 0)
			ereport(WARNING,
					(errcode_for_file_access(),
					 errmsg("could not remove truncated file \"%s\": %m",
							backup),
					 errhint("remove the \"%s\" manually", backup)));
	}
	else
	{
		elog(DEBUG2, "arrow-redo: rename [%s]->[%s]", backup, redo->pathname);
		if (rename(backup, redo->pathname) != 0)
			ereport(WARNING,
					(errcode_for_file_access(),
					 errmsg("could not restore backup file \"%s\": %m",
							backup),
					 errhint("please restore \"%s\" to \"%s\" manually",
							 backup, redo->pathname)));
		arrowInvalidateMetadataCache(&redo->key, true);
	}
}

static void
__applyArrowInsertRedoLog(arrowWriteRedoLog *redo, bool is_commit)
{
	int		fdesc;
	
	if (is_commit)
		return;

	/* special case, if it was an empty file */
	if (redo->footer_offset == 0 &&
		redo->footer_length == 0)
	{
		if (unlink(redo->pathname) != 0)
			ereport(WARNING,
					(errcode_for_file_access(),
					 errmsg("failed on truncate('%s'): %m", redo->pathname),
					 errdetail("could not apply REDO image, therefore, garbages are still remained")));
		return;
	}

	fdesc = open(redo->pathname, O_RDWR);
	if (fdesc < 0)
	{
		ereport(WARNING,
				(errcode_for_file_access(),
				 errmsg("failed on open('%s'): %m", redo->pathname),
				 errdetail("could not apply REDO image, therefore, arrow file might be corrupted")));
	}
	else if (lseek(fdesc, redo->footer_offset, SEEK_SET) < 0)
	{
		ereport(WARNING,
				(errcode_for_file_access(),
				 errmsg("failed on lseek('%s'): %m", redo->pathname),
				 errdetail("could not apply REDO image, therefore, arrow file might be corrupted")));
	}
	else if (__writeFile(fdesc,
						 redo->footer_backup,
						 redo->footer_length) != redo->footer_length)
	{
		ereport(WARNING,
				(errcode_for_file_access(),
				 errmsg("failed on write('%s'): %m", redo->pathname),
				 errdetail("could not apply REDO image, therefore, arrow file might be corrupted")));
	}
	else if (ftruncate(fdesc, (redo->footer_offset +
							   redo->footer_length)) != 0)
	{
		ereport(WARNING,
				(errcode_for_file_access(),
				 errmsg("failed on ftruncate('%s'): %m", redo->pathname),
				 errdetail("could not apply REDO image, therefore, arrow file might be corrupted")));
	}
	/* invalidation of the metadata-cache */
	arrowInvalidateMetadataCache(&redo->key, true);

	close(fdesc);

	elog(DEBUG2, "arrow_fdw: REDO log applied (xid=%u, cid=%u, file=[%s], offset=%zu, length=%zu)", redo->xid, redo->cid, redo->pathname, redo->footer_offset, redo->footer_length);
}

static void
__cleanupArrowWriteMVCCLog(TransactionId curr_xid, dlist_head *mvcc_slot)
{
	dlist_mutable_iter iter;

	dlist_foreach_modify(iter, mvcc_slot)
	{
		arrowWriteMVCCLog *mvcc = dlist_container(arrowWriteMVCCLog,
												  chain, iter.cur);
		if (mvcc->xid == curr_xid)
		{
			dlist_delete(&mvcc->chain);
			elog(DEBUG2, "arrow: release mvcc-log (st_dev=%u, st_ino=%u), xid=%u, cid=%u, record_batch=%u",
				 (uint32)mvcc->key.st_dev, (uint32)mvcc->key.st_ino,
				 (uint32)mvcc->xid, (uint32)mvcc->cid, mvcc->record_batch);
			pfree(mvcc);
		}
	}
}

/*
 * __arrowFdwXactCallback
 */
static void
__arrowFdwXactCallback(TransactionId curr_xid, bool is_commit)
{
	arrowWriteRedoLog  *redo;
	dlist_mutable_iter	iter;
	CommandId			curr_cid = InvalidCommandId;
	uint32				index;
	bool				locked[ARROW_METADATA_HASH_NSLOTS];
	LWLock			   *locks[ARROW_METADATA_HASH_NSLOTS];
	uint32				lcount = 0;

	if (curr_xid == InvalidTransactionId ||
		dlist_is_empty(&arrow_write_redo_list))
		return;

	memset(locked, 0, sizeof(locked));
	dlist_foreach_modify(iter, &arrow_write_redo_list)
	{
		redo = dlist_container(arrowWriteRedoLog, chain, iter.cur);
		if (redo->xid != curr_xid)
			continue;
		if (curr_cid != InvalidCommandId &&
			curr_cid < redo->cid)
			elog(WARNING, "Bug? Order of REDO log is not be correct. ABORT transaction might generate wrong image restored.");

		index = redo->key.hash % ARROW_METADATA_HASH_NSLOTS;
		if (!locked[index])
		{
			LWLock	   *lock = &arrow_metadata_state->lock_slots[index];
			dlist_head *slot = &arrow_metadata_state->mvcc_slots[index];

			LWLockAcquire(lock, LW_EXCLUSIVE);
			__cleanupArrowWriteMVCCLog(curr_xid, slot);
			locked[index] = true;
			locks[lcount++] = lock;
		}
		if (redo->is_truncate)
			__applyArrowTruncateRedoLog(redo, is_commit);
		else
			__applyArrowInsertRedoLog(redo, is_commit);

		dlist_delete(&redo->chain);
		pfree(redo);
	}

	for (index=0; index < lcount; index++)
		LWLockRelease(locks[index]);
}

/*
 * arrowFdwXactCallback
 */
static void
arrowFdwXactCallback(XactEvent event, void *arg)
{
	TransactionId	curr_xid = GetCurrentTransactionIdIfAny();

	if (event == XACT_EVENT_COMMIT)
		__arrowFdwXactCallback(curr_xid, true);
	else if (event == XACT_EVENT_ABORT)
		__arrowFdwXactCallback(curr_xid, false);
}

/*
 * arrowFdwSubXactCallback
 */
static void
arrowFdwSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
						SubTransactionId parentSubid, void *arg)
{
	TransactionId	curr_xid = GetCurrentTransactionIdIfAny();

	if (event == SUBXACT_EVENT_COMMIT_SUB)
		__arrowFdwXactCallback(curr_xid, true);
	else if (event == SUBXACT_EVENT_ABORT_SUB)
		__arrowFdwXactCallback(curr_xid, false);
}

/*
 * pgstrom_request_arrow_fdw
 */
static void
pgstrom_request_arrow_fdw(void)
{
	if (shmem_request_next)
		shmem_request_next();
	RequestAddinShmemSpace(MAXALIGN(sizeof(arrowMetadataState)));
}

/*
 * pgstrom_startup_arrow_fdw
 */
static void
pgstrom_startup_arrow_fdw(void)
{
	bool	found;
	int		i;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	arrow_metadata_state =
		ShmemInitStruct("arrow_metadata_state",
						MAXALIGN(sizeof(arrowMetadataState)),
						&found);
	if (!IsUnderPostmaster)
	{
		SpinLockInit(&arrow_metadata_state->lru_lock);
		dlist_init(&arrow_metadata_state->lru_list);
		pg_atomic_init_u64(&arrow_metadata_state->consumed, 0UL);
		for (i=0; i < ARROW_METADATA_HASH_NSLOTS; i++)
		{
			LWLockInitialize(&arrow_metadata_state->lock_slots[i], -1);
			dlist_init(&arrow_metadata_state->hash_slots[i]);
			dlist_init(&arrow_metadata_state->mvcc_slots[i]);
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
	/* ANALYZE support */
	r->AnalyzeForeignTable			= ArrowAnalyzeForeignTable;
	/* IMPORT FOREIGN SCHEMA support */
	r->ImportForeignSchema			= ArrowImportForeignSchema;
#if PG_VERSION_NUM >= 140000
	r->ExecForeignTruncate			= ArrowExecForeignTruncate;
#endif
	/* CPU Parallel support */
	r->IsForeignScanParallelSafe	= ArrowIsForeignScanParallelSafe;
	r->EstimateDSMForeignScan		= ArrowEstimateDSMForeignScan;
	r->InitializeDSMForeignScan		= ArrowInitializeDSMForeignScan;
	r->ReInitializeDSMForeignScan	= ArrowReInitializeDSMForeignScan;
	r->InitializeWorkerForeignScan	= ArrowInitializeWorkerForeignScan;
	r->ShutdownForeignScan			= ArrowShutdownForeignScan;
	/* INSERT/DELETE support */
	r->PlanForeignModify			= ArrowPlanForeignModify;
	r->BeginForeignModify			= ArrowBeginForeignModify;
	r->ExecForeignInsert			= ArrowExecForeignInsert;
	r->EndForeignModify				= ArrowEndForeignModify;
#if PG_VERSION_NUM >= 110000
	r->BeginForeignInsert			= ArrowBeginForeignInsert;
	r->EndForeignInsert				= ArrowEndForeignInsert;
#endif
	r->ExplainForeignModify			= ArrowExplainForeignModify;

	/*
	 * Turn on/off arrow_fdw
	 */
	DefineCustomBoolVariable("arrow_fdw.enabled",
							 "Enables the planner's use of Arrow_Fdw",
							 NULL,
							 &arrow_fdw_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/*
	 * Turn on/off min/max statistics hint
	 */
	DefineCustomBoolVariable("arrow_fdw.stats_hint_enabled",
							 "Enables min/max statistics hint, if any",
							 NULL,
							 &arrow_fdw_stats_hint_enabled,
							 true,
							 PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);
	/*
	 * Configurations for arrow_fdw metadata cache
	 */
	DefineCustomIntVariable("arrow_fdw.metadata_cache_size",
							"size of shared metadata cache for arrow files",
							NULL,
							&arrow_metadata_cache_size_kb,
							131072,		/* 128MB */
							32768,		/* 32MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	arrow_metadata_cache_size = (size_t)arrow_metadata_cache_size_kb << 10;

	/*
	 * Limit of RecordBatch size for writing
	 */
	DefineCustomIntVariable("arrow_fdw.record_batch_size",
							"maximum size of record batch on writing",
							NULL,
							&arrow_record_batch_size_kb,
							256 * 1024,		/* default: 256MB */
							4 * 1024,		/* min: 4MB */
							2048 * 1024,	/* max: 2GB */
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);

	/* shared memory size */
	shmem_request_next = shmem_request_hook;
	shmem_request_hook = pgstrom_request_arrow_fdw;
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_arrow_fdw;

	/* transaction callback */
	RegisterXactCallback(arrowFdwXactCallback, NULL);
	RegisterSubXactCallback(arrowFdwSubXactCallback, NULL);
	
	/* misc init */
	dlist_init(&arrow_write_redo_list);
}
