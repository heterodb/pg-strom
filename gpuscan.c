/*
 * gpuscan.c
 *
 * Sequential scan accelerated by GPU processors
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/sysattr.h"
#include "commands/explain.h"
#include "miscadmin.h"
#include "nodes/bitmapset.h"
#include "nodes/execnodes.h"
#include "nodes/plannodes.h"
#include "nodes/relation.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/plancat.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/spccache.h"
#include "pg_strom.h"
#include "opencl_gpuscan.h"

static add_scan_path_hook_type	add_scan_path_next;
static CustomPathMethods		gpuscan_path_methods;
static CustomPlanMethods		gpuscan_plan_methods;
static bool						enable_gpuscan;

typedef struct {
	CustomPath	cpath;
	List	   *dev_quals;		/* RestrictInfo run on device */
	List	   *host_quals;		/* RestrictInfo run on host */
	Bitmapset  *dev_attnums;	/* attrs referenced in device */
	Bitmapset  *host_attnums;	/* attrs referenced in host */
} GpuScanPath;

typedef struct {
	CustomPlan	cplan;
	Index		scanrelid;		/* index of the range table */
	const char *kern_source;	/* source of opencl kernel */
	int			extra_flags;	/* extra libraries to be included */
	List	   *used_params;	/* list of Const/Param in use */
	List	   *used_vars;		/* list of Var in use */
	List	   *dev_clauses;	/* clauses to be run on device */
	Bitmapset  *dev_attnums;	/* attrs referenced in device */
	Bitmapset  *host_attnums;	/* attrs referenced in host */
} GpuScanPlan;

/*
 * Gpuscan has three strategy to scan a relation.
 * a) cache-only scan, if all the variables being referenced in target-list
 *    and scan-qualifiers are on the t-tree columnar cache.
 *    It is capable to return a columner-store, instead of individual rows,
 *    if upper plan node is also managed by PG-Strom.
 * b) hybrid-scan, if Var references by scan-qualifiers are on cache, but
 *    ones by target-list are not. It runs first screening by device, then
 *    fetch a tuple from the shared buffers.
 * c) heap-only scan, if all the variables in scan-qualifier are not on
 *    the cache, all we can do is read tuples from shared-buffer to the
 *    row-store, then picking it up.
 * In case of (a) and (b), gpuscan needs to be responsible to MVCC checks;
 * that is not done on the first evaluation timing.
 * In case of (c), it may construct a columnar cache entry that caches the
 * required columns.
 */
#define GpuScanMode_CacheOnlyScan	0x0001
#define GpuScanMode_HybridScan		0x0002
#define GpuScanMode_HeapOnlyScan	0x0003
#define GpuScanMode_CreateCache		0x0004

typedef struct {
	CustomPlanState		cps;
	Relation			scan_rel;
	HeapScanDesc		scan_desc;
	TupleTableSlot	   *scan_slot;
	int					scan_mode;
	pgstrom_queue	   *mqueue;
	kern_parambuf	   *kparambuf;
	Datum				dprog_key;
	kern_colmeta	   *dev_colmeta;
	pgstrom_gpuscan	   *curr_chunk;
	uint32				curr_index;
	int					num_running;
	dlist_head			ready_chunks;
} GpuScanState;

/* static functions */
static void clserv_process_gpuscan_row(pgstrom_message *msg);
static void clserv_put_gpuscan_row(pgstrom_message *msg);

/*
 * cost_gpuscan
 *
 * cost estimation for GpuScan
 */
static void
cost_gpuscan(GpuScanPath *gpu_path, PlannerInfo *root,
			 RelOptInfo *baserel, ParamPathInfo *param_info,
			 List *dev_quals, List *host_quals)
{
	Path	   *path = &gpu_path->cpath.path;
	Cost		startup_cost = 0;
	Cost		run_cost = 0;
	double		spc_seq_page_cost;
	QualCost	dev_cost;
	QualCost	host_cost;
	Cost		gpu_per_tuple;
	Cost		cpu_per_tuple;
	Selectivity	dev_sel;

	/* Should only be applied to base relations */
	Assert(baserel->relid > 0);
	Assert(baserel->rtekind == RTE_RELATION);

	/* Mark the path with the correct row estimate */
	if (param_info)
		path->rows = param_info->ppi_rows;
	else
		path->rows = baserel->rows;

	if (!enable_gpuscan)
		startup_cost += disable_cost;

	/* fetch estimated page cost for tablespace containing table */
	get_tablespace_page_costs(baserel->reltablespace,
							  NULL,
							  &spc_seq_page_cost);
	/*
	 * disk costs
	 * XXX - needs to adjust after columner cache in case of bare heapscan,
	 * or partial heapscan if targetlist references out of cached columns
	 */
    run_cost += spc_seq_page_cost * baserel->pages;

	/* GPU costs */
	cost_qual_eval(&dev_cost, dev_quals, root);
	dev_sel = clauselist_selectivity(root, dev_quals, 0, JOIN_INNER, NULL);

	/*
	 * XXX - very rough estimation towards GPU startup and device calculation
	 *       to be adjusted according to device info
	 *
	 * TODO: startup cost takes NITEMS_PER_CHUNK * width to be carried, but
	 * only first chunk because data transfer is concurrently done, if NOT
	 * integrated GPU
	 * TODO: per_tuple calculation cost shall be divided by parallelism of
	 * average opencl spec.
	 */
	dev_cost.startup += 10000;
	dev_cost.per_tuple /= 100;

	/* CPU costs */
	cost_qual_eval(&host_cost, host_quals, root);
	if (param_info)
	{
		QualCost	param_cost;

		/* Include costs of pushed-down clauses */
		cost_qual_eval(&param_cost, param_info->ppi_clauses, root);
		host_cost.startup += param_cost.startup;
		host_cost.per_tuple += param_cost.per_tuple;
	}

	/* total path cost */
	startup_cost += dev_cost.startup + host_cost.startup;
	cpu_per_tuple = cpu_tuple_cost + host_cost.per_tuple;
	gpu_per_tuple = cpu_tuple_cost / 100 + dev_cost.per_tuple;
	run_cost += (gpu_per_tuple * baserel->tuples +
				 cpu_per_tuple * dev_sel * baserel->tuples);

    gpu_path->cpath.path.startup_cost = startup_cost;
    gpu_path->cpath.path.total_cost = startup_cost + run_cost;
}

static void
gpuscan_add_scan_path(PlannerInfo *root,
					  RelOptInfo *baserel,
					  RangeTblEntry *rte)
{
	GpuScanPath	   *pathnode;
	List		   *dev_quals = NIL;
	List		   *host_quals = NIL;
	Bitmapset	   *dev_attnums = NULL;
	Bitmapset	   *host_attnums = NULL;
	ListCell	   *cell;
	codegen_context	context;

	/* only base relation we can handle */
	if (baserel->rtekind != RTE_RELATION || baserel->relid == 0)
		return;

	/* check whether qualifier can run on GPU device */
	memset(&context, 0, sizeof(codegen_context));
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_codegen_available_expression(rinfo->clause))
		{
			pull_varattnos((Node *)rinfo->clause,
						   baserel->relid,
						   &dev_attnums);
			dev_quals = lappend(dev_quals, rinfo);
		}
		else
		{
			pull_varattnos((Node *)rinfo->clause,
						   baserel->relid,
						   &host_attnums);
			host_quals = lappend(host_quals, rinfo);
		}
	}
	/* also, picks up Var nodes in the target list */
	pull_varattnos((Node *)baserel->reltargetlist,
				   baserel->relid,
				   &host_attnums);
	/*
	 * FIXME: needs to pay attention for projection cost.
	 * It may make sense to use build_physical_tlist, if host_attnums
	 * are much wider than dev_attnums.
	 * Anyway, it needs investigation of the actual behavior.
	 */

	/* XXX - check whether columnar cache may be available */

	/*
	 * Construction of a custom-plan node.
	 */
	pathnode = palloc0(sizeof(GpuScanPath));
	pathnode->cpath.path.type = T_CustomPath;
	pathnode->cpath.path.pathtype = T_CustomPlan;
	pathnode->cpath.path.parent = baserel;
	pathnode->cpath.path.param_info
		= get_baserel_parampathinfo(root, baserel, baserel->lateral_relids);
	pathnode->cpath.path.pathkeys = NIL;	/* gpuscan has unsorted result */

	cost_gpuscan(pathnode, root, baserel,
				 pathnode->cpath.path.param_info,
				 dev_quals, host_quals);

	pathnode->dev_quals = dev_quals;
	pathnode->host_quals = host_quals;
	pathnode->dev_attnums = dev_attnums;
	pathnode->host_attnums = host_attnums;

	add_path(baserel, &pathnode->cpath.path);
}

static char *
gpuscan_codegen_quals(PlannerInfo *root, List *dev_quals,
					  codegen_context *context)
{
	StringInfoData	str;
	ListCell	   *cell;
	int				index;
	devtype_info   *dtype;
	char		   *expr_code;

	memset(context, 0, sizeof(codegen_context));
	if (dev_quals == NIL)
		return NULL;

	expr_code = pgstrom_codegen_expression((Node *)dev_quals, context);

	Assert(expr_code != NULL);

	initStringInfo(&str);

	/* Put param/const definitions */
	index = 0;
	foreach (cell, context->used_params)
	{
		if (IsA(lfirst(cell), Const))
		{
			Const  *con = lfirst(cell);

			dtype = pgstrom_devtype_lookup(con->consttype);
			Assert(dtype != NULL);

			appendStringInfo(&str,
							 "#define KPARAM_%u\t"
							 "pg_%s_param(kparams,%d)\n",
							 index, dtype->type_ident, index);
		}
		else if (IsA(lfirst(cell), Param))
		{
			Param  *param = lfirst(cell);

			dtype = pgstrom_devtype_lookup(param->paramtype);
			Assert(dtype != NULL);

			appendStringInfo(&str,
							 "#define KPARAM_%u\t"
							 "pg_%s_param(kparams,%d)\n",
							 index, dtype->type_ident, index);
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(lfirst(cell)));
		index++;
	}

	/* Put Var definition for row-store */
	index = 0;
	foreach (cell, context->used_vars)
	{
		Var	   *var = lfirst(cell);

		dtype = pgstrom_devtype_lookup(var->vartype);
		Assert(dtype != NULL);

		if (dtype->type_flags & DEVTYPE_IS_VARLENA)
			appendStringInfo(&str,
					 "#define KVAR_%u\t"
							 "pg_%s_vref(kcs,toast,%u,get_global_id(0))\n",
							 index, dtype->type_ident, index);
		else
			appendStringInfo(&str,
							 "#define KVAR_%u\t"
							 "pg_%s_vref(kcs,%u,get_global_id(0))\n",
							 index, dtype->type_ident, index);
		index++;
	}

	/* columns to be referenced */
	appendStringInfo(&str,
					 "\n"
					 "static __constant cl_ushort pg_used_vars[]={");
	foreach (cell, context->used_vars)
	{
		Var	   *var = lfirst(cell);

		appendStringInfo(&str, "%s%u",
						 cell == list_head(context->used_vars) ? "" : ", ",
						 var->varattno);
	}
	appendStringInfo(&str, "};\n\n");

	/* qualifier definition with row-store */
	appendStringInfo(&str,
					 "__kernel void\n"
					 "gpuscan_qual_cs(__global kern_gpuscan *kgscan,\n"
					 "                __global kern_column_store *kcs,\n"
					 "                __global kern_toastbuf *toast,\n"
					 "                __local void *local_workmem)\n"
					 "{\n"
					 "  pg_bool_t   rc;\n"
					 "  cl_int      errcode;\n"
					 "  __global kern_parambuf *kparams\n"
					 "    = KERN_GPUSCAN_PARAMBUF(kgscan);\n"
					 "\n"
					 "  gpuscan_local_init(local_workmem);\n"
					 "  if (get_global_id(0) < kcs->nrows)\n"
					 "    rc = %s;\n"
					 "  else\n"
					 "    rc.isnull = CL_TRUE;\n"
					 "  kern_set_error(!rc.isnull && rc.value != 0\n"
					 "                 ? StromError_Success\n"
					 "                 : StromError_RowFiltered);\n"
					 "  gpuscan_writeback_result(gpuscan);\n"
					 "}\n"
					 "\n"
					 "__kernel void\n"
					 "gpuscan_qual_rs(__global kern_gpuscan *gpuscan,\n"
					 "                __global kern_row_store *krs,\n"
					 "                __global kern_column_store *kcs,\n"
					 "                __local void *local_workmem)\n"
					 "{\n"
					 "  kern_row_to_column(krs,kcs,\n"
					 "                     lengthof(used_vars),\n"
					 "                     used_vars,\n"
					 "                     local_workmem);\n"
					 "  gpuscan_qual_cs(gpuscan,kparams,kcs,\n"
					 "                  (kern_toastbuf *)krs,\n"
					 "                  local_workmem);\n"
					 "}\n", expr_code);
	return str.data;
}

static CustomPlan *
gpuscan_create_plan(PlannerInfo *root, CustomPath *best_path)
{
	RelOptInfo	   *rel = best_path->path.parent;
	GpuScanPath	   *gpath = (GpuScanPath *)best_path;
	GpuScanPlan	   *gscan;
	List		   *tlist;
	List		   *host_clauses;
	List		   *dev_clauses;
	char		   *kern_source;
	codegen_context	context;

	/*
	 * See the comments in create_scan_plan(). We may be able to omit
	 * projection of the table tuples, if possible.
	 */
	if (use_physical_tlist(root, rel))
	{
		tlist = build_physical_tlist(root, rel);
		if (tlist == NIL)
			tlist = build_path_tlist(root, &best_path->path);
	}
	else
		tlist = build_path_tlist(root, &best_path->path);

	/* it should be a base relation */
	Assert(rel->relid > 0);
	Assert(rel->rtekind == RTE_RELATION);

	/* Sort clauses into best execution order */
	host_clauses = order_qual_clauses(root, gpath->host_quals);
	dev_clauses = order_qual_clauses(root, gpath->dev_quals);

	/* Reduce RestrictInfo list to bare expressions; ignore pseudoconstants */
	host_clauses = extract_actual_clauses(host_clauses, false);
	dev_clauses = extract_actual_clauses(dev_clauses, false);

	/* Replace any outer-relation variables with nestloop params */
	if (best_path->path.param_info)
	{
		host_clauses = (List *)
			replace_nestloop_params(root, (Node *) host_clauses);
		dev_clauses = (List *)
			replace_nestloop_params(root, (Node *) dev_clauses);
	}

	/*
	 * Construct OpenCL kernel code - A kernel code contains two forms of
	 * entrypoints; for row-store and column-store. OpenCL intermediator
	 * invoked proper kernel function according to the class of data store.
	 * Once a kernel function for row-store is called, it translates the
	 * data format into column-store, then kicks jobs for row-evaluation.
	 * This design is optimized to process column-oriented data format on
	 * the relation cache.
	 */
	kern_source = gpuscan_codegen_quals(root, gpath->dev_quals, &context);

	/*
	 * Construction of GpuScanPlan node; on top of CustomPlan node
	 */
	gscan = palloc0(sizeof(GpuScanPlan));
	gscan->cplan.plan.type = T_CustomPlan;
	gscan->cplan.plan.targetlist = tlist;
	gscan->cplan.plan.qual = host_clauses;
	gscan->cplan.plan.lefttree = NULL;
	gscan->cplan.plan.righttree = NULL;

	gscan->scanrelid = rel->relid;
	gscan->kern_source = kern_source;
	gscan->extra_flags = context.extra_flags;
	gscan->used_params = context.used_params;
	gscan->used_vars = context.used_vars;
	gscan->dev_clauses = dev_clauses;
	gscan->dev_attnums = gpath->dev_attnums;
	gscan->host_attnums = gpath->host_attnums;

	return &gscan->cplan;
}

/* copy from outfuncs.c */
static void
_outBitmapset(StringInfo str, const Bitmapset *bms)
{
	Bitmapset  *tmpset;
	int			x;

	appendStringInfoChar(str, '(');
	appendStringInfoChar(str, 'b');
	tmpset = bms_copy(bms);
	while ((x = bms_first_member(tmpset)) >= 0)
		appendStringInfo(str, " %d", x);
	bms_free(tmpset);
	appendStringInfoChar(str, ')');
}

/* copy from outfuncs.c */
static void
_outToken(StringInfo str, const char *s)
{
	if (s == NULL || *s == '\0')
	{
		appendStringInfoString(str, "<>");
		return;
	}

	/*
	 * Look for characters or patterns that are treated specially by read.c
	 * (either in pg_strtok() or in nodeRead()), and therefore need a
	 * protective backslash.
	 */
	/* These characters only need to be quoted at the start of the string */
	if (*s == '<' ||
		*s == '\"' ||
		isdigit((unsigned char) *s) ||
		((*s == '+' || *s == '-') &&
		 (isdigit((unsigned char) s[1]) || s[1] == '.')))
		appendStringInfoChar(str, '\\');
	while (*s)
	{
		/* These chars must be backslashed anywhere in the string */
		if (*s == ' ' || *s == '\n' || *s == '\t' ||
			*s == '(' || *s == ')' || *s == '{' || *s == '}' ||
			*s == '\\')
			appendStringInfoChar(str, '\\');
		appendStringInfoChar(str, *s++);
	}
}

static void
gpuscan_textout_path(StringInfo str, Node *node)
{
	GpuScanPath	   *pathnode = (GpuScanPath *) node;
	char		   *temp;

	/* dev_quals */
	temp = nodeToString(pathnode->dev_quals);
	appendStringInfo(str, " :dev_quals %s", temp);
	pfree(temp);

	/* host_quals */
	temp = nodeToString(pathnode->host_quals);
	appendStringInfo(str, " :host_quals %s", temp);
	pfree(temp);

	/* dev_attnums */
	appendStringInfo(str, " :dev_attnums");
	_outBitmapset(str, pathnode->dev_attnums);

	/* host_attnums */
	appendStringInfo(str, " :host_attnums");
	_outBitmapset(str, pathnode->host_attnums);
}

static void
gpuscan_set_plan_ref(PlannerInfo *root,
					 CustomPlan *custom_plan,
					 int rtoffset)
{
	GpuScanPlan	   *gscan = (GpuScanPlan *)custom_plan;

	gscan->scanrelid += rtoffset;
	gscan->cplan.plan.targetlist = (List *)
		fix_scan_expr(root, (Node *)gscan->cplan.plan.targetlist, rtoffset);
	gscan->cplan.plan.qual = (List *)
		fix_scan_expr(root, (Node *)gscan->cplan.plan.qual, rtoffset);
	gscan->used_vars = (List *)
		fix_scan_expr(root, (Node *)gscan->used_vars, rtoffset);
	gscan->dev_clauses = (List *)
		fix_scan_expr(root, (Node *)gscan->dev_clauses, rtoffset);
}

static void
gpuscan_finalize_plan(PlannerInfo *root,
					  CustomPlan *custom_plan,
					  Bitmapset **paramids,
					  Bitmapset **valid_params,
					  Bitmapset **scan_params)
{
	*paramids = bms_add_members(*paramids, *scan_params);
}

static  CustomPlanState *
gpuscan_begin(CustomPlan *node, EState *estate, int eflags)
{
	GpuScanPlan	   *gsplan = (GpuScanPlan *) node;
	Index			scanrelid = gsplan->scanrelid;
	GpuScanState   *gss;
	TupleDesc		tupdesc;
	Bitmapset	   *tempset;
	AttrNumber		anum;

	/* gpuscan should not have inner/outer plan now */
	Assert(outerPlan(node) == NULL);
    Assert(innerPlan(node) == NULL);

	/*
	 * create a state structure
	 */
	gss = palloc0(sizeof(GpuScanState));
	gss->cps.ps.type = T_CustomPlanState;
	gss->cps.ps.plan = (Plan *) node;
	gss->cps.ps.state = estate;

	/*
	 * create expression context
	 */
	ExecAssignExprContext(estate, &gss->cps.ps);

	/*
	 * initialize child expressions
	 */
	gss->cps.ps.targetlist = (List *)
		ExecInitExpr((Expr *) node->plan.targetlist, &gss->cps.ps);
	gss->cps.ps.qual = (List *)
		ExecInitExpr((Expr *) node->plan.qual, &gss->cps.ps);

	/*
	 * tuple table initialization
	 */
	ExecInitResultTupleSlot(estate, &gss->cps.ps);
	gss->scan_slot = ExecAllocTableSlot(&estate->es_tupleTable); // needed?

	/*
	 * initialize scan relation
	 */
	gss->scan_rel = ExecOpenScanRelation(estate, scanrelid, eflags);
	gss->scan_desc = heap_beginscan(gss->scan_rel,
									estate->es_snapshot,
									0,
									NULL);
	tupdesc = RelationGetDescr(gss->scan_rel);
	ExecSetSlotDescriptor(gss->scan_slot, tupdesc);

	/*
	 * Initialize result tuple type and projection info.
	 */
	ExecAssignResultTypeFromTL(&gss->cps.ps);
	if (tlist_matches_tupdesc(&gss->cps.ps,
							  node->plan.targetlist,
							  scanrelid,
							  tupdesc))
		gss->cps.ps.ps_ProjInfo = NULL;
	else
		ExecAssignProjectionInfo(&gss->cps.ps, tupdesc);

	/*
	 * OK, initialization of common part is over.
	 * Let's have GPU stuff initialization
	 */
	gss->scan_mode = GpuScanMode_HeapOnlyScan;
	gss->mqueue = pgstrom_create_queue();
	gss->kparambuf = pgstrom_create_kern_parambuf(gsplan->used_params,
												  gss->cps.ps.ps_ExprContext);
	if (gsplan->kern_source)
		gss->dprog_key = pgstrom_get_devprog_key(gsplan->kern_source,
												 gsplan->extra_flags);
	else
		gss->dprog_key = 0;	/* it might happen if tc-cache got supported */

	gss->dev_colmeta = palloc(sizeof(kern_colmeta) * tupdesc->natts);
	for (anum=0; anum < tupdesc->natts; anum++)
	{
		Form_pg_attribute attr = tupdesc->attrs[anum];
		kern_colmeta   *colmeta = &gss->dev_colmeta[anum];
		AttrNumber		attidx
			= attr->attnum - FirstLowInvalidHeapAttributeNumber;

		if (attr->attnotnull)
			colmeta->flags |= KERN_COLMETA_ATTNOTNULL;
		if (bms_is_member(attidx, gsplan->dev_attnums))
			colmeta->flags |= KERN_COLMETA_ATTREFERENCED;

		if (attr->attalign == 'c')
			colmeta->attalign = sizeof(cl_char);
		else if (attr->attalign == 's')
			colmeta->attalign = sizeof(cl_short);
		else if (attr->attalign == 'i')
			colmeta->attalign = sizeof(cl_int);
		else if (attr->attalign == 'd')
			colmeta->attalign = sizeof(cl_long);
		else
			elog(ERROR, "unexpected attribute alignment: %c", attr->attalign);
		colmeta->attlen = attr->attlen;
		colmeta->cs_ofs = -1;	/* to be calculated for each row_store */
	}
	gss->dev_attnums = gsplan->dev_attnums;
	gss->curr_chunk = NULL;
	gss->curr_index = 0;
	gss->num_running = 0;
	dlist_init(&gss->ready_chunks);

	return &gss->cps;
}

static pgstrom_gpuscan *
pgstrom_load_gpuscan_row(GpuScanState *gss)
{
	pgstrom_gpuscan	   *gscan;
	pgstrom_row_store  *rstore;
	kern_result		   *kresult;
	kern_column_store  *kcs_head;
	ListCell		   *cell;
	HeapTuple	tuple;
	Size		offset;
	Size		usage;
	Size		length;
	int			index;
	int			anum;
	int			ncols = RelationGetNumberOfAttributes(gss->scan_rel);
	int			nrows;
	Bitmapset  *tempset;
	ScanDirection direction = gss->cps.ps.state->es_direction;

	rstore = pgstrom_shmem_alloc(ROWSTORE_DEFAULT_SIZE);
	if (!rstore)
		elog(ERROR, "out of shared memory");

	/*
	 * We put header portion of kern_column_store next to the kern_row_store
	 * as source of copy for in-kernel column store. It has offset of column
	 * array, but contents shall be set up by kernel prior to evaluation of
	 * qualifier expression.
	 */
	rstore->stag = StromTag_RowStore;
	rstore->kern_len =
		STROMALIGN_DOWN(ROWSTORE_DEFAULT_SIZE -
						offsetof(pgstrom_row_store, kern) -
						offsetof(pgstrom_column_store, colmeta[cs_ncols]));
	rstore->kern.ncols = rs_ncols;
	rstore->kern.nrows = 0;
	memcpy(rstore->kern.colmeta,
		   gss->dev_colmeta,
		   sizeof(kern_colmeta) * rs_ncols);

	/*
	 * OK, load tuples and put it on the row-store.
	 * Offset array of rs_tuples begins from the column-metadata
	 */
	offset = offsetof(kern_row_store, colmeta[ncols]);
	usage = rstore->kern_len;

	while (HeapTupleIsValid(tuple = heap_getnext(gss->scan_desc, direction)))
	{
		Size		length = HEAPTUPLESIZE + MAXALIGN(tuple->t_len);
		rs_tuple   *rs_tup;
		cl_uint	   *rs_ofs;

		if (usage - length < offset + sizeof(cl_uint))
		{
			/* rewind the heap-tuple */
			heap_getnext(gss->scan_desc, -direction);
			break;
		}
		usage -= length;
		rs_tup = (rs_tuple *)((uintptr_t)&rstore->kern + usage);
		memcpy(&rs_tup->htup, tuple, sizeof(HeapTupleData));
		rs_tup->htup.t_data = &rs_tup->data;
		memcpy(&rs_tup->data, tuple->t_data, tuple->t_len);

		rs_ofs = (cl_uint *)((uintptr_t)&rstore->kern + offset);
		*rs_ofs = usage;
		offset += sizeof(cl_uint);

		rstore->kern.nrows++;
	}
	nrows = rstore->kern.nrows;

	/* if table scan reached end of the relation, close the ScanDesc */
	if (!HeapTupleIsValid(tuple))
	{
		heap_endscan(gss->scan_desc);
		gss->scan_desc = NULL;
	}

	/*
	 * On tail of the shared-memory block, we put header portion of
	 * the kern_column_store; to be copied to in-kernel structure.
	 */
	kcs_head = (kern_column_store *)((char *)(&rstore->kern) +
									 rstore->kern_len);
	kcs_head->ncols = bms_num_members(gss->dev_attnums);
	kcs_head->nrows = nrows;

	index = 0;
	offset = offsetof(kern_column_store, colmeta[kcs_head->ncols]);
	tempset = bms_copy(gss->dev_attnums);
	while ((anum = bms_first_member(tempset)) >= 0)
	{
		kern_colmeta   *colmeta;

		anum += FirstLowInvalidHeapAttributeNumber;
		Assert(anum > 0 && anum <= ncols);

		colmeta = &kcs_head->colmeta[index];
		memcpy(colmeta, &rstore->kern.colmeta[anum-1], sizeof(kern_colmeta));
		colmeta->cs_ofs = offset;
		if ((colmeta->flags & KERN_COLMETA_ATTNOTNULL) == 0)
			offset += STROMALIGN((nrows + 7) / 8);
		offset += STROMALIGN(nrows * (colmeta->attlen > 0
									  ? colmeta->attlen
									  : sizeof(cl_uint)));
		index++;
	}
	bmc_free(tempset);
	Assert(index == kcs_head->ncols);

	/*
	 * OK, pgstrom_row_store was fully setup.
	 * Let's create a gpuscan object with the row-store.
	 */
	length = offsetof(pgstrom_gpuscan, kern.kparam) +
		STROMALIGN(gss->kparambuf->length) +
		STROMALIGN(offsetof(kern_result, results[nrows]));
	gscan = pgstrom_shmem_alloc(length);
	if (!gscan)
	{
		pgstrom_shmem_free(rstore);
		elog(ERROR, "out of shared memory");
	}
	/* fields of pgstrom_gpuscan */
	gscan->msg.stag = StromTag_GpuScan;
	SpinLockInit(&gscan->msg.lock);
	gscan->msg.refcnt = 1;
	gscan->msg.respq = pgstrom_get_queue(gss->mqueue);
	gscan->msg.cb_process = clserv_process_gpuscan_row;
	gscan->msg.cb_release = clserv_put_gpuscan_row;
	gscan->dprog_key = pgstrom_retain_devprog_key(gss->dprog_key);
	dlist_init(&gscan->rc_store);
	dlist_push_head(&gscan->rc_store, &rstore->chain);
	/* kern_parambuf part */
	memcpy(&gscan->kern.kparam, gss->kparambuf, gss->kparambuf->length);
	gscan->kern.roffset = STROMALIGN(gss->kparambuf->length);
	/* kern_resultbuf part */
	kresult = (kern_resultbuf *)((uintptr_t)&gscan->kern +
								 gscan->kern.roffset);
	kresult->nitems = 0;
	kresult->errcode = 0;

	/* track local object */
	pgstrom_track_object(&gscan->msg.stag);

	return gscan;
}

static bool
gpuscan_next_tuple_row(GpuScanState *gss, TupleTableSlot *slot)
{
	pgstrom_gpuscan	*gscan = gss->curr_chunk;
	kern_result		*kresult;
	rs_tuple		*rs_tup;
	cl_int			 index;

	if (!gscan)
		return false;

	kresult = pgstrom_gpuscan_kresult(gscan);
	if (gss->curr_index < kresult->nitems)
	{
		pgstrom_row_store *rstore
			= dlist_container(pgstrom_row_store, chain,
							  dlist_head_node(&gscan->rc_store));
		Assert(rstore->stag == StromTag_RowStore);

		index = kresult->results[gss->curr_index++];
		Assert(index < rstore->kern.nrows);

		rs_tup = pgstrom_row_store_rstuple(rstore, index);
		ExecStoreTuple(&rs_tup->htup, slot, InvalidBuffer, false);
		return true;
	}
	return false;
}

static TupleTableSlot *
gpuscan_exec(CustomPlanState *node)
{
	GpuScanState   *gss = (GpuScanState *) node;
	TupleTableSlot *slot = gss->scan_slot;

	ExecClearTuple(slot);

	while (!gss->curr_chunk || !gpuscan_next_tuple_row(gss, slot))
	{
		pgstrom_message	   *msg;
		pgstrom_gpuscan	   *gscan;
		kern_result		   *kresult;

		/*
		 * Release the current gpuscan chunk being already scanned
		 */
		if (gss->curr_chunk)
		{
			pgstrom_message	   *msg = &gss->curr_chunk.msg;

			msg->cb_release(msg);
			gss->curr_chunk = NULL;
			gss->curr_index = 0;
		}

		/*
		 * Dequeue the current gpuscan chunks being already processed
		 */
		while ((msg = pgstrom_try_dequeue_message(gss->mqueue)) != NULL)
		{
			Assert(gss->num_running > 0);
			gss->num_running--;
			dlist_push_head(&gss->ready_chunks, &msg->chain);
		}

		/*
		 * Try to keep number of gpuscan chunks being asynchronously executed
		 * larger than minimum multiplicity, unless it does not exceed
		 * maximum one and OpenCL server does not return a new response.
		 */
		while (gss->scan_desc != NULL &&
			   gss->num_running <= pgstrom_max_async_chunks)
		{
			pgstrom_gpuscan	*gscan = pgstrom_load_gpuscan_row(gss);

			if (!pgstrom_enqueue_message(&gscan->msg))
			{
				gscan->msg.cb_release(&gscan->msg);
				elog(ERROR, "failed to enqueue pgstrom_gpuscan message");
			}
			gss->num_running++;

			if (gss->num_running > pgstrom_min_async_chunks &&
				(msg = pgstrom_try_dequeue_message(gss->mqueue)) != NULL)
			{
				gss->num_running--;
				dlist_push_head(&gss->ready_chunks, &msg->chain);
				break;
			}
		}

		/*
		 * Wait for server's response if no available chunks were replied.
		 */
		if (dlist_is_empty(&gss->ready_chunks))
		{
			/* OK, no more chunks to be scanned */
			if (gss->num_running == 0)
				break;

			msg = pgstrom_dequeue_message(gss->mqueue);
			if (!msg)
				elog(ERROR, "message queue wait timeout");
			gss->num_running--;
			dlist_push_head(&gss->ready_chunks, &msg->chain);
		}

		/*
		 * Picks up next available chunks if any
		 */
		Assert(!dlist_is_empty(&gss->ready_chunks));
		gscan = dlist_container(pgstrom_gpuscan, msg.chain,
								dlist_pop_head_node(&gss->ready_chunks));
		Assert(gscan->msg.stag == StromTag_GpuScan);

		/*
         * Raise an error, if chunk-level error was reported.
         */
		if (gscan->msg.errcode != StromError_Success)
		{
			cl_int	errcode = gscan->msg.errcode;

			if (errcode ==StromError_ProgramCompile)
			{
				const char *buildlog
					= pgstrom_get_devprog_errmsg(gscan->dprog_key);

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)",
								pgstrom_strerror(errcode)),
						 errdetail("%s", buildlog)));
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)",
								pgstrom_strerror(errcode))));
			}
		}
		gss->curr_chunk = gscan;
		gss->curr_index = 0;
	}
	return slot;
}

static Node *
gpuscan_exec_multi(CustomPlanState *node)
{
	elog(ERROR, "not implemented yet");
}

static void
gpuscan_end(CustomPlanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	if (gss->dprog_key)
		pgstrom_put_devprog_key(gss->dprog_key);
	pgstrom_close_queue(gss->mqueue);

	/*
	 * Free the exprcontext
	 */
	ExecFreeExprContext(&gss->cps.ps);

	/*
	 * clean out the tuple table
	 */
	ExecClearTuple(gss->cps.ps.ps_ResultTupleSlot);
	ExecClearTuple(gss->scan_slot);

	/*
	 * close heap scan and relation
	 */
	heap_endscan(gss->scan_desc);
	heap_close(gss->scan_rel, NoLock);
}

static void
gpuscan_rescan(CustomPlanState *node)
{
	elog(ERROR, "not implemented yet");
}

static void
gpuscan_explain_rel(CustomPlanState *node, ExplainState *es)
{
	GpuScanState   *gss = (GpuScanState *)node;
	GpuScanPlan	   *gsplan = (GpuScanPlan *)gss->cps.ps.plan;
	Relation		rel = gss->scan_rel;
	RangeTblEntry  *rte;
	char		   *refname;
	char		   *objectname;
	char		   *namespace = NULL;

	rte = rt_fetch(gsplan->scanrelid, es->rtable);
	refname = (char *) list_nth(es->rtable_names,
								gsplan->scanrelid - 1);
	if (refname == NULL)
		refname = rte->eref->aliasname;

	Assert(rte->rtekind == RTE_RELATION);
	objectname = RelationGetRelationName(rel);
	if (es->verbose)
		namespace = get_namespace_name(RelationGetNamespace(rel));

	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		appendStringInfoString(es->str, " on");
		if (namespace != NULL)
			appendStringInfo(es->str, " %s.%s",
							 quote_identifier(namespace),
							 quote_identifier(objectname));
		else
			appendStringInfo(es->str, " %s",
							 quote_identifier(objectname));
	}
	else
	{
		ExplainPropertyText("Relation Name", objectname, es);
		if (namespace != NULL)
			ExplainPropertyText("Schema", namespace, es);
		ExplainPropertyText("Alias", refname, es);
	}
}

static void
gpuscan_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{}

static Bitmapset *
gpuscan_get_relids(CustomPlanState *node)
{
	GpuScanPlan	   *gsp = (GpuScanPlan *)node->ps.plan;

	return bms_make_singleton(gsp->scanrelid);
}

static void
gpuscan_textout_plan(StringInfo str, const CustomPlan *node)
{
	GpuScanPlan	   *plannode = (GpuScanPlan *)node;
	char		   *temp;

	appendStringInfo(str, " :scanrelid %u", plannode->scanrelid);

	appendStringInfo(str, " :kern_source ");
	_outToken(str, plannode->kern_source);

	appendStringInfo(str, " :extra_flags %u", plannode->scanrelid);

	temp = nodeToString(plannode->used_params);
	appendStringInfo(str, " :used_params %s", temp);
	pfree(temp);

	temp = nodeToString(plannode->used_vars);
	appendStringInfo(str, " :used_vars %s", temp);
	pfree(temp);

	temp = nodeToString(plannode->dev_clauses);
	appendStringInfo(str, " :dev_clauses %s", temp);
	pfree(temp);

	appendStringInfo(str, " :dev_attnums ");
	_outBitmapset(str, plannode->dev_attnums);

	appendStringInfo(str, " :host_attnums ");
	_outBitmapset(str, plannode->host_attnums);
}

static CustomPlan *
gpuscan_copy_plan(const CustomPlan *from)
{
	GpuScanPlan	   *oldnode = (GpuScanPlan *)from;
	GpuScanPlan	   *newnode = palloc(sizeof(GpuScanPlan));

	CopyCustomPlanCommon((Node *)from, (Node *)newnode);
	newnode->scanrelid = oldnode->scanrelid;
	newnode->used_params = copyObject(oldnode->used_params);
	newnode->used_vars = copyObject(oldnode->used_vars);
	newnode->extra_flags = oldnode->extra_flags;
	newnode->dev_clauses = oldnode->dev_clauses;
	newnode->dev_attnums = bms_copy(oldnode->dev_attnums);
	newnode->host_attnums = bms_copy(oldnode->host_attnums);

	return &newnode->cplan;
}

void
pgstrom_init_gpuscan(void)
{
	/* GUC definition */
	DefineCustomBoolVariable("pgstrom.enable_gpuscan",
							 "Enables the planner's use of GPU-scan plans.",
							 NULL,
							 &enable_gpuscan,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* setup path methods */
	gpuscan_path_methods.CustomName			= "GpuScan";
	gpuscan_path_methods.CreateCustomPlan	= gpuscan_create_plan;
	gpuscan_path_methods.TextOutCustomPath	= gpuscan_textout_path;

	/* setup plan methods */
	gpuscan_plan_methods.CustomName			= "GpuScan";
	gpuscan_plan_methods.SetCustomPlanRef	= gpuscan_set_plan_ref;
	gpuscan_plan_methods.SupportBackwardScan= NULL;
	gpuscan_plan_methods.FinalizeCustomPlan	= gpuscan_finalize_plan;
	gpuscan_plan_methods.BeginCustomPlan	= gpuscan_begin;
	gpuscan_plan_methods.ExecCustomPlan		= gpuscan_exec;
	gpuscan_plan_methods.MultiExecCustomPlan= gpuscan_exec_multi;
	gpuscan_plan_methods.EndCustomPlan		= gpuscan_end;
	gpuscan_plan_methods.ReScanCustomPlan	= gpuscan_rescan;
	gpuscan_plan_methods.ExplainCustomPlanTargetRel	= gpuscan_explain_rel;
	gpuscan_plan_methods.ExplainCustomPlan	= gpuscan_explain;
	gpuscan_plan_methods.GetRelidsCustomPlan= gpuscan_get_relids;
	gpuscan_plan_methods.GetSpecialCustomVar= NULL;
	gpuscan_plan_methods.TextOutCustomPlan	= gpuscan_textout_plan;
	gpuscan_plan_methods.CopyCustomPlan		= gpuscan_copy_plan;

	/* hook registration */
	add_scan_path_next = add_scan_path_hook;
	add_scan_path_hook = gpuscan_add_scan_path;
}

/*
 * GpuScan message handler
 * ------------------------------------------------------------
 * Note that below routines are executed in the context of OpenCL
 * intermediation server, thus, usual PostgreSQL internal APIs are
 * not available, and need to pay attention that routines work in
 * another process's address space.
 */


typedef struct
{
	pgstrom_message	*msg;
	cl_program		program;
	cl_kernel		kernel;
	cl_mem			m_gpuscan;
	cl_mem			m_rstore;
	cl_mem			m_cstore;

	kern_gpuscan   *kgscan;
	kern_row_store *krstore;
	cl_event		events[FLEXIBLE_ARRAY_MEMBER];
} clstate_gpuscan_row;


static void
clserv_process_gpuscan_row(pgstrom_message *msg)
{
	pgstrom_gpuscan	   *gscan = (pgstrom_gpuscan *)msg;
	pgstrom_row_store  *rstore;
	clstate_gpuscan_row *clgss;
	kern_colmeta	   *colmeta;
	cl_uint				nrows;
	cl_uint				ncols;
	cl_uint				i;
	size_t				length;
	size_t				wkgrp_sz;
	size_t				gwork_ofs;
	size_t				gwork_sz;
	size_t				lwork_sz;

	/* only one store shall be attached! */
	Assert(dlist_head_node(&gscan->rc_store) ==
		   dlist_tail_node(&gscan->rc_store));
	rstore = dlist_container(pgstrom_row_store, chain,
							 dlist_head_node(&gscan->rc_store));

	/* state object of gpuscan with row-store */
	clgss = malloc(sizeof(clstate_gpuscan_row));
	if (!clgss)
	{
		msg->errcode = StromError_OutOfMemory;
		pgstrom_reply_message(msg);
		return;
	}
	memset(clgss, 0, sizeof(clstate_gpuscan_row));
	clgss->msg = &gscan->msg;
	clgss->kgscan  = &gscan->kern;
	clgss->krstore = &rstore->kern;

	nrows = clgss->krstore->nrows;
	ncols = clgss->krstore->ncols;

	/*
	 * First of all, it looks up a program object to be run on
	 * the supplied row-store. We may have three cases.
	 * 1) NULL; it means the required program is under asynchronous
	 *    build, and the message is kept on its internal structure
	 *    to be enqueued again. In this case, we have nothing to do
	 *    any more on the invocation.
	 * 2) BAD_OPENCL_PROGRAM; it means previous compile was failed
	 *    and unavailable to run this program anyway. So, we need
	 *    to reply StromError_ProgramCompile error to inform the
	 *    backend this program.
	 * 3) valid cl_program object; it is an ideal result. pre-compiled
	 *    program object was on the program cache, and cl_program
	 *    object is ready to use.
	 */
	clgss->program = clserv_lookup_device_program(gscan->dprog_key, msg);
	if (!clgss->program)
		return;
	if (clgss->program == BAD_OPENCL_PROGRAM)
		goto error;

	/*
	 * In this case, we use a kernel for row-store; that internally
	 * translate row-format into column-format, then evaluate the
	 * supplied qualifier.
	 */
	clgss->kernel = clCreateKernel(clgss->program,
								   "gpuscan_qual_rs",
								   &rc);
	if (rc != CL_SUCCESS)
		goto error;

	/* allocation of device memory for kern_gpuscan argument */
	length = STROMALIGN(KERN_GPUSCAN_LENGTH(clgss->kgscan, nrows));
	clgss->m_gpuscan = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  length,
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
		goto error;

	/* allocation of device memory for kern_row_store argument */
	length = STROMALIGN(KERN_ROWSTORE_LENGTH(rstore));
	clgss->m_rstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 length,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
		goto error;

	/* allocation of device memory for kern_column_store argument */
	length = STROMALIGN(offsetof(kern_column_store, colmeta[ncols]));
	for (i=0; i < ncols; i++)
	{
		kern_colmeta   *colmeta = &clgss->krstore->colmeta[i];

		if (colmeta->flags & KERN_COLMETA_ATTREFERENCED)
			length += (STROMALIGN((nrows + 7) / 8) +
					   STROMALIGN(nrows * (colmeta->attlen > 0
										   ? colmeta->attlen
										   : sizeof(cl_uint))));
	}
	clgss->m_cstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 length,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
		goto error;

	/*
	 * Calculate an optimized workgroup-size for the main logic.
	 * (Note that kernel_prep shall run single-threaded)
	 */
	rc = clGetKernelWorkGroupInfo(clgss->kernel_qual,
								  opencl_devices[dindex],
								  CL_KERNEL_WORK_GROUP_SIZE,
								  sizeof(wkgrp_sz),
								  &wkgrp_sz,
								  NULL);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clGetKernelWorkGroupInfo: %s",
			 opencl_strerror(rc));
		goto error;
	}

	/*
	 * OK, all the device memory and kernel objects acquired.
	 * Let's prepare kernel invocation.
	 *
	 * The first call is:
	 * __kernel void
	 * gpuscan_qual_rs_prep(__global kern_row_store *krs,
	 *                      __global kern_column_store *kcs)
	 */
	rc = clSetKernelArg(clgss->kernel_prep,
						0,	/* kern_row_store * */
						sizeof(cl_mem),
						&clgss->m_rstore);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clgss->kernel_prep,
						1,	/* kern_column_store */
						sizeof(cl_mem),
						&clgss->m_cstore);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * The second call is:
	 * __kernel void
	 * gpuscan_qual_rs(__global kern_gpuscan *gpuscan,
	 *                 __global kern_row_store *krs,
	 *                 __global kern_column_store *kcs,
	 *                 __local void *local_workmem)
	 */
	rc = clSetKernelArg(clgss->kernel_qual,
						0,	/* kern_gpuscan * */
						sizeof(cl_mem),
						&clgss->m_gpuscan);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clgss->kernel_qual,
						1,	/* kern_row_store */
						sizeof(cl_mem),
						&clgss->m_rstore);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clgss->kernel_qual,
						2,	/* kern_column_store */
						sizeof(cl_mem),
						&clgss->m_cstore);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clgss->kernel_qual,
						3,	/* local_workmem */
						2 * sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * OK, enqueue DMA transfer, kernel execution x2, then DMA writeback.
	 *
	 * (1) gpuscan and row_store shall be copied from host to device
	 * (2) kernel_prep shall be launched
	 * (3) kernel_qual shall be launched
	 * (4) kern_result shall be written back
	 */
	length = clgss->kgscan.roffset + offsetof(kern_result, results[0]);
	rc = clEnqueueWriteBuffer(opencl_cmdq[dindex],
							  clgss->m_gpuscan,
							  CL_FALSE,
							  0,
							  length,
							  clgss->kgscan,
							  0,
							  NULL,
							  &clgss->events[0]);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	length = KERN_ROWSTORE_LENGTH(rstore);
	rc = clEnqueueWriteBuffer(opencl_cmdq[dindex],
							  clgss->m_rstore,
							  CL_FALSE,
							  0,
							  length,
							  clgss->krstore,
							  0,
							  NULL,
							  &clgss->events[1]);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * Kick gpuscan_qual_rs_prep() call
	 */
	gwork_ofs = 0;
	gwork_sz = 1;
	lwork_sz = 1;

	rc = clEnqueueNDRangeKernel(opencl_cmdq[dindex],
								clgss->kernel_prep,
								1,
								&gwork_ofs,
								&gwork_sz,
								&lwork_sz,
								2,
								&clgss->events[0],
								&clgss->events[2]);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clEnqueueNDRangeKernel: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * Kick gpuscan_qual_rs() call
	 */
	gwork_ofs = 0;
	gwork_sz = (nrows + wkgrp_sz - 1) / wkgrp_sz;
	lwork_sz = wkgrp_sz;

	rc = clEnqueueNDRangeKernel(opencl_cmdq[dindex],
								clgss->kernel_qual,
								1,
								&gwork_ofs,
								&gwork_sz,
								&lwork_sz,
								1,
								&clgss->events[2],
								&clgss->events[3]);
	if (rc != CL_SUCCESS)
	{
		elog(LOG, "failed on clEnqueueNDRangeKernel: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * Write back the result-buffer
	 */
	length = clgss->kgscan.roffset + offsetof(kern_result, results[0]);
	rc = clEnqueueReadBuffer(opencl_cmdq[dindex],
							 clgss->m_gpuscan,
							 CL_FALSE,
							 clgss->kgscan.roffset,
							 offsetof(kern_result, results[nrows]),
							 (char *)clgss->kgscan + clgss->kgscan.roffset,
							 1,
							 &clgss->events[3],
							 &clgss->events[4]);

	/*
	 * Last, registers a callback routine that replies the message
	 * to the backend
	 */
	rc = clSetEventCallback(clgss->events[4],
							CL_COMPLETE,
							my_callback,
							clgss);
	if (rc != CL_SUCCESS)
	{
		/*
		 * If an error occured, we need to synchronize the message
		 *
		 */



	}
	return;

error:


	if (clgss->kernel_qual)
		clReleaseKernel(clgss->kernel_qual);
	if (clgss->kernel_prep)
		clReleaseKernel(clgss->kernel_prep);
	if (clgss->program)
		clReleaseProgram(program);
	msg->errcode = StromError_OpenCLInterface;
	pgstrom_reply_message(msg);
}

/*
 * clserv_put_gpuscan_row
 *
 * Callback handler when reference counter of gpuscan is decreased.
 * It also unlinks a device program and release row-store being
 * associated.
 * Also note that this routine might be invoked by OpenCL server.
 */
static void
clserv_put_gpuscan_row(pgstrom_message *msg)
{
	pgstrom_gpuscan	   *gscan = (pgstrom_gpuscan *)msg;
	bool				gpuscan_release = false;

	SpinLockAcquire(&msg->lock);
	if (--gscan->msg.refcnt == 0)
		gpuscan_release = true;
	SpinLockRelease(&msg->lock);

	/* untrack local resource */
	pgstrom_untrack_object(&gscan->msg.stag);

	if (gpuscan_release)
	{
		dlist_mutable_iter	iter;

		pgstrom_put_devprog_key(gscan->dprog_key);

		dlist_foreach_modify(iter, &gscan->rc_store)
		{
			pgstrom_row_store  *rstore
				= dlist_container(pgstrom_row_store, chain, iter.cur);
			Assert(rstore->stag == StromTag_RowStore);
			pgstrom_shmem_free(rstore);
		}
		pgstrom_shmem_free(gscan);
	}
}
