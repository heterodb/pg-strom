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
#include "node/relation.h"
#include "pg_strom.h"

static add_scan_path_hook_type	add_scan_path_next;
static CustomPathMethods		gpuscan_path_methods;
static CustomPlanMethods		gpuscan_plan_methods;
static bool						enable_gpuscan;

typedef struct {
	CustomPath	cpath;
	List	   *dev_quals;		/* RestrictInfo run on device */
	List	   *host_quals;		/* RestrictInfo run on host */
	Bitmapset  *dev_attrs;		/* attrs referenced in device */
	Bitmapset  *host_attrs;		/* attrs referenced in host */
} GpuScanPath;

typedef struct {
	CustomPlan	cplan;
	Index		scanrelid;		/* index of the range table */
	List	   *used_params;	/* list of Const/Param in use */
	List	   *used_vars;		/* list of Var in use */
	int			extra_flags;	/* extra libraries to be included */
	List	   *dev_clauses;	/* clauses to be run on device */
	Bitmapset  *dev_attrs;		/* attrs referenced in device */
	Bitmapset  *host_attrs;		/* attrs referenced in host */
} GpuScanPlan;

typedef struct {
	PlanState	ps;
	Relation	rel;
	

} GpuScanState;


static void
cost_gpuscan(GpuScanPath *gpu_path, PlannerInfo *root,
			 RelOptInfo *baserel, ParamPathInfo *param_info,
			 List *dev_quals, List *host_quals)
{
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
	Bitmapset	   *dev_attrs = NULL;
	Bitmapset	   *host_attrs = NULL;
	bool			has_sysattr = false;
	ListCell	   *cell;
	codegen_context	context;

	/* check whether qualifier can run on GPU device */
	memset(&context, 0, sizeof(codegen_context));
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_codegen_available_expression(rinfo->clause))
		{
			pull_varattnos(rinfo->clause, &dev_attrs);
			dev_quals = lappend(dev_quals, rinfo);
		}
		else
		{
			pull_varattnos(rinfo->clause, &host_attrs);
			host_quals = lappend(host_quals, rinfo);
		}
	}
	/* also, picks up Var nodes in the target list */
	pull_varattnos(baserel->reltargetlist, &host_attrs);

	/*
	 * FIXME: needs to pay attention for projection cost.
	 * It may make sense to use build_physical_tlist, if host_attrs
	 * are much wider than dev_attrs.
	 * Anyway, it needs investigation of the actual behavior.
	 */

	/* XXX - check whether columnar cache may be available */

	/*
	 * Construction of a custom-plan node.
	 */
	pathnode = palloc0(GpuScanPath);
	pathnode->cpath.path.type = T_CustomPath;
	pathnode->cpath.path.pathtype = T_CustomPlan;
	pathnode->cpath.path.parent = baserel;
	pathnode->cpath.path.param_info
		= get_baserel_parampathinfo(root, rel, rel->lateral_relids);
	pathnode->cpath.path.pathkeys = NIL;	/* gpuscan has unsorted result */

	cost_gpuscan(pathnode, root, rel,
				 pathnode->cpath.path->param_info,
				 dev_quals, host_quals);

	pathnode->dev_quals = dev_quals;
	pathnode->host_quals = host_quals;
	pathnode->dev_attrs = dev_attrs;
	pathnode->host_attrs = host_attrs;

	add_paths(rel, &pathnode->cpath.path);
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
					 "gpuscan_qual_cs(__global kern_gpuscan *gpuscan,\n"
					 "                __global kern_parambuf *kparams,\n"
					 "                __global kern_column_store *kcs,\n"
					 "                __global kern_toastbuf *toast,\n"
					 "                __local void *local_workmem)\n"
					 "{\n"
					 "  pg_bool_t   rc;\n"
					 "  cl_int      errcode;\n"
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
					 "gpuscan_qual_rs_prep(__global kern_row_store *krs,\n"
					 "                     __global kern_column_store *kcs)\n"
					 "{\n"
					 "  kern_row_to_column_prep(krs,kcs,\n"
					 "                          lengthof(used_vars),\n"
					 "                          used_vars);\n"
					 "}\n"
					 "\n"
					 "__kernel void\n"
					 "gpuscan_qual_rs(__global kern_gpuscan *gpuscan,\n"
					 "                __global kern_parambuf *kparams,\n"
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
	Datum			dprog_key;
	codegen_context	context;

	/*
	 * See the comments in create_scan_plan(). We may be able to omit
	 * projection of the table tuples, if possible.
	 */
	if (use_physical_tlist(root, rel))
	{
		tlist = build_physical_tlist(root, rel);
		if (tlist == NIL)
			tlist = build_path_tlist(root, best_path);
	}
	else
		tlist = build_path_tlist(root, best_path);

	/* it should be a base relation */
	Assert(rel->relid > 0);
	Assert(rel->relkind == RTE_RELATION);

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
	kern_source = gpuscan_codegen_quals(gpath->dev_quals, &context);
	dprog_key = pgstrom_get_devprog_key(kern_source, context->incl_flags);

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
	gscan->used_params = context->used_params;
	gscan->used_vars = context->used_vars;
	gscan->extra_flags = context->extra_flags;
	gscan->dev_clauses = dev_clauses;
	gscan->dev_attrs = gpath->dev_attrs;
	gscan->host_attrs = gpath->host_attrs;

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

static void
gpuscan_textout_path(StringInfo str, Node *node)
{
	GpuScanPath	   *pathnode = (GpuScanPath *) node;
	Bitmapset	   *tempset;
	char		   *temp;
	int				x;

	/* dev_quals */
	temp = nodeToString(pathnode->dev_quals);
	appendStringInfo(str, " :dev_quals %s", temp);
	pfree(tmep);

	/* host_quals */
	temp = nodeToString(pathnode->host_quals);
	appendStringInfo(str, " :host_quals %s", temp);
	pfree(temp);

	/* dev_attrs */
	appendStringInfo(str, " :dev_attrs");
	_outBitmapset(str, pathnode->dev_attrs);

	/* host_attrs */
	appendStringInfo(str, " :host_attrs");
	_outBitmapset(str, pathnode->host_attrs);
}

static void
gpuscan_set_plan_ref(PlannerInfo *root,
					 CustomPlan *custom_plan,
					 int rtoffset)
{
	GpuScanPlan	   *gscan = (GpuScanPlan *)custom_plan;

	gscan->scanrelid += rtoffset;
	gscan->cplan.plan.targetlist = (List *)
		fix_scan_expr(root, (List *)gscan->cplan.plan.targetlist, rtoffset);
	gscan->cplan.plan.qual = (List *)
		fix_scan_expr(root, (List *)gscan->cplan.plan.qual, rtoffset);
	gscan->used_vars = (List *)
		fix_scan_expr(root, (List *)gscan->used_vars, rtoffset);
	gscan->dev_clauses = (List *)
		fix_scan_expr(root, (List *)gscan->dev_clauses, rtoffset);
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
gpuscan_begin(CustomPlan *custom_plan, EState *estate, int eflags)
{}

static TupleTableSlot *
gpuscan_exec(CustomPlanState *node)
{}

static Node *
gpuscan_exec_multi(CustomPlanState *node)
{}

static void
gpuscan_end(CustomPlanState *node)
{}

static void
gpuscan_rescan(CustomPlanState *node)
{}

static void
gpuscan_explain_rel(CustomPlanState *node, ExplainState *es)
{}

static void
gpuscan_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{}

static Bitmapset *
gpuscan_get_relids(CustomPlanState *node)
{}

static void
gpuscan_textout_plan(StringInfo str, const CustomPlan *node)
{
	GpuScanPlan	   *plannode = (GpuScanPlan *)node;
	char		   *temp;

	appendStringInfoChar(str, " :scanrelid %u", plannode->scanrelid);

	temp = nodeToString(plannode->used_params);
	appendStringInfo(str, " :used_params %s", temp);
	pfree(temp);

	temp = nodeToString(plannode->used_vars);
	appendStringInfo(str, " :used_vars %s", temp);
	pfree(temp);

	appendStringInfo(str, " :extra_flags %u", plannode->scanrelid);

	temp = nodeToString(plannode->dev_clauses);
	appendStringInfo(str, " :dev_clauses %s", temp);
	pfree(temp);

	appendStringInfo(str, " :dev_attrs ");
	_outBitmapset(str, plannode->dev_attrs);

	appendStringInfo(str, " :host_attrs ");
	_outBitmapset(str, plannode->host_attrs);
}

static CustomPlan *
gpuscan_copy_plan(const CustomPlan *from)
{
	GpuScanPlan	   *newnode = palloc(sizeof(GpuScanPlan));

	CopyCustomPlanCommon(from, newnode);
	newnode->scanrelid = from->scanrelid;
	newnode->used_params = copyObject(from->used_params);
	newnode->used_vars = copyObject(from->used_vars);
	newnode->extra_flags = from->extra_flags;
	newnode->dev_clauses = from->dev_clauses;
	newnode->dev_attrs = bms_copy(from->dev_attrs);
	newnode->host_attrs = bms_copy(from->host_attrs);

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
