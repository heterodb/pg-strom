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
	CustomPath	cplan;
	List	   *dev_quals;		/* RestrictInfo run on device */
	List	   *host_quals;		/* RestrictInfo run on host */
	Bitmapset  *dev_attrs;		/* attrs referenced in device */
	Bitmapset  *host_attrs;		/* attrs referenced in host */
} GpuScanPath;

typedef struct {
	CustomPlan	cplan;
	Index		scanrelid;		/* index of the range table */
	List	   *type_defs;		/* list of devtype_info in use */
	List	   *func_defs;		/* list of devfunc_info in use */
	List	   *used_params;	/* list of Const/Param in use */
	List	   *used_vars;		/* list of Var in use */
	List	   *dev_quals;		/* RestrictInfo run on device */
	List	   *host_quals;		/* RestrictInfo run on host */
	Bitmapset  *dev_attrs;		/* attrs referenced in device */
	Bitmapset  *host_attrs;		/* attrs referenced in host */
	int			incl_flags;		/* external libraries to be included */
} GpuScanPlan;

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
	index = 1;
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
	index = 1;
	foreach (cell, context->used_vars)
	{
		Var	   *var = lfirst(cell);

		dtype = pgstrom_devtype_lookup(var->vartype);
		Assert(dtype != NULL);

		appendStringInfo(&str,
						 "#define KVAR_%u\t"
						 "pg_%s_vref_rs(krs,%u,get_global_id(0))\n",
						 index, dtype->type_ident, var->varattno);
		index++;
	}

	/* qualifier definition with row-store */
	appendStringInfo(&str,
					 "__kernel void\n"
					 "gpuscan_qual_rs(__global kern_parambuf *kparams,\n"
					 "                __global kern_row_store *krs,\n"
					 "                __global kern_gpuscan *gpuscan,\n"
					 "                __local cl_int *karg_local_buf)\n"
					 "{\n"
					 "  pg_bool_t   rc;\n"
					 "  cl_int      errcode;\n"
					 "\n"
					 "  if (get_global_id(0) >= krs->nros)\n"
					 "    return;\n"
					 "  gpuscan_local_init(karg_local_buf);\n"
					 "  rc = %s;\n"
					 "  kern_set_error(!rc.isnull && rc.value != 0\n"
					 "                 ? StromError_Success\n"
					 "                 : StromError_RowFiltered);\n"
					 "  gpuscan_writeback_result(gpuscan);\n"
					 "}\n\n", expr_code);

	/* Put Var definition for column-store */
	index = 1;
	foreach (cell, context->used_vars)
	{
		Var	   *var = lfirst(cell);

		dtype = pgstrom_devtype_lookup(var->vartype);
		Assert(dtype != NULL);

		appendStringInfo(&str, "#undef KVAR_%u\n", index);
		if (dtype->type_flags & DEVTYPE_IS_VARLENA)
			appendStringInfo(&str,
							 "#define KVAR_%u\t"
							 "pg_%s_vref_cs(krs,toast,%u,get_global_id(0))\n",
							 index, dtype->type_ident, var->varattno);
		else
			appendStringInfo(&str,
							 "#define KVAR_%u\t"
							 "pg_%s_vref_cs(krs,%u,get_global_id(0))\n",
							 index, dtype->type_ident, var->varattno);
		index++;
	}

	/* qualifier definition with column-store */
	appendStringInfo(&str,
					 "__kernel void\n"
					 "gpuscan_qual_cs(__global kern_parambuf *kparams,\n"
					 "                __global kern_column_store *kcs,\n"
					 "                __global kern_gpuscan *gpuscan,\n"
					 "                __global kern_toastbuf *toast,\n"
					 "                __local cl_int *karg_local_buf)\n"
					 "{\n"
					 "  pg_bool_t   rc;\n"
					 "  cl_int      errcode;\n"
					 "\n"
					 "  if (get_global_id(0) >= kcs->nrows)\n"
					 "    return;\n"
					 "  gpuscan_local_init(karg_local_buf);\n"
					 "  rc = %s;\n"
					 "  kern_set_error(!rc.isnull && rc.value != 0\n"
					 "                 ? StromError_Success\n"
					 "                 : StromError_RowFiltered);\n"
					 "  gpuscan_writeback_result(gpuscan);\n"
					 "}\n", expr_code);
	return str.data;
}

static Datum
gpuscan_lookup_devprog(const char *kernel_source, int incl_flags,
					   List *used_params, TupleDesc tupdesc)
{
	ListCell   *cell;
	Oid		   *old_params;
	Oid		   *old_vars;
	int			index;
	pg_crc32	crc;

	/* setting up parameter's oid array */
	oid_params = palloc(sizeof(Oid) * list_length(used_params));
	index = 0;
	foreach (cell, used_params)
	{
		Node   *node = lfirst(cell);

		if (IsA(node, Const))
			oid_params[index] = ((Const *) node)->consttype;
		else if (IsA(node, Param))
			oid_params[index] = ((Param *) node)->paramtype;
		else
			elog(ERROR, "unexpected node: %s", nodeToString(node));
		index++;
	}

	/* setting up relation's oid array */
	oid_vars = palloc(sizeof(Oid) * tupdesc->natts);
	for (index=0; index < tupdesc->natts; index++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[index];

		oid_vars[index] = attr->atttypid;
	}

	/* setting up crc code of kernel source */
	INIT_CRC32(crc);
	COMP_CRC32(crc, &incl_flags, sizeof(int));
	COMP_CRC32(crc, kernel_source, strlen(kernel_source));
	FIN_CRC32(crc);





	
}

static CustomPlan *
gpuscan_create_plan(PlannerInfo *root, CustomPath *best_path)
{
	RelOptInfo	   *rel = best_path->path.parent;
	GpuScanPath	   *gpath = (GpuScanPath *)best_path;
	GpuScanPlan	   *gscan;
	List		   *tlist;
	List		   *scan_clause;
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
			tlist = build_path_tlist(root, best_path);
	}
	else
		tlist = build_path_tlist(root, best_path);

	/* it should be a base relation */
	Assert(rel->relid > 0);
	Assert(rel->relkind == RTE_RELATION);

	/* Sort clauses into best execution order */
	scan_clauses = order_qual_clauses(root, gpath->host_quals);

	/* Reduce RestrictInfo list to bare expressions; ignore pseudoconstants */
	scan_clauses = extract_actual_clauses(scan_clauses, false);

	/* Replace any outer-relation variables with nestloop params */
	if (best_path->path.param_info)
	{
		scan_clauses = (List *)
			replace_nestloop_params(root, (Node *) scan_clauses);
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



	/*
	 * XXX - now construct a kernel code in text form
	 */



	gscan = palloc0(sizeof(GpuScanPlan));
	gscan->cplan.plan.type = T_CustomPlan;
	gscan->cplan.plan.targetlist = tlist;
	gscan->cplan.plan.qual = scan_clauses;
	gscan->cplan.plan.lefttree = NULL;
	gscan->cplan.plan.righttree = NULL;

	gscan->scanrelid = rel->relid;




	gscan->scanrelid = rel->relid;
	gscan->type_defs = gpath->type_defs;
	gscan->func_defs = gpath->func_defs;
	gscan->used_params = gpath->used_params;
	gscan->used_vars = gpath->used_vars;
	gscan->dev_attrs = gpath->dev_attrs;
	gscan->host_attrs = gpath->host_attrs;

	gscan->dev_quals = gpath->dev_quals;





typedef struct {
	CustomPlan	cplan;
	Index		scanrelid;
	List	   *type_defs;		/* list of devtype_info in use */
	List	   *func_defs;		/* list of devfunc_info in use */
	List	   *used_params;	/* list of Const/Param in use */
	List	   *used_vars;		/* list of Var in use */
	List	   *dev_quals;		/* RestrictInfo run on device */
	List	   *host_quals;		/* RestrictInfo run on host */
	Bitmapset  *dev_attrs;		/* attrs referenced in device */
	Bitmapset  *host_attrs;		/* attrs referenced in host */
	int			incl_flags;		/* external libraries to be included */
} GpuScanPlan;

}

static void
gpuscan_textout_path(StringInfo str, Node *node)
{
	GpuScanPath	   *pathnode = (GpuScanPath *) node;
}

static void
gpuscan_set_plan_ref(PlannerInfo *root,
					 CustomPlan *custom_plan,
					 int rtoffset)
{}

static void
gpuscan_finalize_plan(CustomPlan *custom_plan)
{}

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
{}

static CustomPlan *
gpuscan_copy_plan(const CustomPlan *from)
{}

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
