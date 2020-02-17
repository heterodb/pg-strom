/*
 * gstore_fdw.c
 *
 * On GPU column based data store as FDW provider.
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#include "cuda_gpusort.h"

/*
 * GpuStorePlanInfo
 */
typedef struct
{
	List	   *host_quals;
	List	   *dev_quals;
	size_t		raw_nrows;		/* # of rows kept in GpuStoreFdw */
	size_t		dma_nrows;		/* # of rows to be backed from the device */
	Bitmapset  *outer_refs;		/* attributes to be backed to host */
	List	   *sort_keys;		/* list of Vars */
	List	   *sort_order;		/* BTXXXXStrategyNumber */
	List	   *sort_null_first;/* null-first? */
	/* table options */
	int			pinning;		/* GPU device number */
	int			format;			/* GSTORE_FDW_FORMAT__*  */
	/* kernel code */
	List	   *used_params;	/* list of referenced param-id */
	char	   *kern_source;	/* source of the CUDA kernel */
	cl_uint		extra_flags;	/* extra libraries to be included */
	cl_uint		varlena_bufsz;	/* nbytes of the expected result tuple size */
	cl_uint		proj_tuple_sz;	/* expected average tuple size */
} GpuStoreFdwInfo;

/*
 *  GpuStoreExecState - state object for scan/insert/update/delete
 */
typedef struct
{
	GpuStoreBuffer *gs_buffer;
	cl_ulong		gs_index;
	AttrNumber		ctid_anum;	/* only UPDATE or DELETE */
	GpuContext	   *gcontext;
	ProgramId		program_id;
	kern_parambuf  *kparams;
	kern_gpusort   *kgpusort;
	bool			has_sortkeys;
} GpuStoreExecState;

/* ---- static variables ---- */
static Oid		reggstore_type_oid = InvalidOid;
static bool		enable_gpusort;			/* GUC */

Datum pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_in(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_out(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_recv(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_send(PG_FUNCTION_ARGS);

static inline void
form_gpustore_fdw_info(GpuStoreFdwInfo *gsf_info,
					   List **p_fdw_exprs, List **p_fdw_privs)
{
	List	   *exprs = NIL;
	List	   *privs = NIL;
	List	   *outer_refs_list = NIL;
	int			j;

	exprs = lappend(exprs, gsf_info->host_quals);
	exprs = lappend(exprs, gsf_info->dev_quals);
	privs = lappend(privs, makeInteger(gsf_info->raw_nrows));
	privs = lappend(privs, makeInteger(gsf_info->dma_nrows));
	j = -1;
	while ((j = bms_next_member(gsf_info->outer_refs, j)) >= 0)
		outer_refs_list = lappend_int(outer_refs_list, j);
	privs = lappend(privs, outer_refs_list);
	exprs = lappend(exprs, gsf_info->sort_keys);
	privs = lappend(privs, gsf_info->sort_order);
	privs = lappend(privs, gsf_info->sort_null_first);
	privs = lappend(privs, makeInteger(gsf_info->pinning));
	privs = lappend(privs, makeInteger(gsf_info->format));
	exprs = lappend(exprs, gsf_info->used_params);
	privs = lappend(privs, makeString(gsf_info->kern_source));
	privs = lappend(privs, makeInteger(gsf_info->extra_flags));
	privs = lappend(privs, makeInteger(gsf_info->varlena_bufsz));
	privs = lappend(privs, makeInteger(gsf_info->proj_tuple_sz));

	*p_fdw_exprs = exprs;
	*p_fdw_privs = privs;
}

static inline GpuStoreFdwInfo *
deform_gpustore_fdw_info(ForeignScan *fscan)
{
	GpuStoreFdwInfo *gsf_info = palloc0(sizeof(GpuStoreFdwInfo));
	List	   *exprs = fscan->fdw_exprs;
	List	   *privs = fscan->fdw_private;
	int			pindex = 0;
	int			eindex = 0;
	List	   *temp;
	ListCell   *lc;
	Bitmapset  *outer_refs = NULL;

	gsf_info->host_quals  = list_nth(exprs, eindex++);
	gsf_info->dev_quals   = list_nth(exprs, eindex++);
	gsf_info->raw_nrows   = intVal(list_nth(privs, pindex++));
	gsf_info->dma_nrows   = intVal(list_nth(privs, pindex++));
	temp = list_nth(privs, pindex++);
	foreach (lc, temp)
		outer_refs = bms_add_member(outer_refs, lfirst_int(lc));
	gsf_info->outer_refs  = outer_refs;
	gsf_info->sort_keys   = list_nth(exprs, eindex++);
	gsf_info->sort_order  = list_nth(privs, pindex++);
	gsf_info->sort_null_first = list_nth(privs, pindex++);
	gsf_info->pinning     = intVal(list_nth(privs, pindex++));
	gsf_info->format      = intVal(list_nth(privs, pindex++));
	gsf_info->used_params = list_nth(exprs, eindex++);
	gsf_info->kern_source = strVal(list_nth(privs, pindex++));
	gsf_info->extra_flags = intVal(list_nth(privs, pindex++));
	gsf_info->varlena_bufsz = intVal(list_nth(privs, pindex++));
	gsf_info->proj_tuple_sz = intVal(list_nth(privs, pindex++));

	return gsf_info;
}

/*
 * gstoreGetForeignRelSize
 */
static void
gstoreGetForeignRelSize(PlannerInfo *root,
						RelOptInfo *baserel,
						Oid ftable_oid)
{
	GpuStoreFdwInfo *gsf_info;
	Snapshot	snapshot;
	Size		rawsize;
	Size		nitems;
	double		selectivity;
	List	   *tmp_quals;
	List	   *dev_quals = NIL;
	List	   *host_quals = NIL;
	Bitmapset  *compressed = NULL;
	ListCell   *lc;
	int			anum;

	/* setup GpuStoreFdwInfo */
	gsf_info = palloc0(sizeof(GpuStoreFdwInfo));
	gstore_fdw_table_options(ftable_oid,
							 &gsf_info->pinning,
							 &gsf_info->format);
	for (anum=1; anum <= baserel->max_attr; anum++)
	{
		int		comp;

		gstore_fdw_column_options(ftable_oid, anum, &comp);
		if (comp != GSTORE_COMPRESSION__NONE)
			compressed = bms_add_member(compressed, anum -
										FirstLowInvalidHeapAttributeNumber);
	}
	/* pickup host/device quals */
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(lc);
		Bitmapset	   *varattnos = NULL;

		if (pgstrom_device_expression(root, baserel, rinfo->clause))
		{
			/*
			 * MEMO: Right now, we don't allow to reference compressed
			 * varlena datum by device-side SQL code.
			 */
			pull_varattnos((Node *)rinfo->clause,
						   baserel->relid,
						   &varattnos);
			if (!bms_overlap(varattnos, compressed))
				dev_quals = lappend(dev_quals, rinfo);
			else
				host_quals = lappend(dev_quals, rinfo);
		}
		else
			host_quals = lappend(dev_quals, rinfo);
	}
	/* estimate number of result rows */
	snapshot = RegisterSnapshot(GetTransactionSnapshot());
	GpuStoreBufferGetSize(ftable_oid, snapshot, &rawsize, &nitems);
	UnregisterSnapshot(snapshot);

	tmp_quals = extract_actual_clauses(baserel->baserestrictinfo, false);
	selectivity = clauselist_selectivity(root,
										 tmp_quals,
										 baserel->relid,
										 JOIN_INNER,
										 NULL);
	baserel->rows  = selectivity * (double)nitems;
	baserel->pages = (rawsize + BLCKSZ - 1) / BLCKSZ;

	if (host_quals == NIL)
		gsf_info->dma_nrows = baserel->rows;
	else if (dev_quals != NIL)
	{
		tmp_quals = extract_actual_clauses(dev_quals, false);
		selectivity = clauselist_selectivity(root,
											 tmp_quals,
											 baserel->relid,
											 JOIN_INNER,
											 NULL);
		gsf_info->dma_nrows = selectivity * (double)nitems;
	}
	else
		gsf_info->dma_nrows = (double) nitems;

	gsf_info->raw_nrows  = nitems;
	gsf_info->host_quals = extract_actual_clauses(host_quals, false);
	gsf_info->dev_quals  = extract_actual_clauses(dev_quals, false);

	/* attributes to be referenced in the host code */
	pull_varattnos((Node *)baserel->reltarget->exprs,
				   baserel->relid,
				   &gsf_info->outer_refs);
	pull_varattnos((Node *)gsf_info->host_quals,
				   baserel->relid,
				   &gsf_info->outer_refs);
	baserel->fdw_private = gsf_info;
}

/*
 * gstoreCreateForeignPath
 */
static void
gstoreCreateForeignPath(PlannerInfo *root,
						RelOptInfo *baserel,
						Oid ftable_oid,
						Bitmapset *outer_refs,
						List *host_quals,
						List *dev_quals,
						double raw_nrows,
						double dma_nrows,
						List *query_pathkeys)
{
	ForeignPath *fpath;
	ParamPathInfo *param_info;
	double		gpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	Cost		startup_cost = 0.0;
	Cost		run_cost = 0.0;
	size_t		dma_size;
	size_t		htup_size;
	int			j, anum;
	QualCost	qcost;
	double		path_rows;
	List	   *useful_pathkeys = NIL;
	List	   *sort_keys = NIL;
	List	   *sort_order = NIL;
	List	   *sort_null_first = NIL;
	GpuStoreFdwInfo *gsf_info;

	/* Cost for GPU setup, if any */
	if (dev_quals != NIL || query_pathkeys != NIL)
		startup_cost += pgstrom_gpu_setup_cost;
	/* Cost for GPU qualifiers, if any */
	if (dev_quals)
	{
		cost_qual_eval_node(&qcost, (Node *)dev_quals, root);
		startup_cost += qcost.startup;
		run_cost += qcost.per_tuple * gpu_ratio * raw_nrows;
	}
	/* Cost for DMA (device-->host) */
	htup_size = MAXALIGN(offsetof(HeapTupleHeaderData, t_bits) +
						 BITMAPLEN(baserel->max_attr));
	j = -1;
	while ((j = bms_next_member(outer_refs, j)) >= 0)
	{
		anum = j + FirstLowInvalidHeapAttributeNumber;

		if (anum < InvalidAttrNumber)
			continue;
		if (anum == InvalidAttrNumber)
		{
			dma_size = baserel->pages * BLCKSZ;
			break;
		}
		if (anum < baserel->min_attr || anum > baserel->max_attr)
			elog(ERROR, "Bug? attribute number %d is out of range", anum);
		htup_size += baserel->attr_widths[anum - baserel->min_attr];
	}
	dma_size = KDS_ESTIMATE_ROW_LENGTH(baserel->max_attr,
									   dma_nrows, htup_size);
	run_cost += pgstrom_gpu_dma_cost *
		((double)dma_size / (double)pgstrom_chunk_size());
	/* Cost for CPU qualifiers, if any */
	if (host_quals)
	{
		cost_qual_eval_node(&qcost, (Node *)host_quals, root);
		startup_cost += qcost.startup;
		run_cost += qcost.per_tuple * dma_nrows;
	}
	/* Cost for baserel parameters */
	param_info = get_baserel_parampathinfo(root, baserel, NULL);
	if (param_info)
	{
		cost_qual_eval(&qcost, param_info->ppi_clauses, root);
		startup_cost += qcost.startup;
		run_cost += qcost.per_tuple * dma_nrows;

		path_rows = param_info->ppi_rows;
	}
	else
		path_rows = baserel->rows;

	/* Cost for GpuSort */
	if (query_pathkeys != NIL)
	{
		ListCell   *lc1, *lc2;
		Cost		comparison_cost = 2.0 * pgstrom_gpu_operator_cost;

		foreach (lc1, query_pathkeys)
		{
			PathKey	   *pathkey = lfirst(lc1);
			EquivalenceClass *pathkey_ec = pathkey->pk_eclass;

			foreach (lc2, pathkey_ec->ec_members)
			{
				EquivalenceMember *em = lfirst(lc2);
				Var	   *var;

				/* reference to other table? */
				if (!bms_is_subset(em->em_relids, baserel->relids))
					continue;
				/* sort by constant? it makes no sense for GpuSort */
				if (bms_is_empty(em->em_relids))
					continue;
				/*
				 * GpuSort can support only simple variable reference,
				 * because sorting is earlier than projection.
				 */
				if (!IsA(em->em_expr, Var))
					continue;
				/* sanity checks */
				var = (Var *)em->em_expr;
				if (var->varno != baserel->relid ||
					var->varattno <= 0 ||
					var->varattno >  baserel->max_attr)
					continue;

				/*
				 * Varlena data types have special optimization - offset of
				 * values to extra buffer on KDS are preliminary sorted on
				 * GPU-size when GpuStore is constructed.
				 */
				if (get_typlen(var->vartype) == -1)
				{
					TypeCacheEntry *tcache
						= lookup_type_cache(var->vartype,
											TYPECACHE_CMP_PROC);
					if (!OidIsValid(tcache->cmp_proc))
						continue;
				}
				else
				{
					devtype_info *dtype = pgstrom_devtype_lookup(var->vartype);

					if (!dtype ||
						!pgstrom_devfunc_lookup_type_compare(dtype,
															 var->varcollid))
						continue;
				}
				/* OK, this is suitable key for GpuSort */
				sort_keys = lappend(sort_keys, copyObject(var));
				sort_order = lappend_int(sort_order, pathkey->pk_strategy);
				sort_null_first = lappend_int(sort_null_first,
											  (int)pathkey->pk_nulls_first);
				useful_pathkeys = lappend(useful_pathkeys, pathkey);
				break;
			}
		}
		if (useful_pathkeys == NIL)
			return;
		if (dma_nrows > 1.0)
		{
			double	log2 = log(dma_nrows) / 0.693147180559945;
			startup_cost += comparison_cost * dma_nrows * log2;
		}
	}

	/* setup GpuStoreFdwInfo with modification */
	gsf_info = palloc0(sizeof(GpuStoreFdwInfo));
	memcpy(gsf_info, baserel->fdw_private, sizeof(GpuStoreFdwInfo));
	gsf_info->host_quals = host_quals;
	gsf_info->dev_quals  = dev_quals;
	gsf_info->raw_nrows  = raw_nrows;
	gsf_info->dma_nrows  = dma_nrows;
	gsf_info->outer_refs = outer_refs;
	gsf_info->sort_keys  = sort_keys;
	gsf_info->sort_order = sort_order;
	gsf_info->sort_null_first = sort_null_first;
	gsf_info->proj_tuple_sz = htup_size;

	fpath = create_foreignscan_path(root,
									baserel,
									NULL,	/* default pathtarget */
									path_rows,
									startup_cost,
									startup_cost + run_cost,
									useful_pathkeys,
									NULL,	/* no outer rel */
									NULL,	/* no extra plan */
									list_make1(gsf_info));
	add_path(baserel, (Path *)fpath);
}

/*
 * gstoreGetForeignPaths
 */
static void
gstoreGetForeignPaths(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Oid foreigntableid)
{
	GpuStoreFdwInfo *gsf_info = (GpuStoreFdwInfo *)baserel->fdw_private;
	List		   *any_quals;
	Bitmapset	   *outer_refs_nodev;

	/* outer_refs when dev_quals are skipped */
	if (!gsf_info->dev_quals)
		outer_refs_nodev = gsf_info->outer_refs;
	else
	{
		outer_refs_nodev = bms_copy(gsf_info->outer_refs);
		pull_varattnos((Node *)gsf_info->dev_quals,
					   baserel->relid,
					   &outer_refs_nodev);
	}

	/* no device qual execution, no device side sorting */
	any_quals = extract_actual_clauses(baserel->baserestrictinfo, false);
	gstoreCreateForeignPath(root, baserel, foreigntableid,
							outer_refs_nodev,
							any_quals, NIL,
							gsf_info->raw_nrows,
							gsf_info->raw_nrows,
							NIL);
	if (!pgstrom_enabled)
		return;

	/* device qual execution, but no device side sorting */
	if (enable_gpuscan && gsf_info->dev_quals)
	{
		gstoreCreateForeignPath(root, baserel, foreigntableid,
								gsf_info->outer_refs,
								gsf_info->host_quals,
								gsf_info->dev_quals,
								gsf_info->raw_nrows,
								gsf_info->dma_nrows,
								NIL);
	}

	/* device side sorting */
	if (enable_gpusort && root->query_pathkeys)
	{
		/* without device qual execution */
		gstoreCreateForeignPath(root, baserel, foreigntableid,
								outer_refs_nodev,
								any_quals, NIL,
								gsf_info->raw_nrows,
								gsf_info->raw_nrows,
								root->query_pathkeys);
		/* with device qual execution */
		if (enable_gpuscan && gsf_info->dev_quals)
		{
			gstoreCreateForeignPath(root, baserel, foreigntableid,
									gsf_info->outer_refs,
									gsf_info->host_quals,
									gsf_info->dev_quals,
									gsf_info->raw_nrows,
									gsf_info->dma_nrows,
									root->query_pathkeys);
		}
	}
}

/*
 * gstore_codegen_qual_eval
 */
static void
gstore_codegen_qual_eval(StringInfo kern,
						 codegen_context *context,
						 RelOptInfo *baserel,
						 List *dev_quals_list)
{
	StringInfoData	decl;
	StringInfoData	body;

	initStringInfo(&decl);
	initStringInfo(&body);
	if (dev_quals_list != NIL)
	{
		Node	   *dev_quals;
		char	   *expr_code;
		ListCell   *lc;

		/* WHERE-clause */
		dev_quals = (Node *)make_flat_ands_explicit(dev_quals_list);
		expr_code = pgstrom_codegen_expression(dev_quals, context);
		/* Const/Param declarations */
		pgstrom_codegen_param_declarations(&decl, context);
		/* Sanity check of used_vars */
		foreach (lc, context->used_vars)
		{
			devtype_info *dtype;
			Var	   *var = lfirst(lc);

			Assert(var->varno == baserel->relid);
			if (var->varattno <= 0)
				elog(ERROR, "Bug? system column appeared in expression");
			dtype = pgstrom_devtype_lookup_and_track(var->vartype,
													 context);
			appendStringInfo(
				&decl,
				"  pg_%s_t %s_%u;\n",
				dtype->type_name,
				context->var_label,
				var->varattno);
			appendStringInfo(
				&body,
				"  addr = kern_get_datum_column(kds,%u,row_index);\n"
				"  pg_datum_ref(kcxt,%s_%u,addr); //pg_%s_t\n",
				var->varattno - 1,
				context->var_label,
				var->varattno,
				dtype->type_name);
		}
		appendStringInfo(
			&body,
			"  return EVAL(%s);\n",
			expr_code);
	}
	else
	{
		appendStringInfo(
			&body,
			"  return true;\n");
	}
	appendStringInfo(
		kern,
		"DEVICE_FUNCTION(cl_bool)\n"
		"gpusort_quals_eval(kern_context *kcxt,\n"
		"                   kern_data_store *kds,\n"
		"                   cl_uint row_index)\n"
		"{\n"
		"  void *addr __attribute__((unused));\n"
		"%s%s"
		"}\n\n",
		decl.data,
		body.data);
	pfree(decl.data);
	pfree(body.data);
}

/*
 * gstore_codegen_keycomp
 */
static void
gstore_codegen_keycomp(StringInfo kern,
					   codegen_context *context,
					   RelOptInfo *baserel,
					   Oid ftable_oid,
					   GpuStoreFdwInfo *gsf_info)
{
	StringInfoData	body;
	List	   *type_oid_list = NIL;
	ListCell   *lc1, *lc2, *lc3;

	initStringInfo(&body);
	forthree (lc1, gsf_info->sort_keys,
			  lc2, gsf_info->sort_order,
			  lc3, gsf_info->sort_null_first)
	{
		Var	   *var = lfirst(lc1);
		int		order = lfirst_int(lc2);
		bool	nulls_first = lfirst_int(lc3);
		int16	typlen;
		bool	typbyval;
		devtype_info *dtype;
		devfunc_info *dfunc;

		Assert(IsA(var, Var));
		appendStringInfo(
			&body,
			"  /* -- compare %s attribute -- */\n"
			"  xaddr = kern_get_datum_column(kds_src, %u, x_index);\n"
			"  yaddr = kern_get_datum_column(kds_src, %u, y_index);\n",
			get_attname(ftable_oid, var->varattno, false),
			var->varattno - 1,
			var->varattno - 1);
		if (order != BTGreaterStrategyNumber &&
			order != BTLessStrategyNumber)
			elog(ERROR, "nexpected sort support strategy: %d", order);

		get_typlenbyval(var->vartype, &typlen, &typbyval);
		if (typlen == -1)
		{
			/*
			 * MEMO: Special optimization for variable-length types.
			 * Because varlena-dictionary is preliminary sorted on
			 * buffer creation time, so comparison of pointers are
			 * sufficient to determine which is larger/smaller.
			 */
			appendStringInfo(
				&body,
				"  if (xaddr && yaddr)\n"
				"  {\n"
				"    if (xaddr != yaddr)\n"
				"      return xaddr %s yaddr ? -1 : 1;\n"
				"  }\n"
				"  else if (!xaddr && yaddr)\n"
				"    return %d;\n"
				"  else if (xaddr && !yaddr)\n"
				"    return %d;\n",
				order == BTLessStrategyNumber ? "<" : ">",
				nulls_first ? -1 : 1,
				nulls_first ?  1 : -1);
		}
		else
		{
			devtype_info   *darg1;
			devtype_info   *darg2;
			char		   *cast_darg1 = NULL;
			char		   *cast_darg2 = NULL;

			dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
			if (!dtype)
				elog(ERROR, "Bug? type %s is not supported on device",
					 format_type_be(var->vartype));
			dfunc = pgstrom_devfunc_lookup_type_compare(dtype, var->varcollid);
			pgstrom_devfunc_track(context, dfunc);
			darg1 = linitial(dfunc->func_args);
			darg2 = lsecond(dfunc->func_args);
			if (dtype->type_oid != darg1->type_oid)
			{
				if (!pgstrom_devtype_can_relabel(dtype->type_oid,
												 darg1->type_oid))
					elog(ERROR, "Bug? no binary compatible cast for %s -> %s",
						 format_type_be(dtype->type_oid),
						 format_type_be(darg1->type_oid));
				cast_darg1 = psprintf("to_%s", darg1->type_name);
			}
			if (dtype->type_oid != darg2->type_oid)
			{
				if (!pgstrom_devtype_can_relabel(dtype->type_oid,
												 darg2->type_oid))
					elog(ERROR, "Bug? no binary compatible cast for %s -> %s",
						 format_type_be(dtype->type_oid),
						 format_type_be(darg2->type_oid));
				cast_darg2 = psprintf("to_%s", darg2->type_name);
			}
			type_oid_list = list_append_unique_oid(type_oid_list,
												   dtype->type_oid);
			appendStringInfo(
				&body,
				"  pg_datum_ref(kcxt, xval.%s_v, xaddr);\n"
				"  pg_datum_ref(kcxt, yval.%s_v, yaddr);\n"
				"  if (!xval.%s_v.isnull && !yval.%s_v.isnull)\n"
				"  {\n"
				"    comp = pgfn_%s(kcxt, %s(xval.%s_v), %s(yval.%s_v));\n"
				"    assert(!comp.isnull);\n"
				"    if (comp.value != 0)\n"
				"      return %scomp.value;\n"
				"  }\n"
				"  else if (xval.%s_v.isnull && !yval.%s_v.isnull)\n"
				"    return %d;\n"
				"  else if (!xval.%s_v.isnull && !yval.%s_v.isnull)\n"
				"    return %d;\n",
				dtype->type_name,
				dtype->type_name,
				dtype->type_name, dtype->type_name,
				dfunc->func_devname,
				cast_darg1 ? cast_darg1 : "", dtype->type_name,
				cast_darg2 ? cast_darg2 : "", dtype->type_name,
				order == BTLessStrategyNumber ? "" : "-",
				dtype->type_name, dtype->type_name,
				nulls_first ? -1 :  1,
				dtype->type_name, dtype->type_name,
				nulls_first ?  1 : -1);
			if (cast_darg1)
				pfree(cast_darg1);
			if (cast_darg2)
				pfree(cast_darg2);
		}
	}

	appendStringInfoString(
		kern,
		"DEVICE_FUNCTION(cl_int)\n"
		"gpusort_keycomp(kern_context *kcxt,\n"
		"                kern_data_store *kds_src,\n"
		"                cl_uint x_index,\n"
		"                cl_uint y_index)\n"
		"{\n"
		"  void *xaddr       __attribute__((unused));\n"
		"  void *yaddr       __attribute__((unused));\n"
		"  pg_int4_t comp    __attribute__((unused));\n");
	pgstrom_union_type_declarations(kern, "xval", type_oid_list);
	pgstrom_union_type_declarations(kern, "yval", type_oid_list);
	appendStringInfo(
		kern,
		"\n"
		"  assert(kds_src->format == KDS_FORMAT_COLUMN);\n"
		"  assert(x_index < kds_src->nitems &&\n"
		"         y_index < kds_src->nitems);\n"
		"%s"
		"  return 0;\n"
		"}\n\n",
		body.data);
	pfree(body.data);
}

/*
 * gstoreGetForeignPlan
 */
static ForeignScan *
gstoreGetForeignPlan(PlannerInfo *root,
					 RelOptInfo *baserel,
					 Oid ftable_oid,
					 ForeignPath *best_path,
					 List *tlist,
					 List *scan_clauses,
					 Plan *outer_plan)
{
	GpuStoreFdwInfo *gsf_info = linitial(best_path->fdw_private);
	codegen_context context;
	StringInfoData	kern;
	List		   *fdw_exprs;
	List		   *fdw_privs;

	/* kernel code generation */
	initStringInfo(&kern);
	pgstrom_init_codegen_context(&context, root, baserel);
	gstore_codegen_qual_eval(&kern, &context, baserel,
							 gsf_info->dev_quals);
	gstore_codegen_keycomp(&kern, &context, baserel,
						   ftable_oid, gsf_info);

	/* update GpuStoreFdwInfo */
	gsf_info->used_params = context.used_params;
	gsf_info->extra_flags = DEVKERNEL_NEEDS_GPUSORT | context.extra_flags;
	gsf_info->kern_source = kern.data;
	gsf_info->varlena_bufsz = context.varlena_bufsz;

	form_gpustore_fdw_info(gsf_info, &fdw_exprs, &fdw_privs);
	return make_foreignscan(tlist,					/* plan.targetlist */
							gsf_info->host_quals,	/* plan.qual */
							baserel->relid,			/* scanrelid */
							fdw_exprs,				/* fdw_exprs */
							fdw_privs,				/* fdw_private */
							NIL,					/* fdw_scan_tlist */
							gsf_info->dev_quals,	/* fdw_recheck_quals */
							NULL);					/* outer_plan */
}

/*
 * gstoreAddForeignUpdateTargets
 */
static void
gstoreAddForeignUpdateTargets(Query *parsetree,
							  RangeTblEntry *target_rte,
							  Relation target_relation)
{
	Var			*var;
	TargetEntry *tle;

	/*
	 * We carry row_index as ctid system column
	 */

	/* Make a Var representing the desired value */
	var = makeVar(parsetree->resultRelation,
				  SelfItemPointerAttributeNumber,
				  TIDOID,
				  -1,
				  InvalidOid,
				  0);

	/* Wrap it in a resjunk TLE with the right name ... */
	tle = makeTargetEntry((Expr *) var,
						  list_length(parsetree->targetList) + 1,
						  "ctid",
						  true);

	/* ... and add it to the query's targetlist */
	parsetree->targetList = lappend(parsetree->targetList, tle);
}

/*
 * gstoreBeginForeignScan
 */
static void
gstoreBeginForeignScan(ForeignScanState *node, int eflags)
{
	ForeignScan	   *fscan = (ForeignScan *)node->ss.ps.plan;
	GpuStoreFdwInfo *gsf_info = deform_gpustore_fdw_info(fscan);
	GpuStoreExecState *gstate;
	GpuContext	   *gcontext = NULL;
	ProgramId		program_id = INVALID_PROGRAM_ID;
	kern_parambuf  *kparams = NULL;
	bool			has_sortkeys = false;

	gstate = palloc0(sizeof(GpuStoreExecState));
	if (gsf_info->dev_quals != NIL ||
		gsf_info->sort_keys != NIL)
	{
		StringInfoData kern_define;
		bool		explain_only
			= ((eflags & EXEC_FLAG_EXPLAIN_ONLY) != 0);

		initStringInfo(&kern_define);
		pgstrom_build_session_info(&kern_define,
								   NULL,
								   gsf_info->extra_flags);
		gcontext = AllocGpuContext(gsf_info->pinning,
								   false, false, false);
		program_id = pgstrom_create_cuda_program(gcontext,
												 gsf_info->extra_flags,
												 gsf_info->varlena_bufsz,
												 gsf_info->kern_source,
												 kern_define.data,
												 false,
												 explain_only);
		kparams = construct_kern_parambuf(gsf_info->used_params,
										  node->ss.ps.ps_ExprContext,
										  NIL);
		pfree(kern_define.data);

		if (gsf_info->sort_keys != NIL)
			has_sortkeys = true;
	}
	gstate->gcontext   = gcontext;
	gstate->program_id = program_id;
	gstate->kparams    = kparams;
	gstate->has_sortkeys = has_sortkeys;

	node->fdw_state = (void *) gstate;
}

/*
 * gstoreLaunchScanSortKernel
 */
static void
gstoreLaunchScanSortKernel(GpuContext *gcontext,
						   CUmodule cuda_module,
						   CUdeviceptr m_gpusort,
						   CUdeviceptr m_kds_src,
						   bool run_gpusort)
{
	kern_gpusort   *kgpusort = (kern_gpusort *) m_gpusort;
	CUfunction		kern_gpusort_setup;
	CUfunction		kern_gpusort_local;
	CUfunction		kern_gpusort_step;
	CUfunction		kern_gpusort_merge;
	CUresult		rc;
	cl_uint			nitems;
	cl_int			grid_sz;
	cl_int			block_sz;
	void		   *kern_args[4];

	rc = cuModuleGetFunction(&kern_gpusort_setup,
							 cuda_module,
							 "kern_gpusort_setup_column");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	if (run_gpusort)
	{
		rc = cuModuleGetFunction(&kern_gpusort_local,
								 cuda_module,
								 "kern_gpusort_bitonic_local");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

		rc = cuModuleGetFunction(&kern_gpusort_step,
								 cuda_module,
								 "kern_gpusort_bitonic_step");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

		rc = cuModuleGetFunction(&kern_gpusort_merge,
								 cuda_module,
								 "kern_gpusort_bitonic_merge");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));
	}

	/*
	 * KERNEL_FUNCTION(void)
	 * kern_gpusort_setup_column(kern_gpusort *kgpusort,
	 *                      kern_data_store *kds_src)
	 */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 kern_gpusort_setup,
							 gcontext->cuda_device,
							 0, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuOptimalBlockSize: %s", errorText(rc));

	kern_args[0] = &m_gpusort;
	kern_args[1] = &m_kds_src;
	rc = cuLaunchKernel(kern_gpusort_setup,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						sizeof(cl_uint) * MAXTHREADS_PER_BLOCK,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));

	if (run_gpusort)
	{
		gpusortResultIndex *kresults;
		cl_uint		nhalf;
		cl_uint		i, j;

		rc = cuStreamSynchronize(NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuStreamSynchronize: %s", errorText(rc));
		if (kgpusort->kerror.errcode != ERRCODE_STROM_SUCCESS)
		{
			/* TODO: CPU fallback handling */
			errcode(kgpusort->kerror.errcode & ~ERRCODE_FLAGS_CPU_FALLBACK);
			elog_start(kgpusort->kerror.filename,
					   kgpusort->kerror.lineno,
					   kgpusort->kerror.funcname);
			elog_finish(ERROR, "%s - %c%c%c%c%c",
						kgpusort->kerror.message,
						PGUNSIXBIT(kgpusort->kerror.errcode),
						PGUNSIXBIT(kgpusort->kerror.errcode >> 6),
						PGUNSIXBIT(kgpusort->kerror.errcode >> 12),
						PGUNSIXBIT(kgpusort->kerror.errcode >> 18),
						PGUNSIXBIT(kgpusort->kerror.errcode >> 24));
		}
		kresults = KERN_GPUSORT_RESULT_INDEX(kgpusort);
		nitems = kresults->nitems;
		/* nhalf is the least power of two larger than the nitems */
		nhalf = 1UL << (get_next_log2(nitems + 1) - 1);

		block_sz = MAXTHREADS_PER_BLOCK;
		grid_sz = Max(nhalf / MAXTHREADS_PER_BLOCK, 1);

		/*
		 * make a sorting block up to (2 * BITONIC_MAX_LOCAL_SZ)
		 *
		 * KERNEL_FUNCTION_MAXTHREADS(void)
		 * kern_gpusort_bitonic_local(kern_gpusort *kgpusort,
		 *                            kern_data_store *kds_src)
		 */
		kern_args[0] = &m_gpusort;
		kern_args[1] = &m_kds_src;
		rc = cuLaunchKernel(kern_gpusort_local,
							grid_sz, 1, 1,
							block_sz, 1, 1,
							0,
							CU_STREAM_PER_THREAD,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		/* inter blocks bitonic sorting */
		for (i = BITONIC_MAX_LOCAL_SZ; i < nhalf; i *= 2)
		{
			for (j = 2 * i; j > BITONIC_MAX_LOCAL_SZ; j /= 2)
			{
				cl_uint		unitsz = 2 * j;
				cl_bool		reversing = ((j == 2 * i) ? true : false);

				/*
				 * KERNEL_FUNCTION_MAXTHREADS(void)
				 * kern_gpustore_bitonic_step(kern_gpusort *kgpusort,
				 *                            kern_data_store *kds_src,
				 *                            cl_uint unitsz,
				 *                            cl_bool reversing)
				 */
				kern_args[0] = &m_gpusort;
				kern_args[1] = &m_kds_src;
				kern_args[2] = &unitsz;
				kern_args[3] = &reversing;
				rc = cuLaunchKernel(kern_gpusort_step,
									grid_sz, 1, 1,
									block_sz, 1, 1,
									0,
									CU_STREAM_PER_THREAD,
									kern_args,
									NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
			}

			/*
			 * KERNEL_FUNCTION_MAXTHREADS(void)
			 * kern_gpusort_bitonic_merge(kern_gpusort *kgpusort,
			 *                            kern_data_store *kds_src)
			 */
			kern_args[0] = &m_gpusort;
			kern_args[1] = &m_kds_src;
			rc = cuLaunchKernel(kern_gpusort_merge,
								grid_sz, 1, 1,
								block_sz, 1, 1,
								0,
								CU_STREAM_PER_THREAD,
								kern_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		}
	}

	rc = cuStreamSynchronize(NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuStreamSynchronize: %s", errorText(rc));
}

/*
 * gstoreProcessScanSortKernel
 */
static void
gstoreProcessScanSortKernel(GpuStoreExecState *gstate)
{
	GpuContext	   *gcontext = gstate->gcontext;
	GpuStoreBuffer *gs_buffer = gstate->gs_buffer;
	kern_parambuf  *kparams = gstate->kparams;
	size_t			length;
	size_t			nitems;
	CUdeviceptr		m_gpusort;
	CUdeviceptr		m_kds_src;
	CUmodule		cuda_module;
	CUresult		rc;
	
	ActivateGpuContextNoWorkers(gcontext);
	/*
	 * setup kern_gpusort (including gpusortResultIndex)
	 */
	nitems = GpuStoreBufferGetNitems(gs_buffer);
	length = (STROMALIGN(offsetof(kern_gpusort, kparams)) +
			  STROMALIGN(kparams->length) +
			  STROMALIGN(offsetof(gpusortResultIndex, results)));
	rc = gpuMemAllocManaged(gcontext,
							&m_gpusort,
							STROMALIGN(length + sizeof(cl_uint) * nitems),
							CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));
	memset((void *)m_gpusort, 0, length);
	memcpy(KERN_GPUSORT_PARAMBUF(m_gpusort),
		   kparams,
		   kparams->length);
	gstate->kgpusort = (kern_gpusort *) m_gpusort;
	gstate->kgpusort->nitems_in = nitems;

	/*
	 * map device memory
	 */
	m_kds_src = GpuStoreBufferOpenDevPtr(gcontext, gs_buffer);

	/*
	 * kick GPU kernel(s)
	 */
	cuda_module = GpuContextLookupModule(gcontext, gstate->program_id);
	gstoreLaunchScanSortKernel(gcontext,
							   cuda_module,
							   m_gpusort,
							   m_kds_src,
							   gstate->has_sortkeys);
	/*
	 * unmap device memory
	 */
	rc = gpuIpcCloseMemHandle(gcontext, m_kds_src);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuIpcCloseMemHandle: %s", errorText(rc));
}

/*
 * gstoreExecForeignScan
 */
static TupleTableSlot *
gstoreExecForeignScan(ForeignScanState *node)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) node->fdw_state;
	Relation		frel = node->ss.ss_currentRelation;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	EState		   *estate = node->ss.ps.state;
	Snapshot		snapshot = estate->es_snapshot;
	ForeignScan	   *fscan = (ForeignScan *)node->ss.ps.plan;
	cl_uint			row_index;

	if (!gstate->gs_buffer)
		gstate->gs_buffer = GpuStoreBufferCreate(frel, snapshot);
lnext:
	if (gstate->gcontext)
	{
		gpusortResultIndex *kresults;

		if (!gstate->kgpusort)
			gstoreProcessScanSortKernel(gstate);
		kresults = KERN_GPUSORT_RESULT_INDEX(gstate->kgpusort);
		if (gstate->gs_index >= kresults->nitems)
			return NULL;
		row_index = kresults->results[gstate->gs_index++];
	}
	else
		row_index = gstate->gs_index++;

	if (GpuStoreBufferGetTuple(frel,
							   snapshot,
							   slot,
							   gstate->gs_buffer,
							   row_index,
							   fscan->fsSystemCol) > 0)
		goto lnext;

	return slot;
}

/*
 * gstoreReScanForeignScan
 */
static void
gstoreReScanForeignScan(ForeignScanState *node)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) node->fdw_state;

	gstate->gs_index = 0;
}

/*
 * gstoreEndForeignScan
 */
static void
gstoreEndForeignScan(ForeignScanState *node)
{
	GpuStoreExecState  *gstate = (GpuStoreExecState *) node->fdw_state;

	if (gstate->gcontext)
	{
		if (gstate->program_id != INVALID_PROGRAM_ID)
			pgstrom_put_cuda_program(gstate->gcontext,
									 gstate->program_id);
		if (gstate->kgpusort)
			gpuMemFree(gstate->gcontext, (CUdeviceptr) gstate->kgpusort);

		PutGpuContext(gstate->gcontext);
	}
}

/*
 * gstoreExplainForeignScan
 */
static void
gstoreExplainForeignScan(ForeignScanState *node, ExplainState *es)
{
	GpuStoreExecState  *gstate = (GpuStoreExecState *) node->fdw_state;
	GpuStoreFdwInfo	   *gsf_info;
	List			   *dcontext;
	char			   *temp;

	gsf_info = deform_gpustore_fdw_info((ForeignScan *)node->ss.ps.plan);

	/* setup deparsing context */
	dcontext = set_deparse_context_planstate(es->deparse_cxt,
											 (Node *)&node->ss.ps,
											 NIL); //XXX spec bug?
	/* device quelifiers, if any */
	if (gsf_info->dev_quals != NIL)
	{
		temp = deparse_expression((Node *)gsf_info->dev_quals,
								  dcontext, es->verbose, false);
		ExplainPropertyText("GPU Filter", temp, es);

		//Rows Removed by GPU Filter if EXPLAIN ANALYZE
	}

	/* sorting keys, if any */
	if (gsf_info->sort_keys != NIL)
	{
		StringInfoData buf;
		ListCell   *lc1, *lc2, *lc3;

		initStringInfo(&buf);
		forthree (lc1, gsf_info->sort_keys,
				  lc2, gsf_info->sort_order,
				  lc3, gsf_info->sort_null_first)
		{
			Node   *expr = lfirst(lc1);
			int		__order = lfirst_int(lc2);
			int		null_first = lfirst_int(lc3);
			const char *order;

			if (__order == BTLessStrategyNumber)
				order = "asc";
			else if (__order == BTGreaterStrategyNumber)
				order = "desc";
			else
				order = "???";

			temp = deparse_expression(expr, dcontext, es->verbose, false);
			if (es->verbose)
				appendStringInfo(&buf, "%s%s %s nulls %s",
								 buf.len > 0 ? ", " : "",
								 temp, order,
								 null_first ? "first" : "last");
			else
				appendStringInfo(&buf, "%s%s",
								 buf.len > 0 ? ", " : "",
								 temp);
		}
		ExplainPropertyText("GpuSort keys", buf.data, es);

		pfree(buf.data);
	}

	/* Source path of the GPU kernel */
	if (es->verbose &&
		gstate->program_id != INVALID_PROGRAM_ID &&
		pgstrom_debug_kernel_source)
	{
		const char *cuda_source = pgstrom_cuda_source_file(gstate->program_id);
		const char *cuda_binary = pgstrom_cuda_binary_file(gstate->program_id);

		if (cuda_source)
			ExplainPropertyText("Kernel Source", cuda_source, es);
        if (cuda_binary)
            ExplainPropertyText("Kernel Binary", cuda_binary, es);
	}
}

/*
 * gstorePlanForeignModify
 */
static List *
gstorePlanForeignModify(PlannerInfo *root,
						ModifyTable *plan,
						Index resultRelation,
						int subplan_index)
{
	CmdType		operation = plan->operation;

	if (operation != CMD_INSERT &&
		operation != CMD_UPDATE &&
		operation != CMD_DELETE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("gstore_fdw: not a supported operation")));
	return NIL;
}

/*
 * gstoreBeginForeignModify
 */
static void
gstoreBeginForeignModify(ModifyTableState *mtstate,
						 ResultRelInfo *rrinfo,
						 List *fdw_private,
						 int subplan_index,
						 int eflags)
{
	GpuStoreExecState *gstate = palloc0(sizeof(GpuStoreExecState));
	Relation	frel = rrinfo->ri_RelationDesc;
	CmdType		operation = mtstate->operation;

	/*
	 * NOTE: gstore_fdw does not support update operations by multiple
	 * concurrent transactions. So, we require stronger lock than usual
	 * INSERT/UPDATE/DELETE operations. It may lead unexpected deadlock,
	 * in spite of the per-tuple update capability.
	 */
	LockRelationOid(RelationGetRelid(frel), ShareUpdateExclusiveLock);

	/* Find the ctid resjunk column in the subplan's result */
	if (operation == CMD_UPDATE || operation == CMD_DELETE)
	{
		Plan	   *subplan = mtstate->mt_plans[subplan_index]->plan;
		AttrNumber	ctid_anum;

		ctid_anum = ExecFindJunkAttributeInTlist(subplan->targetlist, "ctid");
		if (!AttributeNumberIsValid(ctid_anum))
			elog(ERROR, "could not find junk ctid column");
		gstate->ctid_anum = ctid_anum;
	}
	rrinfo->ri_FdwState = gstate;
}

/*
 * gstoreExecForeignInsert
 */
static TupleTableSlot *
gstoreExecForeignInsert(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Snapshot		snapshot = estate->es_snapshot;
	Relation		frel = rrinfo->ri_RelationDesc;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = GpuStoreBufferCreate(frel, snapshot);

	GpuStoreBufferAppendRow(gstate->gs_buffer,
							RelationGetDescr(frel),
							snapshot,
							slot);
	return slot;
}

/*
 * gstoreExecForeignUpdate
 */
static TupleTableSlot *
gstoreExecForeignUpdate(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Relation		frel = rrinfo->ri_RelationDesc;
	Snapshot		snapshot = estate->es_snapshot;
	Datum			datum;
	bool			isnull;
	ItemPointer		t_self;
	size_t			old_index;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = GpuStoreBufferCreate(frel, snapshot);

	/* remove old version of the row */
	datum = ExecGetJunkAttribute(planSlot,
								 gstate->ctid_anum,
								 &isnull);
	if (isnull)
		elog(ERROR, "gstore_fdw: ctid is null");
	t_self = (ItemPointer)DatumGetPointer(datum);
	old_index = ((cl_ulong)t_self->ip_blkid.bi_hi << 32 |
				 (cl_ulong)t_self->ip_blkid.bi_lo << 16 |
				 (cl_ulong)t_self->ip_posid);
	GpuStoreBufferRemoveRow(gstate->gs_buffer,
							RelationGetDescr(frel),
							snapshot,
							old_index);

	/* insert new version of the row */
	GpuStoreBufferAppendRow(gstate->gs_buffer,
                            RelationGetDescr(frel),
							snapshot,
                            slot);
	return slot;
}

/*
 * gstoreExecForeignDelete
 */
static TupleTableSlot *
gstoreExecForeignDelete(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Relation		frel = rrinfo->ri_RelationDesc;
	Snapshot		snapshot = estate->es_snapshot;
	Datum			datum;
	bool			isnull;
	ItemPointer		t_self;
	size_t			old_index;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = GpuStoreBufferCreate(frel, snapshot);

	/* remove old version of the row */
	datum = ExecGetJunkAttribute(planSlot,
								 gstate->ctid_anum,
								 &isnull);
	if (isnull)
		elog(ERROR, "gstore_fdw: ctid is null");
	t_self = (ItemPointer)DatumGetPointer(datum);
	old_index = ((cl_ulong)t_self->ip_blkid.bi_hi << 32 |
				 (cl_ulong)t_self->ip_blkid.bi_lo << 16 |
				 (cl_ulong)t_self->ip_posid);
	GpuStoreBufferRemoveRow(gstate->gs_buffer,
							RelationGetDescr(frel),
							snapshot,
							old_index);
	return slot;
}

/*
 * gstoreEndForeignModify
 */
static void
gstoreEndForeignModify(EState *estate,
					   ResultRelInfo *rrinfo)
{
	//GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
}

/*
 * relation_is_gstore_fdw
 */
bool
relation_is_gstore_fdw(Oid table_oid)
{
	HeapTuple	tup;
	Oid			fserv_oid;
	Oid			fdw_oid;
	Oid			handler_oid;
	PGFunction	handler_fn;
	Datum		datum;
	char	   *prosrc;
	char	   *probin;
	bool		isnull;
	/* it should be foreign table, of course */
	if (get_rel_relkind(table_oid) != RELKIND_FOREIGN_TABLE)
		return false;
	/* pull OID of foreign-server */
	tup = SearchSysCache1(FOREIGNTABLEREL, ObjectIdGetDatum(table_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign table %u", table_oid);
	fserv_oid = ((Form_pg_foreign_table) GETSTRUCT(tup))->ftserver;
	ReleaseSysCache(tup);

	/* pull OID of foreign-data-wrapper */
	tup = SearchSysCache1(FOREIGNSERVEROID, ObjectIdGetDatum(fserv_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "foreign server with OID %u does not exist", fserv_oid);
	fdw_oid = ((Form_pg_foreign_server) GETSTRUCT(tup))->srvfdw;
	ReleaseSysCache(tup);

	/* pull OID of FDW handler function */
	tup = SearchSysCache1(FOREIGNDATAWRAPPEROID, ObjectIdGetDatum(fdw_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign-data wrapper %u",fdw_oid);
	handler_oid = ((Form_pg_foreign_data_wrapper) GETSTRUCT(tup))->fdwhandler;
	ReleaseSysCache(tup);
	/* pull library path & function name */
	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(handler_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", handler_oid);
	if (((Form_pg_proc) GETSTRUCT(tup))->prolang != ClanguageId)
		elog(ERROR, "FDW handler function is not written with C-language");

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "null prosrc for C function %u", handler_oid);
	prosrc = TextDatumGetCString(datum);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_probin, &isnull);
	if (isnull)
		elog(ERROR, "null probin for C function %u", handler_oid);
	probin = TextDatumGetCString(datum);
	ReleaseSysCache(tup);
	/* check whether function pointer is identical */
	handler_fn = load_external_function(probin, prosrc, true, NULL);
	if (handler_fn != pgstrom_gstore_fdw_handler)
		return false;
	/* OK, it is GpuStore foreign table */
	return true;
}

/*
 * gstore_fdw_table_options
 */
static void
__gstore_fdw_table_options(List *options,
						   int *p_pinning,
						   int *p_format)
{
	ListCell   *lc;
	int			pinning = -1;
	int			format = -1;

	foreach (lc, options)
	{
		DefElem	   *defel = lfirst(lc);

		if (strcmp(defel->defname, "pinning") == 0)
		{
			if (pinning >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"pinning\" option appears twice")));
			pinning = atoi(defGetString(defel));
			if (pinning < 0 || pinning >= numDevAttrs)
				ereport(ERROR,
						(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						 errmsg("\"pinning\" on unavailable GPU device")));
		}
		else if (strcmp(defel->defname, "format") == 0)
		{
			char   *format_name;

			if (format >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"format\" option appears twice")));
			format_name = defGetString(defel);
			if (strcmp(format_name, "pgstrom") == 0 ||
				strcmp(format_name, "default") == 0)
				format = GSTORE_FDW_FORMAT__PGSTROM;
			else
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("gstore_fdw: format \"%s\" is unknown",
								format_name)));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_SYNTAX_ERROR),
					 errmsg("gstore_fdw: unknown option \"%s\"",
							defel->defname)));
		}
	}
	if (pinning < 0)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("gstore_fdw: No pinning GPU device"),
				 errhint("use 'pinning' option to specify GPU device")));

	/* put default if not specified */
	if (format < 0)
		format = GSTORE_FDW_FORMAT__PGSTROM;
	/* set results */
	if (p_pinning)
		*p_pinning = pinning;
	if (p_format)
		*p_format = format;
}

void
gstore_fdw_table_options(Oid gstore_oid, int *p_pinning, int *p_format)
{
	HeapTuple	tup;
	Datum		datum;
	bool		isnull;
	List	   *options = NIL;

	tup = SearchSysCache1(FOREIGNTABLEREL, ObjectIdGetDatum(gstore_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign table %u", gstore_oid);
	datum = SysCacheGetAttr(FOREIGNTABLEREL, tup,
							Anum_pg_foreign_table_ftoptions,
							&isnull);
	if (!isnull)
		options = untransformRelOptions(datum);
	__gstore_fdw_table_options(options, p_pinning, p_format);
	ReleaseSysCache(tup);
}

/*
 * gstore_fdw_column_options
 */
static void
__gstore_fdw_column_options(List *options, int *p_compression)
{
	ListCell   *lc;
	char	   *temp;
	int			compression = -1;

	foreach (lc, options)
	{
		DefElem	   *defel = lfirst(lc);

		if (strcmp(defel->defname, "compression") == 0)
		{
			if (compression >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"compression\" option appears twice")));
			temp = defGetString(defel);
			if (pg_strcasecmp(temp, "none") == 0)
				compression = GSTORE_COMPRESSION__NONE;
			else if (pg_strcasecmp(temp, "pglz") == 0)
				compression = GSTORE_COMPRESSION__PGLZ;
			else
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("unknown compression logic: %s", temp)));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_SYNTAX_ERROR),
					 errmsg("gstore_fdw: unknown option \"%s\"",
							defel->defname)));
		}
	}
	/* set default, if no valid options were supplied */
	if (compression < 0)
		compression = GSTORE_COMPRESSION__NONE;
	/* set results */
	if (p_compression)
		*p_compression = compression;
}

void
gstore_fdw_column_options(Oid gstore_oid, AttrNumber attnum,
						  int *p_compression)
{
	List	   *options = GetForeignColumnOptions(gstore_oid, attnum);

	__gstore_fdw_column_options(options, p_compression);
}

/*
 * pgstrom_gstore_fdw_validator
 */
Datum
pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS)
{
	List	   *options = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid			catalog = PG_GETARG_OID(1);

	switch (catalog)
	{
		case ForeignTableRelationId:
			__gstore_fdw_table_options(options, NULL, NULL);
			break;

		case AttributeRelationId:
			__gstore_fdw_column_options(options, NULL);
			break;

		case ForeignServerRelationId:
			if (options)
				elog(ERROR, "gstore_fdw: no options are supported on SERVER");
			break;

		case ForeignDataWrapperRelationId:
			if (options)
				elog(ERROR, "gstore_fdw: no options are supported on FOREIGN DATA WRAPPER");
			break;

		default:
			elog(ERROR, "gstore_fdw: no options are supported on catalog %s",
				 get_rel_name(catalog));
			break;
	}
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_validator);

/*
 * pgstrom_gstore_fdw_handler
 */
Datum
pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS)
{
	FdwRoutine *routine = makeNode(FdwRoutine);

	/* functions for scanning foreign tables */
	routine->GetForeignRelSize	= gstoreGetForeignRelSize;
	routine->GetForeignPaths	= gstoreGetForeignPaths;
	routine->GetForeignPlan		= gstoreGetForeignPlan;
	routine->AddForeignUpdateTargets = gstoreAddForeignUpdateTargets;
	routine->BeginForeignScan	= gstoreBeginForeignScan;
	routine->IterateForeignScan	= gstoreExecForeignScan;
	routine->ReScanForeignScan	= gstoreReScanForeignScan;
	routine->EndForeignScan		= gstoreEndForeignScan;
	routine->ExplainForeignScan = gstoreExplainForeignScan;

	/* functions for INSERT/UPDATE/DELETE foreign tables */

	routine->PlanForeignModify	= gstorePlanForeignModify;
	routine->BeginForeignModify	= gstoreBeginForeignModify;
	routine->ExecForeignInsert	= gstoreExecForeignInsert;
	routine->ExecForeignUpdate  = gstoreExecForeignUpdate;
	routine->ExecForeignDelete	= gstoreExecForeignDelete;
	routine->EndForeignModify	= gstoreEndForeignModify;

	PG_RETURN_POINTER(routine);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_handler);

/*
 * pgstrom_reggstore_in
 */
Datum
pgstrom_reggstore_in(PG_FUNCTION_ARGS)
{
	Datum	datum = regclassin(fcinfo);

	if (!relation_is_gstore_fdw(DatumGetObjectId(datum)))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						DatumGetObjectId(datum))));
	PG_RETURN_DATUM(datum);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_in);

/*
 * pgstrom_reggstore_out
 */
Datum
pgstrom_reggstore_out(PG_FUNCTION_ARGS)
{
	Oid		relid = PG_GETARG_OID(0);

	if (!relation_is_gstore_fdw(relid))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						relid)));
	return regclassout(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_out);

/*
 * pgstrom_reggstore_recv
 */
Datum
pgstrom_reggstore_recv(PG_FUNCTION_ARGS)
{
	/* exactly the same as oidrecv, so share code */
	Datum	datum = oidrecv(fcinfo);

	if (!relation_is_gstore_fdw(DatumGetObjectId(datum)))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						DatumGetObjectId(datum))));
	PG_RETURN_DATUM(datum);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_recv);

/*
 * pgstrom_reggstore_send
 */
Datum
pgstrom_reggstore_send(PG_FUNCTION_ARGS)
{
	Oid		relid = PG_GETARG_OID(0);

	if (!relation_is_gstore_fdw(relid))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						relid)));
	/* Exactly the same as oidsend, so share code */
	return oidsend(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_send);

/*
 * get_reggstore_type_oid
 */
Oid
get_reggstore_type_oid(void)
{
	if (!OidIsValid(reggstore_type_oid))
	{
		Oid		temp_oid;

		temp_oid = get_type_oid("reggstore", PG_PUBLIC_NAMESPACE, true);
		if (!OidIsValid(temp_oid) ||
			!type_is_reggstore(temp_oid))
			elog(ERROR, "type \"reggstore\" is not defined");
		reggstore_type_oid = temp_oid;
	}
	return reggstore_type_oid;
}

/*
 * reset_reggstore_type_oid
 */
static void
reset_reggstore_type_oid(Datum arg, int cacheid, uint32 hashvalue)
{
	reggstore_type_oid = InvalidOid;
}

/*
 * type_is_reggstore
 */
bool
type_is_reggstore(Oid type_oid)
{
	Oid			typinput;
	HeapTuple	tup;
	char	   *prosrc;
	char	   *probin;
	Datum		datum;
	bool		isnull;
	PGFunction	handler_fn;

	tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	typinput = ((Form_pg_type) GETSTRUCT(tup))->typinput;
	ReleaseSysCache(tup);

	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(typinput));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", typinput);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "null prosrc for C function %u", typinput);
	prosrc = TextDatumGetCString(datum);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_probin, &isnull);
	if (isnull)
		elog(ERROR, "null probin for C function %u", typinput);
	probin = TextDatumGetCString(datum);
	ReleaseSysCache(tup);

	/* check whether function pointer is identical */
	handler_fn = load_external_function(probin, prosrc, true, NULL);
	if (handler_fn != pgstrom_reggstore_in)
		return false;
	/* ok, it is reggstore type */
	return true;
}

/*
 * pgstrom_init_gstore_fdw
 */
void
pgstrom_init_gstore_fdw(void)
{
	/* pg_strom.enable_gpusort */
	DefineCustomBoolVariable("pg_strom.enable_gpusort",
							 "Enables use of GPU to sort rows on Gstore_Fdw",
							 NULL,
							 &enable_gpusort,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* invalidation of reggstore_oid variable */
	CacheRegisterSyscacheCallback(TYPEOID, reset_reggstore_type_oid, 0);
}
