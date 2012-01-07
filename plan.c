/*
 * plan.c
 *
 * Routines to plan streamed query execution.
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/var.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "pg_strom.h"

/*
 * static declarations
 */
static bool
make_device_qual_code(Node *node, StringInfo qual_source,
					  Oid base_relid,
					  List **type_decl, List **func_decl);

/*
 * make_device_function_code
 *
 * Helper function of make_device_qual_code; that generate source code to
 * invoke a device function or use a device operator.
 */
static bool
make_device_function_code(Oid func_oid, List *args,
						  Oid func_resulttype,
						  StringInfo qual_source,
						  Oid base_relid,
						  List **type_decl, List **func_decl)
{
	PgStromDevFuncInfo *dfunc = pgstrom_devfunc_lookup(func_oid);
	PgStromDevTypeInfo *dtype;
	ListCell   *cell;
	int			i;

   	switch (dfunc->func_kind)
	{
		case 'c':	/* function as constant */
			Assert(list_length(args) == 0);
			appendStringInfo(qual_source, "%s", dfunc->func_ident);
			break;

		case 'l':	/* function as left-operator */
			Assert(list_length(args) == 1);
			appendStringInfo(qual_source, "(%s", dfunc->func_ident);
			if (!make_device_qual_code((Node *) linitial(args),
									   qual_source, base_relid,
									   type_decl, func_decl))
				return false;
			appendStringInfo(qual_source, ")");
			break;

		case 'r':	/* function as right-operator */
			Assert(list_length(args) == 1);
			appendStringInfo(qual_source, "(");
			if (!make_device_qual_code((Node *) linitial(args),
									   qual_source, base_relid,
									   type_decl, func_decl))
				return false;
			appendStringInfo(qual_source, "%s)", dfunc->func_ident);
			break;

		case 'b':	/* function as left-right-operator */
			Assert(list_length(args) == 2);
			appendStringInfo(qual_source, "(");
			if (!make_device_qual_code((Node *) linitial(args),
									   qual_source, base_relid,
									   type_decl, func_decl))
				return false;
			appendStringInfo(qual_source, " %s ", dfunc->func_ident);
			if (!make_device_qual_code((Node *) lsecond(args),
									   qual_source, base_relid,
									   type_decl, func_decl))
				return false;
			appendStringInfo(qual_source, ")");
			break;

		case 'F':	/* function as built-in device function */
			appendStringInfo(qual_source, "%s(", dfunc->func_ident);
			foreach (cell, args)
			{
				if (cell != list_head(args))
					appendStringInfo(qual_source, ", ");
				if (!make_device_qual_code((Node *) lfirst(cell),
										   qual_source, base_relid,
										   type_decl, func_decl))
					return false;
			}
			appendStringInfo(qual_source, ")");
			break;

		case 'f':	/* function as self-defined device function */
			appendStringInfo(qual_source,
							 "%s(&errors, bitmask",
							 dfunc->func_ident);
			foreach (cell, args)
			{
				appendStringInfo(qual_source, ", ");
				if (!make_device_qual_code((Node *) lfirst(cell),
										   qual_source, base_relid,
										   type_decl, func_decl))
					return false;
			}
			appendStringInfo(qual_source, ")");
			break;

		default:
			elog(ERROR, "unexpected func_kind : %c of \"%s\"",
				 dfunc->func_kind, dfunc->func_ident);
			break;
	}
	if (dfunc->func_source)
		*func_decl = list_append_unique_ptr(*func_decl,
											dfunc->func_source);
	dtype = pgstrom_devtype_lookup(func_resulttype);
	if (dtype->type_source)
		*type_decl = list_append_unique_ptr(*type_decl,
											dtype->type_source);
	for (i=0; i < dfunc->func_nargs; i++)
	{
		dtype = pgstrom_devtype_lookup(dfunc->func_argtypes[i]);
		if (dtype->type_source)
			*type_decl = list_append_unique_ptr(*type_decl,
												dtype->type_source);
	}
	return true;
}

/*
 * make_device_var_code
 *
 * Helper function of make_device_qual_code; that generate references
 * to a particular variable.
 */
static bool
make_device_var_code(Var *var, StringInfo qual_source,
					 Oid base_relid,
					 List **type_decl, List **func_decl)
{
	PgStromDevTypeInfo *dtype;
	Form_pg_attribute	attr;
	HeapTuple	tup;

	dtype = pgstrom_devtype_lookup(var->vartype);
	if (dtype->type_source)
		*type_decl = list_append_unique_ptr(*type_decl,
											dtype->type_source);
	/*
	 * A trivial optimization with NOT NULL constraint; if we can make
	 * sure all the values being valid, no need to use varref_* device
	 * function to set a bit of "errors" due to null values.
	 */
	tup = SearchSysCache2(ATTNUM,
						  ObjectIdGetDatum(base_relid),
						  Int16GetDatum(var->varattno));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for attribute %d of relation %u",
			 var->varattno, base_relid);
	attr = (Form_pg_attribute) GETSTRUCT(tup);

	if (attr->attnotnull)
		appendStringInfo(qual_source, "cv%d", var->varattno);
	else
	{
		if (dtype->type_varref)
			*func_decl = list_append_unique_ptr(*func_decl,
												dtype->type_varref);
	   	appendStringInfo(qual_source,
						 "varref_%s(&errors, bitmask, cn%d, cv%d)",
						 dtype->type_ident, var->varattno, var->varattno);
	}
	ReleaseSysCache(tup);

	return true;
}

static bool
make_device_bool_code(BoolExpr *b, StringInfo qual_source,
					  Oid base_relid,
					  List **type_decl, List **func_decl)
{
	StringInfoData	temp;
	ListCell	   *cell;
	int				count;

	switch (b->boolop)
	{
		case AND_EXPR:
			appendStringInfo(qual_source, "(");
			foreach (cell, b->args)
			{
				if (cell != list_head(b->args))
					appendStringInfo(qual_source, " && ");
				if (!make_device_qual_code((Node *)lfirst(cell),
										   qual_source,
										   base_relid,
										   type_decl, func_decl))
					return false;
				appendStringInfo(qual_source, ")");
			}
			break;

		case OR_EXPR:
			initStringInfo(&temp);
			count = 0;

			foreach (cell, b->args)
			{
				List	   *type_decl_temp = NIL;
				List	   *func_decl_temp = NIL;
				ListCell   *l;

				resetStringInfo(&temp);

				if (make_device_qual_code((Node *) lfirst(cell),
										  &temp,
										  base_relid,
										  &type_decl_temp,
										  &func_decl_temp))
				{
					if (count++ == 0)
						appendStringInfo(qual_source, "(%s", temp.data);
					else
						appendStringInfo(qual_source, " || %s", temp.data);

					foreach (l, type_decl_temp)
						*type_decl = lappend(*type_decl, lfirst(l));
					foreach (l, func_decl_temp)
						*func_decl = lappend(*func_decl, lfirst(l));
				}
			}
			if (count == 0)
				return false;
			appendStringInfo(qual_source, ")");
			break;

		case NOT_EXPR:
			Assert(list_length(b->args) == 1);
			appendStringInfo(qual_source, "!");
			if (!make_device_qual_code((Node *) linitial(b->args),
									   qual_source,
									   base_relid,
									   type_decl, func_decl))
				return false;
			break;

		default:
			elog(ERROR, "unexpected operation of BoolExpr: %d", b->boolop);
			break;
	}
	return true;
}

/*
 * make_device_qual_code
 *
 * It generate source code on the device corresponding to the supplied
 * qualifier. It assumes the qualifier already passes all the checks in
 * the is_device_executable_qual()
 */
static bool
make_device_qual_code(Node *node, StringInfo qual_source,
					  Oid base_relid,
					  List **type_decl, List **func_decl)
{
	if (node == NULL)
		return false;

	if (IsA(node, Const))
	{
		PgStromDevTypeInfo *tdev;
		Const  *c = (Const *) node;

		if (c->constisnull)
			return false;

		pgstrom_devtype_format(qual_source,
							   c->consttype,
							   c->constvalue);
		tdev = pgstrom_devtype_lookup(c->consttype);
		if (tdev->type_source)
			*type_decl = list_append_unique_ptr(*type_decl,
												tdev->type_source);
	}
	else if (IsA(node, Var))
	{
		if (!make_device_var_code((Var *)node, qual_source,
								  base_relid,
								  type_decl, func_decl))
			return false;
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr		   *f = (FuncExpr *) node;

		if (!make_device_function_code(f->funcid, f->args,
									   f->funcresulttype,
									   qual_source, base_relid,
									   type_decl, func_decl))
			return false;
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr *op = (OpExpr *) node;
		Oid		funcid = get_opcode(op->opno);

		if (!make_device_function_code(funcid, op->args,
									   op->opresulttype,
									   qual_source, base_relid,
									   type_decl, func_decl))
			return false;
	}
	else if (IsA(node, BoolExpr))
	{
		if (!make_device_bool_code((BoolExpr *)node,
								   qual_source, base_relid,
								   type_decl, func_decl))
			return false;
	}
	else
		elog(ERROR, "unexpected node: %s", nodeToString(node));

	return true;
}

static void
make_device_source(Oid base_relid, List *device_quals,
				   Bitmapset *clause_cols, List **private)
{
	StringInfoData	kern;
	StringInfoData	decl;
	StringInfoData	qual;
	StringInfoData	blk1;
	StringInfoData	blk2;
	Relids		columns = NULL;
	Relids		tempset;
	AttrNumber	attnum;
	DefElem	   *defel;
	ListCell   *cell;
	List	   *qual_list = NIL;
	List	   *type_decl = NIL;
	List	   *func_decl = NIL;

	initStringInfo(&kern);
	initStringInfo(&decl);
	initStringInfo(&qual);
	initStringInfo(&blk1);
	initStringInfo(&blk2);

	/*
	 * Enumelate all the variables being referenced
	 */
	foreach (cell, device_quals)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		Assert(IsA(rinfo, RestrictInfo));

		qual_list = lappend(qual_list, rinfo->clause);
	}

	switch (list_length(qual_list))
	{
		case 0:
			/*
			 * All the tuples shall be visible without execute qualifier
			 */
			return;
		case 1:
			if (!make_device_qual_code(linitial(qual_list),
									   &qual, base_relid,
									   &type_decl, &func_decl))
			{
				/* All the tuples shall be invisible */
				defel = makeDefElem("nevermatch",
									(Node *)makeInteger(TRUE));
				*private = lappend(*private, (Node *) defel);
				return;
			}
			break;
		default:
			if (!make_device_qual_code((Node *) makeBoolExpr(AND_EXPR,
															 qual_list, 0),
									   &qual, base_relid,
									   &type_decl, &func_decl))
			{
				/* All the tuples shall be invisible */
				defel = makeDefElem("nevermatch",
									(Node *)makeInteger(TRUE));
				*private = lappend(*private, (Node *) defel);
				return;
			}
			break;
	}

	appendStringInfo(&kern,
			 "__global__ void\n"
			 "pgstrom_qual(unsigned char rowmap[]");
	appendStringInfo(&blk1,
			 "    int offset_base = blockIdx.x * blockDim.x + threadIdx.x;\n"
			 "    int offset = offset_base * 8;\n"
			 "    unsigned char result = rowmap[offset_base];\n"
			 "    unsigned char errors = 0;\n");
	appendStringInfo(&blk2,
			 "    for (bitmask=1; bitmask < 256; bitmask <<= 1)\n"
			 "    {\n");

	tempset = bms_copy(clause_cols);
	while ((attnum = bms_first_member(tempset)) > 0)
	{
		PgStromDevTypeInfo *tinfo
			= pgstrom_devtype_lookup(get_atttype(base_relid, attnum));

		appendStringInfo(&kern, ",\n"
						 "             %s c%d_values[],\n"
						 "             unsigned char c%d_nulls[]",
						 tinfo->type_ident, attnum, attnum);
		appendStringInfo(&blk1,
						 "    unsigned char cn%d = c%d_nulls[offset_base];\n",
						 attnum, attnum);
		appendStringInfo(&blk2,
						 "        %s cv%d = c%d_values[offset];\n",
						 tinfo->type_ident, attnum, attnum);
	}
	appendStringInfo(&kern, ")\n"
					 "{\n"
					 "%s"
					 "    int bitmask;\n"
					 "\n"
					 "%s"
					 "\n"
					 "        if ((result & bitmask) && !%s)\n"
					 "            result &= ~bitmask;\n"
					 "        offset++;\n"
					 "    }\n"
					 "    rowmap[offset_base] = (result & ~errors);\n"
					 "}", blk1.data, blk2.data, qual.data);
	/*
	 * Declarations
	 */
	appendStringInfo(&decl,
					 "typedef unsigned long size_t;\n"
					 "typedef long __clock_t;\n"
					 "typedef __clock_t clock_t;\n"
					 "#include \"crt/device_runtime.h\"\n\n");
	foreach (cell, type_decl)
		appendStringInfo(&decl, "%s;\n", (char *) lfirst(cell));
	if (type_decl != NIL)
		appendStringInfo(&decl, "\n");
	foreach (cell, func_decl)
		appendStringInfo(&decl, "%s\n", (char *) lfirst(cell));

	/* kernel function follows by declaration part */
	appendStringInfo(&decl, "%s", kern.data);

	/*
	 * Set up private members being referenced in executor stage
	 */
	defel = makeDefElem("kernel_source", (Node *) makeString(decl.data));
	*private = lappend(*private, (Node *)defel);

	while ((attnum = bms_first_member(clause_cols)) > 0)
	{
		defel = makeDefElem("clause_cols",
							(Node *) makeInteger(attnum));
		*private = lappend(*private, (Node *)defel);
	}
	bms_free(columns);

	pfree(blk1.data);
	pfree(blk2.data);
	pfree(qual.data);
	pfree(kern.data);
}

/*
 * is_device_executable_qual
 *
 * It checks whether the supplied qualifier is available to run on GPU
 * device, or not. All the executable functions, operators are listed
 * on devinfo.c. If the qualifier contains any unlisted node, it is
 * not executable on device.
 */
static bool
is_device_executable_qual_walker(Node *node, void *context)
{
	if (node == NULL)
		return false;
	if (IsA(node, Const))
	{
		Const *con = (Const *)node;

		if (!pgstrom_devtype_lookup(con->consttype))
			return true;
	}
	else if (IsA(node, Var))
	{
		Var		   *v = (Var *)node;
		void	  **context_args = (void **)context;
		RelOptInfo *baserel = context_args[0];
		Bitmapset **clause_cols = context_args[1];

		if (v->varno != baserel->relid)
			return true;	/* should not be happen... */
		if (v->varlevelsup != 0)
			return true;	/* should not be happen... */
		if (v->varattno < 1)
			return true;	/* system columns are not supported */
		if (!pgstrom_devtype_lookup(v->vartype))
			return true;	/* unsupported data type? */

		*clause_cols = bms_add_member(*clause_cols, v->varattno);
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr   *f = (FuncExpr *)node;

		if (!pgstrom_devfunc_lookup(f->funcid))
			return true;
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr *op = (OpExpr *)node;

		if (!pgstrom_devfunc_lookup(get_opcode(op->opno)))
			return true;
	}
	else if (IsA(node, BoolExpr))
	{
		/* any bool expr acceptable */
	}
#if 0
	else if (IsA(node, RelabelType))
	{
		RelabelType	*rl = (RelabelType *)node;

		if (!pgstrom_devcast_lookup(exprType((Node *)rl->arg),
									rl->resulttype))
			return true;
	}
	else if (IsA(node, CoerceViaIO))
	{
		CoerceViaIO	*cv = (CoerceViaIO *)node;

		if (!pgstrom_devcast_lookup(exprType((Node *)cv->arg),
									cv->resulttype))
			return true;
	}
#endif
	else
		return true;

	return expression_tree_walker(node,
								  is_device_executable_qual_walker,
								  context);
}

static bool
is_device_executable_qual(RelOptInfo *baserel, RestrictInfo *rinfo,
						  Bitmapset **clause_cols)
{
	void   *context[2];

	if (!bms_singleton_member(rinfo->required_relids))
		return false;

	context[0] = baserel;
	context[1] = clause_cols;

	return !is_device_executable_qual_walker((Node *) rinfo->clause,
											 (void *) &context);
}

/*
 * pgstrom_plan_foreign_scan
 *
 * FDW handler to generate execute plan of PG-Strom.
 */
FdwPlan *
pgstrom_plan_foreign_scan(Oid foreignTblOid,
						  PlannerInfo *root,
						  RelOptInfo *baserel)
{
	FdwPlan	   *fdwplan;
	List	   *private = NIL;
	List	   *host_quals = NIL;
	List	   *device_quals = NIL;
	Bitmapset  *clause_cols = NULL;
	ListCell   *cell;
	DefElem	   *defel;
	AttrNumber	i;

	/*
	 * Save the referenced columns
	 */
	for (i = baserel->min_attr; i <= baserel->max_attr; i++)
	{
		if (bms_is_empty(baserel->attr_needed[i - baserel->min_attr]))
			continue;

		defel = makeDefElem("required_cols", (Node *) makeInteger(i));
		private = lappend(private, (Node *) defel);
	}

	/*
	 * Check whether device computable qualifier
	 */
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_get_num_devices() > 0 &&
			is_device_executable_qual(baserel, rinfo, &clause_cols))
			device_quals = lappend(device_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}

	baserel->baserestrictinfo = host_quals;
	if (device_quals != NIL)
		make_device_source(foreignTblOid, device_quals, clause_cols, &private);

	/*
	 * Set up FdwPlan
	 *
	 * TODO: more pratical information shall be returned to the planner.
	 */
	fdwplan = makeNode(FdwPlan);
	fdwplan->startup_cost = 0.0;
	fdwplan->total_cost = 0.0;
	fdwplan->fdw_private = private;

	return fdwplan;
}

/*
 * pgstrom_explain_foreign_scan 
 *
 * FDW handler to generate the result of EXPLAIN statement
 */
void
pgstrom_explain_foreign_scan(ForeignScanState *fss,
							 ExplainState *es)
{
	ForeignScan	   *fscan = (ForeignScan *) fss->ss.ps.plan;
	Relation		rel = fss->ss.ss_currentRelation;
	ListCell	   *cell;
	DefElem		   *nevermatch = NULL;
	DefElem		   *kernel_source = NULL;
	Relids			clause_cols = NULL;
	Relids			required_cols = NULL;
	StringInfoData	str;
	AttrNumber		attnum;
	Form_pg_attribute	attr;

	foreach (cell, fscan->fdwplan->fdw_private)
	{
		DefElem	   *defel = (DefElem *)lfirst(cell);

		if (strcmp(defel->defname, "nevermatch") == 0)
			nevermatch = defel;
		else if (strcmp(defel->defname, "kernel_source") == 0)
			kernel_source = defel;
		else if (strcmp(defel->defname, "clause_cols") == 0)
		{
			clause_cols = bms_add_member(clause_cols,
										 intVal(defel->arg));
		}
		else if (strcmp(defel->defname, "required_cols") == 0)
		{
			required_cols = bms_add_member(required_cols,
										   intVal(defel->arg));
		}
	}
	initStringInfo(&str);

	if (!bms_is_empty(required_cols))
	{
		resetStringInfo(&str);
		while ((attnum = bms_first_member(required_cols)) > 0)
		{
			attr = RelationGetDescr(rel)->attrs[attnum - 1];
			appendStringInfo(&str, "%s%s",
							 (str.len > 0 ? ", " : ""),
							 NameStr(attr->attname));
		}
		ExplainPropertyText(" Required Cols ", str.data, es);
	}

	if (nevermatch && intVal(nevermatch->arg) == TRUE)
	{
		ExplainPropertyText("  Filter", "false", es);
	}
	else if (kernel_source)
	{
		char   *p;
		char	temp[24];
		int		lineno = 1;

		if (!bms_is_empty(clause_cols))
		{
			resetStringInfo(&str);
			while ((attnum = bms_first_member(clause_cols)) > 0)
			{
				attr = RelationGetDescr(rel)->attrs[attnum - 1];
				appendStringInfo(&str, "%s%s",
								 (str.len > 0 ? ", " : ""),
								 NameStr(attr->attname));
			}
			ExplainPropertyText("Used in clause ", str.data, es);
		}

		resetStringInfo(&str);
		for (p = strVal(kernel_source->arg); *p != '\0'; p++)
		{
			if (*p == '\n')
			{
				snprintf(temp, sizeof(temp), "% 4d", lineno++);
				ExplainPropertyText(temp, str.data, es);
				resetStringInfo(&str);
			}
			else
				appendStringInfoChar(&str, *p);
		}
		if (str.len > 0)
		{
			snprintf(temp, sizeof(temp), "% 4d", lineno++);
			ExplainPropertyText(temp, str.data, es);
		}
	}
	pfree(str.data);
}
