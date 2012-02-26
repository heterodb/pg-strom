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
#include "access/sysattr.h"
#include "catalog/pg_type.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "optimizer/var.h"
#include "pg_strom.h"
#include "opencl_catalog.h"

static bool
is_gpu_executable_qual_walker(Node *node, void *context)
{
	if (node == NULL)
		return false;
	if (IsA(node, Const))
	{
		Const  *c = (Const *) node;

		/* is it a supported data type by GPU? */
		if (!pgstrom_gpu_type_lookup(c->consttype))
			return true;
	}
	else if (IsA(node, Var))
	{
		RelOptInfo *baserel = (RelOptInfo *) context;
		Var		   *v = (Var *) node;

		if (v->varno != baserel->relid)
			return true;	/* should not happen */
		if (v->varlevelsup != 0)
			return true;	/* should not happen */
		if (v->varattno < 1)
			return true;	/* system columns are not supported */

		/* is it a supported data type by GPU? */
		if (!pgstrom_gpu_type_lookup(v->vartype))
			return true;
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr   *f = (FuncExpr *) node;

		/* is it a supported function/operator? */
		if (!pgstrom_gpu_func_lookup(f->funcid))
			return true;
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) node;

		/* is it a supported function/operator? */
		if (!pgstrom_gpu_func_lookup(get_opcode(op->opno)))
			return true;
	}
	else if (IsA(node, BoolExpr))
	{
		BoolExpr   *b = (BoolExpr *) node;

		if (b->boolop != AND_EXPR &&
			b->boolop != OR_EXPR &&
			b->boolop != NOT_EXPR)
			return true;
	}
	else
		return true;

	return expression_tree_walker(node,
								  is_gpu_executable_qual_walker,
								  context);
}

static bool
is_gpu_executable_qual(RelOptInfo *baserel, RestrictInfo *rinfo)
{
	if (bms_membership(rinfo->clause_relids) == BMS_MULTIPLE)
		return false;	/* should not happen */

	if (is_gpu_executable_qual_walker((Node *) rinfo->clause,
									  (void *) baserel))
		return false;

	return true;
}

static void make_gpu_commands_walker(Node *node, StringInfo cmds, int regidx,
									 Bitmapset **gpu_cols);

static void
make_gpu_func_commands(Oid func_oid, List *func_args,
					   StringInfo cmds, int regidx, Bitmapset **gpu_cols)
{
	GpuTypeInfo	   *gtype;
	GpuFuncInfo	   *gfunc;
	ListCell	   *cell;
	int			   *regargs;
	int				i = 0;

	gfunc = pgstrom_gpu_func_lookup(func_oid);
	Assert(gfunc != NULL);

	regargs = alloca(sizeof(int) * (1 + gfunc->func_nargs));
	regargs[i++] = regidx;

	gtype = pgstrom_gpu_type_lookup(gfunc->func_rettype);
	Assert(gtype != NULL);
	regidx += (gtype->type_x2regs ? 2 : 1);

	i = 1;
	foreach (cell, func_args)
	{
		make_gpu_commands_walker(lfirst(cell), cmds, regidx, gpu_cols);
		regargs[i] = regidx;

		Assert(exprType(lfirst(cell)) == gfunc->func_argtypes[i-1]);
		gtype = pgstrom_gpu_type_lookup(gfunc->func_argtypes[i-1]);
		Assert(gtype != NULL);

		regidx += (gtype->type_x2regs ? 2 : 1);
		i++;
	}
	Assert(gfunc->func_nargs == i - 1);

	appendStringInfo(cmds, "%08x", gfunc->func_cmd);
	for (i=0; i <= gfunc->func_nargs; i++)
		appendStringInfo(cmds, "%08x", regargs[i]);
}

static void
make_gpu_commands_walker(Node *node, StringInfo cmds, int regidx,
						 Bitmapset **gpu_cols)
{
	GpuTypeInfo	   *gtype;
	ListCell	   *cell;

	if (node == NULL)
		return;
	if (IsA(node, Const))
	{
		Const  *c = (Const *) node;

		gtype = pgstrom_gpu_type_lookup(c->consttype);
		Assert(gtype != NULL);

		if (c->constisnull)
			appendStringInfo(cmds, "%08x%08x", GPUCMD_CONREF_NULL, regidx);
		else if (!gtype->type_x2regs)
			appendStringInfo(cmds, "%08x%08x%08x",
							 gtype->type_conref, regidx,
							 DatumGetInt32(c->constvalue));
		else
		{
			union {
				uint32	reg32[2];
				uint64	reg64;
			} temp;

			temp.reg64 = DatumGetInt64(c->constvalue);
			appendStringInfo(cmds, "%08x%08x%08x%08x",
							 gtype->type_conref, regidx,
							 temp.reg32[0], temp.reg32[1]);
		}
	}
	else if (IsA(node, Var))
	{
		Var	   *v = (Var *) node;

		gtype = pgstrom_gpu_type_lookup(v->vartype);
		Assert(gtype != NULL);

		appendStringInfo(cmds, "%08x%08x%08x",
						 gtype->type_varref, regidx, v->varattno);
		*gpu_cols = bms_add_member(*gpu_cols, v->varattno);
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr   *f = (FuncExpr *) node;

		make_gpu_func_commands(f->funcid, f->args, cmds, regidx, gpu_cols);
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) node;

		make_gpu_func_commands(get_opcode(op->opno),
							   op->args, cmds, regidx, gpu_cols);
	}
	else if (IsA(node, BoolExpr))
	{
		BoolExpr   *bx = (BoolExpr *) node;

		if (bx->boolop == NOT_EXPR)
		{
			Assert(list_length(bx->args) == 1);
			Assert(exprType(linitial(bx->args)) == BOOLOID);

			make_gpu_commands_walker(linitial(bx->args), cmds, regidx+1,
									 gpu_cols);
			appendStringInfo(cmds, "%08x%08x", GPUCMD_BOOLOP_NOT, regidx);
		}
		else if (bx->boolop == AND_EXPR || bx->boolop == OR_EXPR)
		{
			int		shift = 0;

			Assert(list_length(bx->args) > 1);
			foreach (cell, bx->args)
			{
				Assert(exprType(lfirst(cell)) == BOOLOID);

				make_gpu_commands_walker(lfirst(cell), cmds, regidx + shift,
										 gpu_cols);
				shift++;
			}
			Assert(list_length(bx->args) == shift);
			while (shift >= 2)
			{
				appendStringInfo(cmds, "%08x%08x%08x",
								 (bx->boolop == AND_EXPR ?
								  GPUCMD_BOOLOP_AND : GPUCMD_BOOLOP_OR),
								 regidx + shift - 2, regidx + shift - 1);
				shift--;
			}			
		}
		else
			elog(ERROR, "PG-Strom: unexpected BoolOp %d", (int) bx->boolop);
	}
	else
		elog(ERROR, "PG-Strom: unexpected node type: %d", nodeTag(node));
}

static char *
make_gpu_commands(List *gpu_quals, Bitmapset **gpu_cols)
{
	StringInfoData	cmds;
	RestrictInfo   *rinfo;

	initStringInfo(&cmds);

	Assert(list_length(gpu_quals) > 0);
	if (list_length(gpu_quals) == 1)
	{
		rinfo = linitial(gpu_quals);
		make_gpu_commands_walker((Node *)rinfo->clause, &cmds, 0, gpu_cols);
	}
	else
	{
		List	   *quals = NIL;
		ListCell   *cell;

		foreach (cell, gpu_quals)
		{
			rinfo = lfirst(cell);
			quals = lappend(quals, rinfo->clause);
		}
		make_gpu_commands_walker((Node *)makeBoolExpr(AND_EXPR, quals, -1),
								 &cmds, 0, gpu_cols);
	}
	appendStringInfo(&cmds, "%08x", GPUCMD_TERMINAL_COMMAND);
	return cmds.data;
}

static bool
is_cpu_executable_qual(RelOptInfo *baserel, RestrictInfo *rinfo)
{
	return false;
}

static char *
make_cpu_commands(List *cpu_quals, Bitmapset **cpu_cols)
{
	return NULL;
}

FdwPlan *
pgstrom_plan_foreign_scan(Oid ftableOid,
						  PlannerInfo *root,
						  RelOptInfo *baserel)
{
	FdwPlan	   *fdwplan;
	List	   *host_quals = NIL;
	List	   *gpu_quals = NIL;
	List	   *cpu_quals = NIL;
	List	   *private = NIL;
	Bitmapset  *required_cols = NULL;
	ListCell   *cell;
	AttrNumber	attno;
	char	   *cmdstr;
	DefElem	   *defel;

	/*
	 * check whether GPU/CPU executable qualifier, or not
	 */
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (is_gpu_executable_qual(baserel, rinfo))
			gpu_quals = lappend(gpu_quals, rinfo);
		else if (is_cpu_executable_qual(baserel, rinfo))
			cpu_quals = lappend(cpu_quals, rinfo);
		else
		{
			pull_varattnos((Node *)rinfo->clause,
						   baserel->relid, &required_cols);
			host_quals = lappend(host_quals, rinfo);
		}
	}
	baserel->baserestrictinfo = host_quals;

	/*
	 * Generate command series executed with GPU/CPU, if any
	 */
	if (gpu_quals)
	{
		Bitmapset  *gpu_cols = NULL;

		cmdstr = make_gpu_commands(gpu_quals, &gpu_cols);
		defel = makeDefElem("gpu_cmds", (Node *) makeString(cmdstr));
		private = lappend(private, defel);

		while ((attno = bms_first_member(gpu_cols)) >= 0)
		{
			defel = makeDefElem("gpu_cols", (Node *) makeInteger(attno));
			private = lappend(private, defel);
		}
		bms_free(gpu_cols);
	}
	if (cpu_quals)
	{
		Bitmapset  *cpu_cols = NULL;

		cmdstr = make_cpu_commands(cpu_quals, &cpu_cols);
		defel = makeDefElem("cpu_cmds", (Node *) makeString(cmdstr));
		private = lappend(private, defel);

		while ((attno = bms_first_member(cpu_cols)) >= 0)
		{
			defel = makeDefElem("cpu_cols", (Node *) makeInteger(attno));
			private = lappend(private, defel);
		}
		bms_free(cpu_cols);
	}

	/*
	 * Save the referenced columns with both of targetlist and host quals
	 */
	for (attno = baserel->min_attr; attno <= baserel->max_attr; attno++)
	{
		if (!bms_is_empty(baserel->attr_needed[attno - baserel->min_attr]))
			required_cols = bms_add_member(required_cols,
						attno - FirstLowInvalidHeapAttributeNumber);
	}

	while ((attno = bms_first_member(required_cols)) >= 0)
	{
		attno += FirstLowInvalidHeapAttributeNumber;
		if (attno < 0)
			continue;
		defel = makeDefElem("required_cols", (Node *) makeInteger(attno));
		private = lappend(private, defel);
	}
	bms_free(required_cols);

	/*
	 * Construct FdwPlan object
	 */
	fdwplan = makeNode(FdwPlan);
	fdwplan->fdw_private = private;

	return fdwplan;
}

void
pgstrom_explain_foreign_scan(ForeignScanState *fss,
							 ExplainState *es)
{
	ForeignScan	   *fscan = (ForeignScan *) fss->ss.ps.plan;
	Relation		relation = fss->ss.ss_currentRelation;
	char		   *gpu_cmds = NULL;
	char		   *cpu_cmds = NULL;
	Bitmapset	   *gpu_cols = NULL;
	Bitmapset	   *cpu_cols = NULL;
	Bitmapset	   *required_cols = NULL;
	ListCell	   *cell;
	StringInfoData	str;
	AttrNumber		attno;
	Form_pg_attribute attr;

	foreach (cell, fscan->fdwplan->fdw_private)
	{
		DefElem	   *defel = (DefElem *) lfirst(cell);

		if (strcmp(defel->defname, "gpu_cmds") == 0)
			gpu_cmds = strVal(defel->arg);
		else if (strcmp(defel->defname, "cpu_cmds") == 0)
			cpu_cmds = strVal(defel->arg);
		else if (strcmp(defel->defname, "gpu_cols") == 0)
			gpu_cols = bms_add_member(gpu_cols, intVal(defel->arg));
		else if (strcmp(defel->defname, "cpu_cols") == 0)
			cpu_cols = bms_add_member(cpu_cols, intVal(defel->arg));
		else if (strcmp(defel->defname, "required_cols") == 0)
			required_cols = bms_add_member(required_cols, intVal(defel->arg));
		else
			elog(ERROR, "unexpected parameter: %s", defel->defname);
	}
	initStringInfo(&str);

	if (!bms_is_empty(required_cols))
	{
		resetStringInfo(&str);
		while ((attno = bms_first_member(required_cols)) > 0)
		{
			attr = RelationGetDescr(relation)->attrs[attno - 1];
			appendStringInfo(&str, "%s%s",
							 str.len > 0 ? ", " : "",
							 NameStr(attr->attname));
		}
		ExplainPropertyText("required cols ", str.data, es);
	}

	if (!bms_is_empty(gpu_cols))
	{
		resetStringInfo(&str);
		while ((attno = bms_first_member(gpu_cols)) > 0)
		{
			attr = RelationGetDescr(relation)->attrs[attno - 1];
			appendStringInfo(&str, "%s%s",
							 str.len > 0 ? ", " : "",
							 NameStr(attr->attname));
		}
		ExplainPropertyText("GPU load cols ", str.data, es);
	}

	if (!bms_is_empty(cpu_cols))
	{
		resetStringInfo(&str);
		while ((attno = bms_first_member(cpu_cols)) > 0)
		{
			attr = RelationGetDescr(relation)->attrs[attno - 1];
			appendStringInfo(&str, "%s%s",
							 str.len > 0 ? ", " : "",
							 NameStr(attr->attname));
		}
		ExplainPropertyText("CPU load cols ", str.data, es);
	}

	if (gpu_cmds != NULL)
	{
		int	   *cmds;
		char   *p;
		char	temp[1024];
		int		skip;
		bool	first = true;

		resetStringInfo(&str);
		for (p = gpu_cmds; *p != '\0'; p += 8)
		{
			int		code;

			strncpy(temp, p, 8);
			temp[8] = '\0';

			sscanf(temp, "%x", &code);
			appendBinaryStringInfo(&str, (const char *)&code, sizeof(int));
		}
		cmds = (int *)str.data;

		for (skip = 0; skip >= 0; cmds += skip)
		{
			skip = pgstrom_gpu_command_string(RelationGetRelid(relation),
											  cmds, temp, sizeof(temp));
			if (first)
				ExplainPropertyText("OpenCL command", temp, es);
			else
				ExplainPropertyText("              ", temp, es);
			first = false;
		}
	}

	if (cpu_cmds != NULL)
	{
		/* add cpu command output */
	}
}



#if 0
#include "postgres.h"
#include "access/sysattr.h"
#include "catalog/pg_type.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/var.h"
#include "utils/builtins.h"
#include "utils/int8.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "pg_strom.h"

typedef struct {
	Oid			base_relid;
	List	   *type_decl;
	List	   *func_decl;
	List	   *const_decl;
	Bitmapset  *clause_cols;
} MakeDeviceQualContext;

static bool
make_device_qual_code(Node *node, StringInfo qual_source,
					  MakeDeviceQualContext *context);

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
						  MakeDeviceQualContext *context)
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
									   qual_source, context))
				return false;
			appendStringInfo(qual_source, ")");
			break;

		case 'r':	/* function as right-operator */
			Assert(list_length(args) == 1);
			appendStringInfo(qual_source, "(");
			if (!make_device_qual_code((Node *) linitial(args),
									   qual_source, context))
				return false;
			appendStringInfo(qual_source, "%s)", dfunc->func_ident);
			break;

		case 'b':	/* function as left-right-operator */
			Assert(list_length(args) == 2);
			appendStringInfo(qual_source, "(");
			if (!make_device_qual_code((Node *) linitial(args),
									   qual_source, context))
				return false;
			appendStringInfo(qual_source, " %s ", dfunc->func_ident);
			if (!make_device_qual_code((Node *) lsecond(args),
									   qual_source, context))
				return false;
			appendStringInfo(qual_source, ")");
			break;

		case 'f':	/* function as built-in device function */
			appendStringInfo(qual_source, "%s(", dfunc->func_ident);
			foreach (cell, args)
			{
				if (cell != list_head(args))
					appendStringInfo(qual_source, ", ");
				if (!make_device_qual_code((Node *) lfirst(cell),
										   qual_source, context))
					return false;
			}
			appendStringInfo(qual_source, ")");
			break;

		case 'F':	/* function as self-defined device function */
			appendStringInfo(qual_source,
							 "%s(&errors, bitmask",
							 dfunc->func_ident);
			foreach (cell, args)
			{
				appendStringInfo(qual_source, ", ");
				if (!make_device_qual_code((Node *) lfirst(cell),
										   qual_source, context))
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
		context->func_decl
			= list_append_unique_ptr(context->func_decl,
									 dfunc->func_source);
	dtype = pgstrom_devtype_lookup(func_resulttype);
	if (dtype->type_source)
		context->type_decl
			= list_append_unique_ptr(context->type_decl,
									 dtype->type_source);
	for (i=0; i < dfunc->func_nargs; i++)
	{
		dtype = pgstrom_devtype_lookup(dfunc->func_argtypes[i]);
		if (dtype->type_source)
			context->type_decl
				= list_append_unique_ptr(context->type_decl,
										 dtype->type_source);
	}
	return true;
}

/*
 * make_device_const_code
 *
 * Helper function of make_device_qual_code; that generates a reference
 * to a particular variable.
 * We deliver constance values to GPU device using constant memory of the
 * device, instead of putting them on the generated source code directly,
 * because it allows to reduce times to compile GPU binary on run-time.
 */
static bool
make_device_const_code(Const *con, StringInfo qual_source,
					   MakeDeviceQualContext *context)
{
	PgStromDevTypeInfo *dtype;

	if (con->constisnull)
		return false;

	dtype = pgstrom_devtype_lookup(con->consttype);
   	if (dtype->type_source)
		context->type_decl =
			list_append_unique_ptr(context->type_decl,
								   dtype->type_source);
	if (dtype->type_conref)
		context->func_decl =
			list_append_unique_ptr(context->func_decl,
								   dtype->type_conref);

	appendStringInfo(qual_source, "conref_%s(constval[%d])",
					 dtype->type_ident,
					 list_length(context->const_decl));

	if (get_typbyval(con->consttype))
	{
		context->const_decl = lappend(context->const_decl,
									  makeInteger(con->constvalue));
	}
	else
	{
		/*
		 * XXX - 64bit const value is not available to pack with
		 * makeInteger on 32bit platform, so move it as string.
		 */
		char   *temp;

		Assert(con->consttype == INT8OID || con->consttype == FLOAT8OID);

		temp = DatumGetCString(DirectFunctionCall1(int8out,
												   con->constvalue));
		context->const_decl = lappend(context->const_decl,
									  makeString(temp));
	}
	return true;
}

/*
 * make_device_var_code
 *
 * Helper function of make_device_qual_code; that generates a reference
 * to a particular variable.
 */
static bool
make_device_var_code(Var *var, StringInfo qual_source,
					 MakeDeviceQualContext *context)
{
	PgStromDevTypeInfo *dtype;
	Form_pg_attribute	attr;
	HeapTuple	tup;

	dtype = pgstrom_devtype_lookup(var->vartype);
	if (dtype->type_source)
		context->type_decl
			= list_append_unique_ptr(context->type_decl,
									 dtype->type_source);

	/*
	 * Mark attribute number being used in the qualifier
	 */
	context->clause_cols = bms_add_member(context->clause_cols,
										  var->varattno);

	/*
	 * A trivial optimization with NOT NULL constraint; if we can make
	 * sure all the values being valid, no need to use varref_* device
	 * function to set a bit of "errors" due to null values.
	 */
	tup = SearchSysCache2(ATTNUM,
						  ObjectIdGetDatum(context->base_relid),
						  Int16GetDatum(var->varattno));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for attribute %d of relation %u",
			 var->varattno, context->base_relid);
	attr = (Form_pg_attribute) GETSTRUCT(tup);

	if (attr->attnotnull)
		appendStringInfo(qual_source, "cv%d", var->varattno);
	else
	{
		if (dtype->type_varref)
			context->func_decl
				= list_append_unique_ptr(context->func_decl,
										 dtype->type_varref);
	   	appendStringInfo(qual_source,
						 "varref_%s(&errors, bitmask, cn%d, cv%d)",
						 dtype->type_ident, var->varattno, var->varattno);
	}
	ReleaseSysCache(tup);

	return true;
}

/*
 * make_device_bool_code
 *
 * Helper function of make_device_qual_code; that generate a conjunction
 * of multiple qualifiers with AND, OR or NOT operators.
 */
static bool
make_device_bool_code(BoolExpr *b, StringInfo qual_source,
					  MakeDeviceQualContext *context)
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
										   qual_source, context))
					return false;
				appendStringInfo(qual_source, ")");
			}
			break;

		case OR_EXPR:
			initStringInfo(&temp);
			count = 0;

			foreach (cell, b->args)
			{
				MakeDeviceQualContext	cxt_temp;
				ListCell   *l;

				cxt_temp.base_relid = context->base_relid;
				cxt_temp.type_decl = NIL;
				cxt_temp.func_decl = NIL;
				cxt_temp.clause_cols = NULL;

				resetStringInfo(&temp);

				if (make_device_qual_code((Node *) lfirst(cell),
										  &temp, &cxt_temp))
				{
					if (count++ == 0)
						appendStringInfo(qual_source, "(%s", temp.data);
					else
						appendStringInfo(qual_source, " || %s", temp.data);

					foreach (l, cxt_temp.type_decl)
						context->type_decl  =
							list_append_unique_ptr(context->type_decl,
												   lfirst(l));
					foreach (l, cxt_temp.func_decl)
						context->func_decl =
							list_append_unique_ptr(context->func_decl,
												   lfirst(l));
					context->clause_cols = bms_union(context->clause_cols,
													 cxt_temp.clause_cols);
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
									   qual_source, context))
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
					  MakeDeviceQualContext *context)
{
	if (node == NULL)
		return false;

	if (IsA(node, Const))
	{
		if (!make_device_const_code((Const *)node, qual_source, context))
			return false;
	}
	else if (IsA(node, Var))
	{
		if (!make_device_var_code((Var *)node, qual_source, context))
			return false;
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr		   *f = (FuncExpr *) node;

		if (!make_device_function_code(f->funcid, f->args,
									   f->funcresulttype,
									   qual_source, context))
			return false;
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr *op = (OpExpr *) node;
		Oid		funcid = get_opcode(op->opno);

		if (!make_device_function_code(funcid, op->args,
									   op->opresulttype,
									   qual_source, context))
			return false;
	}
	else if (IsA(node, BoolExpr))
	{
		if (!make_device_bool_code((BoolExpr *)node,
								   qual_source, context))
			return false;
	}
	else
		elog(ERROR, "unexpected node: %s", nodeToString(node));

	return true;
}

static void
make_device_qual_source(Oid base_relid, List *device_quals,
						uint32 devinfo_flags, List **private)
{
	MakeDeviceQualContext cxt;
	StringInfoData	kern;
	StringInfoData	decl;
	StringInfoData	qual;
	StringInfoData	blk1;
	StringInfoData	blk2;
	Bitmapset	   *tempset;
	AttrNumber	attnum;
	DefElem	   *defel;
	ListCell   *cell;
	List	   *qual_list = NIL;

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

	cxt.base_relid = base_relid;
	cxt.type_decl = NIL;
	cxt.func_decl = NIL;
	cxt.const_decl = NIL;
	cxt.clause_cols = NULL;

	switch (list_length(qual_list))
	{
		case 0:
			/*
			 * All the tuples shall be visible without execute qualifier
			 */
			return;
		case 1:
			if (!make_device_qual_code(linitial(qual_list), &qual, &cxt))
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
									   &qual, &cxt))
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
			 "pgstrom_qual(unsigned int nitems,\n"
			 "             unsigned char rowmap[]");
	appendStringInfo(&blk1,
			 "    int offset_base = blockIdx.x * blockDim.x + threadIdx.x;\n"
			 "    int offset = offset_base * 8;\n"
			 "    unsigned char result = rowmap[offset_base];\n"
			 "    unsigned char errors = 0;\n");
	appendStringInfo(&blk2,
			 "    for (bitmask=1; bitmask < 256; bitmask <<= 1)\n"
			 "    {\n");

	tempset = bms_copy(cxt.clause_cols);
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
	bms_free(tempset);

	appendStringInfo(&kern, ")\n"
					 "{\n"
					 "%s"
					 "    int bitmask;\n"
					 "\n"
					 "    if (offset >= nitems)\n"
					 "        return;\n"
					 "%s"
					 "\n"
					 "        if ((result & bitmask) == 0 &&\n"
					 "            !%s)\n"
					 "            result |= bitmask;\n"
					 "        offset++;\n"
					 "    }\n"
					 "    rowmap[offset_base] = (result | errors);\n"
					 "}", blk1.data, blk2.data, qual.data);
	/*
	 * Declarations
	 */
	appendStringInfo(&decl,
					 "typedef unsigned long size_t;\n"
					 "typedef long __clock_t;\n"
					 "typedef __clock_t clock_t;\n"
					 "#include \"crt/device_runtime.h\"\n");
	if (devinfo_flags & DEVINFO_FLAGS_INC_MATHFUNC_H)
		appendStringInfo(&decl,
						 "#include \"math_functions.h\"\n");
	appendStringInfo(&decl, "\n");

	foreach (cell, cxt.type_decl)
		appendStringInfo(&decl, "%s;\n", (char *) lfirst(cell));
	if (cxt.type_decl != NIL)
		appendStringInfo(&decl, "\n");
	foreach (cell, cxt.func_decl)
		appendStringInfo(&decl, "%s\n", (char *) lfirst(cell));
	if (cxt.const_decl != NIL)
	{
		appendStringInfo(&decl, "__constant__ long constval[%d];\n\n",
						 list_length(cxt.const_decl));
		*private = lappend(*private,
						   makeDefElem("const_values",
									   (Node *)cxt.const_decl));
	}

	/* kernel function follows by declaration part */
	appendStringInfo(&decl, "%s", kern.data);

	/*
	 * Set up private members being referenced in executor stage
	 */
	defel = makeDefElem("kernel_source", (Node *) makeString(decl.data));
	*private = lappend(*private, (Node *)defel);

	tempset = bms_copy(cxt.clause_cols);
	while ((attnum = bms_first_member(tempset)) > 0)
	{
		defel = makeDefElem("clause_cols",
							(Node *) makeInteger(attnum));
		*private = lappend(*private, (Node *)defel);
	}
	bms_free(tempset);

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
typedef struct {
	RelOptInfo *baserel;
	uint32		devinfo_flags;
} IsDeviceExecutableQualContext;

static bool
is_device_executable_qual_walker(Node *node, void *context)
{
	IsDeviceExecutableQualContext *cxt = context;
	PgStromDevTypeInfo *dtype;
	PgStromDevFuncInfo *dfunc;

	if (node == NULL)
		return false;
	if (IsA(node, Const))
	{
		Const *con = (Const *)node;

		/* is it a supported data type? */
		dtype = pgstrom_devtype_lookup(con->consttype);
		if (!dtype)
			return true;
		cxt->devinfo_flags |= dtype->type_flags;
	}
	else if (IsA(node, Var))
	{
		Var		   *v = (Var *)node;

		if (v->varno != cxt->baserel->relid)
			return true;	/* should not be happen... */
		if (v->varlevelsup != 0)
			return true;	/* should not be happen... */
		if (v->varattno < 1)
			return true;	/* system columns are not supported */

		/* is it a supported data type? */
		dtype = pgstrom_devtype_lookup(v->vartype);
		if (!dtype)
			return true;
		cxt->devinfo_flags |= dtype->type_flags;
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr   *f = (FuncExpr *)node;

		/* is it a supported function/operator? */
		dfunc = pgstrom_devfunc_lookup(f->funcid);
		if (!dfunc)
			return true;
		cxt->devinfo_flags |= dfunc->func_flags;
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr *op = (OpExpr *)node;

		/* is it a supported function/operator? */
        dfunc = pgstrom_devfunc_lookup(get_opcode(op->opno));
		if (!dfunc)
			return true;
		cxt->devinfo_flags |= dfunc->func_flags;
	}
	else if (IsA(node, BoolExpr))
	{
		BoolExpr *b = (BoolExpr *)node;

		if (b->boolop != AND_EXPR &&
			b->boolop != OR_EXPR &&
			b->boolop != NOT_EXPR)
			return true;
	}
	else
		return true;

	return expression_tree_walker(node,
								  is_device_executable_qual_walker,
								  context);
}

static bool
is_device_executable_qual(RelOptInfo *baserel,
						  RestrictInfo *rinfo,
						  uint32 *devinfo_flags)
{
	IsDeviceExecutableQualContext	cxt;

	if (!bms_singleton_member(rinfo->required_relids))
		return false;	/* should not be happen */

	cxt.baserel = baserel;
	cxt.devinfo_flags = 0;

	if (is_device_executable_qual_walker((Node *) rinfo->clause,
										 (void *) &cxt))
		return false;

	/*
	 * in the case of the supplied qualifier needs double-FP support,
	 * at least, one GPU device must support this feature.
	 */
	if (cxt.devinfo_flags & DEVINFO_FLAGS_DOUBLE_FP)
	{
		/*
		 * TODO: we should put check 64bit-FP support here.
		 * If not, this routine should return false, to execute
		 * this qualifier by CPU
		 */
	}
	*devinfo_flags = cxt.devinfo_flags;
	return true;
}

/*
 * pgstrom_cost_estimation
 *
 * Estimate query execution cost, and set them
 */
static void
pgstrom_cost_estimation(FdwPlan *fdwplan, RelOptInfo *baserel,
						Oid ftableOid, List *host_quals, List *device_quals)
{
	/* TODO: more practical cost estimation */
	fdwplan->startup_cost = 100.0;
	fdwplan->total_cost = 0.0;

	baserel->rows = 20000000.0;
	baserel->width = 64;
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
	uint32		devinfo_flags = 0;
	Bitmapset  *required_cols = NULL;
	ListCell   *cell;
	DefElem	   *defel;
	AttrNumber	i;

	/*
	 * Check whether device computable qualifier
	 */
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_get_num_devices() > 0 &&
			is_device_executable_qual(baserel, rinfo, &devinfo_flags))
			device_quals = lappend(device_quals, rinfo);
		else
		{
			pull_varattnos((Node *)rinfo->clause,
						   baserel->relid, &required_cols);
			host_quals = lappend(host_quals, rinfo);
		}
	}
	baserel->baserestrictinfo = host_quals;
	if (device_quals != NIL)
		make_device_qual_source(foreignTblOid, device_quals,
								devinfo_flags, &private);

	/*
	 * Save the referenced columns by both of targelist and host quals
	 */
	for (i = baserel->min_attr; i <= baserel->max_attr; i++)
	{
		if (!bms_is_empty(baserel->attr_needed[i - baserel->min_attr]))
			required_cols = bms_add_member(required_cols,
								i - FirstLowInvalidHeapAttributeNumber);
	}

	while ((i = bms_first_member(required_cols)) > 0)
	{
		i += FirstLowInvalidHeapAttributeNumber;
		if (i > InvalidAttrNumber)
		{
			defel = makeDefElem("required_cols", (Node *) makeInteger(i));
			private = lappend(private, (Node *) defel);
		}
	}

	/*
	 * Construct FdwPlan object
	 */
	fdwplan = makeNode(FdwPlan);
	fdwplan->fdw_private = private;
	pgstrom_cost_estimation(fdwplan, baserel, foreignTblOid,
							host_quals, device_quals);

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
	DefElem		   *const_values = NULL;
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
		else if (strcmp(defel->defname, "const_values") == 0)
			const_values = defel;
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
		char	temp[64];
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

		if (const_values)
		{
			List   *const_values_list = (List *)const_values->arg;
			char	const_buf[64];
			int		const_idx = 0;

			foreach (cell, const_values_list)
			{
				Assert(IsA(lfirst(cell), Integer) ||
					   IsA(lfirst(cell), String));

				if (IsA(lfirst(cell), Integer))
				{
					pg_lltoa((int64) intVal(lfirst(cell)), const_buf);
					snprintf(temp, sizeof(temp), "constval[%d]", const_idx++);
					ExplainPropertyText(temp, const_buf, es);
				}
				else
				{
					snprintf(temp, sizeof(temp), "constval[%d]", const_idx++);
					ExplainPropertyText(temp, strVal(lfirst(cell)), es);
				}
			}
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
#endif
