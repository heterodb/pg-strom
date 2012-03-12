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
#include "cuda_cmds.h"

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

static void push_cmd1(StringInfo cmds, int cmd1)
{
	appendBinaryStringInfo(cmds, (const char *)&cmd1, sizeof(cmd1));
}

static void push_cmd2(StringInfo cmds, int cmd1, int cmd2)
{
	appendBinaryStringInfo(cmds, (const char *)&cmd1, sizeof(cmd1));
	appendBinaryStringInfo(cmds, (const char *)&cmd2, sizeof(cmd2));
}

static void push_cmd3(StringInfo cmds, int cmd1, int cmd2, int cmd3)
{
	appendBinaryStringInfo(cmds, (const char *)&cmd1, sizeof(cmd1));
	appendBinaryStringInfo(cmds, (const char *)&cmd2, sizeof(cmd2));
	appendBinaryStringInfo(cmds, (const char *)&cmd3, sizeof(cmd3));
}

static void push_cmd4(StringInfo cmds, int cmd1, int cmd2, int cmd3, int cmd4)
{
	appendBinaryStringInfo(cmds, (const char *)&cmd1, sizeof(cmd1));
	appendBinaryStringInfo(cmds, (const char *)&cmd2, sizeof(cmd2));
	appendBinaryStringInfo(cmds, (const char *)&cmd3, sizeof(cmd3));
	appendBinaryStringInfo(cmds, (const char *)&cmd4, sizeof(cmd4));
}

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

	gtype = pgstrom_gpu_type_lookup(gfunc->func_rettype);
	Assert(gtype != NULL);

	/*
	 * XXX - 64bit variables have to be stored on the virtual registed
	 * indexed with even number, because unaligned access makes run-
	 * time error on device side.
	 */
	if (gtype->type_x2regs)
	{
		regidx = (regidx + 1) & ~(0x0001);
		regargs[i++] = regidx;
		regidx += 2;
	}
	else
	{
		regargs[i++] = regidx;
		regidx++;
	}

	i = 1;
	foreach (cell, func_args)
	{
		Assert(exprType(lfirst(cell)) == gfunc->func_argtypes[i-1]);
		gtype = pgstrom_gpu_type_lookup(exprType(lfirst(cell)));
		Assert(gtype != NULL);
		if (gtype->type_x2regs)
			regidx = (regidx + 1) & ~(0x0001);

		make_gpu_commands_walker(lfirst(cell), cmds, regidx, gpu_cols);
		regargs[i] = regidx;

		regidx += (gtype->type_x2regs ? 2 : 1);
		i++;
	}
	Assert(gfunc->func_nargs == i - 1);

	push_cmd1(cmds, gfunc->func_cmd);
	for (i=0; i <= gfunc->func_nargs; i++)
		push_cmd1(cmds, regargs[i]);
}

static void
make_gpu_commands_walker(Node *node, StringInfo cmds, int regidx,
						 Bitmapset **gpu_cols)
{
	GpuTypeInfo	   *gtype;
	ListCell	   *cell;
	union {
		uint32	reg32[2];
		uint64	reg64;
	} xreg;

	if (node == NULL)
		return;
	if (IsA(node, Const))
	{
		Const  *c = (Const *) node;

		gtype = pgstrom_gpu_type_lookup(c->consttype);
		Assert(gtype != NULL);

		if (c->constisnull)
			push_cmd2(cmds, GPUCMD_CONREF_NULL, regidx);
		else if (!gtype->type_x2regs)
			push_cmd3(cmds,
					  gtype->type_conref, regidx,
					  DatumGetInt32(c->constvalue));
		else
		{
			xreg.reg64 = DatumGetInt64(c->constvalue);
			push_cmd4(cmds,
					  gtype->type_conref, regidx,
					  xreg.reg32[0], xreg.reg32[1]);
		}
	}
	else if (IsA(node, Var))
	{
		Var	   *v = (Var *) node;

		gtype = pgstrom_gpu_type_lookup(v->vartype);
		Assert(gtype != NULL);

		push_cmd3(cmds,
				  gtype->type_varref, regidx, v->varattno - 1);
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
			push_cmd2(cmds, GPUCMD_BOOLOP_NOT, regidx);
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
				push_cmd3(cmds,
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

static bytea *
make_gpu_commands(List *gpu_quals, Bitmapset **gpu_cols)
{
	StringInfoData	cmds;
	RestrictInfo   *rinfo;
	int				code;

	initStringInfo(&cmds);
	appendStringInfoSpaces(&cmds, VARHDRSZ);

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
	code = GPUCMD_TERMINAL_COMMAND;
	appendBinaryStringInfo(&cmds, (const char *)&code, sizeof(code));
	SET_VARSIZE(cmds.data, cmds.len);
	return (bytea *)cmds.data;
}

static bool
is_cpu_executable_qual(RelOptInfo *baserel, RestrictInfo *rinfo)
{
	return false;
}

static bytea *
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
	bytea	   *cmds_bytea;
	Const	   *cmds_const;
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

		cmds_bytea = make_gpu_commands(gpu_quals, &gpu_cols);
		cmds_const = makeConst(BYTEAOID, -1, InvalidOid,
							   VARSIZE(cmds_bytea),
							   PointerGetDatum(cmds_bytea),
							   false, false);
		defel = makeDefElem("gpu_cmds", (Node *) cmds_const);
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

		cmds_bytea = make_cpu_commands(cpu_quals, &cpu_cols);
		cmds_const = makeConst(BYTEAOID, -1, InvalidOid,
							   VARSIZE(cmds_bytea),
							   PointerGetDatum(cmds_bytea),
							   false, false);
		defel = makeDefElem("cpu_cmds", (Node *) cmds_const);
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
	Const		   *gpu_cmds = NULL;
	Const		   *cpu_cmds = NULL;
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
			gpu_cmds = (Const *) defel->arg;
		else if (strcmp(defel->defname, "cpu_cmds") == 0)
			cpu_cmds = (Const *) defel->arg;
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
		ExplainPropertyText("Required cols ", str.data, es);
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
		char	temp[1024];
		int	   *cmds;
		int		skip;
		bool	first = true;

		Assert(IsA(gpu_cmds, Const));

		cmds = (int *)VARDATA((bytea *)(gpu_cmds->constvalue));
		for (skip = 0; skip >= 0; cmds += skip)
		{
			skip = pgstrom_gpu_command_string(RelationGetRelid(relation),
											  cmds, temp, sizeof(temp));
			if (first)
				ExplainPropertyText("CUDA commands ", temp, es);
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
