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
#include "pg_strom.h"

static bool
make_device_qual_code(Node *node, StringInfo qual_source,
					  Oid base_relid,
					  List **type_decl, List **func_decl);

static bool
make_device_function_code(Oid func_oid, List *args,
						  StringInfo qual_source,
						  Oid base_relid,
						  List **type_decl, List **func_decl)
{
	PgStromDevFuncInfo *fdev = pgstrom_devfunc_lookup(func_oid);
	ListCell   *cell;

   	switch (fdev->func_kind)
	{
		case 'c':	/* function as constant */
			Assert(list_length(args) == 0);
			appendStringInfo(qual_source, "%s", fdev->func_ident);
			break;

		case 'l':	/* function as left-operator */
			Assert(list_length(args) == 1);
			appendStringInfo(qual_source, "(%s", fdev->func_ident);
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
			appendStringInfo(qual_source, "%s)", fdev->func_ident);
			break;

		case 'b':	/* function as left-right-operator */
			Assert(list_length(args) == 2);
			appendStringInfo(qual_source, "(");
			if (!make_device_qual_code((Node *) linitial(args),
									   qual_source, base_relid,
									   type_decl, func_decl))
				return false;
			appendStringInfo(qual_source, " %s ", fdev->func_ident);
			if (!make_device_qual_code((Node *) lsecond(args),
									   qual_source, base_relid,
									   type_decl, func_decl))
				return false;
			appendStringInfo(qual_source, ")");
			break;

		case 'f':	/* function as device function */
			appendStringInfo(qual_source, "%s(", fdev->func_ident);
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
		default:
			elog(ERROR, "unexpected func_kind : %c of \"%s\"",
				 fdev->func_kind, fdev->func_ident);
			break;
	}
	return true;
}

static bool
make_device_qual_code(Node *node, StringInfo qual_source,
					  Oid base_relid,
					  List **type_decl, List **func_decl)
{
	if (node == NULL)
		return false;

	if (IsA(node, Const))
	{
		Const  *c = (Const *) node;

		if (c->constisnull)
			return false;
		else
			pgstrom_devtype_format(qual_source,
								   c->consttype,
								   c->constvalue);
	}
	else if (IsA(node, Var))
	{
		Var *v = (Var *) node;
		appendStringInfo(qual_source, "%s_cs[offset]",
						 get_relid_attribute_name(base_relid, v->varattno));
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr		   *f = (FuncExpr *) node;

		make_device_function_code(f->funcid, f->args,
								  qual_source, base_relid,
								  type_decl, func_decl);
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr *op = (OpExpr *) node;

		make_device_function_code(get_opcode(op->opno), op->args,
								  qual_source, base_relid,
								  type_decl, func_decl);
	}
	else if (IsA(node, RelabelType))
	{}
	else if (IsA(node, CoerceViaIO))
	{}
	else if (IsA(node, BoolExpr))
	{
		BoolExpr   *b = (BoolExpr *)node;
		ListCell   *cell;

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
				}
				appendStringInfo(qual_source, ")");
				break;
			case OR_EXPR:
				{
					StringInfoData	temp_src;
					int		count = 0;

					foreach (cell, b->args)
					{
						initStringInfo(&temp_src);

						if (make_device_qual_code((Node *) lfirst(cell),
												  &temp_src, base_relid,
												  type_decl, func_decl))
						{
							if (count++ == 0)
								appendStringInfo(qual_source, "(%s",
												 temp_src.data);
							else
								appendStringInfo(qual_source, " || %s",
												 temp_src.data);
						}
						pfree(temp_src.data);
					}

					if (count == 0)
						appendStringInfo(qual_source, "false");
					else
						appendStringInfo(qual_source, ")");
				}
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
				break;
		}
	}
	else
		elog(ERROR, "unexpected node: %s", nodeToString(node));

	return true;
}

static char *
make_device_source(Oid base_relid, List *device_quals, Relids *columns)
{
	StringInfoData	source;
	StringInfoData	qualsrc;
	StringInfoData	subblk1;
	StringInfoData	subblk2;
	Relids		tempset;
	AttrNumber	attnum;
	ListCell   *cell;
	List	   *qual_list = NIL;
	List	   *type_decl = NIL;
	List	   *func_decl = NIL;

	initStringInfo(&source);
	initStringInfo(&qualsrc);
	initStringInfo(&subblk1);
	initStringInfo(&subblk2);

	/*
	 * Kernel function shall be declared as follows:
	 *
	 *
	 * __kernel void pgstrom_qual(long rowid,
	 *                            __global uchar *rowmap,
	 *                            __global <atttype> *<attname>_cs,
	 *                            __global uchar *<attname>_nulls,
	 *                                    :
	 *                            __global <atttype> *<attname>_cs,
	 *                            __global uchar *<attname>_nulls)
	 * {
	 *     int offset = get_global_id(0);
	 *     uchar result = rowmap[offset];
	 *     uchar <attname>_isnull = (<attname>_nl ? <attname>_nl[offset] : 0);
	 *             :
	 *     uchar <attname>_isnull = (<attname>_nl ? <attname>_nl[offset] : 0);
	 *     int   bitmask;
	 *
	 *     for (bitmask = 1; bitmask < 256; bitmask <<= 1)
	 *     {
	 *         if ((result & bitmask) &&
	 *             (<attname>_isnull & bitmask) == 0 &&
	 *                :
	 *             (<attname>_isnull & bitmask) == 0 &&
	 *             (!(qual) || error))
	 *             result &= ~bitmask;
	 *         offset++;
	 *     }
	 *     rowmap[get_global_id(0)] = result;
	 * }
	 */

	/*
	 * Enumelate all the variables being referenced
	 */
	foreach (cell, device_quals)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		Assert(IsA(rinfo, RestrictInfo));

		*columns = bms_union(*columns, pull_varnos((Node *)rinfo->clause));

		qual_list = lappend(qual_list, rinfo->clause);
	}

	switch (list_length(qual_list))
	{
		case 0:
			return NULL;		/* no need to check */
		case 1:
			if (!make_device_qual_code(linitial(qual_list),
									   &qualsrc, base_relid,
									   &type_decl, &func_decl))
				return NULL;	/* no need to run */
			break;
		default:
			if (!make_device_qual_code((Node *) makeBoolExpr(AND_EXPR,
															 qual_list, 0),
									   &qualsrc, base_relid,
									   &type_decl, &func_decl))
				return NULL;	/* no need to run */
			break;
	}

	appendStringInfo(&source,
					 "__kernel void\n"
					 "pgstrom_qual(long rowid,\n"
					 "             __global uchar *rowmap");
	appendStringInfo(&subblk1,
					 "    int   offset = get_global_id(0) << 3;\n"
					 "    uchar result = rowmap[offset];\n");
	appendStringInfo(&subblk2,
					 "    for (bitmask = 1; bitmask < 256; bitmask <<= 1)\n"
					 "    {\n"
					 "        int error = 0;\n"
					 "        if ((result & bitmask) &&\n");

	tempset = bms_copy(*columns);
	while ((attnum = bms_first_member(tempset)) > 0)
	{
		PgStromDevTypeInfo *tinfo
			= pgstrom_devtype_lookup(get_atttype(base_relid, attnum));
		const char	   *attname = get_attname(base_relid, attnum);

		appendStringInfo(&source, ",\n"
						 "             __global %s *%s_cs,\n"
						 "             __global uchar *%s_nl",
						 tinfo->type_ident,
						 attname, attname);
		appendStringInfo(&subblk1,
						 "    uchar %s_isnull = (%s_nl ? %s_nl[offset] : 0);\n",
						 attname, attname, attname);
		appendStringInfo(&subblk2,
						 "            (%s_isnull & bitmask) == 0 &&\n",
						 attname);
	}
	bms_free(tempset);

	appendStringInfo(&source, ")\n"
					 "{\n"
					 "%s"
					 "    int   bitmask;\n"
					 "\n"
					 "%s"
					 "            (!%s || error))\n"
					 "            result &= ~bitmask;\n"
					 "        offset++;\n"
					 "    }\n"
					 "    rowmap[get_global_id(0) << 3] = result;\n"
					 "}\n",
					 subblk1.data, subblk2.data, qualsrc.data);

	pfree(subblk1.data);
	pfree(subblk2.data);
	pfree(qualsrc.data);

	return source.data;
}

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
		RelOptInfo *baserel = (RelOptInfo *)context;
		Var	*v = (Var *)node;

		if (v->varno != baserel->relid)
			return true;	/* should not be happen... */
		if (v->varlevelsup != 0)
			return true;	/* should not be happen... */
		if (v->varattno < 1)
			return true;	/* system columns are not supported */
		if (!pgstrom_devtype_lookup(v->vartype))
			return true;	/* unsupported data type? */
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
	else
		return true;

	return expression_tree_walker(node,
								  is_device_executable_qual_walker,
								  context);
}

static bool
is_device_executable_qual(RelOptInfo *baserel, RestrictInfo *rinfo)
{
	if (!bms_singleton_member(rinfo->required_relids))
		return false;

	return !is_device_executable_qual_walker((Node *) rinfo->clause,
											 (void *) baserel);
}

FdwPlan *
pgstrom_plan_foreign_scan(Oid foreignTblOid,
						  PlannerInfo *root,
						  RelOptInfo *baserel)
{
	FdwPlan	   *fdwplan;
	List	   *private = NIL;
	List	   *host_quals = NIL;
	List	   *device_quals = NIL;
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

		defel = makeDefElem("cols_needed", (Node *) makeInteger(i));
		private = lappend(private, (Node *) defel);
	}

	/*
	 * Check whether device computable qualifier
	 */
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (is_device_executable_qual(baserel, rinfo))
			device_quals = lappend(device_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}

	baserel->baserestrictinfo = host_quals;
	if (device_quals != NIL)
	{
		Relids	columns = NULL;
		char   *device_source;

		device_source = make_device_source(foreignTblOid,
										   device_quals,
										   &columns);
		defel = makeDefElem("device_source",
							(Node *) makeString(device_source));
		private = lappend(private, (Node *) defel);

		while ((i = bms_first_member(columns)) > 0)
		{
			defel = makeDefElem("device_column", (Node *) makeInteger(i));
			private = lappend(private, (Node *) defel);
		}
	}

	/*
	 * Set up FdwPlan
	 */
	fdwplan = makeNode(FdwPlan);
	fdwplan->startup_cost = 1.0;
	fdwplan->startup_cost = 2.0;
	fdwplan->fdw_private = private;

	return fdwplan;
}


void
pgstrom_explain_foreign_scan(ForeignScanState *fss,
							 ExplainState *es)
{
	ForeignScan	   *fscan = (ForeignScan *) fss->ss.ps.plan;
	Relation		rel = fss->ss.ss_currentRelation;
	ListCell	   *cell;

	ExplainPropertyText("Stream Relation", RelationGetRelationName(rel), es);

	foreach (cell, fscan->fdwplan->fdw_private)
	{
		DefElem	   *defel = (DefElem *)lfirst(cell);
		char	   *buf, *p1, *p2;

		if (strcmp(defel->defname, "device_source") != 0)
			continue;

		buf = pstrdup(strVal(defel->arg));

		for (p1=buf; *p1 != '\0'; p1 = p2)
		{
			p2 = strchr(p1, '\n');
			if (!p2)
				break;
			*p2++ = '\0';
			ExplainPropertyText("  ", p1, es);
		}
		pfree(buf);
		break;
	}
}
