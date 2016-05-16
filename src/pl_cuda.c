/*
 * pl_cuda.c
 *
 * PL/CUDA SQL function support
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#include "postgres.h"
#include "builtins.h"
#include "catalog/pg_prog.h"
#include "utils/syscache.h"
#include "pg_strom.h"



typedef struct plcuda_func_info
{
	/* number of the threads to kick */
	Oid				func_num_threads;
	Size			value_num_threads;
	/* amount of shmem to allocate */
	Oid				func_shmem_size;
	Size			value_shmem_size;
	/* amount of device memory to allocate */
	Oid				func_buffer_size;
	Size			value_buffer_size;
	/* emergency fallback if GPU returned CpuReCheck error */
	Oid				func_cpu_fallback;
	/* kernel source code */
	cl_uint			extra_flags;
	const char	   *kern_decl;
	const char	   *kern_body;
} plcuda_func_info;

/*
 * XXX - to be revised to use ExtensibleNode in 9.6 based implementation
 */
static text *
form_plcuda_func_info(plcuda_func_info *cf_info)
{
	List   *l = NIL;

	l = lappend(l, makeInteger(cf_info->func_num_threads));
	l = lappend(l, makeInteger(cf_info->value_num_threads));
	l = lappend(l, makeInteger(cf_info->func_shmem_size));
	l = lappend(l, makeInteger(cf_info->value_shmem_size));
	l = lappend(l, makeInteger(cf_info->func_buffer_size));
	l = lappend(l, makeInteger(cf_info->value_buffer_size));
	l = lappend(l, makeInteger(cf_info->func_cpu_fallback));
	l = lappend(l, makeInteger(cf_info->extra_flags));
	l = lappend(l, makeString(cf_info->kern_decl));
	l = lappend(l, makeString(cf_info->kern_body));

	return cstring_to_text(nodeToString(l));
}

static plcuda_func_info *
deform_plcuda_func_info(text *cf_info_text)
{
	plcuda_func_info *cf_info = palloc0(sizeof(plcuda_func_info));
	List	   *l = stringToNode(VARDATA(cf_info_text));
	cl_uint		index = 0;

	cf_info->func_num_threads = intVal(list_nth(l, index++));
	cf_info->value_num_threads = intVal(list_nth(l, index++));
	cf_info->func_shmem_size = intVal(list_nth(l, index++));
	cf_info->value_shmem_size = intVal(list_nth(l, index++));
	cf_info->func_buffer_size = intVal(list_nth(l, index++));
	cf_info->value_buffer_size = intVal(list_nth(l, index++));
	cf_info->func_cpu_fallback = intVal(list_nth(l, index++));
	cf_info->extra_flags = intVal(list_nth(l, index++));
	cf_info->kern_decl = strVal(list_nth(l, index++));
	cf_info->kern_body = strVal(list_nth(l, index++));

	return cf_info;
}




static List *
plcuda_parse_tokens(const char *buffer)
{
	List	   *l = NIL;
	char	   *pos = buffer;
	char		quote = '\0';
	StringInfoData token;

	initStringInfo(&token);
	while (*pos != '\0')
	{
		if (token.len == 0)
		{
			Assert(quote == '\0');
			if (*pos == '"' || *pos == '\'')
			{
				quote = *pos;
				appendStringInfoChar(&token, *++pos);
			}
			else if (*pos == '\\')
			{
				if (*++pos == '\0')
					return NIL;
				appendStringInfoChar(&token, *pos);
			}
			else
			{
				appendStringInfoChar(&token, *pos);
			}
		}
		else
		{
			if (*pos == '\\')
			{
				if (*++pos == '\0')
					return NIL;
				appendStringInfoChar(&token, *pos);
			}
			else if (quote != '\0')
			{
				if (*pos == quote)
				{
					l = lappend(l, pstrdup(token.data));
					resetStringInfo(&token);
					quote = '\0';
				}
				else
					appendStringInfoChar(&token, *pos);
			}
			else
			{
				if (isspace(*pos))
				{
					l = lappend(l, pstrdup(token.data));
					resetStringInfo(&token);
					quote = '\0';
				}
				else if (*pos == '\"' || *pos == '\\')
				{
					l = lappend(l, pstrdup(token.data));
					resetStringInfo(&token);
					quote = *pos;

					if (*++pos == '\0')
						return NIL;
					appendStringInfoChar(&token, *pos);
				}
				else
					appendStringInfoChar(&token, *pos);
			}
		}
		pos++;
	}
	/* last token, if not quoted */
	if (token.len > 0)
	{
		if (quote != '\0')
			return NIL;
		l = lappend(l, pstrdup(token.data));
	}
	return l;
}

static bool
plcuda_lookup_helper(List *l, oidvector *arg_types, Oid result_type,
					 Oid *p_func_oid, Size *p_size_value)
{
	const char *plcuda_cmd = linitial(l);
	List	   *names;
	Oid			helper_oid;
	HeapTuple	helper_tup;
	Form_pg_proc helper_form;

	if (list_length(l) == 2)
	{
		/* a constant value, or a function in search path
		 * #plcuda_xxxx [<value> | <function>]
		 */
		const char *ident = lsecond(l);

		if (p_size_value)
		{
			/* check whether ident is const value, or not */
			const char *pos = ident;

			while (isdigit(*pos))
				pos++;
			if (*pos == '\0')
			{
				*p_size_value = atol(ident);
				return true;
			}
		}
		names = list_make1(makeString(ident));
	}
	else if (list_length(l) == 4)
	{
		/* function in a particular schema:
		 * #plcuda_xxxx <schema> . <function>
		 */
		const char *nspname = lsecond(l);
		const char *dot = lthird(l);
		const char *proname = lfourth(l);

		if (strcmp(dot, ".") != 0)
			return false;

		names = list_make2(makeString(nspname),
						   makeString(proname));
	}
	else
		return false;

	helper_oid = LookupFuncName(names,
								arg_types->dim1,
								arg_types->values,
								true);
	if (!OidIsValid(helper_oid))
		return false;

	helper_tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(helper_oid));
	if (!HeapTupleIsValid(helper_tup))
		elog(ERROR, "cache lookup failed for function %u", helper_oid);
	helper_form = (Form_pg_proc) GETSTRUCT(helper_tup);


	ReleaseSysCache(helper_tup);

	return true;
}

/*
 * MEMO: structure of pl/cuda function source
 *
 * #plcuda_declare (optional)
 *     :
 * code to be located prior to the function itself
 *     :
 * #plcuda_begin
 *     :
 * function body
 *     :
 * #plcuda_end
 *
 * (additional options)
 * #plcuda_include
 * #plcuda_num_threads {<value>|<function>}
 * #plcuda_shmem_size {<value>|<function>}
 * #plcuda_buffer_size {<value>|<function>}
 * #plcuda_cpu_fallback {<function>}
 */
static void
plcuda_code_validation(plcuda_func_info *cf_info,
					   Form_pg_proc proc_form, char *proc_source)
{
	char		   *line;
	int				lineno;
	StringInfoData	decl;
	StringInfoData	body;
	StringInfoData	emsg;
	bool			notice_out_of_block = true;
	bool			has_num_threads = false;
	bool			has_shmem_size = false;
	bool			has_buffer_size = false;
	enum {
		PLCUDA_PHASE_INIT,
		PLCUDA_PHASE_DECLARE,	/* meet #plcuda_declare */
		PLCUDA_PHASE_BEGIN,		/* meet #plcuda_begin */
		PLCUDA_PHASE_END,		/* meet #plcuda_end */
	} phase = PLCUDA_PHASE_INIT,

	initStringInfo(&decl);
	initStringInfo(&body);
	initStringInfo(&emsg);

	for (line = strtok(source, "\n"), lineno = 1;
		 line != NULL;
		 line = strtok(NULL, "\n"), lineno++)
	{
		if (strncmp(line, "#plcuda_", 8) == 0)
		{
			List	   *l = plcuda_parse_tokens(line);
			const char *plcuda_cmd;

			if (list_length(l) < 1)
			{
				appendStringInfo(&emsg, "\n%u: pl/cuda parse error:\n",
								 lineno, line);
				continue;
			}
			plcuda_cmd = (const char *)linitial(l);

			if (strcmp(plcuda_cmd, "#plcuda_declare") == 0)
			{
				if (list_length(l) != 1)
					appendStringInfo(&emsg, "\n%u: "
									 "%s took no parameters",
									 lineno, plcuda_cmd);
				else if (phase != PLCUDA_PHASE_INIT)
					appendStringInfo(&emsg, "\n%u: "
									 "%s appeared at wrong location",
									 lineno, plcuda_cmd);
				else
				{
					phase = PLCUDA_PHASE_DECLARE;
					notice_out_of_block = true;		/* reset error status */
				}
			}
			else if (strcmp(plcuda_cmd, "#plcuda_begin") == 0)
			{
				if (list_length(l) != 1)
					appendStringInfo(&emsg, "\n%u: "
									 "%s took no parameters",
									 lineno, plcuda_cmd);
				else if (phase != PLCUDA_PHASE_INIT &&
						 phase != PLCUDA_PHASE_DECLARE)
					appendStringInfo(&emsg, "\n%u: "
									 "%s appeared at wrong location",
									 lineno, plcuda_cmd);
				else
				{
					phase = PLCUDA_PHASE_BEGIN;
					notice_out_of_block = true;		/* reset error status */
				}
			}
			else if (strcmp(plcuda_cmd, "#plcuda_end") == 0)
			{
				if (list_length(l) != 1)
					appendStringInfo(&emsg, "\n%u: "
									 "%s took no parameters",
									 lineno, plcuda_cmd);
				else if (phase != PLCUDA_PHASE_BEGIN)
					appendStringInfo(&emsg, "\n%u: "
									 "%s appeared at wrong location",
									 lineno, plcuda_cmd);
				else
				{
					phase = PLCUDA_PHASE_END;
					notice_out_of_block = true;		/* reset error status */
				}
			}
			else if (strcmp(plcuda_cmd, "#plcuda_include") == 0)
			{
				const char *target;

				if (list_length(l) != 2)
					appendStringInfo(&emsg, "\n%u: "
									 "%s wrong syntax",
									 lineno, plcuda_cmd);
				target = lsecond(l);
				if (strcmp(target, "cuda_dynpara.h") == 0)
					cf_info->extra_flags |= DEVKERNEL_NEEDS_DYNPARA;
				else if (strcmp(target, "cuda_matrix.h") == 0)
					cf_info->extra_flags |= DEVKERNEL_NEEDS_MATRIX;
				else if (strcmp(target, "cuda_timelib.h") == 0)
					cf_info->extra_flags |= DEVKERNEL_NEEDS_TIMELIB;
				else if (strcmp(target, "cuda_textlib.h") == 0)
					cf_info->extra_flags |= DEVKERNEL_NEEDS_TEXTLIB;
				else if (strcmp(target, "cuda_numeric.h") == 0)
					cf_info->extra_flags |= DEVKERNEL_NEEDS_NUMERIC;
				else if (strcmp(target, "cuda_mathlib.h") == 0)
					cf_info->extra_flags |= DEVKERNEL_NEEDS_MATHLIB;
				else if (strcmp(target, "cuda_money.h") == 0)
					cf_info->extra_flags |= DEVKERNEL_NEEDS_MONEY;
				else
					appendStringInfo(&emsg, "\n%u: "
									 "%s unknown include target: %s",
									 lineno, plcuda_cmd, target);
			}
			else if (strcmp(plcuda_cmd, "#plcuda_num_threads") == 0)
			{
				if (has_num_threads)
					appendStringInfo(&emsg, "\n%u: %s appeared twice",
									 lineno, plcuda_cmd);
				else
				{
					has_num_threads = true;
					Assert(procForm->pronargs == procForm->proargtypes.dim1);
					if (!plcuda_lookup_helper(l, &procForm->proargtypes,
											  INT4OID,
											  &cf_info->func_num_threads,
											  &cf_info->value_num_threads))
					{
						appendStringInfo(&emsg, "\n%u: "
										 "%s took invalid identifier: ",
										 lineno, plcuda_cmd);
						for (lc = lnext(list_head(l));
							 lc != NULL;
							 lc = lnext(lc))
							appendStringInfo(&emsg, "%s",
											 quote_identifier(lfirst(lc)));
					}
				}
			}
			else if (strcmp(plcuda_cmd, "#plcuda_shmem_size") == 0)
			{
				if (has_shmem_size)
					appendStringInfo(&emsg, "\n%u: %s appeared twice",
									 lineno, plcuda_cmd);
				else
				{
					has_shmem_size = true;
					Assert(procForm->pronargs == procForm->proargtypes.dim1);
					if (!plcuda_lookup_helper(l, &procForm->proargtypes,
											  INT4OID,
											  &cf_info->func_shmem_size,
											  &cf_info->value_shmem_size))
					{
						appendStringInfo(&emsg, "\n%u: "
										 "%s took invalid identifier: ",
										 lineno, plcuda_cmd);
						for (lc = lnext(list_head(l));
							 lc != NULL;
							 lc = lnext(lc))
							appendStringInfo(&emsg, "%s",
											 quote_identifier(lfirst(lc)));
					}
				}
			}
			else if (strcmp(plcuda_cmd, "#plcuda_buffer_size") == 0)
			{
				if (has_buffer_size)
					appendStringInfo(&emsg, "\n%u: %s appeared twice",
									 lineno, plcuda_cmd);
				else
				{
					has_buffer_size = true;
					if (!plcuda_lookup_helper(l, &procForm->proargtypes,
											  INT4OID,
											  &cf_info->func_buffer_size,
											  &cf_info->value_shmem_size))
					{
						appendStringInfo(&emsg, "\n%u: "
										 "%s took invalid identifier: ",
										 lineno, plcuda_cmd);
						for (lc = lnext(list_head(l));
							 lc != NULL;
							 lc = lnext(lc))
							appendStringInfo(&emsg, "%s",
											 quote_identifier(lfirst(lc)));
					}
				}
			}
			else if (strcmp(plcuda_cmd, "#plcuda_cpu_fallback") == 0)
			{
				if (has_cpu_fallback)
					appendStringInfo(&emsg, "\n%u: %s appeared twice",
                                     lineno, plcuda_cmd);
				else
				{
					has_cpu_fallback = true;
					if (!plcuda_lookup_helper(l, &procForm->proargtypes,
											  procForm->prorettype,
											  &cf_info->func_cpu_fallback,
											  NULL))
					{
						appendStringInfo(&emsg, "\n%u: "
										 "%s took invalid identifier: ",
										 lineno, plcuda_cmd);
						for (lc = lnext(list_head(l));
							 lc != NULL;
							 lc = lnext(lc))
							appendStringInfo(&emsg, "%s",
											 quote_identifier(lfirst(lc)));
					}
				}
			}
			else
				appendStringInfo(&emsg, "\n%u: unknown #plcuda_* command: %s",
								 lineno, plcuda_cmd);
		}
		else
		{
			/* not a line of PL/CUDA command */
			if (phase == PLCUDA_PHASE_DECLARE_BLOCK)
				appendStringInfo(&decl, "%s\n", line);
			else if (phase == PLCUDA_PHASE_BODY_BLOCK)
				appendStringInfo(&body, "%s\n", line);
			else if (notice_out_of_block)
			{
				appendStringInfo(&emsg, "\n%u: "
								 "code is out of the valid block:\n%s",
								 lineno, line);
				notice_out_of_block = false;
			}
		}
	}

	if (emsg.len > 0)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("pl/cuda function syntax error\n%s", emsg.data)));

	appendStringInfo(&decl, "\n%s", body.data);
	cf_info->kern_source = decl.data;
	pfree(body.data);
	pfree(emsg.data);

	cf_info->kfunc_decl = decl.data;
	cf_info->kfunc_body = decl.data;
}

static void
plcuda_setup_cuda_program(GpuContext *gcontext, plcuda_func_info *cf_info)
{
	gcontext = pgstrom_get_gpucontext();
	cuda_modules = plcuda_load_cuda_program(gcontext,
											cf_info.kern_source,
											cf_info.extra_flags);
	plcuda_code_trybuild(gcontext, &cf_info);
	pgstrom_put_gpucontext(gcontext);
	
	/*
	 * Try to compile the pl/cuda code
	 */




	//
	// need to revise __pgstrom_load_cuda_program definition
	//
	// cuda_module has to be released on release of GPU context
	//
}


Datum
plcuda_function_validator(PG_FUNCTION_ARGS)
{
	Oid				func_oid = PG_GETARG_OID(0);
	Relation		rel;
	TupleDesc		tupdesc;
	HeapTuple		tuple;
	Form_pg_proc	procForm;
	bool			isnull[Natts_pg_proc];
	Datum			values[Natts_pg_proc];
	plcuda_func_info cf_info;
	devtype_info   *dtype;
	const char	   *proc_source;
	cl_uint			extra_flags = 0;
	cl_uint			i;

	rel = heap_open(ProcedureRelationId, RowExclusiveLock);
	tupdesc = RelationGetDescr(rel);

	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	procForm = (Form_pg_proc) GETSTRUCT(tuple);

	/*
	 * Sanity check of PL/CUDA functions
	 */
	if (procForm->proisagg)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Unable to use PL/CUDA for aggregate functions")));
	if (procForm->proiswindow)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Unable to use PL/CUDA for window functions")));
	if (procForm->proretset)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Righ now, PL/CUDA function does not support set returning function")));

	dtype = pgstrom_devtype_lookup(procForm->prorettype);
	if (!dtype)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Result type \"%s\" is not device executable",
						format_type_be(procForm->prorettype))));
	extra_flags |= dtype->type_flags;

	for (i=0; i < proForm->pronargs; i++)
	{
		Oid		argtype_oid = proForm->proargtypes.values[i];

		dtype = pgstrom_devtype_lookup(argtype_oid);
		if (!dtype)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("Argument type \"%s\" is not device executable",
							format_type_be(argtype_oid))));
		extra_flags |= dtype->type_flags;
	}

	heap_deform_tuple(tuple, RelationGetDescr(rel), values, isnull);
	if (isnull[Anum_pg_proc_prosrc - 1])
		elog(ERROR, "Bug? no program source was supplied");
	if (!isnull[Anum_pg_proc_probin - 1])
		elog(ERROR, "Bug? pg_proc.probin has non-NULL preset value");

	/*
	 * Do syntax checks and construction of plcuda_info
	 */
	memset(&cf_info, 0, sizeof(plcuda_func_info));
	cf_info->extra_flags = extra_flags;

	proc_source = TextDatumGetCString(values[Anum_pg_proc_prosrc - 1]);
	plcuda_code_validation(&cf_info, procForm, proc_source);
	


	plcuda_code_compile(&cf_info);

	/*
	 * OK, supplied function is compilable. Update the catalog.
	 */
	isnull[Anum_pg_proc_probin - 1] = false;
	values[Anum_pg_proc_probin - 1] =
		PointerGetDatum(form_plcuda_func_info(&cf_info));

	newtup = heap_form_tuple(RelationGetDescr(rel), values, isnull);
	simple_heap_update(rel, &tuple->t_self, tuple);

	CatalogUpdateIndexes(rel, tuple);

	/*
	 * Add dependency for hint routines
	 */
	myself.classId = ProcedureRelationId;
	myself.objectId = func_oid;
	myself.objectSubId = 0;

	/* dependency to num_threads function */
	if (OidIsValid(plcuda_info.func_num_threads))
	{
		referenced.classId = ProcedureRelationId;
		referenced.objectId = plcuda_info.func_num_threads;
		referenced.objectSubId = 0;
		recordDependencyOn(&myself, &referenced, DEPENDENCY_NORMAL);
	}

	/* dependency to shmem_size function */
	if (OidIsValid(plcuda_info.func_shmem_size))
	{
		referenced.classId = ProcedureRelationId;
		referenced.objectId = plcuda_info.func_shmem_size;
		referenced.objectSubId = 0;
		recordDependencyOn(&myself, &referenced, DEPENDENCY_NORMAL);
	}

	/* dependency to buffer_size function */
	if (OidIsValid(plcuda_info.func_buffer_size))
	{
		referenced.classId = ProcedureRelationId;
		referenced.objectId = plcuda_info.func_buffer_size;
		referenced.objectSubId = 0;
		recordDependencyOn(&myself, &referenced, DEPENDENCY_NORMAL);
	}





	ReleaseSysCache(tuple);
	heap_close(rel, RowExclusiveLock);

	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(plcuda_function_validator);





Datum
plcuda_function_handler(PG_FUNCTION_ARGS)
{
	// construct kernel code

	// setup param buffer

	elog(ERROR, "not implemented yet");

	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(plcuda_function_handler);
