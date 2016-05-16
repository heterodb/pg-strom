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
	cl_uint		extra_flags;
	/* kernel function for decl */
	const char *kern_decl;
	/* kernel function for prep */
	const char *kern_prep;
	bool		kern_prep_max_threads;
	Oid			fn_prep_num_threads;
	Size		val_prep_num_threads;
	Oid			fn_prep_shmem_size;
	Size		varl_prep_shmem_size;

	/* kernel function for body */
	const char *kern_body;
	bool		kern_body_max_threads;
	Oid			fn_body_num_threads;
	Size		val_body_num_threads;
	Oid			fn_body_shmem_size;
	Size		varl_body_shmem_size;

	/* kernel function for post */
	const char *kern_post;
	bool		kern_post_max_threads;
	Oid			fn_post_num_threads;
	Size		val_post_num_threads;
	Oid			fn_post_shmem_size;
	Size		val_post_shmem_size;

	/* device memory size for working buffer */
	Oid			fn_working_bufsz;
	Size		val_working_bufsz;

	/* device memory size for result buffer */
	Oid			fn_results_bufsz;
	Size		val_results_bufsz;
} plcuda_func_info;

/*
 * XXX - to be revised to use ExtensibleNode in 9.6 based implementation
 */
static text *
form_plcuda_func_info(plcuda_func_info *cf_info)
{
	List   *l = NIL;

	l = lappend(l, makeInteger(cf_info->extra_flags));
	/* declarations */
	l = lappend(l, makeString(cf_info->kern_decl));
	/* prep kernel */
	l = lappend(l, makeString(cf_info->kern_prep));
	l = lappend(l, makeInteger(cf_info->kern_prep_max_threads));
	l = lappend(l, makeInteger(cf_info->fn_prep_num_threads));
	l = lappend(l, makeInteger(cf_info->val_prep_num_threads));
	l = lappend(l, makeInteger(cf_info->fn_prep_shmem_size));
	l = lappend(l, makeInteger(cf_info->val_prep_shmem_size));
	/* body kernel */
	l = lappend(l, makeString(cf_info->kern_body));
	l = lappend(l, makeInteger(cf_info->kern_body_max_threads));
	l = lappend(l, makeInteger(cf_info->fn_body_num_threads));
	l = lappend(l, makeInteger(cf_info->val_body_num_threads));
	l = lappend(l, makeInteger(cf_info->fn_body_shmem_size));
	l = lappend(l, makeInteger(cf_info->val_body_shmem_size));
	/* post kernel */
	l = lappend(l, makeString(cf_info->kern_post));
	l = lappend(l, makeInteger(cf_info->kern_post_max_threads));
	l = lappend(l, makeInteger(cf_info->fn_post_num_threads));
	l = lappend(l, makeInteger(cf_info->val_post_num_threads));
	l = lappend(l, makeInteger(cf_info->fn_post_shmem_size));
	l = lappend(l, makeInteger(cf_info->val_post_shmem_size));
	/* working buffer */
	l = lappend(l, makeInteger(cf_info->fn_working_bufsz));
	l = lappend(l, makeInteger(cf_info->val_working_bufsz));
	/* results buffer */
	l = lappend(l, makeInteger(cf_info->fn_results_bufsz));
	l = lappend(l, makeInteger(cf_info->val_results_bufsz));

	return cstring_to_text(nodeToString(l));
}

static plcuda_func_info *
deform_plcuda_func_info(text *cf_info_text)
{
	plcuda_func_info *cf_info = palloc0(sizeof(plcuda_func_info));
	List	   *l = stringToNode(VARDATA(cf_info_text));
	cl_uint		index = 0;

	cf_info->extra_flags = intVal(list_nth(l, index++));
	/* declarations */
	cf_info->kern_decl = strVal(list_nth(l, index++));
	/* prep kernel */
	cf_info->kern_prep = strVal(list_nth(l, index++));
	cf_info->fn_prep_num_threads = intVal(list_nth(l, index++));
	cf_info->val_prep_num_threads = intVal(list_nth(l, index++));
	cf_info->fn_prep_shmem_size = intVal(list_nth(l, index++));
	cf_info->val_prep_shmem_size = intVal(list_nth(l, index++));
	/* body kernel */
	cf_info->kern_body = strVal(list_nth(l, index++));
	cf_info->fn_body_num_threads = intVal(list_nth(l, index++));
	cf_info->val_body_num_threads = intVal(list_nth(l, index++));
	cf_info->fn_body_shmem_size = intVal(list_nth(l, index++));
	cf_info->val_body_shmem_size = intVal(list_nth(l, index++));
	/* post kernel */
	cf_info->kern_post = strVal(list_nth(l, index++));
	cf_info->fn_post_num_threads = intVal(list_nth(l, index++));
	cf_info->val_post_num_threads = intVal(list_nth(l, index++));
	cf_info->fn_post_shmem_size = intVal(list_nth(l, index++));
	cf_info->val_post_shmem_size = intVal(list_nth(l, index++));
	/* working buffer */
	cf_info->fn_working_bufsz = intVal(list_nth(l, index++));
	cf_info->val_working_bufsz = intVal(list_nth(l, index++));
	/* results buffer */
	cf_info->fn_results_bufsz = intVal(list_nth(l, index++));
	cf_info->val_results_bufsz = intVal(list_nth(l, index++));

	return cf_info;
}



/*
 * structure to communicate between host <--> device
 *
 * kern_parambuf is used to excange arguments
 *
 * KPARAM_0       ... buffer for the result
 * KPARAM_X (X>0) ... introduce 
 */
typedef struct
{
	kern_errorbuf	kerror;
	cl_uint			num_threads;
	cl_uint			shmem_size;
	cl_uint			result_size;
	cl_uint			buffer_size;
	kern_parambuf	kparams;
} kern_plcuda;


/*
 * plcuda_parse_cmdline
 *
 * It parse the line of '#plcuda_xxx'
 */
static bool
plcuda_parse_cmd_options(const char *linebuf, List *p_options)
{
	List	   *l = NIL;
	char	   *pos = linebuf;
	char		quote = '\0';
	List	   *options = NIL;
	StringInfoData token;

	initStringInfo(&token);
	while (*pos != '\0')
	{
		if (*pos == '\\')
		{
			if (*++pos == '\0')
				return false;
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
		else if (*pos == ',')
		{
			if (token.len > 0)
			{
				l = lappend(l, pstrdup(token.data));
				resetStringInfo(&token);
			}
			if (l == NIL)
				return false;
			options = lappend(options, l);
			l = NIL;
		}
		else if (*pos == '.')
		{
			if (token.len > 0)
			{
				l = lappend(l, pstrdup(token.data));
				resetStringInfo(&token);
			}
			/* single character token for delimiter */
			l = lappend(l, pnstrdup(pos, 1));
		}
		else if (*pos == '"' || *pos == '\'')
		{
			if (token.len > 0)
			{
				l = lappend(l, pstrdup(token.data));
				resetStringInfo(&token);
			}
			quote = *pos;
		}
		else if (token.len > 0)
		{
			if (isspace(*pos))
			{
				l = lappend(l, pstrdup(token.data));
				resetStringInfo(&token);
			}
			else
				appendStringInfoChar(&token, tolower(*pos));
		}
		else
		{
			if (!isspace(*pos))
				appendStringInfoChar(&token, tolower(*pos));
		}
	}

	if (quote != '\0')
		return false;		/* syntax error; EOL inside quote */
	if (token.len > 0)
		l = lappend(l, pstrdup(token.data));
	if (l != NIL)
		options = lappend(options, l);
	else if (options != NIL)
		return false;		/* syntax error; EOL by ',' */

	*p_options = options;
	return plcuda_cmd;
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

static inline char *
ident_to_cstring(List *ident)
{
	StringInfoData	buf;
	ListCell	   *lc;

	initStringInfo(&buf);
	foreach (lc, ident)
		appendStringInfo(&buf, " %s", quote_identifier(lfirst(lc)));
	return buf.data;
}

/*
 * MEMO: structure of pl/cuda function source
 *
 * #plcuda_decl (optional)
 *      :  any declaration code
 * #plcuda_end 
 *
 * #plcuda_prep  [<num_threads>[, <shmem_size>[, max_threads]]]
 *      :  initial setup of working/result buffer
 * #plcuda_end
 *
 * #plcuda_body  [<num_threads>[,<shmem_size>[, max_threads]]]
 *      :  main logic of pl/cuda function
 * #plcuda_end
 *
 * #plcuda_final [<num_threads>[,<shmem_size>[, max_threads]]]
 *      :  final setup of result buffer
 * #plcuda_end
 *
 * (additional options)
 * #plcuda_include "cuda_xxx.h"
 * #plcuda_results_size {<value>|<function>}     (default: 0)
 * #plcuda_buffer_size {<value>|<function>}      (default: 0)
 * #plcuda_cpu_fallback {<function>}             (default: no fallback)
 */
static void
plcuda_code_validation(plcuda_func_info *cf_info,
					   Form_pg_proc proc_form, char *proc_source)
{
	StringInfoData	decl;
	StringInfoData	prep;
	StringInfoData	body;
	StringInfoData	post;
	StringInfoData	emsg;
	StringInfo		curr = NULL;
	int			nargs = proc_form->pronargs;
	oidvector  *argtypes = &proc_form->proargtypes;
	char	   *line;
	int			lineno;
	bool		has_decl_block = false;
	bool		has_prep_block = false;
	bool		has_body_block = false;
	bool		has_post_block = false;
	bool		has_working_bufsz = false;
	bool		has_results_bufsz = false;

	initStringInfo(&decl);
	initStringInfo(&prep);
	initStringInfo(&body);
	initStringInfo(&post);
	initStringInfo(&emsg);

	for (line = strtok(source, "\n"), lineno = 1;
		 line != NULL;
		 line = strtok(NULL, "\n"), lineno++)
	{
		const char *cmd;
		const char *pos;
		List	   *l;

		/* put a non pl/cuda command line*/
		if (strncmp(line, "#plcuda_", 8) != 0)
		{
			if (curr != NULL)
				appendStringInfo(curr, "%s\n", line);
			else
				appendStringInfo(&emsg, "\n%u: "
								 "code is out of valid block:\n%s",
								 lineno, line);
			continue;
		}
		/* pick up command name */
		for (pos = line; !isspace(*pos) && *pos != '\0'; pos++);
		cmd = pnstrdup(line, pos - line);
		/* parse pl/cuda command options */
		if (!plcuda_parse_cmd_options(pos, &options))
		{
			appendStringInfo(&emsg, "\n%u: pl/cuda parse error:\n",
							 lineno, line);
			continue;
		}

		if (strcmp(cmd, "#plcuda_decl") == 0)
		{
			if (has_decl_block)
			{
				appendStringInfo(&emsg, "\n%u: %s appeared twice",
								 lineno, line);
				continue;
			}

			if (list_length(options) > 0)
			{
				appendStringInfo(&emsg, "\n%u: %s takes no parameters",
								 lineno, line);
				continue;
			}
			curr = &decl;
			has_decl_block = true;
		}
		else if (strcmp(cmd, "#plcuda_prep") == 0)
		{
			if (has_prep_block)
			{
				appendStringInfo(&emsg, "\n%u: %s appeared twice",
								 lineno, cmd);
				continue;
			}

			switch (list_length(options))
			{
				case 3:
					ident = lthird(options);
					if (list_length(ident) != 1)
						appendStringInfo(&emsg, "\n%u:%s was not valid",
										 lineno, linitial(ident));
					else if (strcmp(linitial(ident), "max_threads") == 0)
						cf_info->kern_prep_max_threads = true;
					else
						appendStringInfo(&emsg, "\n%u:%s was unknown",
										 lineno, linitial(ident));
				case 2:
					if (!plcuda_lookup_helper(lsecond(options),
											  nargs, argtypes,
											  &cf_info->fn_prep_shmem_size,
											  &cf_info->val_prep_shmem_size))
						appendStringInfo(&emsg, "\n%u:%s was not valid",
								lineno, ident_to_cstring(lsecond(options)));
				case 1:
					if (!plcuda_lookup_helper(linitial(options),
											  nargs, argtypes,
											  &cf_info->fn_prep_num_threads,
											  &cf_info->val_prep_num_threads))
						appendStringInfo(&emsg, "\n%u:%s was not valid",
								lineno, ident_to_cstring(linitial(options)));
				case 0:
					break;
				default:
					appendStringInfo(&emsg, "\n%u: %s had too much parameters",
									 lineno, cmd);
					break;
			}
		}
		else if (strcmp(cmd, "#plcuda_body") == 0)
		{
			if (has_prep_block)
			{
				appendStringInfo(&emsg, "\n%u: %s appeared twice",
								 lineno, cmd);
				continue;
			}

			switch (list_length(options))
			{
				case 3:
					ident = lthird(options);
					if (list_length(ident) != 1)
						appendStringInfo(&emsg, "\n%u:%s was not valid",
										 lineno, linitial(ident));
					else if (strcmp(linitial(ident), "max_threads") == 0)
						cf_info->kern_body_max_threads = true;
					else
						appendStringInfo(&emsg, "\n%u:%s was unknown",
										 lineno, linitial(ident));
				case 2:
					if (!plcuda_lookup_helper(lsecond(options),
											  nargs, argtypes,
											  &cf_info->fn_body_shmem_size,
											  &cf_info->val_body_shmem_size))
						appendStringInfo(&emsg, "\n%u:%s was not valid",
								lineno, ident_to_cstring(lsecond(options)));
				case 1:
					if (!plcuda_lookup_helper(linitial(options),
											  nargs, argtypes,
											  &cf_info->fn_body_num_threads,
											  &cf_info->val_body_num_threads))
						appendStringInfo(&emsg, "\n%u:%s was not valid",
								lineno, ident_to_cstring(linitial(options)));
				case 0:
					break;
				default:
					appendStringInfo(&emsg, "\n%u: %s had too much parameters",
									 lineno, cmd);
					break;
			}
		}
		else if (strcmp(cmd, "#plcuda_post") == 0)
		{
			if (has_prep_block)
			{
				appendStringInfo(&emsg, "\n%u: %s appeared twice",
								 lineno, cmd);
				continue;
			}

			switch (list_length(options))
			{
				case 3:
					ident = lthird(options);
					if (list_length(ident) != 1)
						appendStringInfo(&emsg, "\n%u:%s was not valid",
										 lineno, linitial(ident));
					else if (strcmp(linitial(ident), "max_threads") == 0)
						cf_info->kern_post_max_threads = true;
					else
						appendStringInfo(&emsg, "\n%u:%s was unknown",
										 lineno, linitial(ident));
				case 2:
					if (!plcuda_lookup_helper(lsecond(options),
											  nargs, argtypes,
											  &cf_info->fn_post_shmem_size,
											  &cf_info->val_post_shmem_size))
						appendStringInfo(&emsg, "\n%u:%s was not valid",
								lineno, ident_to_cstring(lsecond(options)));
				case 1:
					if (!plcuda_lookup_helper(linitial(options),
											  nargs, argtypes,
											  &cf_info->fn_post_num_threads,
											  &cf_info->val_post_num_threads))
						appendStringInfo(&emsg, "\n%u:%s was not valid",
								lineno, ident_to_cstring(linitial(options)));
				case 0:
					break;
				default:
					appendStringInfo(&emsg, "\n%u: %s had too much parameters",
									 lineno, cmd);
					break;
			}
		}
		else if (strcmp(cmd, "#plcuda_end") == 0)
		{
			if (list_length(options) > 0)
				appendStringInfo(&emsg, "\n%u: %s takes no parameters",
								 lineno, line);
			curr = NULL;
		}
		else if (strcmp(cmd, "#plcuda_working_bufsz") == 0)
		{
			if (has_working_bufsz)
				appendStringInfo(&emsg, "\n%u: %s appears twice",
								 lineno, cmd);
			else if (list_length(options) != 1)
				appendStringInfo(&emsg, "\n%u: %s wrong syntax",
								 lineno, cmd);
			else if (plcuda_lookup_helper(linitial(options),
										  nargs, argtypes,
										  &cf_info->fn_working_bufsz,
										  &cf_info->val_working_bufsz))
				has_working_bufsz = true;
			else
				appendStringInfo(&emsg, "\n%u:%s was not valid",
								 lineno, ident_to_cstring(linitial(options)));
		}
		else if (strcmp(cmd, "#plcuda_results_size") == 0)
		{
			if (has_results_bufsz)
				appendStringInfo(&emsg, "\n%u: %s appears twice",
								 lineno, cmd);
			else if (list_length(options) != 1)
				appendStringInfo(&emsg, "\n%u: %s wrong syntax",
								 lineno, cmd);
			else if (plcuda_lookup_helper(linitial(options),
										  nargs, argtypes,
										  &cf_info->fn_results_bufsz,
										  &cf_info->val_results_bufsz))
				has_results_bufsz = true;
			else
				appendStringInfo(&emsg, "\n%u:%s was not valid",
								 lineno, ident_to_cstring(linitial(options)));
		}
		else if (strcmp(cmd, "#plcuda_include") == 0)
		{
			const char *target;

			if (list_length(options) != 1)
				appendStringInfo(&emsg, "\n%u: %s wrong syntax", lineno, cmd);

			target = linitial(l);
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
				appendStringInfo(&emsg, "\n%u: %s unknown include target: %s",
								 lineno, plcuda_cmd, target);
		}
	}

	if (emsg.len > 0)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("pl/cuda function syntax error\n%s", emsg.data)));

	if (has_decl_block)
		cf_info->kern_decl = decl.data;
	if (has_prep_block)
		cf_info->kern_prep = prep.data;
	if (has_body_block)
		cf_info->kern_body = body.data;
	if (has_post_block)
		cf_info->kern_post = body.data;

	pfree(emsg.data);
}

static void
plcuda_setup_cuda_program(GpuContext *gcontext,
						  Form_pg_proc procForm,
						  plcuda_func_info *cf_info)
{
	devtype_info   *dtype_r;
	StringInfoData	kern;


	initStringInfo(&kern);
	/* type definition of kern_plcuda */
	appendStringInfoString(
		&kern,
		"/* structure to communicat between host <--> device */\n"
		"typedef struct {\n"
		"  kern_errorbuf      kerror;\n"
		"  kern_parambuf      kparams;\n"
		"};\n\n");

	/* declaration part */
	appendStringInfoString(&kern, cf_info->kern_decl);

	/* pl/cuda function body */
	dtype_r = pgstrom_devtype_lookup(procForm->prorettype);
	if (!dtype_r)
		elog(ERROR, "cache lookup failed for device type: %s",
			 format_type_be(procForm->prorettype));

	appendStringInfo(&kern,
					 "STATIC_FUNCTION(pg_%s_t)\n"
					 "pgfn_%s_%s(kern_context *kcxt",
					 dtype_r->type_name,
					 get_namespace_name(procForm->pronamespace),
					 NameStr(procForm->proname));

	for (i=0; i < procForm->pronargs; i++)
	{
		Oid		type_oid = procForm->proargtypes.values[i];
		devtype_info *dtype;

		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "cache lookup failed for device type: %s",
				 format_type_be(type_oid));
		appendStringInfo(&kern, ", pg_%s_t arg%d",
						 dtype->type_name, i+1);
	}
	appendStringInfo(&kern, ")\n{\n%s\n}\n\n", cf_info->kern_body);

	/* kernel entrypoint */
	appendStringInfo(
		&kern,
		"KERNEL_FUNCTION%s(void)\n"
		"plcuda_entrypoint(kern_plcuda *kplcuda,\n"
		"                  char *buffer)\n"
		"{\n"
		"    "








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
	cf_info->value_num_threads = 1;		/* default */

	proc_source = TextDatumGetCString(values[Anum_pg_proc_prosrc - 1]);
	plcuda_code_validation(&cf_info, procForm, proc_source);

	gcontext = get_gpucontext();
	plcuda_setup_cuda_program(gcontext, procForm, &cf_info);
	put_gpucontext(gcontext);

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

	if (OidIsValid(plcuda_info.func_results_size))
	{
		referenced.classId = ProcedureRelationId;
		referenced.objectId = plcuda_info.func_results_size;
		referenced.objectSubId = 0;
		recordDependencyOn(&myself, &referenced, DEPENDENCY_NORMAL);
	}

	if (OidIsValid(plcuda_info.func_cpu_fallback))
	{
		referenced.classId = ProcedureRelationId;
		referenced.objectId = plcuda_info.func_cpu_fallback;
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
