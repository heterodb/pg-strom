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
	/* kernel function for preparation */
	/* kernel function for main logic */
	/* kernel function for exit */


	/* number of the threads to kick */
	Oid				func_num_threads;
	Size			value_num_threads;
	/* amount of shmem to allocate */
	Oid				func_shmem_size;
	Size			value_shmem_size;
	/* number of the threads for preparation kernel */
	Oid				func_num_prep_threads;
	Size			value_num_prep_threads;
	/* amount of shmem for preparation kernel */
	Oid				func_prep_shmem_size;
	Size			value_prep_shmem_size;
	/* amount of device memory for variable length result */
	Oid				func_results_size;
	Size			value_results_size;
	/* amount of device memory for working buffer */
	Oid				func_buffer_size;
	Size			value_buffer_size;
	/* emergency fallback if GPU returned CpuReCheck error */
	Oid				func_cpu_fallback;
	/* kernel's attribute */
	bool			kernel_max_threads;
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
	l = lappend(l, makeInteger(cf_info->func_num_prep_threads));
	l = lappend(l, makeInteger(cf_info->value_num_prep_threads));
	l = lappend(l, makeInteger(cf_info->func_prep_shmem_size));
	l = lappend(l, makeInteger(cf_info->value_prep_shmem_size));
	l = lappend(l, makeInteger(cf_info->func_buffer_size));
	l = lappend(l, makeInteger(cf_info->value_buffer_size));
	l = lappend(l, makeInteger(cf_info->func_cpu_fallback));
	l = lappend(l, makeInteger(cf_info->kernel_max_threads));
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
	cf_info->func_num_prep_threads = intVal(list_nth(l, index++));
	cf_info->value_num_prep_threads = intVal(list_nth(l, index++));
	cf_info->func_prep_shmem_size = intVal(list_nth(l, index++));
	cf_info->value_prep_shmem_size = intVal(list_nth(l, index++));
	cf_info->func_buffer_size = intVal(list_nth(l, index++));
	cf_info->value_buffer_size = intVal(list_nth(l, index++));
	cf_info->func_cpu_fallback = intVal(list_nth(l, index++));
	cf_info->kernel_max_threads = intVal(list_nth(l, index++));
	cf_info->extra_flags = intVal(list_nth(l, index++));
	cf_info->kern_decl = strVal(list_nth(l, index++));
	cf_info->kern_body = strVal(list_nth(l, index++));

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

static void
plcuda_parse_blockcmd(ListCell *options,
					  Oid *p_func_num_threads, Size *p_const_num_threads,
					  Oid *p_func_shmem_size, Size *p_const_shmem_size,
					  bool *p_max_threads_attr)
{
	Oid			func_num_threads = InvalidOid;
	Size		const_num_threads = 1;	/* default */
	Oid			func_shmem_size = InvalidOid;
	Size		const_shmem_size = 0;
	bool		max_threads_attr = false;
	ListCell   *lc = options;

	while (lc)
	{
		const char *token = lfirst(lc);

		



		
		
	}



	if (list_length(options) < 1)
		goto out;






	/* num_threads hint */
	if (list_length(options) == 1)
	{
		
		
	}
	else
	{



	}







out:
	*p_func_num_threads = func_num_threads;
	*p_const_num_threads = const_num_threads;
	*p_func_shmem_size = func_shmem_size;
	*p_const_shmem_size = const_shmem_size;
	*p_max_threads_attr = p_max_threads_attr;

	return false;
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
	StringInfoData	finish;
	StringInfoData	emsg;
	StringInfo		curr = NULL;
	char   *line;
	int		lineno;
	bool			notice_out_of_block = true;
	bool			kernel_maxthreads = false;
	bool			has_num_threads = false;
	bool			has_shmem_size = false;
	bool			has_buffer_size = false;
	bool			has_results_size = false;
	enum {
		PLCUDA_PHASE_INIT,
		PLCUDA_PHASE_DECL,		/* meet #plcuda_decl */
		PLCUDA_PHASE_PREP,		/* meet #plcuda_prep */
		PLCUDA_PHASE_BODY,		/* meet #plcuda_body */
		PLCUDA_PHASE_FINISH,	/* meet #plcuda_finish */
	} phase = PLCUDA_PHASE_INIT,

	initStringInfo(&decl);
	initStringInfo(&prep);
	initStringInfo(&body);
	initStringInfo(&finish);
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

		}
		else if (strcmp(cmd, "#plcuda_prep") == 0)
		{}
		else if (strcmp(cmd, "#plcuda_body") == 0)
		{}
		else if (strcmp(cmd, "#plcuda_post") == 0)
		{}
		else if (strcmp(cmd, "#plcuda_end") == 0)
		{}
		else if (strcmp(cmd, "#plcuda_buffer_size") == 0)
		{}
		else if (strcmp(cmd, "#plcuda_results_size") == 0)
		{}
		else if (strcmp(cmd, "#plcuda_include") == 0)
		{}



		if (strcmp(plcuda_cmd, "#plcuda_decl") == 0)
		{
			if (list_length(l) != 1)
			{
				appendStringInfo(&emsg, "\n%u: %s takes no parameters",
								 lineno, cmd);
				continue;
			}

			if (has_decl_block)
			{
				appendStringInfo(&emsg, "\n%u: %s appeared twice or more",
								 lineno, cmd);
				continue;
			}
			curr = &decl;
			has_decl_block = true;
		}
		else if (strcmp(plcuda_cmd, "#plcuda_prep") == 0)
		{
			if (has_prep_block)
			{
				appendStringInfo(&emsg, "\n%u: %s appeared twice or more",
								 lineno, cmd);
				continue;
			}
			

			
		}


 ||
				 strcmp(plcuda_cmd, "#plcuda_main") == 0 ||
				 strcmp(plcuda_cmd, "#plcuda_finish") == 0)
		{
				ListCell   *curr = lnext(list_head(l));

				if (!curr)
				{
					no max threads;
					single threads;
					no shmem;
				}





			{
				if (list_length(l) != 1)
					appendStringInfo(&emsg, "\n%u: "
									 "%s takes no parameters",
									 lineno, plcuda_cmd);
				else if (phase != PLCUDA_PHASE_INIT &&
						 phase != PLCUDA_PHASE_DECLARE)
					appendStringInfo(&emsg, "\n%u: "
									 "%s appeared at wrong location",
									 lineno, plcuda_cmd);
				else
				{
					phase = PLCUDA_PHASE_PREP;
					notice_out_of_block = true;		/* reset error status */
				}
			}
			else if (strcmp(plcuda_cmd, "#plcuda_begin") == 0)
			{
				if (list_length(l) != 1)
					appendStringInfo(&emsg, "\n%u: "
									 "%s takes no parameters",
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
			else if (strcmp(plcuda_cmd, "#plcuda_kernel_attrs") == 0)
			{
				if (list_length(l) < 2)
					appendStringInfo(&emsg, "\n%u: %s has no attributes",
									 lineno, plcuda_cmd);
				else
				{
					for (lc = lnext(list_head(l)); lc != NULL; lc = lnext(lc))
					{
						const char *attr = lfirst(lc);

						if (strcmp(attr, "maxthreads") == 0)
							kernel_maxthreads = true;
						else
						{
							appendStringInfo(&emsg,
											 "\n%u: %s - unknown attribute %s",
											 lineno, plcuda_cmd, attr);
						}
					}
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
			else if (strcmp(plcuda_cmd, "#plcuda_num_prep_threads") == 0)
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
											&cf_info->func_num_prep_threads,
											&cf_info->value_num_prep_threads))
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
			else if (strcmp(plcuda_cmd, "#plcuda_prep_shmem_size") == 0)
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
											  &cf_info->func_prep_shmem_size,
											  &cf_info->value_prep_shmem_size))
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
											  &cf_info->value_buffer_size))
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
			else if (strcmp(plcuda_cmd, "#plcuda_results_size") == 0)
			{
				if (has_results_size)
					appendStringInfo(&emsg, "\n%u: %s appeared twice",
									 lineno, plcuda_cmd);
				else
				{
					has_results_size = true;
					if (!plcuda_lookup_helper(l, &procForm->proargtypes,
											  INT4OID,
											  &cf_info->func_results_size,
											  &cf_info->value_results_size))
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
