/*
 * pl_cuda.c
 *
 * PL/CUDA SQL function support
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#include "access/tuptoaster.h"
#include "access/xact.h"
#include "catalog/dependency.h"
#include "catalog/indexing.h"
#include "catalog/objectaddress.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "parser/parse_func.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "pg_strom.h"
#include "cuda_plcuda.h"

/*
 * plcudaTaskState
 */
typedef struct plcudaTaskState
{
	GpuTaskState_v2	gts;		/* dummy */
	ResourceOwner	owner;
	dlist_node		chain;
	kern_plcuda	   *kplcuda_head;
	void		   *last_results_buf;	/* results buffer last used */
	/* kernel requirement */
	cl_uint		extra_flags;		/* flags to standard includes */
	/* kernel declarations */
	char	   *kern_decl;
	/* kernel prep function */
	char	   *kern_prep;
	Oid			fn_prep_kern_blocksz;
	long		val_prep_kern_blocksz;
	Oid			fn_prep_num_threads;
	Size		val_prep_num_threads;
	Oid			fn_prep_shmem_unitsz;
	Size		val_prep_shmem_unitsz;
	Oid			fn_prep_shmem_blocksz;
	Size		val_prep_shmem_blocksz;
	/* kernel function */
	char	   *kern_main;
	Oid			fn_main_kern_blocksz;
	long		val_main_kern_blocksz;
	Oid			fn_main_num_threads;
	Size		val_main_num_threads;
	Oid			fn_main_shmem_unitsz;
	Size		val_main_shmem_unitsz;
	Oid			fn_main_shmem_blocksz;
	Size		val_main_shmem_blocksz;
	/* kernel post function */
	char	   *kern_post;
	Oid			fn_post_kern_blocksz;
	long		val_post_kern_blocksz;
	Oid			fn_post_num_threads;
	Size		val_post_num_threads;
	Oid			fn_post_shmem_unitsz;
	Size		val_post_shmem_unitsz;
	Oid			fn_post_shmem_blocksz;
	Size		val_post_shmem_blocksz;
	/* device memory size for working buffer */
	Oid			fn_working_bufsz;
	long		val_working_bufsz;
	/* device memory size for result buffer */
	Oid			fn_results_bufsz;
	long		val_results_bufsz;
	/* comprehensive functions */
	Oid			fn_sanity_check;
	Oid			fn_cpu_fallback;
} plcudaTaskState;

/*
 *  plcudaTask
 */
typedef struct plcudaTask
{
	GpuTask_v2		task;
	bool			exec_prep_kernel;
	bool			exec_post_kernel;
	bool			has_cpu_fallback;
	void		   *h_results_buf;	/* results buffer in host-side */
	CUfunction		kern_plcuda_prep;
	CUfunction		kern_plcuda_main;
	CUfunction		kern_plcuda_post;
	CUdeviceptr		m_kern_plcuda;
	CUdeviceptr		m_results_buf;
	CUdeviceptr		m_working_buf;
	CUevent			ev_dma_send_start;
	CUevent			ev_dma_send_stop;
	CUevent			ev_dma_recv_start;
	CUevent			ev_dma_recv_stop;
	/* TODO: performance counter here */
	kern_plcuda		kern;
} plcudaTask;

/*
 * static functions
 */
static plcudaTaskState *plcuda_exec_begin(HeapTuple protup,
										  FunctionCallInfo fcinfo);

static void plcuda_exec_end(plcudaTaskState *plts);

/* tracker of plcudaState */
static dlist_head	plcuda_state_list;

/*
 * plcuda_parse_cmdline
 *
 * It parse the line of '#plcuda_xxx'
 */
static bool
plcuda_parse_cmd_options(const char *linebuf, List **p_options)
{
	const char *pos = linebuf;
	char		quote = '\0';
	List	   *options = NIL;
	StringInfoData token;

	initStringInfo(&token);
	for (pos = linebuf; *pos != '\0'; pos++)
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
				options = lappend(options, pstrdup(token.data));
				resetStringInfo(&token);
				quote = '\0';
			}
			else
				appendStringInfoChar(&token, *pos);
		}
		else if (*pos == '.')
		{
			if (token.len > 0)
			{
				options = lappend(options, pstrdup(token.data));
				resetStringInfo(&token);
			}
			if (options == NIL)
				return false;
		}
		else if (*pos == '"' || *pos == '\'')
		{
			if (token.len > 0)
			{
				options = lappend(options, pstrdup(token.data));
				resetStringInfo(&token);
			}
			quote = *pos;
		}
		else if (token.len > 0)
		{
			if (isspace(*pos))
			{
				options = lappend(options, pstrdup(token.data));
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
		options = lappend(options, pstrdup(token.data));

	*p_options = options;
	return true;
}

static bool
plcuda_lookup_helper(List *options, oidvector *arg_types, Oid result_type,
					 Oid *p_func_oid, long *p_size_value)
{
	List	   *names;

	if (list_length(options) == 1)
	{
		/* a constant value, or a function in search path */
		char   *ident = linitial(options);
		char   *pos = ident;

		if (p_size_value)
		{
			for (pos = ident; isdigit(*pos); pos++);
			if (*pos == '\0')
			{
				if (p_func_oid)
					*p_func_oid = InvalidOid;	/* no function */
				*p_size_value = atol(ident);
                return true;
			}
		}
		names = list_make1(makeString(ident));
	}
	else if (list_length(options) == 3)
	{
		/* function in a particular schema */
		char   *nspname = linitial(options);
		char   *dot = lsecond(options);
		char   *proname = lthird(options);

		if (strcmp(dot, ".") != 0)
			return false;

		names = list_make2(makeString(nspname),
						   makeString(proname));
	}
	else
		return false;

	if (p_func_oid)
	{
		Oid		helper_oid = LookupFuncName(names,
											arg_types->dim1,
											arg_types->values,
											true);
		if (!OidIsValid(helper_oid))
			return false;
		if (result_type != get_func_rettype(helper_oid))
			return false;

		/* OK, helper function is valid */
		*p_func_oid = helper_oid;
		if (p_size_value)
			*p_size_value = 0;	/* no const value */

		return true;
	}
	return false;
}

static inline char *
ident_to_cstring(List *ident)
{
	StringInfoData	buf;
	ListCell	   *lc;

	initStringInfo(&buf);
	foreach (lc, ident)
	{
		if (buf.len > 0)
			appendStringInfoChar(&buf, ' ');
		appendStringInfo(&buf, "%s", quote_identifier(lfirst(lc)));
	}
	return buf.data;
}

/*
 * MEMO: structure of pl/cuda function source
 *
 * #plcuda_decl (optional)
 *      :  any declaration code
 * #plcuda_end 
 *
 * #plcuda_prep (optional)
 * #plcuda_num_threads (value|function)
 * #plcuda_shmem_size  (value|function)
 * #plcuda_kernel_blocksz
 *      :
 * #plcuda_end
 *
 * #plcuda_begin
 * #plcuda_num_threads (value|function)
 * #plcuda_shmem_size  (value|function)
 * #plcuda_kernel_blocksz
 *      :
 * #plcuda_end
 *
 * #plcuda_post (optional)
 * #plcuda_num_threads (value|function)
 * #plcuda_shmem_size  (value|function)
 * #plcuda_kernel_blocksz
 *      :
 * #plcuda_end
 *
 * (additional options)
 * #plcuda_include "cuda_xxx.h"
 * #plcuda_include <function>
 * #plcuda_results_bufsz {<value>|<function>}     (default: 0)
 * #plcuda_working_bufsz {<value>|<function>}      (default: 0)
 * #plcuda_sanity_check {<function>}             (default: no fallback)
 * #plcuda_cpu_fallback {<function>}             (default: no fallback)
 */
static void
plcuda_code_validation(plcudaTaskState *plts,	/* dummy GTS */
					   Oid proowner,
					   oidvector *proargtypes,
					   Oid prorettype,
					   char *source)
{
	StringInfoData	decl_src;
	StringInfoData	prep_src;
	StringInfoData	main_src;
	StringInfoData	post_src;
	StringInfoData	emsg;
	StringInfo		curr = NULL;
	devtype_info   *dtype;
	char		   *line;
	int				i, lineno;
	bool			not_exec_now = !OidIsValid(proowner);
	bool			has_decl_block = false;
	bool			has_prep_block = false;
	bool			has_main_block = false;
	bool			has_post_block = false;
	bool			has_working_bufsz = false;
	bool			has_results_bufsz = false;
	bool			has_sanity_check = false;
	bool			has_cpu_fallback = false;

	initStringInfo(&decl_src);
	initStringInfo(&prep_src);
	initStringInfo(&main_src);
	initStringInfo(&post_src);
	initStringInfo(&emsg);

#define EMSG(fmt,...)		appendStringInfo(&emsg, "\n%u: " fmt,	\
											 lineno, __VA_ARGS__)
#define NOTE(fmt,...)		elog(NOTICE, "%u: " fmt, lineno, __VA_ARGS__)
#define HELPER_PRIV_CHECK(func_oid, ownership)		\
	(!OidIsValid(func_oid) ||						\
	 !OidIsValid(ownership) ||						\
	 pg_proc_ownercheck((func_oid),(ownership)))

	plts->extra_flags = DEVKERNEL_NEEDS_PLCUDA;
	/* check result type */
	dtype = pgstrom_devtype_lookup(prorettype);
	if (!dtype)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Result type \"%s\" is not device executable",
						format_type_be(prorettype))));
	plts->extra_flags |= dtype->type_flags;
	/* check argument types */
	for (i=0; i < proargtypes->dim1; i++)
	{
		Oid		argtype_oid = proargtypes->values[i];

		dtype = pgstrom_devtype_lookup(argtype_oid);
		if (!dtype)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("Result type \"%s\" is not device executable",
							format_type_be(argtype_oid))));
		plts->extra_flags |= dtype->type_flags;
	}

	for (line = strtok(source, "\n"), lineno = 1;
		 line != NULL;
		 line = strtok(NULL, "\n"), lineno++)
	{
		const char *cmd;
		const char *pos;
		char	   *end;
		List	   *options;

		/* Trimming of whitespace in the tail */
		/* NOTE: DOS/Windows uses '\r\n' for line-feed */
		end = line + strlen(line) - 1;
		while (line <= end && isspace(*end))
			*end-- = '\0';

		/* put a non pl/cuda command line*/
		if (strncmp(line, "#plcuda_", 8) != 0)
		{
			if (curr != NULL)
				appendStringInfo(curr, "%s\n", line);
			else
			{
				/* ignore if empty line */
				for (pos = line; !isspace(*pos) && *pos != '\0'; pos++);

				if (*pos != '\0')
					EMSG("code is out of valid block:\n%s", line);
			}
		}
		else
		{
			/* pick up command name */
			for (pos = line; !isspace(*pos) && *pos != '\0'; pos++);
			cmd = pnstrdup(line, pos - line);
			/* parse pl/cuda command options */
			if (!plcuda_parse_cmd_options(pos, &options))
			{
				EMSG("pl/cuda parse error:\n%s", line);
				continue;
			}

			if (strcmp(cmd, "#plcuda_decl") == 0)
			{
				if (has_decl_block)
					EMSG("%s appeared twice", cmd);
				else if (list_length(options) > 0)
					EMSG("syntax error:\n  %s", line);
				else
				{
					curr = &decl_src;
					has_decl_block = true;
				}
			}
			else if (strcmp(cmd, "#plcuda_prep") == 0)
			{
				if (has_prep_block)
					EMSG("%s appeared twice", cmd);
				else if (list_length(options) > 0)
					EMSG("syntax error:\n  %s", line);
				else
				{
					curr = &prep_src;
					has_prep_block = true;
				}
			}
			else if (strcmp(cmd, "#plcuda_begin") == 0)
			{
				if (has_main_block)
					EMSG("%s appeared twice", cmd);
				else if (list_length(options) > 0)
					EMSG("syntax error:\n  %s", line);
				else
				{
					curr = &main_src;
					has_main_block = true;
				}
			}
			else if (strcmp(cmd, "#plcuda_post") == 0)
			{
				if (has_post_block)
					EMSG("%s appeared twice", cmd);
				else if (list_length(options) > 0)
					EMSG("syntax error:\n  %s\n", line);
				else
				{
					curr = &post_src;
					has_post_block = true;
				}
			}
			else if (strcmp(cmd, "#plcuda_end") == 0)
			{
				if (!curr)
					EMSG("%s was used out of code block", cmd);
				else
					curr = NULL;
			}
			else if (strcmp(cmd, "#plcuda_num_threads") == 0)
			{
				Oid		fn_num_threads;
				long	val_num_threads;

				if (!curr)
					EMSG("%s appeared outside of code block", cmd);
				else if (plcuda_lookup_helper(options,
											  proargtypes, INT8OID,
											  &fn_num_threads,
											  &val_num_threads))
				{
					if (!HELPER_PRIV_CHECK(fn_num_threads, proowner))
					{
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
					}
					else if (curr == &prep_src)
					{
						plts->fn_prep_num_threads = fn_num_threads;
						plts->val_prep_num_threads = val_num_threads;
					}
					else if (curr == &main_src)
					{
						plts->fn_main_num_threads = fn_num_threads;
						plts->val_main_num_threads = val_num_threads;
					}
					else if (curr == &post_src)
					{
						plts->fn_post_num_threads = fn_num_threads;
						plts->val_post_num_threads = val_num_threads;
					}
					else
						EMSG("cannot use \"%s\" in this code block", cmd);
				}
				else if (not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_shmem_unitsz") == 0)
			{
				Oid		fn_shmem_unitsz;
				long	val_shmem_unitsz;

				if (!curr)
					EMSG("%s appeared outside of code block", cmd);
				else if (plcuda_lookup_helper(options,
											  proargtypes, INT8OID,
											  &fn_shmem_unitsz,
											  &val_shmem_unitsz))
				{
					if (!HELPER_PRIV_CHECK(fn_shmem_unitsz, proowner))
					{
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
					}
					else if (curr == &prep_src)
					{
						plts->fn_prep_shmem_unitsz = fn_shmem_unitsz;
						plts->val_prep_shmem_unitsz = val_shmem_unitsz;
					}
					else if (curr == &main_src)
					{
						plts->fn_main_shmem_unitsz = fn_shmem_unitsz;
						plts->val_main_shmem_unitsz = val_shmem_unitsz;
					}
					else if (curr == &post_src)
					{
						plts->fn_post_shmem_unitsz = fn_shmem_unitsz;
						plts->val_post_shmem_unitsz = val_shmem_unitsz;
					}
					else
						EMSG("cannot use \"%s\" in this code block", cmd);
				}
				else if (not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_shmem_blocksz") == 0)
			{
				Oid		fn_shmem_blocksz;
				long	val_shmem_blocksz;

				if (!curr)
					EMSG("%s appeared outside of code block", cmd);
				else if (plcuda_lookup_helper(options,
											  proargtypes, INT8OID,
											  &fn_shmem_blocksz,
											  &val_shmem_blocksz))
				{
					if (!HELPER_PRIV_CHECK(fn_shmem_blocksz, proowner))
					{
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
					}
					else if (curr == &prep_src)
					{
						plts->fn_prep_shmem_blocksz = fn_shmem_blocksz;
						plts->val_prep_shmem_blocksz = val_shmem_blocksz;
					}
					else if (curr == &main_src)
					{
						plts->fn_main_shmem_blocksz = fn_shmem_blocksz;
						plts->val_main_shmem_blocksz = val_shmem_blocksz;
					}
					else if (curr == &post_src)
					{
						plts->fn_post_shmem_blocksz = fn_shmem_blocksz;
						plts->val_post_shmem_blocksz = val_shmem_blocksz;
					}
					else
						EMSG("cannot use \"%s\" in this code block", cmd);
				}
				else if (not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_kernel_blocksz") == 0)
			{
				Oid		fn_kern_blocksz;
				long	val_kern_blocksz;

				if (!curr)
					EMSG("%s appeared outside of code block", cmd);
				else if (plcuda_lookup_helper(options,
											  proargtypes, INT8OID,
											  &fn_kern_blocksz,
											  &val_kern_blocksz))
				{
					if (!HELPER_PRIV_CHECK(fn_kern_blocksz, proowner))
					{
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
					}
					else if (curr == &prep_src)
					{
						plts->fn_prep_kern_blocksz = fn_kern_blocksz;
						plts->val_prep_kern_blocksz = val_kern_blocksz;
					}
					else if (curr == &main_src)
					{
						plts->fn_main_kern_blocksz = fn_kern_blocksz;
						plts->val_main_kern_blocksz = val_kern_blocksz;
					}
					else if (curr == &post_src)
					{
						plts->fn_post_kern_blocksz = fn_kern_blocksz;
						plts->val_post_kern_blocksz = val_kern_blocksz;
					}
					else
						EMSG("cannot use \"%s\" in this code block", cmd);
				}
				else if (not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_working_bufsz") == 0)
			{
				if (has_working_bufsz)
					EMSG("%s appeared twice", cmd);
				else if (plcuda_lookup_helper(options,
											  proargtypes, INT8OID,
											  &plts->fn_working_bufsz,
											  &plts->val_working_bufsz))
				{
					if (HELPER_PRIV_CHECK(plts->fn_working_bufsz, proowner))
						has_working_bufsz = true;
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
                         ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_results_bufsz") == 0)
			{
				if (has_results_bufsz)
					EMSG("%s appeared twice", cmd);
				else if (plcuda_lookup_helper(options,
											  proargtypes, INT8OID,
											  &plts->fn_results_bufsz,
											  &plts->val_results_bufsz))
				{
					if (HELPER_PRIV_CHECK(plts->fn_results_bufsz, proowner))
						has_results_bufsz = true;
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_include") == 0)
			{
				cl_uint		extra_flags = 0;

				/* built-in include? */
				if (list_length(options) == 1)
				{
					const char *target = linitial(options);

					if (strcmp(target, "cuda_dynpara.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_DYNPARA;
					else if (strcmp(target, "cuda_matrix.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_MATRIX;
					else if (strcmp(target, "cuda_timelib.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_TIMELIB;
					else if (strcmp(target, "cuda_textlib.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_TEXTLIB;
					else if (strcmp(target, "cuda_numeric.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_NUMERIC;
					else if (strcmp(target, "cuda_mathlib.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_MATHLIB;
					else if (strcmp(target, "cuda_money.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_MONEY;
					else if (strcmp(target, "cuda_curand.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_CURAND;
					else if (strcmp(target, "cuda_cublas.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_CUBLAS;

					if (extra_flags != 0 && (has_decl_block ||
											 has_prep_block ||
											 has_main_block ||
											 has_post_block))
					{
						EMSG("built-in \"%s\" must appear prior to the code block:\n%s",
							 target, line);
					}
				}
				if (extra_flags != 0)
					plts->extra_flags |= extra_flags;
				else if (!curr)
					EMSG("#plcuda_include must appear in code block:\n%s",
						 line);
				else
				{
					Oid		fn_extra_include = InvalidOid;

					/* function that returns text */
					if (plcuda_lookup_helper(options,
											 buildoidvector(NULL, 0),
											 TEXTOID,
											 &fn_extra_include,
											 NULL))
					{
						if (HELPER_PRIV_CHECK(fn_extra_include, proowner))
						{
							Datum	txt = OidFunctionCall0(fn_extra_include);
							char   *str = text_to_cstring(DatumGetTextP(txt));

							appendStringInfoString(curr, str);
						}
						else
							EMSG("permission denied on helper function %s",
								 NameListToString(options));
					}
					else if (not_exec_now)
						NOTE("\"%s\" may be a function but not declared yet",
							 ident_to_cstring(options));
					else
						EMSG("\"%s\" was not a valid function name",
							 ident_to_cstring(options));
				}
			}
			else if (strcmp(cmd, "#plcuda_sanity_check") == 0)
			{
				if (has_sanity_check)
					EMSG("%s appeared twice", cmd);
				else if (plcuda_lookup_helper(options,
											  proargtypes, BOOLOID,
											  &plts->fn_sanity_check,
											  NULL))
				{
					if (HELPER_PRIV_CHECK(plts->fn_sanity_check, proowner))
						has_sanity_check = true;
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid function name",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_cpu_fallback") == 0)
			{
				if (has_cpu_fallback)
					EMSG("%s appeared twice", cmd);
				else if (plcuda_lookup_helper(options,
											  proargtypes, prorettype,
											  &plts->fn_cpu_fallback,
											  NULL))
				{
					if (HELPER_PRIV_CHECK(plts->fn_cpu_fallback, proowner))
						has_cpu_fallback = true;
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid function name",
						 ident_to_cstring(options));
			}
			else
				EMSG("unknown command: %s", line);
		}
	}
	if (curr)
		appendStringInfo(&emsg, "\n%u: code block was not closed", lineno);
	if (!has_main_block)
		appendStringInfo(&emsg, "\n%u: no main code block", lineno);
	if (emsg.len > 0)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("pl/cuda function syntax error\n%s", emsg.data)));

	plts->kern_decl = (has_decl_block ? decl_src.data : NULL);
	plts->kern_prep = (has_prep_block ? prep_src.data : NULL);
	plts->kern_main = main_src.data;
	plts->kern_post = (has_post_block ? post_src.data : NULL);
#undef NMSG
#undef EMSG
#undef HELPER_PRIV_CHECK
	pfree(emsg.data);
}

/*
 * plcuda_codegen
 *
 *
 *
 */
static void
plcuda_codegen_part(StringInfo kern,
					const char *suffix,
					const char *users_code,
					bool kernel_maxthreads,
					Form_pg_proc procForm,
					const char *last_suffix)
{
	devtype_info   *dtype_r;
	devtype_info   *dtype_a;
	int		i;

	appendStringInfo(
		kern,
		"STATIC_INLINE(void)\n"
		"__plcuda_%s_kernel(kern_plcuda *kplcuda,\n"
		"                   void *workbuf,\n"
		"                   void *results,\n"
		"                   kern_context *kcxt)\n"
		"{\n",
		suffix);

	/* declaration of 'retval' variable */
	dtype_r = pgstrom_devtype_lookup(procForm->prorettype);
	if (!dtype_r)
		elog(ERROR, "cache lookup failed for type '%s'",
			 format_type_be(procForm->prorettype));
	appendStringInfo(
		kern,
		"  pg_%s_t *retval __attribute__ ((unused));\n",
		dtype_r->type_name);

	/* declaration of argument variables */
	for (i=0; i < procForm->pronargs; i++)
	{
		Oid		type_oid = procForm->proargtypes.values[i];

		dtype_a = pgstrom_devtype_lookup(type_oid);
		if (!dtype_a)
			elog(ERROR, "cache lookup failed for type '%s'",
				 format_type_be(type_oid));
		appendStringInfo(
			kern,
			"  pg_%s_t arg%u __attribute__((unused));\n",
			dtype_a->type_name, i+1);
	}

	appendStringInfo(
		kern,
		"  assert(sizeof(*retval) <= sizeof(kplcuda->__retval));\n"
		"  retval = (pg_%s_t *)kplcuda->__retval;\n", dtype_r->type_name);
	if (dtype_r->type_length < 0)
		appendStringInfo(
			kern,
			"  assert(retval->isnull || (void *)retval->value == results);\n");

	for (i=0; i < procForm->pronargs; i++)
	{
		Oid		type_oid = procForm->proargtypes.values[i];

		dtype_a = pgstrom_devtype_lookup(type_oid);
		if (!dtype_a)
			elog(ERROR, "cache lookup failed for type '%s'",
				 format_type_be(type_oid));
		appendStringInfo(
			kern,
			"  arg%u = pg_%s_param(kcxt,%d);\n",
			i+1, dtype_a->type_name, i);
	}

	appendStringInfo(
		kern,
		"\n"
		"  /* ---- code by pl/cuda function ---- */\n"
		"%s"
		"  /* ---- code by pl/cuda function ---- */\n"
		"}\n\n",
		users_code);

	appendStringInfo(
		kern,
		"KERNEL_FUNCTION%s(void)\n"
		"plcuda_%s_kernel_entrypoint(kern_plcuda *kplcuda,\n"
		"            void *workbuf,\n"
		"            void *results)\n"
		"{\n"
		"  kern_parambuf *kparams = KERN_PLCUDA_PARAMBUF(kplcuda);\n"
		"  kern_context kcxt;\n"
		"\n"
		"  assert(kplcuda->nargs == kparams->nparams);\n",
		kernel_maxthreads ? "_MAXTHREADS" : "",
		suffix);

	if (last_suffix)
		appendStringInfo(
			kern,
			"  if (kplcuda->kerror_%s.errcode != StromError_Success)\n"
			"    kcxt.e = kplcuda->kerror_%s;\n"
			"  else\n"
			"  {\n"
			"    INIT_KERNEL_CONTEXT(&kcxt,plcuda_%s_kernel,kparams);\n"
			"    __plcuda_%s_kernel(kplcuda, workbuf, results, &kcxt);\n"
			"  }\n",
			last_suffix,
			last_suffix,
			suffix,
			suffix);
	else
		appendStringInfo(
			kern,
			"  INIT_KERNEL_CONTEXT(&kcxt,plcuda_%s_kernel,kparams);\n"
			"  __plcuda_%s_kernel(kplcuda, workbuf, results, &kcxt);\n",
			suffix,
			suffix);

	appendStringInfo(
		kern,
		"  kern_writeback_error_status(&kplcuda->kerror_%s, kcxt.e);\n"
		"}\n\n",
		suffix);
}

static char *
plcuda_codegen(Form_pg_proc procForm, plcudaTaskState *plts)
{
	StringInfoData	kern;
	const char	   *last_stage = NULL;

	initStringInfo(&kern);

	if (plts->kern_decl)
		appendStringInfo(&kern, "%s\n", plts->kern_decl);
	if (plts->kern_prep)
	{
		plcuda_codegen_part(&kern, "prep",
							plts->kern_prep,
							(OidIsValid(plts->fn_prep_kern_blocksz) ||
							 plts->val_prep_kern_blocksz > 0),
							procForm,
							last_stage);
		last_stage = "prep";
	}
	if (plts->kern_main)
	{
		plcuda_codegen_part(&kern, "main",
							plts->kern_main,
							(OidIsValid(plts->fn_main_kern_blocksz) ||
							 plts->val_main_kern_blocksz > 0),
							procForm,
							last_stage);
		last_stage = "main";
	}
	if (plts->kern_post)
	{
		plcuda_codegen_part(&kern, "post",
							plts->kern_post,
							(OidIsValid(plts->fn_post_kern_blocksz) ||
							 plts->val_post_kern_blocksz > 0),
							procForm,
							last_stage);
		last_stage = "post";
	}
	return kern.data;
}


static void
plcuda_cleanup_resources(ResourceReleasePhase phase,
						 bool isCommit,
						 bool isTopLevel,
						 void *private)
{
	dlist_mutable_iter iter;

	if (phase != RESOURCE_RELEASE_BEFORE_LOCKS)
		return;

	dlist_foreach_modify(iter, &plcuda_state_list)
	{
		plcudaTaskState *plts = (plcudaTaskState *)
			dlist_container(plcudaTaskState, chain, iter.cur);
		if (plts->owner == CurrentResourceOwner)
			plcuda_exec_end(plts);
	}
}

static inline kern_colmeta
__setup_kern_colmeta(Oid type_oid, int arg_index)
{
	HeapTuple		tuple;
	Form_pg_type	typeForm;
	kern_colmeta	result;

	tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for type '%s'",
			 format_type_be(type_oid));
	typeForm = (Form_pg_type) GETSTRUCT(tuple);

	result.attbyval = typeForm->typbyval;
	result.attalign = typealign_get_width(typeForm->typalign);
	result.attlen = typeForm->typlen;
	result.attnum = arg_index;
	result.attcacheoff = -1;	/* we don't use attcacheoff */
	result.atttypid = type_oid;
	result.atttypmod = typeForm->typtypmod;

	ReleaseSysCache(tuple);

	return result;
}

static plcudaTaskState *
plcuda_exec_begin(HeapTuple protup, FunctionCallInfo fcinfo)
{
	GpuContext_v2   *gcontext = AllocGpuContext(fcinfo != NULL);
	plcudaTaskState *plts;
	kern_plcuda	   *kplcuda;
	Form_pg_proc	procForm = (Form_pg_proc) GETSTRUCT(protup);
	Datum			prosrc;
	bool			isnull;
	char		   *kern_source;
	char		   *kern_define;
	ProgramId		program_id;
	int				i;

	/* setup a dummy GTS for PL/CUDA (see pgstromInitGpuTaskState) */
	plts = MemoryContextAllocZero(CacheMemoryContext,
								  sizeof(plcudaTaskState));
	plts->gts.gcontext = gcontext;
	plts->gts.task_kind = GpuTaskKind_PL_CUDA;
	plts->gts.revision = 1;
	plts->gts.kern_params = NULL;
	dlist_init(&plts->gts.ready_tasks);
	//FIXME: nobody may be responsible to 'prime_in_gpucontext'

	plts->val_prep_num_threads = 1;
	plts->val_main_num_threads = 1;
	plts->val_post_num_threads = 1;

	/* validate PL/CUDA source code */
	prosrc = SysCacheGetAttr(PROCOID, protup, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "Bug? no program source was supplied");
	plcuda_code_validation(plts,
						   !fcinfo ? procForm->proowner : InvalidOid,
						   &procForm->proargtypes,
						   procForm->prorettype,
						   TextDatumGetCString(prosrc));
	/* construct a flat kernel source to be built */
	kern_source = plcuda_codegen(procForm, plts);
	kern_define = pgstrom_build_session_info(plts->extra_flags, &plts->gts);
	program_id = pgstrom_create_cuda_program(gcontext,
											 plts->extra_flags,
											 kern_source,
											 kern_define,
											 true);
	plts->gts.program_id = program_id;

	/* build template of the kern_plcuda */
	kplcuda = palloc0(offsetof(kern_plcuda,
							   argmeta[procForm->pronargs]));
	kplcuda->nargs = procForm->pronargs;
	kplcuda->retmeta = __setup_kern_colmeta(procForm->prorettype, -1);
	for (i=0; i < procForm->pronargs; i++)
	{
		Oid		argtype_oid = procForm->proargtypes.values[i];

		kplcuda->argmeta[i] = __setup_kern_colmeta(argtype_oid, i);
	}
	/* MEMO: if result type is varlena, initialized to NULL as a default */
	if (kplcuda->retmeta.attlen < 0)
		kplcuda->__retval[sizeof(CUdeviceptr)] = true;
	plts->kplcuda_head = kplcuda;
	/* track plcudaTaskState by local tracker */
	plts->owner = CurrentResourceOwner;
	dlist_push_head(&plcuda_state_list, &plts->chain);

	return plts;
}

static void
plcuda_exec_end(plcudaTaskState *plts)
{
	dlist_delete(&plts->chain);

	if (plts->last_results_buf)
		dmaBufferFree(plts->last_results_buf);
	pgstromReleaseGpuTaskState(&plts->gts);
}

Datum
plcuda_function_validator(PG_FUNCTION_ARGS)
{
	Oid				func_oid = PG_GETARG_OID(0);
	HeapTuple		tuple;
	Form_pg_proc	procForm;
	plcudaTaskState	*plts;

	if (!CheckFunctionValidatorAccess(fcinfo->flinfo->fn_oid, func_oid))
		PG_RETURN_VOID();

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

	/*
	 * Only validation of the CUDA code. Run synchronous code build, then
	 * raise an error if code block has any error.
	 */
	plts = plcuda_exec_begin(tuple, NULL);
	plcuda_exec_end(plts);

	ReleaseSysCache(tuple);

	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(plcuda_function_validator);

static inline Datum
kernel_launch_helper(FunctionCallInfo __fcinfo,	/* PL/CUDA's fcinfo */
					 Oid helper_func_oid,
					 Datum static_config,
					 bool *p_isnull)	/* may be NULL, if don't allow NULL */
{
    FunctionCallInfoData fcinfo;
	FmgrInfo	flinfo;
	AclResult	aclresult;
	Datum		result;

	if (!OidIsValid(helper_func_oid))
		return static_config;

	aclresult = pg_proc_aclcheck(helper_func_oid, GetUserId(), ACL_EXECUTE);
	if (aclresult != ACLCHECK_OK)
		aclcheck_error(aclresult, ACL_KIND_PROC,
					   get_func_name(helper_func_oid));

	fmgr_info(helper_func_oid, &flinfo);
	InitFunctionCallInfoData(fcinfo, &flinfo,
							 __fcinfo->nargs,
							 __fcinfo->fncollation,
							 NULL, NULL);
	memcpy(fcinfo.arg, __fcinfo->arg,
		   sizeof(Datum) * fcinfo.nargs);
	memcpy(fcinfo.argnull, __fcinfo->argnull,
		   sizeof(bool) * fcinfo.nargs);

	result = FunctionCallInvoke(&fcinfo);
	if (fcinfo.isnull)
	{
		if (p_isnull)
			*p_isnull = true;
		else
			elog(ERROR, "helper function %s returned NULL",
				 format_procedure(helper_func_oid));
	}
	return result;
}

/*
 * create_plcuda_task
 */
static plcudaTask *
create_plcuda_task(plcudaTaskState *plts, FunctionCallInfo fcinfo,
				   Size working_bufsz, Size results_bufsz)
{
	GpuContext_v2  *gcontext = plts->gts.gcontext;
	plcudaTask	   *ptask;
	kern_parambuf  *kparams;
	Size			head_length;
	Size			total_length;
	Size			offset;
	int				i;

	/* calculation of the length */
	Assert(fcinfo->nargs == plts->kplcuda_head->nargs);
	head_length = offsetof(plcudaTask,
						   kern.argmeta[fcinfo->nargs]);
	total_length = STROMALIGN(head_length);

	total_length += STROMALIGN(offsetof(kern_parambuf,
										poffset[fcinfo->nargs]));
	for (i=0; i < fcinfo->nargs; i++)
	{
		kern_colmeta	cmeta = plts->kplcuda_head->argmeta[i];

		if (fcinfo->argnull[i])
			continue;
		if (cmeta.attlen > 0)
			total_length += MAXALIGN(cmeta.attlen);
		else
			total_length += MAXALIGN(toast_raw_datum_size(fcinfo->arg[i]));
	}
	total_length = STROMALIGN(total_length);

	/* setup plcudaTask */
	ptask = dmaBufferAlloc(gcontext, total_length);
	memset(ptask, 0, offsetof(plcudaTask, kern));
	pgstromInitGpuTask(&plts->gts, &ptask->task);
	if (results_bufsz > 0)
		ptask->h_results_buf = dmaBufferAlloc(gcontext, results_bufsz);
	/* setup kern_plcuda */
	memcpy(&ptask->kern, plts->kplcuda_head,
		   offsetof(kern_plcuda, argmeta[fcinfo->nargs]));
	ptask->kern.working_bufsz = working_bufsz;
	ptask->kern.working_usage = 0UL;
	ptask->kern.results_bufsz = results_bufsz;
	ptask->kern.results_usage = 0UL;

	/* copy function arguments onto DMA buffer */
	kparams = KERN_PLCUDA_PARAMBUF(&ptask->kern);
	kparams->hostptr = (hostptr_t) kparams;
	kparams->xactStartTimestamp = GetCurrentTransactionStartTimestamp();

	offset = STROMALIGN(offsetof(kern_parambuf,
								 poffset[fcinfo->nargs]));
	for (i=0; i < fcinfo->nargs; i++)
	{
		kern_colmeta	cmeta = plts->kplcuda_head->argmeta[i];

		if (fcinfo->argnull[i])
			kparams->poffset[i] = 0;	/* null */
		else
		{
			kparams->poffset[i] = offset;
			if (cmeta.attbyval)
			{
				Assert(cmeta.attlen > 0);
				memcpy((char *)kparams + offset,
					   &fcinfo->arg[i],
					   cmeta.attlen);
				offset += MAXALIGN(cmeta.attlen);
			}
			else
			{
				char   *vl_ptr = (char *)PG_DETOAST_DATUM(fcinfo->arg[i]);
				Size	vl_len = VARSIZE_ANY(vl_ptr);

				memcpy((char *)kparams + offset, vl_ptr, vl_len);
				offset += MAXALIGN(vl_len);
			}
		}
	}
	kparams->nparams = fcinfo->nargs;
	kparams->length = STROMALIGN(offset);
	Assert(STROMALIGN(offsetof(plcudaTask,
							   kern.argmeta[fcinfo->nargs])) +
		   kparams->length == total_length);
	Assert(!plts->last_results_buf);
	plts->last_results_buf = ptask->h_results_buf;

	return ptask;
}

Datum
plcuda_function_handler(PG_FUNCTION_ARGS)
{
	FmgrInfo	   *flinfo = fcinfo->flinfo;
	plcudaTaskState *plts;
	plcudaTask	   *ptask;
	plcudaTask	   *precv;
	Size			working_bufsz;
	Size			results_bufsz;
	kern_errorbuf	kerror;
	Datum			retval = 0;
	bool			isnull = false;

	if (!flinfo->fn_extra)
	{
		MemoryContext	oldcxt;
		HeapTuple		tuple;

		tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(flinfo->fn_oid));
		if (!HeapTupleIsValid(tuple))
			elog(ERROR, "cache lookup failed for function %u",
				 flinfo->fn_oid);

		oldcxt = MemoryContextSwitchTo(flinfo->fn_mcxt);
		plts = plcuda_exec_begin(tuple, fcinfo);
		MemoryContextSwitchTo(oldcxt);

		ReleaseSysCache(tuple);
		flinfo->fn_extra = plts;
	}
	else
	{
		plts = (plcudaTaskState *) flinfo->fn_extra;
		pgstromRescanGpuTaskState(&plts->gts);
	}

	/* results buffer of last invocation will not be used no longer */
	if (plts->last_results_buf)
	{
		dmaBufferFree(plts->last_results_buf);
		plts->last_results_buf = NULL;
	}

	/* sanitycheck of the supplied arguments, prior to GPU launch */
	if (!DatumGetBool(kernel_launch_helper(fcinfo,
										   plts->fn_sanity_check,
										   BoolGetDatum(true),
										   NULL)))
		elog(ERROR, "function '%s' argument sanity check failed",
			 format_procedure(plts->fn_sanity_check));

	/* determine the kernel launch parameters */
	working_bufsz =
		DatumGetInt64(kernel_launch_helper(fcinfo,
										   plts->fn_working_bufsz,
										   plts->val_working_bufsz,
										   NULL));
	results_bufsz =
		DatumGetInt64(kernel_launch_helper(fcinfo,
										   plts->fn_results_bufsz,
										   plts->val_results_bufsz,
										   NULL));
	elog(DEBUG2, "working_bufsz = %zu, results_bufsz = %zu",
		 working_bufsz, results_bufsz);

	/* construction of plcudaTask structure */
	ptask = create_plcuda_task(plts, fcinfo,
							   working_bufsz,
							   results_bufsz);
	if (plts->kern_prep)
	{
		int64		v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_prep_num_threads,
											   plts->val_prep_num_threads,
											   NULL));
		if (v <= 0)
			elog(ERROR, "kern_prep: invalid number of threads: %ld", v);
		ptask->kern.prep_num_threads = (cl_ulong)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_prep_kern_blocksz,
											   plts->val_prep_kern_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_prep: invalid kernel block size: %ld", v);
		ptask->kern.prep_kern_blocksz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_prep_shmem_unitsz,
											   plts->val_prep_shmem_unitsz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_prep: invalid shared memory required: %ld", v);
		ptask->kern.prep_shmem_unitsz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_prep_shmem_blocksz,
											   plts->val_prep_shmem_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_prep: invalid shared memory required: %ld", v);
		ptask->kern.prep_shmem_blocksz = v;

		elog(DEBUG2, "kern_prep {blocksz=%u, nitems=%lu, shmem=%u,%u}",
			 ptask->kern.prep_kern_blocksz,
			 ptask->kern.prep_num_threads,
			 ptask->kern.prep_shmem_unitsz,
			 ptask->kern.prep_shmem_blocksz);
		ptask->exec_prep_kernel = true;
	}

	if (plts->kern_main)
	{
		int64		v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_main_num_threads,
											   plts->val_main_num_threads,
											   NULL));
		if (v <= 0)
			elog(ERROR, "kern_main: invalid number of threads: %ld", v);
		ptask->kern.main_num_threads = (cl_ulong)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_main_kern_blocksz,
											   plts->val_main_kern_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_main: invalid kernel block size: %ld", v);
		ptask->kern.main_kern_blocksz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_main_shmem_unitsz,
											   plts->val_main_shmem_unitsz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_main: invalid shared memory required: %ld", v);
		ptask->kern.main_shmem_unitsz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_main_shmem_blocksz,
											   plts->val_main_shmem_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_main: invalid shared memory required: %ld", v);
		ptask->kern.main_shmem_blocksz = v;

		elog(DEBUG2, "kern_main {blocksz=%u, nitems=%lu, shmem=%u,%u}",
			 ptask->kern.main_kern_blocksz,
			 ptask->kern.main_num_threads,
			 ptask->kern.main_shmem_unitsz,
			 ptask->kern.main_shmem_blocksz);
	}

	if (plts->kern_post)
	{
		int64		v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_post_num_threads,
											   plts->val_post_num_threads,
											   NULL));
		if (v <= 0)
			elog(ERROR, "kern_post: invalid number of threads: %ld", v);
		ptask->kern.post_num_threads = (cl_ulong)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_post_kern_blocksz,
											   plts->val_post_kern_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_post: invalid kernel block size: %ld", v);
		ptask->kern.post_kern_blocksz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_post_shmem_unitsz,
											   plts->val_post_shmem_unitsz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_post: invalid shared memory required: %ld", v);
		ptask->kern.post_shmem_unitsz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   plts->fn_post_shmem_blocksz,
											   plts->val_post_shmem_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_post: invalid shared memory required: %ld", v);
		ptask->kern.post_shmem_blocksz = v;

		elog(DEBUG2, "kern_post {blocksz=%u, nitems=%lu, shmem=%u,%u}",
			 ptask->kern.post_kern_blocksz,
			 ptask->kern.post_num_threads,
			 ptask->kern.post_shmem_unitsz,
			 ptask->kern.post_shmem_blocksz);
		ptask->exec_post_kernel = true;
	}

	if (!gpuservSendGpuTask(plts->gts.gcontext, &ptask->task))
		elog(ERROR, "failed to send GpuTask to GPU server");
	plts->gts.scan_done = true;

	precv = (plcudaTask *) fetch_next_gputask(&plts->gts);
	if (!precv)
		elog(ERROR, "failed to recv GpuTask from GPU server");
	Assert(precv == ptask);

	/*
	 * Dump the debug counter if valid values are set by kernel function
	 */
	if (precv->kern.plcuda_debug_count0)
		elog(NOTICE, "PL/CUDA debug count0 => %lu",
			 precv->kern.plcuda_debug_count0);
	if (precv->kern.plcuda_debug_count1)
		elog(NOTICE, "PL/CUDA debug count1 => %lu",
			 precv->kern.plcuda_debug_count1);
	if (precv->kern.plcuda_debug_count2)
		elog(NOTICE, "PL/CUDA debug count2 => %lu",
			 precv->kern.plcuda_debug_count2);
	if (precv->kern.plcuda_debug_count3)
		elog(NOTICE, "PL/CUDA debug count3 => %lu",
			 precv->kern.plcuda_debug_count3);
	if (precv->kern.plcuda_debug_count4)
		elog(NOTICE, "PL/CUDA debug count4 => %lu",
			 precv->kern.plcuda_debug_count4);
	if (precv->kern.plcuda_debug_count5)
		elog(NOTICE, "PL/CUDA debug count5 => %lu",
			 precv->kern.plcuda_debug_count5);
	if (precv->kern.plcuda_debug_count6)
		elog(NOTICE, "PL/CUDA debug count6 => %lu",
			 precv->kern.plcuda_debug_count6);
	if (precv->kern.plcuda_debug_count7)
		elog(NOTICE, "PL/CUDA debug count7 => %lu",
			 precv->kern.plcuda_debug_count7);

	if (precv->task.kerror.errcode == StromError_Success)
	{
		if (precv->kern.retmeta.attlen > 0)
		{
			if (precv->kern.__retval[precv->kern.retmeta.attlen])
				isnull = true;
			else if (precv->kern.retmeta.attbyval)
			{
				/* inline fixed-length variable */
				memcpy(&retval, precv->kern.__retval,
					   precv->kern.retmeta.attbyval);
			}
			else
			{
				/* indirect fixed-length variable */
				retval = PointerGetDatum(pnstrdup(precv->kern.__retval,
												  precv->kern.retmeta.attlen));
			}
		}
		else
		{
			Assert(!precv->kern.retmeta.attbyval);
			if (precv->kern.__retval[sizeof(CUdeviceptr)])
				isnull = true;
			else if (!precv->h_results_buf)
				elog(ERROR, "non-NULL result in spite of no results buffer");
			else
				retval = PointerGetDatum(precv->h_results_buf);
		}
	}
	else
	{
		if (precv->h_results_buf)
		{
			dmaBufferFree(precv->h_results_buf);
			precv->h_results_buf = NULL;
			plts->last_results_buf = NULL;
		}

		if (kerror.errcode == StromError_CpuReCheck &&
			OidIsValid(plts->fn_cpu_fallback))
		{
			/* CPU fallback, if any */
			retval = kernel_launch_helper(fcinfo,
										  plts->fn_cpu_fallback,
										  (Datum)0,
										  &isnull);
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("PL/CUDA execution error (%s)",
							errorTextKernel(&kerror))));
		}
	}
	dmaBufferFree(ptask);

	if (isnull)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(retval);
}
PG_FUNCTION_INFO_V1(plcuda_function_handler);

/*
 * plcuda_function_source
 */
Datum
plcuda_function_source(PG_FUNCTION_ARGS)
{
	Oid				func_oid = PG_GETARG_OID(0);
	Oid				lang_oid;
	const char	   *lang_name = "plcuda";
	HeapTuple		tuple;
	Form_pg_proc	procForm;
	Datum			prosrc;
	bool			isnull;
	text		   *plcuda_source;

	lang_oid = GetSysCacheOid1(LANGNAME, CStringGetDatum(lang_name));
	if (!OidIsValid(lang_oid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("language \"%s\" does not exist", lang_name)));

	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	procForm = (Form_pg_proc) GETSTRUCT(tuple);

	if (procForm->prolang != lang_oid)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("procedure \"%s\" is not implemented by \"%s\"",
						format_procedure(func_oid), lang_name)));

	prosrc = SysCacheGetAttr(PROCOID, tuple,
							 Anum_pg_proc_prosrc,
							 &isnull);
	if (isnull)
		elog(ERROR, "Bug? source code of PL/CUDA function has gone");
	plcuda_source = DatumGetTextPCopy(prosrc);
	ReleaseSysCache(tuple);

	PG_RETURN_TEXT_P(plcuda_source);
}
PG_FUNCTION_INFO_V1(plcuda_function_source);

/*
 * plcuda_cleanup_cuda_resources
 */
static void
plcuda_cleanup_cuda_resources(plcudaTask *ptask)
{
	CUresult	rc;

	PFMON_EVENT_DESTROY(ptask, ev_dma_send_start);
	PFMON_EVENT_DESTROY(ptask, ev_dma_send_stop);
	PFMON_EVENT_DESTROY(ptask, ev_dma_recv_start);
	PFMON_EVENT_DESTROY(ptask, ev_dma_recv_stop);

	if (ptask->m_kern_plcuda)
	{
		rc = gpuMemFree_v2(ptask->task.gcontext, ptask->m_kern_plcuda);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on gpuMemFree: %s", errorText(rc));
	}

	if (ptask->m_results_buf)
	{
		rc = gpuMemFree_v2(ptask->task.gcontext, ptask->m_results_buf);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on gpuMemFree: %s", errorText(rc));
	}

	if (ptask->m_working_buf)
	{
		rc = gpuMemFree_v2(ptask->task.gcontext, ptask->m_working_buf);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on gpuMemFree: %s", errorText(rc));
	}
	ptask->kern_plcuda_prep = NULL;
	ptask->kern_plcuda_main = NULL;
	ptask->kern_plcuda_post = NULL;
	ptask->m_kern_plcuda = 0UL;
	ptask->m_results_buf = 0UL;
	ptask->m_working_buf = 0UL;
}

/*
 * plcuda_respond_task
 */
static void
plcuda_respond_task(CUstream stream, CUresult status, void *private)
{
	plcudaTask *ptask = private;
	bool		is_urgent = false;

	if (status == CUDA_SUCCESS)
	{
		if (ptask->exec_prep_kernel &&
			ptask->kern.kerror_prep.errcode != StromError_Success)
			ptask->task.kerror = ptask->kern.kerror_prep;
		else if (ptask->kern.kerror_main.errcode != StromError_Success)
			ptask->task.kerror = ptask->kern.kerror_main;
		else if (ptask->exec_post_kernel &&
				 ptask->kern.kerror_post.errcode != StromError_Success)
			ptask->task.kerror = ptask->kern.kerror_post;
	}
	else
	{
		/* CUDA run-time error - not recoverable */
		ptask->task.kerror.errcode = status;
		ptask->task.kerror.kernel  = StromKernel_CudaRuntime;
		ptask->task.kerror.lineno  = 0;
		is_urgent = true;
	}
	gpuservCompleteGpuTask(&ptask->task, is_urgent);
}

/*
 * plcuda_process_task
 */
static int
__plcuda_process_task(plcudaTask *ptask,
					  CUmodule cuda_module,
					  CUstream cuda_stream)
{
	GpuContext_v2  *gcontext = ptask->task.gcontext;
	void		   *kern_args[3];
	size_t			warp_size;
	size_t			block_size;
	size_t			grid_size;
	CUresult		rc;

	/* property of the device */
	warp_size = devAttrs[gpuserv_cuda_dindex].WARP_SIZE;

	/* plcuda_prep_kernel_entrypoint, if any */
	if (ptask->exec_prep_kernel)
	{
		rc = cuModuleGetFunction(&ptask->kern_plcuda_prep,
								 cuda_module,
								 "plcuda_prep_kernel_entrypoint");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));
	}
	/* plcuda_main_kernel_entrypoint */
	rc = cuModuleGetFunction(&ptask->kern_plcuda_main,
							 cuda_module,
							 "plcuda_main_kernel_entrypoint");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));
	/* plcuda_post_kernel_entrypoint */
	if (ptask->exec_post_kernel)
	{
		rc = cuModuleGetFunction(&ptask->kern_plcuda_post,
								 cuda_module,
								 "plcuda_post_kernel_entrypoint");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));
	}

	/* kern_plcuda structure on the device side */
	rc = gpuMemAlloc_v2(gcontext,
						&ptask->m_kern_plcuda,
						KERN_PLCUDA_DMASEND_LENGTH(&ptask->kern));
	if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		goto out_of_resource;
	else if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAlloc: %s", errorText(rc));

	/* working buffer if required */
	if (ptask->kern.working_bufsz > 0)
	{
		rc = gpuMemAlloc_v2(gcontext,
							&ptask->m_working_buf,
							ptask->kern.working_bufsz);
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
			goto out_of_resource;
		else if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAlloc: %s", errorText(rc));
	}
	else
	{
		ptask->m_working_buf = 0UL;
	}

	/* results buffer if required  */
	if (ptask->kern.results_bufsz > 0)
	{
		rc = gpuMemAlloc_v2(gcontext,
							&ptask->m_results_buf,
							ptask->kern.results_bufsz);
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
			goto out_of_resource;
		else if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAlloc: %s", errorText(rc));
	}
	else
	{
		ptask->m_results_buf = 0UL;
	}

	/* move the control block + argument buffer */
	rc = cuMemcpyHtoDAsync(ptask->m_kern_plcuda,
						   &ptask->kern,
						   KERN_PLCUDA_DMASEND_LENGTH(&ptask->kern),
						   cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));

	/* kernel arguments (common for all thress kernels) */
	kern_args[0] = &ptask->m_kern_plcuda;
	kern_args[1] = &ptask->m_working_buf;
	kern_args[2] = &ptask->m_results_buf;

	/* launch plcuda_prep_kernel_entrypoint */
	if (ptask->exec_prep_kernel)
    {
		if (ptask->kern.prep_kern_blocksz > 0)
		{
			block_size = (ptask->kern.prep_kern_blocksz +
						  warp_size - 1) & ~(warp_size - 1);
			grid_size = (ptask->kern.prep_num_threads +
						 block_size - 1) / block_size;
		}
		else
		{
			optimal_workgroup_size(&grid_size,
								   &block_size,
								   ptask->kern_plcuda_prep,
								   gpuserv_cuda_device,
								   ptask->kern.prep_num_threads,
								   ptask->kern.prep_shmem_blocksz,
								   ptask->kern.prep_shmem_unitsz);
		}

		rc = cuLaunchKernel(ptask->kern_plcuda_prep,
							grid_size, 1, 1,
							block_size, 1, 1,
							ptask->kern.prep_shmem_blocksz +
							ptask->kern.prep_shmem_unitsz * block_size,
							cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("failed on cuLaunchKernel: %s", errorText(rc)),
					 errhint("prep-kernel: grid=%u block=%u shmem=%zu",
							 (cl_uint)grid_size, (cl_uint)block_size,
							 ptask->kern.prep_shmem_blocksz +
							 ptask->kern.prep_shmem_unitsz * block_size)));
		elog(DEBUG2, "PL/CUDA prep-kernel: grid=%u block=%u shmem=%zu",
			 (cl_uint)grid_size, (cl_uint)block_size,
			 ptask->kern.prep_shmem_blocksz +
			 ptask->kern.prep_shmem_unitsz * block_size);
	}

	/* launch plcuda_main_kernel_entrypoint */
	if (ptask->kern.main_kern_blocksz > 0)
	{
		block_size = (ptask->kern.main_kern_blocksz +
					  warp_size - 1) & ~(warp_size - 1);
		grid_size = (ptask->kern.main_num_threads +
					 block_size - 1) / block_size;
	}
	else
	{
		optimal_workgroup_size(&grid_size,
							   &block_size,
							   ptask->kern_plcuda_main,
							   gpuserv_cuda_device,
							   ptask->kern.main_num_threads,
							   ptask->kern.main_shmem_blocksz,
							   ptask->kern.main_shmem_unitsz);
	}

	rc = cuLaunchKernel(ptask->kern_plcuda_main,
						grid_size, 1, 1,
						block_size, 1, 1,
						ptask->kern.main_shmem_blocksz +
						ptask->kern.main_shmem_unitsz * block_size,
						cuda_stream,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed on cuLaunchKernel: %s", errorText(rc)),
				 errhint("main-kernel: grid=%u block=%u shmem=%zu",
						 (cl_uint)grid_size, (cl_uint)block_size,
						 ptask->kern.main_shmem_blocksz +
						 ptask->kern.main_shmem_unitsz * block_size)));
	elog(DEBUG2, "PL/CUDA main-kernel: grid=%u block=%u shmem=%zu",
		 (cl_uint)grid_size, (cl_uint)block_size,
		 ptask->kern.main_shmem_blocksz +
		 ptask->kern.main_shmem_unitsz * block_size);

	/* launch plcuda_post_kernel_entrypoint */
	if (ptask->exec_post_kernel)
    {
		if (ptask->kern.post_kern_blocksz > 0)
		{
			block_size = (ptask->kern.post_kern_blocksz +
						  warp_size - 1) & ~(warp_size - 1);
			grid_size = (ptask->kern.post_num_threads +
						 block_size - 1) / block_size;
		}
		else
		{
			optimal_workgroup_size(&grid_size,
								   &block_size,
								   ptask->kern_plcuda_post,
								   gpuserv_cuda_device,
								   ptask->kern.post_num_threads,
								   ptask->kern.post_shmem_blocksz,
								   ptask->kern.post_shmem_unitsz);
		}

		rc = cuLaunchKernel(ptask->kern_plcuda_post,
							grid_size, 1, 1,
							block_size, 1, 1,
							ptask->kern.post_shmem_blocksz +
							ptask->kern.post_shmem_unitsz * block_size,
							cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("failed on cuLaunchKernel: %s", errorText(rc)),
					 errhint("post-kernel: grid=%u block=%u shmem=%zu",
							 (cl_uint)grid_size, (cl_uint)block_size,
							 ptask->kern.post_shmem_blocksz +
							 ptask->kern.post_shmem_unitsz * block_size)));
		elog(DEBUG2, "PL/CUDA post-kernel: grid=%u block=%u shmem=%zu",
			 (cl_uint)grid_size, (cl_uint)block_size,
			 ptask->kern.post_shmem_blocksz +
			 ptask->kern.post_shmem_unitsz * block_size);
	}

	/* write back the control block */
	rc = cuMemcpyDtoHAsync(&ptask->kern,
						   ptask->m_kern_plcuda,
						   KERN_PLCUDA_DMARECV_LENGTH(&ptask->kern),
						   cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoHAsync: %s", errorText(rc));

	/* write back the result buffer, if any */
	if (ptask->m_results_buf != 0UL)
	{
		Assert(ptask->h_results_buf != NULL);
		rc = cuMemcpyDtoHAsync(ptask->h_results_buf,
							   ptask->m_results_buf,
							   ptask->kern.results_bufsz,
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyDtoHAsync: %s", errorText(rc));
	}

	/* callback registration */
	rc = cuStreamAddCallback(cuda_stream,
							 plcuda_respond_task,
							 ptask, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));
	return 0;

out_of_resource:
	plcuda_cleanup_cuda_resources(ptask);
	return 1;	/* retry later */
}

int
plcuda_process_task(GpuTask_v2 *gtask,
					CUmodule cuda_module,
					CUstream cuda_stream)
{
	plcudaTask *ptask = (plcudaTask *) gtask;
	int			retval;

	PG_TRY();
	{
		retval = __plcuda_process_task(ptask, cuda_module, cuda_stream);
	}
	PG_CATCH();
	{
		plcuda_cleanup_cuda_resources((plcudaTask *) gtask);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return retval;
}

/*
 * plcuda_complete_task
 */
int
plcuda_complete_task(GpuTask_v2 *gtask)
{
	plcudaTask	   *ptask = (plcudaTask *) gtask;

	plcuda_cleanup_cuda_resources(ptask);

	return 0;
}

/*
 * plcuda_release_task
 */
void
plcuda_release_task(GpuTask_v2 *gtask)
{
	plcudaTask	   *ptask = (plcudaTask *) gtask;

	if (ptask->h_results_buf)
		dmaBufferFree(ptask->h_results_buf);
	dmaBufferFree(ptask);
}

/*
 * pgstrom_init_plcuda
 */
void
pgstrom_init_plcuda(void)
{
	dlist_init(&plcuda_state_list);
	RegisterResourceReleaseCallback(plcuda_cleanup_resources, NULL);
}
