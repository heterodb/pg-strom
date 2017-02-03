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
#include "utils/rel.h"
#include "utils/syscache.h"
#include "pg_strom.h"
#include "cuda_plcuda.h"

#if 0
#define PLCUDAINFO_EXNODE_NAME	"plcudaInfo"

typedef struct plcudaInfo
{
	ExtensibleNode	ex;
	/* kernel requirement */
	cl_uint	extra_flags;
	/* kernel declarations */
	char   *kern_decl;
	/* kernel prep function */
	char   *kern_prep;
	Oid		fn_prep_kern_blocksz;
	long	val_prep_kern_blocksz;
	Oid		fn_prep_num_threads;
	Size	val_prep_num_threads;
	Oid		fn_prep_shmem_unitsz;
	Size	val_prep_shmem_unitsz;
	Oid		fn_prep_shmem_blocksz;
	Size	val_prep_shmem_blocksz;

	/* kernel function */
	char   *kern_main;
	Oid		fn_main_kern_blocksz;
	long	val_main_kern_blocksz;
	Oid		fn_main_num_threads;
	Size	val_main_num_threads;
	Oid		fn_main_shmem_unitsz;
	Size	val_main_shmem_unitsz;
	Oid		fn_main_shmem_blocksz;
	Size	val_main_shmem_blocksz;

	/* kernel post function */
    char   *kern_post;
	Oid		fn_post_kern_blocksz;
	long	val_post_kern_blocksz;
	Oid		fn_post_num_threads;
	Size	val_post_num_threads;
	Oid		fn_post_shmem_unitsz;
	Size	val_post_shmem_unitsz;
	Oid		fn_post_shmem_blocksz;
	Size	val_post_shmem_blocksz;

	/* device memory size for working buffer */
	Oid		fn_working_bufsz;
	long	val_working_bufsz;

	/* device memory size for result buffer */
	Oid		fn_results_bufsz;
	long	val_results_bufsz;

	/* comprehensive functions */
	Oid		fn_sanity_check;
	Oid		fn_cpu_fallback;
} plcudaInfo;
#endif

/*
 * plcudaState - runtime state of pl/cuda functions
 */
typedef struct plcudaState
{
	dlist_node		chain;
	ResourceOwner	owner;
	GpuContext	   *gcontext;
	plcudaInfo		cf_info;
	kern_plcuda	   *kplcuda_head;	/* template */
	/* GPU resources */
	cl_int			cuda_index;
	CUfunction		kern_prep;
	CUfunction		kern_main;
	CUfunction		kern_post;
	CUmodule	   *cuda_modules;
} plcudaState;



/*
 * plcudaTaskState
 */
typedef struct plcudaTaskState
{
	GpuTaskState_v2	gts;		/* dummy */
	/* kernel requirement */
	cl_uint	extra_flags;
	List   *extra_includes;		/* OID of extra include functions */
	/* kernel declarations */
	char   *kern_decl;
	/* kernel prep function */
	char   *kern_prep;
	Oid		fn_prep_kern_blocksz;
	long	val_prep_kern_blocksz;
	Oid		fn_prep_num_threads;
	Size	val_prep_num_threads;
	Oid		fn_prep_shmem_unitsz;
	Size	val_prep_shmem_unitsz;
	Oid		fn_prep_shmem_blocksz;
	Size	val_prep_shmem_blocksz;

	/* kernel function */
	char   *kern_main;
	Oid		fn_main_kern_blocksz;
	long	val_main_kern_blocksz;
	Oid		fn_main_num_threads;
	Size	val_main_num_threads;
	Oid		fn_main_shmem_unitsz;
	Size	val_main_shmem_unitsz;
	Oid		fn_main_shmem_blocksz;
	Size	val_main_shmem_blocksz;

	/* kernel post function */
    char   *kern_post;
	Oid		fn_post_kern_blocksz;
	long	val_post_kern_blocksz;
	Oid		fn_post_num_threads;
	Size	val_post_num_threads;
	Oid		fn_post_shmem_unitsz;
	Size	val_post_shmem_unitsz;
	Oid		fn_post_shmem_blocksz;
	Size	val_post_shmem_blocksz;

	/* device memory size for working buffer */
	Oid		fn_working_bufsz;
	long	val_working_bufsz;

	/* device memory size for result buffer */
	Oid		fn_results_bufsz;
	long	val_results_bufsz;

	/* comprehensive functions */
	Oid		fn_sanity_check;
	Oid		fn_cpu_fallback;
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
static plcudaTaskState *plcuda_exec_begin(Form_pg_proc proc_form,
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
 * #plcuda_results_bufsz {<value>|<function>}     (default: 0)
 * #plcuda_working_bufsz {<value>|<function>}      (default: 0)
 * #plcuda_sanity_check {<function>}             (default: no fallback)
 * #plcuda_cpu_fallback {<function>}             (default: no fallback)
 */
static void
plcuda_code_validation(plcudaTaskState *plts,	/* dummy GTS */
					   Oid proowner,
					   oidvector *proargtypes,
					   Oid prorettype)
{
	StringInfoData	decl_src;
	StringInfoData	prep_src;
	StringInfoData	main_src;
	StringInfoData	post_src;
	StringInfoData	emsg;
	StringInfo		curr = NULL;
	char		   *line;
	int				lineno;
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
#define HELPER_PRIV_CHECK(fn_oid, ownership)						\
	(!OidIsValid(fn_oid) ||											\
	 (OidIsValid(ownership)											\
	  ? (pg_proc_ownercheck((fn_oid),(ownership)))					\
	  : (pg_proc_aclcheck((fn_oid),GetUserId(),ACL_EXECUTE) == ACLCHECK_OK)))

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
				else
					EMSG("\"%s\" was not a valid value or function",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_include") == 0)
			{
				if (has_decl_block ||
					has_prep_block ||
					has_main_block ||
					has_post_block)
				{
					EMSG("%s must appear prior to the code block:\n%s",
						 cmd, line);
				}
				else
				{
					cl_uint		extra_flags = 0;

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
					}

					if (extra_flags != 0)
						plts->extra_flags |= extra_flags;
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
								plts->extra_includes =
									lappend_oid(plts->extra_includes,
												fn_extra_include);
							}
							else
								EMSG("permission denied on helper function %s",
									 NameListToString(options));
						}
						else
						{
							EMSG("\"%s\" was not a valid function name",
								 ident_to_cstring(options));
						}
					}
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

	plts->kern_decl = decl_src.data;
	plts->kern_prep = prep_src.data;
	plts->kern_main = main_src.data;
	plts->kern_post = post_src.data;
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
__plcuda_codegen(StringInfo kern,
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
		"__plcuda_%s%s(kern_plcuda *kplcuda,\n"
		"              void *workbuf,\n"
		"              void *results,\n"
		"              kern_context *kcxt)\n"
		"{\n",
		NameStr(procForm->proname), suffix);

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
		"plcuda_%s%s(kern_plcuda *kplcuda,\n"
		"            void *workbuf,\n"
		"            void *results)\n"
		"{\n"
		"  kern_parambuf *kparams = KERN_PLCUDA_PARAMBUF(kplcuda);\n"
		"  kern_context kcxt;\n"
		"\n"
		"  assert(kplcuda->nargs == kparams->nparams);\n",
		kernel_maxthreads ? "_MAXTHREADS" : "",
		NameStr(procForm->proname), suffix);

	if (last_suffix)
		appendStringInfo(
			kern,
			"  if (kplcuda->kerror%s.errcode != StromError_Success)\n"
			"    kcxt.e = kplcuda->kerror%s;\n"
			"  else\n"
			"  {\n"
			"    INIT_KERNEL_CONTEXT(&kcxt,plcuda%s_kernel,kparams);\n"
			"    __plcuda_%s%s(kplcuda, workbuf, results, &kcxt);\n"
			"  }\n",
			last_suffix,
			last_suffix,
			suffix,
			NameStr(procForm->proname), suffix);
	else
		appendStringInfo(
			kern,
			"  INIT_KERNEL_CONTEXT(&kcxt,plcuda%s_kernel,kparams);\n"
			"  __plcuda_%s%s(kplcuda, workbuf, results, &kcxt);\n",
			suffix,
			NameStr(procForm->proname), suffix);

	appendStringInfo(
		kern,
		"  kern_writeback_error_status(&kplcuda->kerror%s, kcxt.e);\n"
		"}\n\n",
		suffix);
}

static char *
plcuda_codegen(Form_pg_proc procForm,
			   plcudaInfo *cf_info)
{
	StringInfoData	kern;
	const char	   *last_stage = NULL;

	initStringInfo(&kern);

	if (cf_info->kern_decl)
		appendStringInfo(&kern, "%s\n", cf_info->kern_decl);
	if (cf_info->kern_prep)
	{
		__plcuda_codegen(&kern, "_prep",
						 cf_info->kern_prep,
						 (OidIsValid(cf_info->fn_prep_kern_blocksz) ||
						  cf_info->val_prep_kern_blocksz > 0),
						 procForm,
						 last_stage);
		last_stage = "_prep";
	}
	if (cf_info->kern_main)
	{
		__plcuda_codegen(&kern, "_main",
						 cf_info->kern_main,
						 (OidIsValid(cf_info->fn_main_kern_blocksz) ||
						  cf_info->val_main_kern_blocksz > 0),
						 procForm,
						 last_stage);
		last_stage = "_main";
	}
	if (cf_info->kern_post)
	{
		__plcuda_codegen(&kern, "_post",
						 cf_info->kern_post,
						 (OidIsValid(cf_info->fn_post_kern_blocksz) ||
						  cf_info->val_post_kern_blocksz > 0),
						 procForm,
						 last_stage);
		last_stage = "_post";
	}
	return kern.data;
}

static void
__plcuda_cleanup_resources(plcudaState *state)
{
	GpuContext	   *gcontext = state->gcontext;
	cl_uint			i, ndevs = gcontext->num_context;

	for (i=0; i < ndevs; i++)
	{
		CUresult	rc;

		if (!state->cuda_modules[i])
			continue;

		rc = cuModuleUnload(state->cuda_modules[i]);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuModuleUnload: %s", errorText(rc));
	}
	pfree(state->cuda_modules);
	pfree(state->kplcuda_head);
	pfree(state);

	pgstrom_put_gpucontext(gcontext);
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
		plcudaState	   *state = (plcudaState *)
			dlist_container(plcudaState, chain, iter.cur);

		if (state->owner == CurrentResourceOwner)
		{
			/* detach state */
			dlist_delete(&state->chain);
			__plcuda_cleanup_resources(state);
		}
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
plcuda_exec_begin(Form_pg_proc procForm, FunctionCallInfo fcinfo)
{
	GpuContext	   *gcontext = pgstrom_get_gpucontext();
	plcudaTaskState *plts;
	char		   *kern_source;
	CUmodule	   *cuda_modules;
	plcudaState	   *state;
	kern_plcuda	   *kplcuda;
	char			namebuf[NAMEDATALEN + 40];
	CUresult		rc;
	Size			length;
	int				i;

	/* setup a dummy GTS for PL/CUDA */
	plts = palloc0(sizeof(plcudaTaskState));
	pgstromInitGpuTaskState(&plts->gts,
							gcontext,
							GpuTaskKind_PL_CUDA,
							NIL,	/* no constants */
							NULL);	/* no EState */

	/* validate PL/CUDA source code */
	prosrc = SysCacheGetAttr(PROCOID, tuple, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "Bug? no program source was supplied");
	plcuda_code_validation(plts,
						   !fcinfo ? procForm->proowner : InvalidOid,
						   &procForm->proargtypes
						   procForm->prorettype,
						   TextDatumGetCString(prosrc));
	/* construct a flat kernel source to be built */
	kern_source = plcuda_codegen(procForm, plts);
	program_id = pgstrom_create_cuda_program(gcontext,
											 plts->extra_flags,
											 plts->kern_source,
											 kern_define,
											 with_connection);
	plts->gts.program_id = program_id;

	wait_for_cuda_program_build(program_id);






	cuda_modules = plcuda_load_cuda_program_legacy(gcontext,
												   kern_source,
												   cf_info->extra_flags);
	/* construct plcudaState */
	state = MemoryContextAllocZero(gcontext->memcxt,
								   sizeof(plcudaState));
	state->owner = CurrentResourceOwner;
	state->gcontext = gcontext;
	memcpy(&state->cf_info, cf_info, sizeof(plcudaInfo));
	state->cf_info.kern_decl = NULL;
	state->cf_info.kern_prep = NULL;
	state->cf_info.kern_main = NULL;
	state->cf_info.kern_post = NULL;
	state->cuda_modules = cuda_modules;

	/* build template of kern_plcuda */
	length = offsetof(kern_plcuda, argmeta[procForm->pronargs]);
	kplcuda = MemoryContextAllocZero(gcontext->memcxt, length);

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
	state->kplcuda_head = kplcuda;

	/* track state */
	dlist_push_head(&plcuda_state_list, &state->chain);

#if 0
	/* resolve kernel functions */
	i = (gcontext->next_context++ % gcontext->num_context);
	state->cuda_index = i;

	rc = cuCtxPushCurrent(gcontext->gpu[i].cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

	if (cf_info->kern_prep)
	{
		snprintf(namebuf, sizeof(namebuf), "plcuda_%s_prep",
				 NameStr(procForm->proname));
		rc = cuModuleGetFunction(&state->kern_prep,
								 state->cuda_modules[i],
								 namebuf);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));
	}

	if (cf_info->kern_main)
	{
		snprintf(namebuf, sizeof(namebuf), "plcuda_%s_main",
				 NameStr(procForm->proname));
		rc = cuModuleGetFunction(&state->kern_main,
								 state->cuda_modules[i],
								 namebuf);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));
	}

	if (cf_info->kern_post)
	{
		snprintf(namebuf, sizeof(namebuf), "plcuda_%s_post",
				 NameStr(procForm->proname));
		rc = cuModuleGetFunction(&state->kern_post,
								 state->cuda_modules[i],
								 namebuf);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));
	}

	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

	return state;
#endif
	return plts;
}

static void
plcuda_exec_end(plcudaState *state)
{
	dlist_delete(&state->chain);
	__plcuda_cleanup_resources(state);
}

Datum
plcuda_function_validator(PG_FUNCTION_ARGS)
{
	Oid				func_oid = PG_GETARG_OID(0);
	Relation		rel;
	HeapTuple		tuple;
	HeapTuple		newtup;
	Form_pg_proc	procForm;
	Datum			prosrc;
	bool			isnull;
	plcudaTaskState	plts;
	devtype_info   *dtype;
	char		   *source;
	cl_uint			extra_flags = DEVKERNEL_NEEDS_PLCUDA;
	cl_uint			i;
	plcudaState	   *state;
	ObjectAddress	myself;
	ObjectAddress	referenced;

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

	if (!pgstrom_devtype_lookup(procForm->prorettype))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Result type \"%s\" is not device executable",
						format_type_be(procForm->prorettype))));

	Assert(procForm->pronargs == procForm->proargtypes.dim1);
	for (i=0; i < procForm->pronargs; i++)
	{
		Oid		argtype_oid = procForm->proargtypes.values[i];

		if (!pgstrom_devtype_lookup(argtype_oid))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("Argument type \"%s\" is not device executable",
							format_type_be(argtype_oid))));
	}
	/* only validation of the CUDA code */
	plts = plcuda_exec_begin(procForm, NULL);
	plcuda_exec_end(plts);





	/*
	 * Do syntax checks and construction of plcudaInfo
	 */
	memset(&plts, 0, sizeof(plcudaTaskState));
	plts.extra_flags = extra_flags;
	plts.val_prep_kern_blocksz = 0;	/* default; performance optimal */
	plts.val_main_kern_blocksz = 0;	/* default; performance optimal */
	plts.val_post_kern_blocksz = 0;	/* default; performance optimal */
	plts.val_prep_num_threads = 1;	/* default */
	plts.val_main_num_threads = 1;	/* default */
	plts.val_post_num_threads = 1;	/* default */

	plcuda_code_validation(&plts,
						   procForm->proowner,
						   &procForm->proargtypes
						   procForm->prorettype,
						   TextDatumGetCString(prosrc));



	state = plcuda_exec_begin(procForm, NULL);	/* validation only */
	plcuda_exec_end(state);



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
	Datum		result;

	if (!OidIsValid(helper_func_oid))
		return static_config;

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

static kern_plcuda *
__build_kern_plcuda(FunctionCallInfo fcinfo,
					plcudaState *state,
					Size working_bufsz,
					Size results_bufsz)
{
	GpuContext	   *gcontext = state->gcontext;
	kern_plcuda	   *kplcuda_head = state->kplcuda_head;
	kern_plcuda	   *kplcuda;
	kern_parambuf  *kparams;
	Size			head_length;
	Size			total_length;
	Size			offset;
	int				i;

	/* calculation of the length */
	Assert(fcinfo->nargs == kplcuda_head->nargs);
	head_length = offsetof(kern_plcuda, argmeta[fcinfo->nargs]);
	total_length = STROMALIGN(head_length) +
		STROMALIGN(offsetof(kern_parambuf, poffset[fcinfo->nargs]));

	for (i=0; i < fcinfo->nargs; i++)
	{
		kern_colmeta	cmeta = kplcuda_head->argmeta[i];

		if (fcinfo->argnull[i])
			continue;
		if (cmeta.attlen > 0)
			total_length += MAXALIGN(cmeta.attlen);
		else
			total_length += MAXALIGN(toast_raw_datum_size(fcinfo->arg[i]));
	}
	total_length = STROMALIGN(total_length);

	/* setup kern_plcuda to be launched */
	kplcuda = MemoryContextAlloc(gcontext->memcxt, total_length);
	memcpy(kplcuda, kplcuda_head, head_length);
	kplcuda->working_bufsz = working_bufsz;
	kplcuda->working_usage = 0UL;
	kplcuda->results_bufsz = results_bufsz;
	kplcuda->results_usage = 0UL;
	kplcuda->total_length = total_length;

	/* copy function argument to DMA buffer */
	kparams = KERN_PLCUDA_PARAMBUF(kplcuda);
	kparams->hostptr = (hostptr_t) kparams;
	kparams->xactStartTimestamp = GetCurrentTransactionStartTimestamp();

	offset = STROMALIGN(offsetof(kern_parambuf,
								 poffset[fcinfo->nargs]));
	for (i=0; i < fcinfo->nargs; i++)
	{
		kern_colmeta	cmeta = kplcuda_head->argmeta[i];

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

	Assert(STROMALIGN(offsetof(kern_plcuda,
							   argmeta[fcinfo->nargs])) +
		   kparams->length == kplcuda->total_length);
	return kplcuda;
}

static Datum
__launch_plcuda_kernels(plcudaState *state,
						kern_plcuda *kplcuda,
						CUdeviceptr m_kern_plcuda,
						CUdeviceptr m_working_buf,
						CUdeviceptr m_results_buf,
						char *h_results_buf,
						kern_errorbuf *p_kerror,
						bool *p_isnull)
{
	GpuContext *gcontext = state->gcontext;
	CUstream	stream = NULL;
	void	   *kern_args[5];
	size_t		grid_size;
	size_t		block_size;
	int			warp_size;
	CUdevice	cuda_device;
	CUresult	rc;
	Datum		retval;

	/* device to be used */
	cuda_device = gcontext->gpu[state->cuda_index].cuda_device;

	rc = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuStreamCreate: %s", errorText(rc));

	rc = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
							  cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));
	Assert((warp_size & (warp_size - 1)) == 0);

	PG_TRY();
	{
		/* control block + argument buffer */
		rc = cuMemcpyHtoDAsync(m_kern_plcuda,
							   kplcuda,
							   KERN_PLCUDA_DMASEND_LENGTH(kplcuda),
							   stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));

		/* kernel arguments (common for all three kernels) */
		kern_args[0] = &m_kern_plcuda;
		kern_args[1] = &m_working_buf;
		kern_args[2] = &m_results_buf;

		/* kernel launch of the prep function */
		if (state->kern_prep)
		{
			if (kplcuda->prep_kern_blocksz > 0)
			{
				block_size = (kplcuda->prep_kern_blocksz +
							  warp_size - 1) & ~(warp_size - 1);
				grid_size = (kplcuda->prep_num_threads +
							 block_size - 1) / block_size;
			}
			else
			{
				optimal_workgroup_size(&grid_size,
									   &block_size,
									   state->kern_prep,
									   cuda_device,
									   kplcuda->prep_num_threads,
									   kplcuda->prep_shmem_blocksz,
									   kplcuda->prep_shmem_unitsz);
			}

			rc = cuLaunchKernel(state->kern_prep,
								grid_size, 1, 1,
								block_size, 1, 1,
								kplcuda->prep_shmem_blocksz +
								kplcuda->prep_shmem_unitsz * block_size,
								stream,
								kern_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("failed on cuLaunchKernel: %s", errorText(rc)),
						 errhint("prep-kernel: grid=%u block=%u shmem=%zu",
								 (cl_uint)grid_size, (cl_uint)block_size,
								 kplcuda->prep_shmem_blocksz +
								 kplcuda->prep_shmem_unitsz * block_size)));

			elog(DEBUG2, "PL/CUDA prep-kernel: grid=%u block=%u shmem=%zu",
				 (cl_uint)grid_size, (cl_uint)block_size,
				 kplcuda->prep_shmem_blocksz +
				 kplcuda->prep_shmem_unitsz * block_size);
		}

		/* kernel launch of the main function */
		if (kplcuda->main_kern_blocksz > 0)
		{
			block_size = (kplcuda->main_kern_blocksz +
						  warp_size - 1) & ~(warp_size - 1);
			grid_size = (kplcuda->main_num_threads +
						 block_size - 1) / block_size;
		}
		else
		{
			optimal_workgroup_size(&grid_size,
								   &block_size,
								   state->kern_main,
								   cuda_device,
								   kplcuda->main_num_threads,
								   kplcuda->main_shmem_blocksz,
								   kplcuda->main_shmem_unitsz);
		}

		rc = cuLaunchKernel(state->kern_main,
							grid_size, 1, 1,
							block_size, 1, 1,
							kplcuda->main_shmem_blocksz +
							kplcuda->main_shmem_unitsz * block_size,
							stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("failed on cuLaunchKernel: %s", errorText(rc)),
					 errhint("main-kernel: grid=%u block=%u shmem=%zu",
							 (cl_uint)grid_size, (cl_uint)block_size,
							 kplcuda->main_shmem_blocksz +
							 kplcuda->main_shmem_unitsz * block_size)));

		elog(DEBUG2, "PL/CUDA main-kernel: grid=%u block=%u shmem=%zu",
			 (cl_uint)grid_size, (cl_uint)block_size,
		   	 kplcuda->main_shmem_blocksz +
		   	 kplcuda->main_shmem_unitsz * block_size);

		/* kernel launch of the post function */
		if (state->kern_post)
		{
			if (kplcuda->post_kern_blocksz > 0)
			{
				block_size = (kplcuda->post_kern_blocksz +
							  warp_size - 1) & ~(warp_size - 1);
				grid_size = (kplcuda->post_num_threads +
							 block_size - 1) / block_size;
			}
			else
			{
				optimal_workgroup_size(&grid_size,
									   &block_size,
									   state->kern_post,
									   cuda_device,
									   kplcuda->post_num_threads,
									   kplcuda->post_shmem_blocksz,
									   kplcuda->post_shmem_unitsz);
			}
			rc = cuLaunchKernel(state->kern_post,
								grid_size, 1, 1,
								block_size, 1, 1,
								kplcuda->post_shmem_blocksz +
								kplcuda->post_shmem_unitsz * block_size,
								stream,
								kern_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("failed on cuLaunchKernel: %s", errorText(rc)),
						 errhint("post-kernel: grid=%u block=%u shmem=%zu",
								 (cl_uint)grid_size, (cl_uint)block_size,
								 kplcuda->post_shmem_blocksz +
								 kplcuda->post_shmem_unitsz * block_size)));

			elog(DEBUG2, "PL/CUDA post-kernel: grid=%u block=%u shmem=%zu",
				 (cl_uint)grid_size, (cl_uint)block_size,
				 kplcuda->post_shmem_blocksz +
				 kplcuda->post_shmem_unitsz * block_size);
		}

		/* write back the control block */
		rc = cuMemcpyDtoHAsync (kplcuda,
								m_kern_plcuda,
								KERN_PLCUDA_DMARECV_LENGTH(kplcuda),
								stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));

		/*
		 * write back the result buffer, if any
		 * (note that h_results_buf is not pinned, thus sync DMA is used)
		 */
		if (h_results_buf)
		{
			rc = cuMemcpyDtoH(h_results_buf,
							  m_results_buf,
							  kplcuda->results_bufsz);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemcpyDtoH: %s", errorText(rc));
			elog(DEBUG2, "PL/CUDA results DMA %zu bytes",
				 kplcuda->results_bufsz);
		}
	}
	PG_CATCH();
	{
		/* ensure concurrent jobs are done */
		rc = cuStreamSynchronize(stream);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuStreamSynchronize: %s", errorText(rc));

		rc = cuStreamDestroy(stream);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuStreamDestroy: %s", errorText(rc));

		rc = cuCtxPopCurrent(NULL);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

		PG_RE_THROW();
	}
	PG_END_TRY();

	/* wait for completion of the jobs */
	rc = cuStreamSynchronize(stream);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuStreamSynchronize: %s", errorText(rc));

	rc = cuStreamDestroy(stream);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuStreamDestroy: %s", errorText(rc));

	/* construction of the result value */
	if (state->kern_post != NULL &&
		kplcuda->kerror_post.errcode != StromError_Success)
	{
		*p_kerror = kplcuda->kerror_post;
	}
	else if (kplcuda->kerror_main.errcode != StromError_Success)
	{
		*p_kerror = kplcuda->kerror_main;
	}
	else if (kplcuda->retmeta.attlen > 0)
	{
		if (kplcuda->__retval[kplcuda->retmeta.attlen])
			*p_isnull = true;
		else if (kplcuda->retmeta.attbyval)
		{
			/* inline fixed-length variable */
			*p_isnull = false;
			memcpy(&retval, kplcuda->__retval, kplcuda->retmeta.attlen);
		}
		else
		{
			/* indirect fixed-length variable */
			*p_isnull = false;
			retval = PointerGetDatum(pnstrdup(kplcuda->__retval,
											  kplcuda->retmeta.attlen));
		}
	}
	else
	{
		Assert(!kplcuda->retmeta.attbyval);
		if (kplcuda->__retval[sizeof(CUdeviceptr)])
			*p_isnull = true;
		else if (m_results_buf == 0UL)
			elog(ERROR, "non-NULL result in spite of no results buffer");
		else
		{
			/* varlena datum that referenced h_resultbuf */
			CUdeviceptr	p = *((CUdeviceptr *)kplcuda->__retval);

			if (p != m_results_buf)
				elog(ERROR, "kernel 'retval' didn't point result buffer");
			*p_isnull = false;
			retval = PointerGetDatum(h_results_buf);
		}
	}
	return retval;
}

Datum
plcuda_function_handler(PG_FUNCTION_ARGS)
{
	FmgrInfo	   *flinfo = fcinfo->flinfo;
	plcudaState	   *state;
	plcudaInfo	   *cf_info;
	Size			working_bufsz;
	Size			results_bufsz;
	kern_plcuda	   *kplcuda;
	kern_errorbuf	kerror;
	CUcontext		cuda_context;
	CUdeviceptr		m_kern_plcuda = 0UL;
	CUdeviceptr		m_working_buf = 0UL;
	CUdeviceptr		m_results_buf = 0UL;
	char		   *h_results_buf = NULL;
	Datum			retval;
	bool			isnull;
	CUresult		rc;

	if (flinfo->fn_extra)
		state = (plcudaState *) flinfo->fn_extra;
	else
	{
		MemoryContext	oldcxt;
		HeapTuple		tuple;

		tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(flinfo->fn_oid));
		if (!HeapTupleIsValid(tuple))
			elog(ERROR, "cache lookup failed for function %u",
				 flinfo->fn_oid);

		oldcxt = MemoryContextSwitchTo(flinfo->fn_mcxt);
		plts = plcuda_exec_begin((Form_pg_proc) GETSTRUCT(tuple), fcinfo);
		MemoryContextSwitchTo(oldcxt);

		ReleaseSysCache(tuple);
		flinfo->fn_extra = state;
	}
	cf_info = &state->cf_info;

	/* sanitycheck of the supplied arguments, prior to GPU launch */
	if (!DatumGetBool(kernel_launch_helper(fcinfo,
										   cf_info->fn_sanity_check,
										   BoolGetDatum(true),
										   NULL)))
		elog(ERROR, "function '%s' argument sanity check failed",
			 format_procedure(cf_info->fn_sanity_check));

	/* determine the kernel launch parameters */
	working_bufsz =
		DatumGetInt64(kernel_launch_helper(fcinfo,
										   cf_info->fn_working_bufsz,
										   cf_info->val_working_bufsz,
										   NULL));
	results_bufsz =
		DatumGetInt64(kernel_launch_helper(fcinfo,
										   cf_info->fn_results_bufsz,
										   cf_info->val_results_bufsz,
										   NULL));
	elog(DEBUG2, "working_bufsz = %zu, results_bufsz = %zu",
		 working_bufsz, results_bufsz);

	/* make kern_plcuda structure */
	kplcuda = __build_kern_plcuda(fcinfo, state,
								  working_bufsz,
								  results_bufsz);

	if (state->kern_prep)
	{
		int64		v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_prep_num_threads,
											   cf_info->val_prep_num_threads,
											   NULL));
		if (v <= 0)
			elog(ERROR, "kern_prep: invalid number of threads: %ld", v);
		kplcuda->prep_num_threads = (cl_ulong)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_prep_kern_blocksz,
											   cf_info->val_prep_kern_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_prep: invalid kernel block size: %ld", v);
		kplcuda->prep_kern_blocksz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_prep_shmem_unitsz,
											   cf_info->val_prep_shmem_unitsz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_prep: invalid shared memory required: %ld", v);
		kplcuda->prep_shmem_unitsz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_prep_shmem_blocksz,
											   cf_info->val_prep_shmem_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_prep: invalid shared memory required: %ld", v);
		kplcuda->prep_shmem_blocksz = v;

		elog(DEBUG2, "kern_prep {blocksz=%u, nitems=%lu, shmem=%u,%u}",
			 kplcuda->prep_kern_blocksz,
			 kplcuda->prep_num_threads,
			 kplcuda->prep_shmem_unitsz,
			 kplcuda->prep_shmem_blocksz);
	}

	if (state->kern_main)
	{
		int64		v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_main_num_threads,
											   cf_info->val_main_num_threads,
											   NULL));
		if (v <= 0)
			elog(ERROR, "kern_main: invalid number of threads: %ld", v);
		kplcuda->main_num_threads = (cl_ulong)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_main_kern_blocksz,
											   cf_info->val_main_kern_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_main: invalid kernel block size: %ld", v);
		kplcuda->main_kern_blocksz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_main_shmem_unitsz,
											   cf_info->val_main_shmem_unitsz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_main: invalid shared memory required: %ld", v);
		kplcuda->main_shmem_unitsz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_main_shmem_blocksz,
											   cf_info->val_main_shmem_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_main: invalid shared memory required: %ld", v);
		kplcuda->main_shmem_blocksz = v;

		elog(DEBUG2, "kern_main {blocksz=%u, nitems=%lu, shmem=%u,%u}",
			 kplcuda->main_kern_blocksz,
			 kplcuda->main_num_threads,
			 kplcuda->main_shmem_unitsz,
			 kplcuda->main_shmem_blocksz);
	}

	if (state->kern_post)
	{
		int64		v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_post_num_threads,
											   cf_info->val_post_num_threads,
											   NULL));
		if (v <= 0)
			elog(ERROR, "kern_post: invalid number of threads: %ld", v);
		kplcuda->post_num_threads = (cl_ulong)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_post_kern_blocksz,
											   cf_info->val_post_kern_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_post: invalid kernel block size: %ld", v);
		kplcuda->post_kern_blocksz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_post_shmem_unitsz,
											   cf_info->val_post_shmem_unitsz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_post: invalid shared memory required: %ld", v);
		kplcuda->post_shmem_unitsz = (cl_uint)v;

		v = DatumGetInt64(kernel_launch_helper(fcinfo,
											   cf_info->fn_post_shmem_blocksz,
											   cf_info->val_post_shmem_blocksz,
											   NULL));
		if (v < 0 || v > INT_MAX)
			elog(ERROR, "kern_post: invalid shared memory required: %ld", v);
		kplcuda->post_shmem_blocksz = v;

		elog(DEBUG2, "kern_post {blocksz=%u, nitems=%lu, shmem=%u,%u}",
			 kplcuda->post_kern_blocksz,
			 kplcuda->post_num_threads,
			 kplcuda->post_shmem_unitsz,
			 kplcuda->post_shmem_blocksz);
	}

	/* set context */
	cuda_context = state->gcontext->gpu[state->cuda_index].cuda_context;
	rc = cuCtxPushCurrent(cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

	PG_TRY();
	{
		/* kern_plcuda structure on the device side */
		m_kern_plcuda = __gpuMemAlloc(state->gcontext,
									  state->cuda_index,
									  kplcuda->total_length);
		if (m_kern_plcuda == 0UL)
			elog(ERROR, "out of device memory; %u bytes required",
				 kplcuda->total_length);

		/* working buffer on the device side */
		if (working_bufsz > 0)
		{
			m_working_buf = __gpuMemAlloc(state->gcontext,
										  state->cuda_index,
										  working_bufsz);
			if (m_working_buf == 0UL)
				elog(ERROR, "out of device memory; %zu bytes required",
					 working_bufsz);
		}

		/* results buffer on the device side */
		if (results_bufsz > 0)
		{
			m_results_buf = __gpuMemAlloc(state->gcontext,
										  state->cuda_index,
										  results_bufsz);
			if (m_results_buf == 0UL)
				elog(ERROR, "out of device memory; %zu bytes required",
					 results_bufsz);

			/*
			 * NOTE: We allocate host-side result buffer on the current
			 * memory context (usually, per tuple), because we have no
			 * way to release host pinned buffer until end of the query
			 * execution.
			 * The current version of PL/CUDA does not support asynchronous
			 * data transfer, thus, its performance penalty is not so much.
			 */
			h_results_buf = palloc(results_bufsz);
		}

		/*
		 * OK, launch a series of CUDA kernels (synchronous invocation)
		 */
		memset(&kerror, 0, sizeof(kern_errorbuf));
		retval = __launch_plcuda_kernels(state, kplcuda,
										 m_kern_plcuda,
										 m_working_buf,
										 m_results_buf,
										 h_results_buf,
										 &kerror,
										 &isnull);
	}
	PG_CATCH();
	{
		__gpuMemFree(state->gcontext,
					 state->cuda_index,
					 m_kern_plcuda);
		if (m_working_buf != 0UL)
			__gpuMemFree(state->gcontext,
						 state->cuda_index,
						 m_working_buf);
		if (m_results_buf != 0UL)
			__gpuMemFree(state->gcontext,
						 state->cuda_index,
						 m_results_buf);

		rc = cuCtxPopCurrent(NULL);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

		PG_RE_THROW();
	}
	PG_END_TRY();

	/* release GPU resources */
	__gpuMemFree(state->gcontext, state->cuda_index, m_kern_plcuda);
	if (m_working_buf != 0UL)
		__gpuMemFree(state->gcontext, state->cuda_index, m_working_buf);
	if (m_results_buf != 0UL)
		__gpuMemFree(state->gcontext, state->cuda_index, m_results_buf);

	/* restore context */
	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

	/*
	 * Dump the debug counter if valid values are set by kernel function
	 */
	if (kplcuda->plcuda_debug_count0)
		elog(NOTICE, "PL/CUDA debug count0 => %lu",
			 kplcuda->plcuda_debug_count0);
	if (kplcuda->plcuda_debug_count1)
		elog(NOTICE, "PL/CUDA debug count1 => %lu",
			 kplcuda->plcuda_debug_count1);
	if (kplcuda->plcuda_debug_count2)
		elog(NOTICE, "PL/CUDA debug count2 => %lu",
			 kplcuda->plcuda_debug_count2);
	if (kplcuda->plcuda_debug_count3)
		elog(NOTICE, "PL/CUDA debug count3 => %lu",
			 kplcuda->plcuda_debug_count3);
	if (kplcuda->plcuda_debug_count4)
		elog(NOTICE, "PL/CUDA debug count4 => %lu",
			 kplcuda->plcuda_debug_count4);
	if (kplcuda->plcuda_debug_count5)
		elog(NOTICE, "PL/CUDA debug count5 => %lu",
			 kplcuda->plcuda_debug_count5);
	if (kplcuda->plcuda_debug_count6)
		elog(NOTICE, "PL/CUDA debug count6 => %lu",
			 kplcuda->plcuda_debug_count6);
	if (kplcuda->plcuda_debug_count7)
		elog(NOTICE, "PL/CUDA debug count7 => %lu",
			 kplcuda->plcuda_debug_count7);
	pfree(kplcuda);

	if (kerror.errcode != StromError_Success)
	{
		if (kerror.errcode == StromError_CpuReCheck &&
			OidIsValid(cf_info->fn_cpu_fallback))
		{
			/* CPU fallback, if any */
			retval = kernel_launch_helper(fcinfo,
										  cf_info->fn_cpu_fallback,
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
	Datum			probin;
	bool			isnull;
	char			vl_head[VARHDRSZ];
	plcudaInfo	   *cf_info;
	StringInfoData	str;

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

	probin = SysCacheGetAttr(PROCOID, tuple,
							 Anum_pg_proc_probin,
							 &isnull);
	if (isnull)
		elog(ERROR, "Bug? plcudaInfo was not built yet");
	cf_info = stringToNode(TextDatumGetCString(probin));

	/* construct source text */
	initStringInfo(&str);
	appendBinaryStringInfo(&str, vl_head, VARHDRSZ);	/* varlena head */

	appendStringInfo(&str, "#include \"cuda_common.h\"\n");
	if (cf_info->extra_flags & DEVKERNEL_NEEDS_DYNPARA)
		appendStringInfo(&str, "#include \"cuda_dynpara.h\"\n");
	if (cf_info->extra_flags & DEVKERNEL_NEEDS_MATRIX)
		appendStringInfo(&str, "#include \"cuda_matrix.h\"\n");
	if (cf_info->extra_flags & DEVKERNEL_NEEDS_TIMELIB)
		appendStringInfo(&str, "#include \"cuda_timelib.h\"\n");
	if (cf_info->extra_flags & DEVKERNEL_NEEDS_TEXTLIB)
		appendStringInfo(&str, "#include \"cuda_textlib.h\"\n");
	if (cf_info->extra_flags & DEVKERNEL_NEEDS_NUMERIC)
		appendStringInfo(&str, "#include \"cuda_numeric.h\"\n");
	if (cf_info->extra_flags & DEVKERNEL_NEEDS_MATHLIB)
		appendStringInfo(&str, "#include \"cuda_mathlib.h\"\n");
	if (cf_info->extra_flags & DEVKERNEL_NEEDS_MONEY)
		appendStringInfo(&str, "#include \"cuda_money.h\"\n");
	if (cf_info->extra_flags & (DEVKERNEL_NEEDS_GPUSCAN |
								DEVKERNEL_NEEDS_GPUJOIN |
								DEVKERNEL_NEEDS_GPUPREAGG |
								DEVKERNEL_NEEDS_GPUSORT))
		elog(WARNING, "Bug? PL/CUDA function needs logic routines");
	if (cf_info->extra_flags & DEVKERNEL_NEEDS_PLCUDA)
		appendStringInfo(&str, "#include \"cuda_plcuda.h\"\n");
	appendStringInfoChar(&str, '\n');

	appendStringInfoString(&str, plcuda_codegen(procForm, cf_info));
	SET_VARSIZE(str.data, str.len);

	ReleaseSysCache(tuple);

	PG_RETURN_TEXT_P(str.data);
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
plcuda_respond_task(CUstream stream, CUresult status, void *private))
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
	GpuContext	   *gcontext = ptask->task.gcontext;
	CUresult		rc;


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
						ptask->kern.total_length);
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
						ptask->kern.main_num_threads +
						ptask->kern.main_shmem_blocksz * block_size,
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

	if (ptask->task.kerror.errcode == StromError_CpuReCheck &&
		ptask->has_cpu_fallback)
	{
		ptask->task.cpu_fallback = true;
		memset(&ptask->task.kerror, 0, sizeof(kern_errorbuf));
	}
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
