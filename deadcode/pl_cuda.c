/*
 * pl_cuda.c
 *
 * PL/CUDA SQL function support
 * ----
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
#include "cuda_plcuda.h"

/*
 * plcudaCodeProperty
 */
typedef struct plcudaCodeProperty
{
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
	/* composite type descriptors, if any */
	List	   *composite_types;
} plcudaCodeProperty;

/*
 * plcudaTaskState
 */
typedef struct plcudaTaskState
{
	GpuTaskState	gts;		/* dummy */
	ResourceOwner	owner;
	dlist_node		chain;
	kern_plcuda	   *kplcuda_head;
	CUdeviceptr		last_results_buf;	/* results buffer last used */
	/* property of the code block */
	plcudaCodeProperty p;
	/* property of the PL/CUDA kernel functions */
	int				kprep_max_blocksz;
	int				kprep_static_shmsz;
	int				kprep_dynamic_shmsz;
	int				kprep_const_memsz;
	int				kprep_local_memsz;
	int				kmain_max_blocksz;
	int				kmain_static_shmsz;
	int				kmain_dynamic_shmsz;
	int				kmain_const_memsz;
	int				kmain_local_memsz;
	int				kpost_max_blocksz;
	int				kpost_static_shmsz;
	int				kpost_dynamic_shmsz;
	int				kpost_const_memsz;
	int				kpost_local_memsz;
} plcudaTaskState;

/*
 * plcudaTask
 */
typedef struct plcudaTask
{
	GpuTask			task;
	bool			exec_prep_kernel;
	bool			exec_post_kernel;
	bool			has_cpu_fallback;
	CUdeviceptr		m_results_buf;	/* results buffer as unified memory */
	List		   *gstore_oid_list;	/* OID of GpuStore foreign table */
	List		   *gstore_devptr_list;	/* CUdeviceptr of GpuStore */
	List		   *gstore_dindex_list;	/* Preferable dindex if any */
	kern_plcuda		kern;
} plcudaTask;

/* static functions */
static plcudaTaskState *plcuda_exec_begin(HeapTuple protup,
										  FunctionCallInfo fcinfo);
static void plcuda_exec_end(plcudaTaskState *plts);
static int  plcuda_process_task(GpuTask *gtask, CUmodule cuda_module);
static void plcuda_release_task(GpuTask *gtask);

/* SQL functions */
Datum plcuda_function_validator(PG_FUNCTION_ARGS);
Datum plcuda_function_handler(PG_FUNCTION_ARGS);
Datum plcuda_function_source(PG_FUNCTION_ARGS);
Datum plcuda_kernel_max_blocksz(PG_FUNCTION_ARGS);
Datum plcuda_kernel_static_shmsz(PG_FUNCTION_ARGS);
Datum plcuda_kernel_dynamic_shmsz(PG_FUNCTION_ARGS);
Datum plcuda_kernel_const_memsz(PG_FUNCTION_ARGS);
Datum plcuda_kernel_local_memsz(PG_FUNCTION_ARGS);

/* Tracker of plcudaState */
static dlist_head	plcuda_state_list;

/* PL/CUDA function property */
static int	plcuda_kfunc_max_blocksz    = -1;
static int	plcuda_kfunc_static_shmsz   = -1;
static int	plcuda_kfunc_dynamic_shmsz  = -1;
static int	plcuda_kfunc_const_memsz    = -1;
static int	plcuda_kfunc_local_memsz    = -1;

Datum
plcuda_kernel_max_blocksz(PG_FUNCTION_ARGS)
{
	if (plcuda_kfunc_max_blocksz < 0)
		elog(ERROR, "%s is called out of the PL/CUDA helper context",
			 format_procedure(fcinfo->flinfo->fn_oid));
	PG_RETURN_INT32(plcuda_kfunc_max_blocksz);
}
PG_FUNCTION_INFO_V1(plcuda_kernel_max_blocksz);

Datum
plcuda_kernel_static_shmsz(PG_FUNCTION_ARGS)
{
	if (plcuda_kfunc_static_shmsz < 0)
		elog(ERROR, "%s is called out of the PL/CUDA helper context",
			 format_procedure(fcinfo->flinfo->fn_oid));
	PG_RETURN_INT32(plcuda_kfunc_static_shmsz);
}
PG_FUNCTION_INFO_V1(plcuda_kernel_static_shmsz);

Datum
plcuda_kernel_dynamic_shmsz(PG_FUNCTION_ARGS)
{
	if (plcuda_kfunc_dynamic_shmsz < 0)
		elog(ERROR, "%s is called out of the PL/CUDA helper context",
			 format_procedure(fcinfo->flinfo->fn_oid));
	PG_RETURN_INT32(plcuda_kfunc_dynamic_shmsz);
}
PG_FUNCTION_INFO_V1(plcuda_kernel_dynamic_shmsz);

Datum
plcuda_kernel_const_memsz(PG_FUNCTION_ARGS)
{
	if (plcuda_kfunc_const_memsz < 0)
		elog(ERROR, "%s is called out of the PL/CUDA helper context",
             format_procedure(fcinfo->flinfo->fn_oid));
	PG_RETURN_INT32(plcuda_kfunc_const_memsz);
}
PG_FUNCTION_INFO_V1(plcuda_kernel_const_memsz);

Datum
plcuda_kernel_local_memsz(PG_FUNCTION_ARGS)
{
	if (plcuda_kfunc_local_memsz < 0)
		elog(ERROR, "%s is called out of the PL/CUDA helper context",
			 format_procedure(fcinfo->flinfo->fn_oid));
	PG_RETURN_INT32(plcuda_kfunc_local_memsz);
}
PG_FUNCTION_INFO_V1(plcuda_kernel_local_memsz);

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
typedef struct {
	Oid					proowner;
	oidvector		   *proargtypes;
	Oid					prorettype;
	StringInfo			curr;
	StringInfoData		decl_src;
	StringInfoData		prep_src;
	StringInfoData		main_src;
	StringInfoData		post_src;
	StringInfoData		emsg;
	FunctionCallInfo	fcinfo;
	bool				not_exec_now;
	bool				has_decl_block;
	bool				has_prep_block;
	bool				has_main_block;
	bool				has_post_block;
	bool				has_working_bufsz;
	bool				has_results_bufsz;
	bool				has_sanity_check;
	bool				has_cpu_fallback;
	List			   *include_func_oids;
	plcudaCodeProperty	p;
} plcuda_code_context;

static void
plcuda_init_code_context(plcuda_code_context *context,
						 Form_pg_proc procForm,
						 FunctionCallInfo fcinfo)
{
	memset(context, 0, sizeof(plcuda_code_context));

	context->proowner	= procForm->proowner;
	context->proargtypes= &procForm->proargtypes;
	context->prorettype	= procForm->prorettype;
	initStringInfo(&context->decl_src);
    initStringInfo(&context->prep_src);
    initStringInfo(&context->main_src);
    initStringInfo(&context->post_src);
    initStringInfo(&context->emsg);
	context->fcinfo = fcinfo;
	context->not_exec_now = !fcinfo;
	/* default setting */
	context->p.extra_flags = DEVKERNEL_NEEDS_PLCUDA;
	context->p.val_prep_num_threads = 1;
	context->p.val_main_num_threads = 1;
	context->p.val_post_num_threads = 1;
}

static void __plcuda_code_include(plcuda_code_context *con,
								  Oid fn_extra_include,
								  const char *source_name, int lineno);

static void
__plcuda_code_validation(plcuda_code_context *con,
						 const char *source_name,
						 char *source)
{
	plcudaCodeProperty *prop = &con->p;
	int			lineno;
	char	   *line;
	char	   *saveptr = NULL;

#define EMSG(fmt,...)									\
	appendStringInfo(&con->emsg, "\n%s%s%u: " fmt,		\
					 !source_name ? "" : source_name,	\
					 !source_name ? "" : ":",			\
					 lineno, ##__VA_ARGS__)
#define NOTE(fmt,...)							\
	elog(NOTICE, "%s%s%u: " fmt,				\
		 !source_name ? "" : source_name,		\
		 !source_name ? "" : ":",				\
		 lineno, ##__VA_ARGS__)

#define HELPER_PRIV_CHECK(func_oid)				\
	(!OidIsValid(func_oid) ||					\
	 con->not_exec_now ||						\
	 pg_proc_ownercheck((func_oid), con->proowner))

	for (line = strtok_r(source, "\n", &saveptr), lineno=1;
		 line != NULL;
		 line = strtok_r(NULL, "\n", &saveptr), lineno++)
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
			if (con->curr != NULL)
				appendStringInfo(con->curr, "%s\n", line);
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
				if (con->has_decl_block)
					EMSG("%s appeared twice", cmd);
				else if (list_length(options) > 0)
					EMSG("syntax error:\n  %s", line);
				else
				{
					con->curr = &con->decl_src;
					con->has_decl_block = true;
				}
			}
			else if (strcmp(cmd, "#plcuda_prep") == 0)
			{
				if (con->has_prep_block)
					EMSG("%s appeared twice", cmd);
				else if (list_length(options) > 0)
					EMSG("syntax error:\n  %s", line);
				else
				{
					con->curr = &con->prep_src;
					con->has_prep_block = true;
				}
			}
			else if (strcmp(cmd, "#plcuda_begin") == 0)
			{
				if (con->has_main_block)
					EMSG("%s appeared twice", cmd);
				else if (list_length(options) > 0)
					EMSG("syntax error:\n  %s", line);
				else
				{
					con->curr = &con->main_src;
					con->has_main_block = true;
				}
			}
			else if (strcmp(cmd, "#plcuda_post") == 0)
			{
				if (con->has_post_block)
					EMSG("%s appeared twice", cmd);
				else if (list_length(options) > 0)
					EMSG("syntax error:\n  %s\n", line);
				else
				{
					con->curr = &con->post_src;
					con->has_post_block = true;
				}
			}
			else if (strcmp(cmd, "#plcuda_end") == 0)
			{
				if (!con->curr)
					EMSG("%s was used out of code block", cmd);
				else
					con->curr = NULL;
			}
			else if (strcmp(cmd, "#plcuda_num_threads") == 0)
			{
				Oid		fn_num_threads;
				long	val_num_threads;

				if (!con->curr)
					EMSG("%s appeared outside of code block", cmd);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes, INT8OID,
											  &fn_num_threads,
											  &val_num_threads))
				{
					if (!HELPER_PRIV_CHECK(fn_num_threads))
					{
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
					}
					else if (con->curr == &con->prep_src)
					{
						prop->fn_prep_num_threads = fn_num_threads;
						prop->val_prep_num_threads = val_num_threads;
					}
					else if (con->curr == &con->main_src)
					{
						prop->fn_main_num_threads = fn_num_threads;
						prop->val_main_num_threads = val_num_threads;
					}
					else if (con->curr == &con->post_src)
					{
						prop->fn_post_num_threads = fn_num_threads;
						prop->val_post_num_threads = val_num_threads;
					}
					else
						EMSG("cannot use \"%s\" in this code block", cmd);
				}
				else if (con->not_exec_now)
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

				if (!con->curr)
					EMSG("%s appeared outside of code block", cmd);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes, INT8OID,
											  &fn_shmem_unitsz,
											  &val_shmem_unitsz))
				{
					if (!HELPER_PRIV_CHECK(fn_shmem_unitsz))
					{
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
					}
					else if (con->curr == &con->prep_src)
					{
						prop->fn_prep_shmem_unitsz = fn_shmem_unitsz;
						prop->val_prep_shmem_unitsz = val_shmem_unitsz;
					}
					else if (con->curr == &con->main_src)
					{
						prop->fn_main_shmem_unitsz = fn_shmem_unitsz;
						prop->val_main_shmem_unitsz = val_shmem_unitsz;
					}
					else if (con->curr == &con->post_src)
					{
						prop->fn_post_shmem_unitsz = fn_shmem_unitsz;
						prop->val_post_shmem_unitsz = val_shmem_unitsz;
					}
					else
						EMSG("cannot use \"%s\" in this code block", cmd);
				}
				else if (con->not_exec_now)
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

				if (!con->curr)
					EMSG("%s appeared outside of code block", cmd);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes, INT8OID,
											  &fn_shmem_blocksz,
											  &val_shmem_blocksz))
				{
					if (!HELPER_PRIV_CHECK(fn_shmem_blocksz))
					{
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
					}
					else if (con->curr == &con->prep_src)
					{
						prop->fn_prep_shmem_blocksz = fn_shmem_blocksz;
						prop->val_prep_shmem_blocksz = val_shmem_blocksz;
					}
					else if (con->curr == &con->main_src)
					{
						prop->fn_main_shmem_blocksz = fn_shmem_blocksz;
						prop->val_main_shmem_blocksz = val_shmem_blocksz;
					}
					else if (con->curr == &con->post_src)
					{
						prop->fn_post_shmem_blocksz = fn_shmem_blocksz;
						prop->val_post_shmem_blocksz = val_shmem_blocksz;
					}
					else
						EMSG("cannot use \"%s\" in this code block", cmd);
				}
				else if (con->not_exec_now)
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

				if (!con->curr)
					EMSG("%s appeared outside of code block", cmd);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes, INT8OID,
											  &fn_kern_blocksz,
											  &val_kern_blocksz))
				{
					if (!HELPER_PRIV_CHECK(fn_kern_blocksz))
					{
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
					}
					else if (con->curr == &con->prep_src)
					{
						prop->fn_prep_kern_blocksz = fn_kern_blocksz;
						prop->val_prep_kern_blocksz = val_kern_blocksz;
					}
					else if (con->curr == &con->main_src)
					{
						prop->fn_main_kern_blocksz = fn_kern_blocksz;
						prop->val_main_kern_blocksz = val_kern_blocksz;
					}
					else if (con->curr == &con->post_src)
					{
						prop->fn_post_kern_blocksz = fn_kern_blocksz;
						prop->val_post_kern_blocksz = val_kern_blocksz;
					}
					else
						EMSG("cannot use \"%s\" in this code block", cmd);
				}
				else if (con->not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_working_bufsz") == 0)
			{
				if (con->has_working_bufsz)
					EMSG("%s appeared twice", cmd);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes, INT8OID,
											  &prop->fn_working_bufsz,
											  &prop->val_working_bufsz))
				{
					if (HELPER_PRIV_CHECK(prop->fn_working_bufsz))
						con->has_working_bufsz = true;
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (con->not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
                         ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_results_bufsz") == 0)
			{
				if (con->has_results_bufsz)
					EMSG("%s appeared twice", cmd);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes, INT8OID,
											  &prop->fn_results_bufsz,
											  &prop->val_results_bufsz))
				{
					if (HELPER_PRIV_CHECK(prop->fn_results_bufsz))
						con->has_results_bufsz = true;
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (con->not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid value or function",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_include") == 0)
			{
				cl_uint		extra_flags = 0;
				Oid			fn_extra_include = InvalidOid;

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
					else if (strcmp(target, "cuda_misc.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_MISC;
					else if (strcmp(target, "cuda_curand.h") == 0)
						extra_flags |= DEVKERNEL_NEEDS_CURAND;
				}
				if (extra_flags != 0)
					prop->extra_flags |= extra_flags;
				else if (!con->curr)
					EMSG("#plcuda_include must appear in code block:\n%s",
						 line);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes,
											  TEXTOID,
											  &fn_extra_include,
											  NULL))
				{
					if (HELPER_PRIV_CHECK(fn_extra_include))
						__plcuda_code_include(con, fn_extra_include,
											  source_name, lineno);
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (con->not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid function name",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_sanity_check") == 0)
			{
				if (con->has_sanity_check)
					EMSG("%s appeared twice", cmd);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes, BOOLOID,
											  &prop->fn_sanity_check,
											  NULL))
				{
					if (HELPER_PRIV_CHECK(prop->fn_sanity_check))
						con->has_sanity_check = true;
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (con->not_exec_now)
					NOTE("\"%s\" may be a function but not declared yet",
						 ident_to_cstring(options));
				else
					EMSG("\"%s\" was not a valid function name",
						 ident_to_cstring(options));
			}
			else if (strcmp(cmd, "#plcuda_cpu_fallback") == 0)
			{
				if (con->has_cpu_fallback)
					EMSG("%s appeared twice", cmd);
				else if (plcuda_lookup_helper(options,
											  con->proargtypes,
											  con->prorettype,
											  &prop->fn_cpu_fallback,
											  NULL))
				{
					if (HELPER_PRIV_CHECK(prop->fn_cpu_fallback))
						con->has_cpu_fallback = true;
					else
						EMSG("permission denied on helper function %s",
							 NameListToString(options));
				}
				else if (con->not_exec_now)
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
}

static void
__plcuda_code_include(plcuda_code_context *con, Oid fn_extra_include,
					  const char *source_name, int lineno)
{
	ListCell   *lc;
	Datum		src;
	const char *func_name = get_func_name(fn_extra_include);

	/* prevent infinite inclusion */
	foreach (lc, con->include_func_oids)
	{
		if (lfirst_oid(lc) == fn_extra_include)
		{
			EMSG("\"%s\" leads infinite inclusion", func_name);
			return;
		}
	}

	if (con->fcinfo)
	{
		FmgrInfo	flinfo;
		FunctionCallInfoData fcinfo;
		FunctionCallInfo __fcinfo = con->fcinfo;

		appendStringInfo(con->curr,
						 "/* ------ BEGIN %s ------ */\n", func_name);
		con->include_func_oids = lappend_oid(con->include_func_oids,
											 fn_extra_include);

		/* see OidFunctionCallXX */
		fmgr_info(fn_extra_include, &flinfo);
		InitFunctionCallInfoData(fcinfo,
								 &flinfo,
								 __fcinfo->nargs,
								 __fcinfo->fncollation,
								 NULL,
								 NULL);
		memcpy(&fcinfo.arg, __fcinfo->arg,
			   __fcinfo->nargs * sizeof(Datum));
		memcpy(&fcinfo.argnull, __fcinfo->argnull,
			   __fcinfo->nargs * sizeof(bool));
		src = FunctionCallInvoke(&fcinfo);
		if (fcinfo.isnull)
			elog(ERROR, "function %u returned NULL", fn_extra_include);

		/* recursive walks of the included source */
		__plcuda_code_validation(con, func_name,
								 text_to_cstring(DatumGetTextP(src)));

		con->include_func_oids = list_delete_oid(con->include_func_oids,
												 fn_extra_include);
		appendStringInfo(con->curr,
						 "/* ------ END %s ------ */\n", func_name);
	}
}
#undef NMSG
#undef EMSG
#undef HELPER_PRIV_CHECK

static void
plcuda_code_validation(plcudaCodeProperty *prop,
					   HeapTuple proc_tuple,
					   FunctionCallInfo fcinfo)
{
	Form_pg_proc	procForm = (Form_pg_proc) GETSTRUCT(proc_tuple);
	plcuda_code_context	context;
	devtype_info   *dtype;
	Datum			prosrc;
	bool			isnull;
	int				i;

	plcuda_init_code_context(&context, procForm, fcinfo);
	/* check result type */
	dtype = pgstrom_devtype_lookup(procForm->prorettype);
	if (dtype)
		context.p.extra_flags |= dtype->type_flags;
	else if (get_typlen(procForm->prorettype) == -1)
	{
		Assert(!get_typbyval(procForm->prorettype));
		if (!fcinfo)
			elog(NOTICE, "Unknown varlena result - PL/CUDA must be responsible to the data format to return");
	}
	else
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Result type \"%s\" is not device executable",
						format_type_be(procForm->prorettype))));

	/* check argument types */
	Assert(procForm->pronargs == procForm->proargtypes.dim1);
	for (i=0; i < procForm->pronargs; i++)
	{
		Oid		argtype_oid = context.proargtypes->values[i];

		if (argtype_oid == REGGSTOREOID)
			continue;	/* OK, only for PL/CUDA argument */

		dtype = pgstrom_devtype_lookup(argtype_oid);
		if (dtype)
			context.p.extra_flags |= dtype->type_flags;
		else if (get_typlen(argtype_oid) == -1)
		{
			Assert(!get_typbyval(argtype_oid));
			if (!fcinfo)
				elog(NOTICE, "Unknown varlena argument%d - PL/CUDA must be responsible to interpretation of the supplied data format", i+1);
		}
		else
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("Argument type \"%s\" is not device executable",
							format_type_be(argtype_oid))));
	}
	/* Fetch CUDA code block */
	prosrc = SysCacheGetAttr(PROCOID, proc_tuple,
							 Anum_pg_proc_prosrc,
							 &isnull);
	if (isnull)
		elog(ERROR, "Bug? no program source was supplied");
	/* walk on the pl/cuda source */
	__plcuda_code_validation(&context, NULL,
							 TextDatumGetCString(prosrc));
	/* raise syntax error if any */
	if (context.curr != NULL)
		appendStringInfo(&context.emsg, "\n???: Code block was not closed");
	if (!context.has_main_block)
		appendStringInfo(&context.emsg, "\n???: No main code block");
	if (context.emsg.len > 0)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("pl/cuda function syntax error\n%s",
						context.emsg.data)));

	memcpy(prop, &context.p, sizeof(plcudaCodeProperty));
	if (context.has_decl_block)
		prop->kern_decl = context.decl_src.data;

	if (context.has_prep_block)
		prop->kern_prep = context.prep_src.data;
	else
		pfree(context.prep_src.data);

	prop->kern_main = context.main_src.data;
	if (context.has_post_block)
		prop->kern_post = context.post_src.data;
	else
		pfree(context.post_src.data);

	pfree(context.emsg.data);
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
	devtype_info   *dtype;
	const char	   *retval_typname;
	int				retval_typlen;
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
	dtype = pgstrom_devtype_lookup(procForm->prorettype);
	if (dtype)
	{
		retval_typname = dtype->type_name;
		retval_typlen  = dtype->type_length;
	}
	else if (get_typlen(procForm->prorettype) == -1)
	{
		/* NOTE: As long as PL/CUDA function is responsible to the data
		 * format of varlena result type, we can allow non-device executable
		 * data type as function result.
		 * Its primary target is composite data type, but not limited to.
		 */
		retval_typname = "varlena";
		retval_typlen  = -1;
	}
	else
	{
		elog(ERROR, "cache lookup failed for type '%s'",
			 format_type_be(procForm->prorettype));
	}
	appendStringInfo(
		kern,
		"  pg_%s_t *retval __attribute__ ((unused));\n",
		retval_typname);

	/* declaration of argument variables */
	for (i=0; i < procForm->pronargs; i++)
	{
		Oid			type_oid = procForm->proargtypes.values[i];
		const char *arg_typename;

		/* special case if reggstore type */
		if (type_oid == REGGSTOREOID)
		{
			appendStringInfo(
				kern,
				"  kern_reggstore_t arg%u __attribute__((unused));\n",
				i+1);
			continue;
		}

		dtype = pgstrom_devtype_lookup(type_oid);
		if (dtype)
			arg_typename = dtype->type_name;
		else if (get_typlen(type_oid) ==  -1)
			arg_typename = "varlena";
		else
			elog(ERROR, "cache lookup failed for type '%s'",
				 format_type_be(type_oid));

		appendStringInfo(
			kern,
			"  pg_%s_t arg%u __attribute__((unused));\n",
			arg_typename, i+1);
	}

	appendStringInfo(
		kern,
		"  assert(sizeof(*retval) <= sizeof(kplcuda->__retval));\n"
		"  retval = (pg_%s_t *)kplcuda->__retval;\n", retval_typname);
	if (retval_typlen < 0)
		appendStringInfoString(
			kern,
			"  assert(retval->isnull ||\n"
			"         (void *)retval->value == results ||\n"
			"         (void *)retval->value == kplcuda->__vl_buffer);\n");

	for (i=0; i < procForm->pronargs; i++)
	{
		Oid			type_oid = procForm->proargtypes.values[i];
		const char *arg_typename;

		/* special case if reggstore type */
		if (type_oid == REGGSTOREOID)
		{
			appendStringInfo(
				kern,
				"  arg%u = pg_reggstore_param(kcxt,%d);\n",
				i+1, i);
			continue;
		}

		dtype = pgstrom_devtype_lookup(type_oid);
		if (dtype)
			arg_typename = dtype->type_name;
		else if (get_typlen(type_oid) ==  -1)
			arg_typename = "varlena";
		else
			elog(ERROR, "cache lookup failed for type '%s'",
				 format_type_be(type_oid));
		appendStringInfo(
			kern,
			"  arg%u = pg_%s_param(kcxt,%d);\n",
			i+1, arg_typename, i);
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
		"  assert(kplcuda->nargs <= kparams->nparams);\n",
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
		"  kern_writeback_error_status(&kplcuda->kerror_%s, &kcxt.e);\n"
		"}\n\n",
		suffix);
}

static void
plcuda_codegen_composite_typedesc(StringInfo kern,
								  Form_pg_proc procForm,
								  plcudaCodeProperty *prop)
{
	ListCell   *lc;
	cl_int		pindex = procForm->pronargs;

	if (prop->composite_types == NIL)
		return;
	appendStringInfoString(
		kern,
		"STATIC_FUNCTION(pg_composite_typedesc *)\n"
		"pg_lookup_composite_typedesc(kern_parambuf *kparams, cl_uint type_oid)\n"
		"{\n"
		"  pg_composite_typedesc *ctdesc = NULL;\n"
		"\n"
		"  switch (type_oid)\n"
		"  {\n");
	foreach (lc, prop->composite_types)
	{
		pg_composite_typedesc *ctdesc = lfirst(lc);
		appendStringInfo(
			kern,
			"    case %d:\n"
			"      ctdesc = (pg_composite_typedesc *)kparam_get_value(kparams, %d);\n"
			"      break;\n",
			ctdesc->type_oid,
			pindex++);
	}
	appendStringInfoString(
		kern,
		"    default:\n"
		"      /* not found */\n"
		"      break;\n"
		"  }\n"
		"  return ctdesc;\n"
		"}\n\n");
}

static char *
plcuda_codegen(Form_pg_proc procForm, plcudaCodeProperty *prop)
{
	StringInfoData	kern;
	const char	   *last_stage = NULL;

	initStringInfo(&kern);

	/* access function for composite types */
	plcuda_codegen_composite_typedesc(&kern, procForm, prop);
	/* misc declarations */
	if (prop->kern_decl)
		appendStringInfo(&kern, "%s\n", prop->kern_decl);
	if (prop->kern_prep)
	{
		plcuda_codegen_part(&kern, "prep",
							prop->kern_prep,
							(OidIsValid(prop->fn_prep_kern_blocksz) ||
							 prop->val_prep_kern_blocksz > 0),
							procForm,
							last_stage);
		last_stage = "prep";
	}
	if (prop->kern_main)
	{
		plcuda_codegen_part(&kern, "main",
							prop->kern_main,
							(OidIsValid(prop->fn_main_kern_blocksz) ||
							 prop->val_main_kern_blocksz > 0),
							procForm,
							last_stage);
		last_stage = "main";
	}
	if (prop->kern_post)
	{
		plcuda_codegen_part(&kern, "post",
							prop->kern_post,
							(OidIsValid(prop->fn_post_kern_blocksz) ||
							 prop->val_post_kern_blocksz > 0),
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

static void
__setup_composite_kern_colmeta(HeapTuple typtup, plcudaCodeProperty *p)
{
	Form_pg_type	typeForm = (Form_pg_type) GETSTRUCT(typtup);
	Oid				type_oid = HeapTupleGetOid(typtup);
	Oid				type_relid = typeForm->typrelid;
	List		   *composite_subtypes = NIL;
	ListCell	   *lc;
	HeapTuple		tp;
	Form_pg_class	classForm;
	Form_pg_attribute attForm;
	cl_int			j, nattrs;
	size_t			vl_len;
	pg_composite_typedesc *ctdesc;

	/* check duplication */
	foreach (lc, p->composite_types)
	{
		ctdesc = lfirst(lc);
		if (ctdesc->type_oid == type_oid)
			return;		/* already built */
	}

	/* build a pg_composite_typedesc */
	Assert(OidIsValid(type_relid));
	tp = SearchSysCache1(RELOID, ObjectIdGetDatum(type_relid));
	if (!HeapTupleIsValid(tp))
		elog(ERROR, "cache lookup failed for relation %u", type_relid);
	classForm = (Form_pg_class) GETSTRUCT(tp);

	Assert(classForm->relkind == RELKIND_COMPOSITE_TYPE &&
		   classForm->reltype == type_oid);
	nattrs = classForm->relnatts;
	ReleaseSysCache(tp);

	vl_len = offsetof(pg_composite_typedesc, colmeta[nattrs]);
	ctdesc = palloc0(vl_len);
	ctdesc->type_oid = type_oid;
	ctdesc->nattrs = nattrs;
	for (j=0; j < nattrs; j++)
	{
		kern_colmeta   *cmeta = &ctdesc->colmeta[j];

		tp = SearchSysCache2(ATTNUM,
							 ObjectIdGetDatum(type_relid),
							 Int16GetDatum(j+1));
		if (!HeapTupleIsValid(tp))
			elog(ERROR, "cache lookup failed for attribute %d of relation %u",
				 j, type_relid);
		attForm = (Form_pg_attribute) GETSTRUCT(tp);

		cmeta->attbyval = attForm->attbyval;
		cmeta->attalign = typealign_get_width(attForm->attalign);
		cmeta->attlen = attForm->attlen;
		cmeta->attnum = attForm->attnum;
		cmeta->attcacheoff = -1;
		cmeta->atttypid = attForm->atttypid;
		cmeta->atttypmod = attForm->atttypmod;
		cmeta->va_offset = 0;
		cmeta->va_length = 0;

		if (get_typtype(attForm->atttypid) == TYPTYPE_COMPOSITE)
			composite_subtypes = list_append_unique_oid(composite_subtypes,
														attForm->atttypid);
		ReleaseSysCache(tp);
	}
	SET_VARSIZE(ctdesc, vl_len);

	p->composite_types = lappend(p->composite_types, ctdesc);

	/* dive into sub-types if any more composite types */
	foreach (lc, composite_subtypes)
	{
		Oid		subtype_oid = lfirst_oid(lc);

		tp = SearchSysCache1(TYPEOID, ObjectIdGetDatum(subtype_oid));
		if (!HeapTupleIsValid(tp))
			elog(ERROR, "cache lookup failed for type %u", subtype_oid);
		__setup_composite_kern_colmeta(tp, p);
		ReleaseSysCache(tp);
	}
	list_free(composite_subtypes);
}

static kern_colmeta
__setup_kern_colmeta(Oid type_oid, int attnum, plcudaCodeProperty *p)
{
	HeapTuple		tuple;
	Form_pg_type	typeForm;
	kern_colmeta	result;

	/* special case handling for reggstore if PL/CUDA used */
	if (type_oid == REGGSTOREOID)
	{
		result.attbyval = true;
		result.attalign = sizeof(cl_long);
		result.attlen = sizeof(CUdeviceptr);
		result.attnum = attnum;
		result.attcacheoff = -1;	/* we don't use attcacheoff */
		result.atttypid = type_oid;
		result.atttypmod = -1;
		result.va_offset = 0;
		result.va_length = 0;

		return result;
	}

	tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for type '%s'",
			 format_type_be(type_oid));
	typeForm = (Form_pg_type) GETSTRUCT(tuple);

	result.attbyval = typeForm->typbyval;
	result.attalign = typealign_get_width(typeForm->typalign);
	result.attlen = typeForm->typlen;
	result.attnum = attnum;
	result.attcacheoff = -1;	/* we don't use attcacheoff */
	result.atttypid = type_oid;
	result.atttypmod = typeForm->typtypmod;
	result.va_offset = 0;
	result.va_length = 0;

	/*
	 * composite type needs extra kern_colmeta array to form/deform
	 * packed tuple in the GPU kernel space.
	 */
	if (typeForm->typtype == TYPTYPE_COMPOSITE)
		__setup_composite_kern_colmeta(tuple, p);
	ReleaseSysCache(tuple);

	return result;
}

static void
plcuda_assign_kernel_properties(plcudaTaskState *plts)
{
	CUmodule	cuda_module;
	CUfunction	cuda_function;
	CUresult	rc;

	cuda_module = pgstrom_load_cuda_program(plts->gts.program_id);
	/* prep kernel */
	rc = cuModuleGetFunction(&cuda_function, cuda_module,
							 "plcuda_prep_kernel_entrypoint");
	if (rc == CUDA_ERROR_NOT_FOUND)
	{
		plts->kprep_max_blocksz = -1;
		plts->kprep_static_shmsz = -1;
		plts->kprep_dynamic_shmsz = -1;
		plts->kprep_const_memsz = -1;
		plts->kprep_local_memsz = -1;
	}
	else if (rc == CUDA_SUCCESS)
	{
		rc = cuFuncGetAttribute(&plts->kprep_max_blocksz,
								CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kprep_static_shmsz,
								CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kprep_dynamic_shmsz,
							   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kprep_const_memsz,
								CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kprep_local_memsz,
								CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
								cuda_function);
	}
	else
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	/* main kernel */
	rc = cuModuleGetFunction(&cuda_function, cuda_module,
							 "plcuda_main_kernel_entrypoint");
	if (rc == CUDA_ERROR_NOT_FOUND)
	{
		plts->kmain_max_blocksz = -1;
        plts->kmain_static_shmsz = -1;
        plts->kmain_dynamic_shmsz = -1;
        plts->kmain_const_memsz = -1;
        plts->kmain_local_memsz = -1;
	}
	else if (rc == CUDA_SUCCESS)
	{
		rc = cuFuncGetAttribute(&plts->kmain_max_blocksz,
								CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kmain_static_shmsz,
								CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kmain_dynamic_shmsz,
							   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kmain_const_memsz,
								CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kmain_local_memsz,
								CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
								cuda_function);
	}
	else
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));


	/* post kernel */
	rc = cuModuleGetFunction(&cuda_function, cuda_module,
							 "plcuda_post_kernel_entrypoint");
	if (rc == CUDA_ERROR_NOT_FOUND)
	{
		plts->kpost_max_blocksz = -1;
		plts->kpost_static_shmsz = -1;
		plts->kpost_dynamic_shmsz = -1;
		plts->kpost_const_memsz = -1;
		plts->kpost_local_memsz = -1;
	}
	else if (rc == CUDA_SUCCESS)
	{
		rc = cuFuncGetAttribute(&plts->kpost_max_blocksz,
								CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kpost_static_shmsz,
								CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kpost_dynamic_shmsz,
							   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kpost_const_memsz,
								CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
								cuda_function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

		rc = cuFuncGetAttribute(&plts->kpost_local_memsz,
								CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
								cuda_function);
	}
	else
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleUnload(cuda_module);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleUnload: %s", errorText(rc));
}

static plcudaTaskState *
plcuda_exec_begin(HeapTuple protup, FunctionCallInfo fcinfo)
{
	Form_pg_proc	procForm = (Form_pg_proc) GETSTRUCT(protup);
	GpuContext	   *gcontext;
	plcudaTaskState *plts;
	kern_plcuda	   *kplcuda;
	size_t			kplcuda_length;
	char		   *kern_source;
	StringInfoData	kern_define;
	ProgramId		program_id;
	int				i, cuda_dindex = -1;

	/*
	 * NOTE: In case when PL/CUDA function call contains gstore_fdw in
	 * the argument list, some of them might be pinned to a particular
	 * GPU device. In this case, we set up GpuContext on the preferable
	 * location.
	 *
	 * FIXME: If a particular PL/CUDA function is invoked multiple times
	 * in a single query, right now, it cannot have gstore_fdw which are
	 * pinned to the different devices from the former invocations cases.
	 * For more ideal performance, GPU kernel should be also built to
	 * the dedicated device instead of the common capability.
	 */
	if (fcinfo)
		cuda_dindex = gstore_fdw_preferable_device(fcinfo);
	gcontext = AllocGpuContext(cuda_dindex, true, true, true);
	/* setup a dummy GTS for PL/CUDA (see pgstromInitGpuTaskState) */
	plts = MemoryContextAllocZero(CurTransactionContext,
								  sizeof(plcudaTaskState));
	plts->gts.gcontext = gcontext;
	plts->gts.task_kind = GpuTaskKind_PL_CUDA;
	plts->gts.kern_params = NULL;
	plts->gts.ccache_refs = NULL;
	plts->gts.cb_process_task = plcuda_process_task;
	plts->gts.cb_release_task = plcuda_release_task;
	dlist_init(&plts->gts.ready_tasks);

	/* validate PL/CUDA source code */
	procForm = (Form_pg_proc) GETSTRUCT(protup);
	plcuda_code_validation(&plts->p, protup, fcinfo);

	/* build the template of the kern_plcuda */
	kplcuda_length = STROMALIGN(offsetof(kern_plcuda,
										 argmeta[procForm->pronargs]));
	kplcuda = palloc0(kplcuda_length);
	kplcuda->length = kplcuda_length;
	kplcuda->nargs = procForm->pronargs;
	kplcuda->retmeta = __setup_kern_colmeta(procForm->prorettype, -1,
											&plts->p);
	for (i=0; i < procForm->pronargs; i++)
	{
		Oid		argtype_oid = procForm->proargtypes.values[i];

		kplcuda->argmeta[i] = __setup_kern_colmeta(argtype_oid, i+1,
												   &plts->p);
	}
	plts->kplcuda_head = kplcuda;

	/* construct a flat kernel source to be built */
	initStringInfo(&kern_define);
	kern_source = plcuda_codegen(procForm, &plts->p);
	pgstrom_build_session_info(&kern_define,
							   &plts->gts,
							   plts->p.extra_flags);
	program_id = pgstrom_create_cuda_program(gcontext,
											 plts->p.extra_flags,
											 0,
											 kern_source,
											 kern_define.data,
											 true,
											 false);
	plts->gts.program_id = program_id;
	pfree(kern_define.data);
	/* track plcudaTaskState by local tracker */
	plts->owner = CurrentResourceOwner;
	dlist_push_head(&plcuda_state_list, &plts->chain);
	/* assign properties of the kernel functions */
	plcuda_assign_kernel_properties(plts);

	return plts;
}

static void
plcuda_exec_end(plcudaTaskState *plts)
{
	dlist_delete(&plts->chain);

	if (plts->last_results_buf)
		gpuMemFree(plts->gts.gcontext,
				   plts->last_results_buf);
	pgstromReleaseGpuTaskState(&plts->gts, NULL);
}

Datum
plcuda_function_validator(PG_FUNCTION_ARGS)
{
	Oid				func_oid = PG_GETARG_OID(0);
	HeapTuple		tuple;
	char			prokind;
	plcudaTaskState	*plts;

	if (!CheckFunctionValidatorAccess(fcinfo->flinfo->fn_oid, func_oid))
		PG_RETURN_VOID();
	/*
	 * Sanity check of PL/CUDA functions
	 */
	prokind = get_func_prokind(func_oid);
	switch (prokind)
	{
		case PROKIND_FUNCTION:
			/* OK */
			break;
		case PROKIND_AGGREGATE:
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("Unable to use PL/CUDA for aggregate functions")));
			break;
		case PROKIND_WINDOW:
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("Unable to use PL/CUDA for window functions")));
			break;
		case PROKIND_PROCEDURE:
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("Unable to use PL/CUDA for procedure")));
			break;
		default:
			elog(ERROR, "Bug? unknown procedure kind: %c", prokind);
			break;
	}

	/*
	 * Only validation of the CUDA code. Run synchronous code build, then
	 * raise an error if code block has any error.
	 */
	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	if (((Form_pg_proc) GETSTRUCT(tuple))->proretset)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Righ now, PL/CUDA function does not support set returning function")));

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
	Datum		result;

	if (!OidIsValid(helper_func_oid))
		return static_config;

	strom_proc_aclcheck(helper_func_oid, GetUserId(), ACL_EXECUTE);

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
	GpuContext	   *gcontext = plts->gts.gcontext;
	plcudaTask	   *ptask;
	kern_parambuf  *kparams;
	kern_plcuda	   *kplcuda_head = plts->kplcuda_head;
	kern_colmeta   *cmeta;
	Size			total_length;
	Size			offset;
	List		   *gstore_oid_list;
	List		   *gstore_devptr_list;
	List		   *gstore_dindex_list;
	CUdeviceptr		m_deviceptr;
	CUresult		rc;
	ListCell	   *lc;
	int				i, nparams;

	/* load gstore_fdw if any */
	gstore_fdw_load_function_args(gcontext,
								  fcinfo,
								  &gstore_oid_list,
								  &gstore_devptr_list,
								  &gstore_dindex_list);
	/* calculation of the length */
	Assert(fcinfo->nargs == kplcuda_head->nargs);
	Assert(kplcuda_head->length
		   == STROMALIGN(offsetof(kern_plcuda,
								  argmeta[kplcuda_head->nargs])));
	total_length = STROMALIGN(offsetof(plcudaTask, kern)
							  + kplcuda_head->length);
	nparams = fcinfo->nargs;
	if (plts->p.composite_types != NIL)
		nparams += list_length(plts->p.composite_types);
	total_length += STROMALIGN(offsetof(kern_parambuf,
										poffset[nparams]));
	for (i=0; i < fcinfo->nargs; i++)
	{
		kern_colmeta	cmeta = kplcuda_head->argmeta[i];

		if (fcinfo->argnull[i])
			continue;
		if (cmeta.atttypid == REGGSTOREOID)
			total_length += MAXALIGN(sizeof(CUdeviceptr));
		else if (cmeta.attlen > 0)
			total_length += MAXALIGN(cmeta.attlen);
		else
			total_length += MAXALIGN(toast_raw_datum_size(fcinfo->arg[i]));
	}
	/* additional pg_composite_typedesc parameters */
	foreach (lc, plts->p.composite_types)
		total_length += MAXALIGN(VARSIZE(lfirst(lc)));

	total_length = STROMALIGN(total_length);

	/* setup plcudaTask */
	rc = gpuMemAllocManaged(gcontext,
							&m_deviceptr,
							total_length,
							CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));
	ptask = (plcudaTask *) m_deviceptr;
	memset(ptask, 0, offsetof(plcudaTask, kern.retmeta));
	pgstromInitGpuTask(&plts->gts, &ptask->task);
	if (results_bufsz > 0)
	{
		rc = gpuMemAllocManaged(gcontext,
								&ptask->m_results_buf,
								results_bufsz,
								CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));
	}
	ptask->gstore_oid_list = gstore_oid_list;
	ptask->gstore_devptr_list = gstore_devptr_list;
	ptask->gstore_dindex_list = gstore_dindex_list;

	/* setup kern_plcuda */
	memcpy(&ptask->kern, kplcuda_head, kplcuda_head->length);
	ptask->kern.working_bufsz = working_bufsz;
	ptask->kern.working_usage = 0UL;
	ptask->kern.results_bufsz = results_bufsz;
	ptask->kern.results_usage = 0UL;

	/* setup default result value */
	cmeta = &ptask->kern.retmeta;
	if (cmeta->attlen > 0)
		ptask->kern.__retval[cmeta->attlen] = true;
	else
	{
		ptask->kern.__retval[sizeof(CUdeviceptr)] = true;
		if (ptask->m_results_buf != 0UL)
			*((CUdeviceptr *)ptask->kern.__retval) = ptask->m_results_buf;
		else
			*((CUdeviceptr *)ptask->kern.__retval) = (CUdeviceptr)ptask->kern.__vl_buffer;
	}

	/* copy function arguments onto DMA buffer */
	kparams = KERN_PLCUDA_PARAMBUF(&ptask->kern);
	kparams->hostptr = (hostptr_t) kparams;
	kparams->xactStartTimestamp = GetCurrentTransactionStartTimestamp();

	offset = STROMALIGN(offsetof(kern_parambuf,
								 poffset[fcinfo->nargs]));
	for (i=0; i < fcinfo->nargs; i++)
	{
		kern_colmeta	cmeta = kplcuda_head->argmeta[i];

		if (fcinfo->argnull[i])
			kparams->poffset[i] = 0;	/* null */
		else if (cmeta.atttypid == REGGSTOREOID)
		{
			ListCell   *lc1, *lc2;
			Oid			gstore_oid = DatumGetObjectId(fcinfo->arg[i]);
			CUdeviceptr	m_deviceptr = 0L;

			forboth (lc1, ptask->gstore_oid_list,
					 lc2, ptask->gstore_devptr_list)
			{
				if (lfirst_oid(lc1) == gstore_oid)
				{
					m_deviceptr = (CUdeviceptr) lfirst(lc2);
					break;
				}
			}
			if (m_deviceptr == 0L)
				kparams->poffset[i] = 0;	/* empty gstore deal as NULL */
			else
			{
				kparams->poffset[i] = offset;
				memcpy((char *)kparams + offset,
					   &m_deviceptr,
					   sizeof(CUdeviceptr));
				offset += MAXALIGN(sizeof(CUdeviceptr));
			}
		}
		else if (cmeta.attbyval)
		{
			kparams->poffset[i] = offset;
			Assert(cmeta.attlen > 0);
			memcpy((char *)kparams + offset,
				   &fcinfo->arg[i],
				   cmeta.attlen);
			offset += MAXALIGN(cmeta.attlen);
		}
		else if (cmeta.attlen > 0)
		{
			kparams->poffset[i] = offset;
			memcpy((char *)kparams + offset,
				   DatumGetPointer(fcinfo->arg[i]),
				   cmeta.attlen);
			offset += MAXALIGN(cmeta.attlen);
		}
		else
		{
			char   *vl_ptr = (char *)PG_DETOAST_DATUM(fcinfo->arg[i]);
			Size	vl_len = VARSIZE_ANY(vl_ptr);

			kparams->poffset[i] = offset;
			memcpy((char *)kparams + offset, vl_ptr, vl_len);
			offset += MAXALIGN(vl_len);
		}
	}
	Assert(i == fcinfo->nargs);

	foreach (lc, plts->p.composite_types)
	{
		pg_composite_typedesc *ctdesc = lfirst(lc);
		size_t		vl_len = VARSIZE(ctdesc);

		kparams->poffset[i++] = offset;
		memcpy((char *)kparams + offset, ctdesc, vl_len);
		offset += MAXALIGN(vl_len);
	}
	Assert(i == nparams);
	kparams->nparams = nparams;
	kparams->length = STROMALIGN(offset);
	Assert(STROMALIGN(offsetof(plcudaTask, kern) +
					  kplcuda_head->length) +
		   kparams->length <= total_length);
	Assert(plts->last_results_buf == 0UL);
	plts->last_results_buf = ptask->m_results_buf;

	return ptask;
}

Datum
plcuda_function_handler(PG_FUNCTION_ARGS)
{
	FmgrInfo	   *flinfo = fcinfo->flinfo;
	plcudaTaskState *plts;
	plcudaTask	   *ptask;
	plcudaTask	   *precv;
	GpuContext	   *gcontext;
	Size			working_bufsz;
	Size			results_bufsz;
	kern_errorbuf	kerror;
	Datum			retval = 0;
	bool			isnull = false;
	ListCell	   *lc1, *lc2, *lc3;

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
		gpuMemFree(plts->gts.gcontext,
				   plts->last_results_buf);
		plts->last_results_buf = 0UL;
	}

	/* sanitycheck of the supplied arguments, prior to GPU launch */
	if (!DatumGetBool(kernel_launch_helper(fcinfo,
										   plts->p.fn_sanity_check,
										   BoolGetDatum(true),
										   NULL)))
		elog(ERROR, "function '%s' argument sanity check failed",
			 format_procedure(plts->p.fn_sanity_check));

	/* determine the kernel launch parameters */
	working_bufsz =
		DatumGetInt64(kernel_launch_helper(fcinfo,
										   plts->p.fn_working_bufsz,
										   plts->p.val_working_bufsz,
										   NULL));
	results_bufsz =
		DatumGetInt64(kernel_launch_helper(fcinfo,
										   plts->p.fn_results_bufsz,
										   plts->p.val_results_bufsz,
										   NULL));
	elog(DEBUG2, "working_bufsz = %zu, results_bufsz = %zu",
		 working_bufsz, results_bufsz);

	/* construction of plcudaTask structure */
	ptask = create_plcuda_task(plts, fcinfo,
							   working_bufsz,
							   results_bufsz);
	if (plts->p.kern_prep)
	{
		int		saved_max_blocksz   = plcuda_kfunc_max_blocksz;
		int		saved_static_shmsz  = plcuda_kfunc_static_shmsz;
		int		saved_dynamic_shmsz = plcuda_kfunc_dynamic_shmsz;
		int		saved_const_memsz   = plcuda_kfunc_const_memsz;
		int		saved_local_memsz   = plcuda_kfunc_local_memsz;
		int64	v;

		PG_TRY();
		{
			plcuda_kfunc_max_blocksz   = plts->kprep_max_blocksz;
			plcuda_kfunc_static_shmsz  = plts->kprep_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = plts->kprep_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = plts->kprep_const_memsz;
			plcuda_kfunc_local_memsz   = plts->kprep_local_memsz;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_prep_num_threads,
										 plts->p.val_prep_num_threads,
										 NULL));
			if (v <= 0)
				elog(ERROR, "kern_prep: invalid number of threads: %ld", v);
			ptask->kern.prep_num_threads = (cl_ulong)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_prep_kern_blocksz,
										 plts->p.val_prep_kern_blocksz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR, "kern_prep: invalid kernel block size: %ld", v);
			ptask->kern.prep_kern_blocksz = (cl_uint)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_prep_shmem_unitsz,
										 plts->p.val_prep_shmem_unitsz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR,
					 "kern_prep: invalid shared memory requirement: %ld", v);
			ptask->kern.prep_shmem_unitsz = (cl_uint)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_prep_shmem_blocksz,
										 plts->p.val_prep_shmem_blocksz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR,
					 "kern_prep: invalid shared memory requirement: %ld", v);
			ptask->kern.prep_shmem_blocksz = v;

			elog(DEBUG2, "kern_prep {blocksz=%u, nitems=%lu, shmem=%u,%u}",
				 ptask->kern.prep_kern_blocksz,
				 ptask->kern.prep_num_threads,
				 ptask->kern.prep_shmem_unitsz,
				 ptask->kern.prep_shmem_blocksz);
			ptask->exec_prep_kernel = true;

			plcuda_kfunc_max_blocksz   = saved_max_blocksz;
			plcuda_kfunc_static_shmsz  = saved_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = saved_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = saved_const_memsz;
			plcuda_kfunc_local_memsz   = saved_local_memsz;
		}
		PG_CATCH();
		{
			plcuda_kfunc_max_blocksz   = saved_max_blocksz;
			plcuda_kfunc_static_shmsz  = saved_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = saved_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = saved_const_memsz;
			plcuda_kfunc_local_memsz   = saved_local_memsz;
			PG_RE_THROW();
		}
		PG_END_TRY();
	}

	if (plts->p.kern_main)
	{
		int		saved_max_blocksz   = plcuda_kfunc_max_blocksz;
		int		saved_static_shmsz  = plcuda_kfunc_static_shmsz;
		int		saved_dynamic_shmsz = plcuda_kfunc_dynamic_shmsz;
		int		saved_const_memsz   = plcuda_kfunc_const_memsz;
		int		saved_local_memsz   = plcuda_kfunc_local_memsz;
		int64	v;

		PG_TRY();
		{
			plcuda_kfunc_max_blocksz   = plts->kmain_max_blocksz;
			plcuda_kfunc_static_shmsz  = plts->kmain_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = plts->kmain_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = plts->kmain_const_memsz;
			plcuda_kfunc_local_memsz   = plts->kmain_local_memsz;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_main_num_threads,
										 plts->p.val_main_num_threads,
										 NULL));
			if (v <= 0)
				elog(ERROR, "kern_main: invalid number of threads: %ld", v);
			ptask->kern.main_num_threads = (cl_ulong)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_main_kern_blocksz,
										 plts->p.val_main_kern_blocksz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR, "kern_main: invalid kernel block size: %ld", v);
			ptask->kern.main_kern_blocksz = (cl_uint)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_main_shmem_unitsz,
										 plts->p.val_main_shmem_unitsz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR,
					 "kern_main: invalid shared memory requirement: %ld", v);
			ptask->kern.main_shmem_unitsz = (cl_uint)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_main_shmem_blocksz,
										 plts->p.val_main_shmem_blocksz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR,
					 "kern_main: invalid shared memory requirement: %ld", v);
			ptask->kern.main_shmem_blocksz = v;

			elog(DEBUG2, "kern_main {blocksz=%u, nitems=%lu, shmem=%u,%u}",
				 ptask->kern.main_kern_blocksz,
				 ptask->kern.main_num_threads,
				 ptask->kern.main_shmem_unitsz,
				 ptask->kern.main_shmem_blocksz);

			plcuda_kfunc_max_blocksz   = saved_max_blocksz;
			plcuda_kfunc_static_shmsz  = saved_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = saved_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = saved_const_memsz;
			plcuda_kfunc_local_memsz   = saved_local_memsz;
		}
		PG_CATCH();
		{
			plcuda_kfunc_max_blocksz   = saved_max_blocksz;
			plcuda_kfunc_static_shmsz  = saved_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = saved_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = saved_const_memsz;
			plcuda_kfunc_local_memsz   = saved_local_memsz;
			PG_RE_THROW();
		}
		PG_END_TRY();
	}

	if (plts->p.kern_post)
	{
		int		saved_max_blocksz   = plcuda_kfunc_max_blocksz;
		int		saved_static_shmsz  = plcuda_kfunc_static_shmsz;
		int		saved_dynamic_shmsz = plcuda_kfunc_dynamic_shmsz;
		int		saved_const_memsz   = plcuda_kfunc_const_memsz;
		int		saved_local_memsz   = plcuda_kfunc_local_memsz;
		int64	v;

		PG_TRY();
		{
			plcuda_kfunc_max_blocksz   = plts->kpost_max_blocksz;
			plcuda_kfunc_static_shmsz  = plts->kpost_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = plts->kpost_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = plts->kpost_const_memsz;
			plcuda_kfunc_local_memsz   = plts->kpost_local_memsz;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_post_num_threads,
										 plts->p.val_post_num_threads,
										 NULL));
			if (v <= 0)
				elog(ERROR, "kern_post: invalid number of threads: %ld", v);
			ptask->kern.post_num_threads = (cl_ulong)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_post_kern_blocksz,
										 plts->p.val_post_kern_blocksz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR, "kern_post: invalid kernel block size: %ld", v);
			ptask->kern.post_kern_blocksz = (cl_uint)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_post_shmem_unitsz,
										 plts->p.val_post_shmem_unitsz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR,
					 "kern_post: invalid shared memory requirement: %ld", v);
			ptask->kern.post_shmem_unitsz = (cl_uint)v;

			v = DatumGetInt64(
					kernel_launch_helper(fcinfo,
										 plts->p.fn_post_shmem_blocksz,
										 plts->p.val_post_shmem_blocksz,
										 NULL));
			if (v < 0 || v > INT_MAX)
				elog(ERROR,
					 "kern_post: invalid shared memory requirement: %ld", v);
			ptask->kern.post_shmem_blocksz = v;

			elog(DEBUG2, "kern_post {blocksz=%u, nitems=%lu, shmem=%u,%u}",
				 ptask->kern.post_kern_blocksz,
				 ptask->kern.post_num_threads,
				 ptask->kern.post_shmem_unitsz,
				 ptask->kern.post_shmem_blocksz);
			ptask->exec_post_kernel = true;

			plcuda_kfunc_max_blocksz   = saved_max_blocksz;
			plcuda_kfunc_static_shmsz  = saved_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = saved_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = saved_const_memsz;
			plcuda_kfunc_local_memsz   = saved_local_memsz;
		}
		PG_CATCH();
		{
			plcuda_kfunc_max_blocksz   = saved_max_blocksz;
			plcuda_kfunc_static_shmsz  = saved_static_shmsz;
			plcuda_kfunc_dynamic_shmsz = saved_dynamic_shmsz;
			plcuda_kfunc_const_memsz   = saved_const_memsz;
			plcuda_kfunc_local_memsz   = saved_local_memsz;
			PG_RE_THROW();
		}
		PG_END_TRY();
	}

	/* Exec PL/CUDA function by GPU */
	gcontext = plts->gts.gcontext;
	pthreadMutexLock(gcontext->mutex);
	dlist_push_tail(&gcontext->pending_tasks, &ptask->task.chain);
	plts->gts.num_running_tasks++;
	pg_atomic_add_fetch_u32(gcontext->global_num_running_tasks, 1);
	pthreadCondSignal(gcontext->cond);
	pthreadMutexUnlock(gcontext->mutex);

	/* Wait for the completion */
	plts->gts.scan_done = true;
	precv = (plcudaTask *) fetch_next_gputask(&plts->gts);
	if (!precv)
		elog(ERROR, "PL/CUDA GPU Task has gone to somewhere...");
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
					   precv->kern.retmeta.attlen);
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
			else
			{
				CUdeviceptr	m_devptr = *((CUdeviceptr *)precv->kern.__retval);

				if (m_devptr != (CUdeviceptr)precv->kern.__vl_buffer &&
					m_devptr != precv->m_results_buf)
					elog(ERROR, "pl/cuda: uncertain result varlena buffer");
				retval = PointerGetDatum(m_devptr);
			}
		}
	}
	else
	{
		if (precv->m_results_buf)
		{
			gpuMemFree(plts->gts.gcontext,
					   precv->m_results_buf);
			precv->m_results_buf = 0UL;
			plts->last_results_buf = 0UL;
		}

		if (kerror.errcode == StromError_CpuReCheck &&
			OidIsValid(plts->p.fn_cpu_fallback))
		{
			/* CPU fallback, if any */
			retval = kernel_launch_helper(fcinfo,
										  plts->p.fn_cpu_fallback,
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
	/* close gstore_fdw if any */
	forthree(lc1, ptask->gstore_oid_list,
			 lc2, ptask->gstore_devptr_list,
			 lc3, ptask->gstore_dindex_list)
	{
		Oid			gstore_oid __attribute__((unused)) = lfirst_oid(lc1);
		CUdeviceptr	m_deviceptr = (CUdeviceptr) lfirst(lc2);
		cl_int		cuda_dindex = lfirst_int(lc3);

		if (cuda_dindex < 0)
			gpuMemFree(plts->gts.gcontext, m_deviceptr);
		else
			gpuIpcCloseMemHandle(plts->gts.gcontext, m_deviceptr);
	}
	gpuMemFree(plts->gts.gcontext, (CUdeviceptr)ptask);

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
	plcudaTaskState *plts;
	char		   *source;
	text		   *result;

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


	plts = plcuda_exec_begin(tuple, NULL);
	source = pgstrom_cuda_source_string(plts->gts.program_id);
	PG_TRY();
	{
		result = cstring_to_text(source);
	}
	PG_CATCH();
	{
		free(source);
		PG_RE_THROW();
	}
	PG_END_TRY();
	free(source);
	plcuda_exec_end(plts);

	ReleaseSysCache(tuple);

	PG_RETURN_TEXT_P(result);
}
PG_FUNCTION_INFO_V1(plcuda_function_source);

/*
 * plcuda_process_task
 */
static int
plcuda_process_task(GpuTask *gtask, CUmodule cuda_module)
{
	plcudaTask	   *ptask = (plcudaTask *) gtask;
	GpuContext	   *gcontext = GpuWorkerCurrentContext;
	void		   *kern_args[3];
	cl_int			warp_size;
	cl_int			block_size;
	cl_int			grid_size;
	CUfunction		kern_plcuda_prep;
	CUfunction		kern_plcuda_main;
	CUfunction		kern_plcuda_post;
	CUdeviceptr		m_kern_plcuda = (CUdeviceptr)&ptask->kern;
	CUdeviceptr		m_results_buf = ptask->m_results_buf;
	CUdeviceptr		m_working_buf = 0UL;
	CUresult		rc;
	int				retval = 100001;

	/* property of the device */
	warp_size = devAttrs[CU_DINDEX_PER_THREAD].WARP_SIZE;

	/* plcuda_prep_kernel_entrypoint, if any */
	if (ptask->exec_prep_kernel)
	{
		rc = cuModuleGetFunction(&kern_plcuda_prep,
								 cuda_module,
								 "plcuda_prep_kernel_entrypoint");
		if (rc != CUDA_SUCCESS)
			werror("failed on cuModuleGetFunction: %s", errorText(rc));
	}
	/* plcuda_main_kernel_entrypoint */
	rc = cuModuleGetFunction(&kern_plcuda_main,
							 cuda_module,
							 "plcuda_main_kernel_entrypoint");
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction: %s", errorText(rc));
	/* plcuda_post_kernel_entrypoint */
	if (ptask->exec_post_kernel)
	{
		rc = cuModuleGetFunction(&kern_plcuda_post,
								 cuda_module,
								 "plcuda_post_kernel_entrypoint");
		if (rc != CUDA_SUCCESS)
			werror("failed on cuModuleGetFunction: %s", errorText(rc));
	}

	/* working buffer if required */
	if (ptask->kern.working_bufsz > 0)
	{
		rc = gpuMemAllocManaged(gcontext,
								&m_working_buf,
								ptask->kern.working_bufsz,
								CU_MEM_ATTACH_GLOBAL);
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
			goto out_of_resource;
		else if (rc != CUDA_SUCCESS)
			werror("failed on gpuMemAllocManaged: %s", errorText(rc));
	}

	/* move the control block + argument buffer */
	rc = cuMemPrefetchAsync((CUdeviceptr)&ptask->kern,
							KERN_PLCUDA_DMASEND_LENGTH(&ptask->kern),
							CU_DEVICE_PER_THREAD,
							CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuMemPrefetchAsync: %s", errorText(rc));

	/* kernel arguments (common for all thress kernels) */
	kern_args[0] = &m_kern_plcuda;
	kern_args[1] = &m_working_buf;
	kern_args[2] = &m_results_buf;

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
			rc = gpuOccupancyMaxPotentialBlockSize(NULL,
												   &block_size,
												   kern_plcuda_prep,
												ptask->kern.prep_shmem_blocksz,
												ptask->kern.prep_shmem_unitsz);
			if (rc != CUDA_SUCCESS)
				werror("failed on gpuOccupancyMaxPotentialBlockSize: %s",
					   errorText(rc));
			grid_size = (ptask->kern.prep_num_threads +
						 block_size - 1) / block_size;
		}
		rc = cuLaunchKernel(kern_plcuda_prep,
							grid_size, 1, 1,
							block_size, 1, 1,
							ptask->kern.prep_shmem_blocksz +
							ptask->kern.prep_shmem_unitsz * block_size,
							NULL,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuLaunchKernel: %s "
				   "(prep-kernel: grid=%d block=%d shmem=%u)",
				   errorText(rc),
				   grid_size, block_size,
				   ptask->kern.prep_shmem_blocksz +
				   ptask->kern.prep_shmem_unitsz * block_size);
		wdebug("PL/CUDA prep-kernel: grid=%d block=%d shmem=%u",
			   grid_size, block_size,
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
		rc = gpuOccupancyMaxPotentialBlockSize(NULL,
											   &block_size,
											   kern_plcuda_main,
											   ptask->kern.main_shmem_blocksz,
											   ptask->kern.main_shmem_unitsz);
		if (rc != CUDA_SUCCESS)
			werror("failed on gpuOccupancyMaxPotentialBlockSize: %s",
				   errorText(rc));
		grid_size = (ptask->kern.prep_num_threads +
					 block_size - 1) / block_size;
	}

	rc = cuLaunchKernel(kern_plcuda_main,
						grid_size, 1, 1,
						block_size, 1, 1,
						ptask->kern.main_shmem_blocksz +
						ptask->kern.main_shmem_unitsz * block_size,
						NULL,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s "
			   "(main-kernel: grid=%d block=%d shmem=%u)",
			   errorText(rc),
			   (cl_uint)grid_size, (cl_uint)block_size,
			   ptask->kern.main_shmem_blocksz +
			   ptask->kern.main_shmem_unitsz * block_size);
	wdebug("PL/CUDA main-kernel: grid=%d block=%d shmem=%u",
		   grid_size, block_size,
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
			rc = gpuOccupancyMaxPotentialBlockSize(NULL,
												   &block_size,
												   kern_plcuda_post,
												ptask->kern.post_shmem_blocksz,
												ptask->kern.post_shmem_unitsz);
			if (rc != CUDA_SUCCESS)
				werror("failed on gpuOccupancyMaxPotentialBlockSize: %s",
					   errorText(rc));
			grid_size = (ptask->kern.post_num_threads +
						 block_size - 1) / block_size;
		}

		rc = cuLaunchKernel(kern_plcuda_post,
							grid_size, 1, 1,
							block_size, 1, 1,
							ptask->kern.post_shmem_blocksz +
							ptask->kern.post_shmem_unitsz * block_size,
							NULL,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuLaunchKernel: %s "
				   "(post-kernel: grid=%d block=%d shmem=%u)",
				   errorText(rc),
				   grid_size, block_size,
				   ptask->kern.post_shmem_blocksz +
				   ptask->kern.post_shmem_unitsz * block_size);
		wdebug("PL/CUDA post-kernel: grid=%d block=%d shmem=%u",
			   grid_size, block_size,
			   ptask->kern.post_shmem_blocksz +
			   ptask->kern.post_shmem_unitsz * block_size);
	}
	/* write back the control block */
	rc = cuMemPrefetchAsync((CUdeviceptr)&ptask->kern,
							KERN_PLCUDA_DMARECV_LENGTH(&ptask->kern),
							CU_DEVICE_CPU,
							CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuMemPrefetchAsync: %s", errorText(rc));

	/* write back the result buffer, if any */
	if (m_results_buf != 0UL)
	{
		rc = cuMemPrefetchAsync(ptask->m_results_buf,
								ptask->kern.results_bufsz,
								CU_DEVICE_CPU,
								CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuMemPrefetchAsync: %s", errorText(rc));
	}

	rc = cuEventRecord(CU_EVENT0_PER_THREAD, CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventRecord: %s", errorText(rc));

	/* Point of synchronization */
	rc = cuEventSynchronize(CU_EVENT0_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventSynchronize: %s", errorText(rc));

	/* check kernel execution status */
	memset(&ptask->task.kerror, 0, sizeof(kern_errorbuf));
	if (ptask->exec_prep_kernel &&
		ptask->kern.kerror_prep.errcode != StromError_Success)
		ptask->task.kerror = ptask->kern.kerror_prep;
	else if (ptask->kern.kerror_main.errcode != StromError_Success)
		ptask->task.kerror = ptask->kern.kerror_main;
	else if (ptask->exec_post_kernel &&
			 ptask->kern.kerror_post.errcode != StromError_Success)
		ptask->task.kerror = ptask->kern.kerror_post;

	retval = 0;

out_of_resource:
	if (m_working_buf != 0UL)
	{
		rc = gpuMemFree(gcontext, m_working_buf);
		if (rc != CUDA_SUCCESS)
			werror("failed on gpuMemFree: %s", errorText(rc));
	}
	return retval;
}

/*
 * plcuda_release_task
 */
static void
plcuda_release_task(GpuTask *gtask)
{
	plcudaTask	   *ptask = (plcudaTask *) gtask;
	GpuContext	   *gcontext = ptask->task.gts->gcontext;

	if (ptask->m_results_buf)
		gpuMemFree(gcontext, ptask->m_results_buf);
	gpuMemFree(gcontext, (CUdeviceptr)ptask);
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
