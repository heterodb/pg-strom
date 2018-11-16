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

Datum plcuda_function_validator(PG_FUNCTION_ARGS);
Datum plcuda_function_handler(PG_FUNCTION_ARGS);
Datum pgsql_table_attr_numbers_by_names(PG_FUNCTION_ARGS);
Datum pgsql_table_attr_number_by_name(PG_FUNCTION_ARGS);
Datum pgsql_table_attr_types_by_names(PG_FUNCTION_ARGS);
Datum pgsql_table_attr_type_by_name(PG_FUNCTION_ARGS);
Datum pgsql_check_attrs_of_types(PG_FUNCTION_ARGS);
Datum pgsql_check_attrs_of_type(PG_FUNCTION_ARGS);
Datum pgsql_check_attr_of_type(PG_FUNCTION_ARGS);

typedef struct
{
	const char	   *proname;
	Oid				proowner;
	oidvector	   *proargtypes;
	Oid				prorettype;
	List		   *all_type_oids;
	List		   *all_type_names;
	const char	   *source;
	int				lineno;
	StringInfo		curr;
	StringInfoData	decl;
	StringInfoData	main;
	StringInfoData	emsg;
	FunctionCallInfo fcinfo;
	MemoryContext	results_memcxt;
	Oid				fn_sanity_check;
	int				include_count;
	List		   *include_func_oids;
	List		   *link_libs;
	char			afname[200];	/* argument file name */
	char			rfname[200];	/* result file name */
	char		   *prog_args[FUNC_MAX_ARGS];
} plcuda_code_context;

static void plcuda_expand_source(plcuda_code_context *con, char *source);
static bool	plcuda_enable_debug;	/* GUC */
static const char *__attr_unused = "__attribute__((unused))";

/*
 * get_type_name
 */
static inline char *
get_type_name(Oid type_oid)
{
	HeapTuple		tup;
	Form_pg_type	typeForm;
	char		   *typeName;
	char		   *pos;

	tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	typeForm = (Form_pg_type) GETSTRUCT(tup);
	typeName = pstrdup(NameStr(typeForm->typname));
	for (pos=typeName; *pos != '\0'; pos++)
	{
		if (!isalnum(*pos) && *pos != '_')
		{
			typeName = psprintf("pgtype_%u", type_oid);
			break;
		}
	}
	ReleaseSysCache(tup);

	return typeName;
}

/*
 * plcuda_init_code_context
 */
static void
plcuda_init_code_context(plcuda_code_context *con,
						 HeapTuple protup,
						 FunctionCallInfo fcinfo,
						 MemoryContext results_memcxt)
{
	Form_pg_proc	proc = (Form_pg_proc) GETSTRUCT(protup);
	List		   *all_type_oids = NIL;
	List		   *all_type_names = NIL;
	ListCell	   *lc1, *lc2;
	int				i;

	memset(con, 0, sizeof(plcuda_code_context));
	con->proname		= NameStr(proc->proname);
	con->proowner		= proc->proowner;
	con->proargtypes	= &proc->proargtypes;
	con->prorettype		= proc->prorettype;
	con->source			= NameStr(proc->proname);
	con->lineno			= 1;
	initStringInfo(&con->emsg);
	con->curr			= NULL;
	con->fcinfo			= fcinfo;	/* NULL if only validation */
	con->results_memcxt	= results_memcxt;
	con->include_count	= 0;
	con->include_func_oids = NIL;

	/* makes type oid/name list */
	all_type_oids = list_make1_oid(proc->prorettype);
	all_type_names = list_make1(get_type_name(proc->prorettype));
	for (i=0; i < proc->proargtypes.dim1; i++)
	{
		Oid		type_oid = proc->proargtypes.values[i];
		char   *type_name = get_type_name(type_oid);

		forboth (lc1, all_type_oids,
				 lc2, all_type_names)
		{
			if (type_oid == lfirst_oid(lc1))
				break;
			if (strcmp(type_name, lfirst(lc2)) == 0)
			{
				appendStringInfo(
					&con->emsg,
					"\n%s:  Different types but have same name are used: %s",
					con->source,
					type_name);
				break;
			}
		}
		if (!lc1 && !lc2)
		{
			all_type_oids = lappend_oid(all_type_oids, type_oid);
			all_type_names = lappend(all_type_names, type_name);
		}
	}
	con->all_type_oids = all_type_oids;
	con->all_type_names = all_type_names;
}

#define EMSG(fmt,...)									\
	appendStringInfo(&con->emsg, "\n%s:%d  " fmt,		\
					 con->source, con->lineno, ##__VA_ARGS__)

/*
 * plcuda_lookup_helper
 */
static Oid
plcuda_lookup_helper(plcuda_code_context *con,
					 const char *cmd, List *options, Oid result_type)
{
	List	   *names = NIL;
	ListCell   *lc;
	Oid			func_oid = InvalidOid;
	Oid			type_oid = InvalidOid;
	StringInfoData temp;

	if (list_length(options) == 1)
	{
		char   *ident = linitial(options);

		names = list_make1(makeString(ident));
	}
	else if (list_length(options) == 3)
	{
		/* function in a particular schema */
		char   *nspname = linitial(options);
		char   *dot = lsecond(options);
		char   *proname = lthird(options);

		if (strcmp(dot, ".") == 0)
			names = list_make2(makeString(nspname),
							   makeString(proname));
	}

	if (names != NIL)
	{
		func_oid = LookupFuncName(names,
								  con->proargtypes->dim1,
								  con->proargtypes->values,
								  true);
		if (!OidIsValid(func_oid))
		{
			EMSG("function %s was not found", NameListToString(names));
			return InvalidOid;
		}
		type_oid = get_func_rettype(func_oid);
		if (result_type != type_oid)
		{
			EMSG("function %s has unexpected result typs: %s, instead of %s",
				 NameListToString(names),
				 format_type_be(type_oid),
				 format_type_be(result_type));
			return InvalidOid;
		}
		if (!pg_proc_ownercheck(func_oid, con->proowner))
		{
			EMSG("permission denied on helper function %s",
				 NameListToString(options));
			return InvalidOid;
		}
		return func_oid;
	}
	initStringInfo(&temp);
	foreach (lc, options)
		appendStringInfo(&temp, " %s", quote_identifier(lfirst(lc)));
	EMSG("%s has invalid identifier: %s", cmd, temp.data);
	pfree(temp.data);
	return InvalidOid;
}

/*
 * plcuda_parse_cmd_options - parses the '#plcuda_xxx' line
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

/*
 * plcuda_code_include
 */
static void
plcuda_code_include(plcuda_code_context *con, Oid fn_extra_include)
{
	const char	   *func_name = get_func_name(fn_extra_include);
	FmgrInfo		flinfo;
	FunctionCallInfoData fcinfo;
	FunctionCallInfo __fcinfo = con->fcinfo;
	Datum			value;
	ListCell	   *lc;

	/* prevent infinite inclusion */
	foreach (lc, con->include_func_oids)
	{
		if (lfirst_oid(lc) == fn_extra_include)
		{
			EMSG("\"%s\" leads infinite inclusion", func_name);
			return;
		}
	}

	/* see OidFunctionCallXX */
	Assert(__fcinfo != NULL);
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
	value = FunctionCallInvoke(&fcinfo);
	if (fcinfo.isnull)
	{
		EMSG("function %s returned NULL", format_procedure(fn_extra_include));
	}
	else
	{
		const char *source_saved = con->source;
		int			lineno_saved = con->lineno;

		appendStringInfo(con->curr,
						 "/* ------ BEGIN %s ------ */\n", func_name);
		con->include_func_oids = lappend_oid(con->include_func_oids,
											 fn_extra_include);
		con->source = func_name;
		con->lineno = 1;
		plcuda_expand_source(con, TextDatumGetCString(value));
		con->lineno = lineno_saved;
		con->source = source_saved;
		con->include_func_oids =
			list_truncate(con->include_func_oids,
						  con->include_func_oids->length - 1);
		appendStringInfo(con->curr,
						 "/* ------ END %s ------ */\n",
						 func_name);
	}
}

/*
 * plcuda_expand_source
 */
static void
plcuda_expand_source(plcuda_code_context *con, char *source)
{
	char   *source_pos;
	char   *line, *pos;
	bool	had_code_out_of_block_error = false;

	for (line = strtok_r(source, "\n\r", &source_pos), con->lineno=1;
		 line != NULL;
		 line = strtok_r(NULL, "\n\r", &source_pos), con->lineno++)
	{
		char   *end = line + strlen(line) - 1;
		char   *cmd;
		List   *options;

		/* trimming the whitespace in the tail */
		end = line + strlen(line) - 1;
		while (line <= end && isspace(*end))
			*end-- = '\0';

		if (strncmp(line, "#plcuda_", 8) != 0)
		{
			if (con->curr)
				appendStringInfo(con->curr, "%s\n", line);
			else if (!had_code_out_of_block_error)
			{
				EMSG("code out of valid blocks");
				had_code_out_of_block_error = true;
			}
			continue;
		}
		/* pick up '#plcuda_' command line */
		for (pos = line; !isspace(*pos) && *pos != '\0'; pos++);
		cmd = pnstrdup(line, pos - line);
		if (!plcuda_parse_cmd_options(pos, &options))
		{
			EMSG("pl/cuda command parse error:\n%s", line);
			continue;
		}

		if (strcmp(cmd, "#plcuda_decl") == 0)
		{
			if (con->decl.data != NULL)
				EMSG("%s appeared twice", cmd);
			else if (list_length(options) > 0)
				EMSG("%s cannot takes options", cmd);
			else
			{
				initStringInfo(&con->decl);
				con->curr = &con->decl;
				had_code_out_of_block_error = false;
			}
		}
		else if (strcmp(cmd, "#plcuda_begin") == 0)
		{
			if (con->main.data != NULL)
				EMSG("%s appeared twice", cmd);
			else if (list_length(options) > 0)
				EMSG("%s cannot takes options", cmd);
			else
			{
				initStringInfo(&con->main);
				con->curr = &con->main;
				had_code_out_of_block_error = false;
			}
		}
		else if (strcmp(cmd, "#plcuda_end") == 0)
		{
			if (con->curr != &con->decl &&
				con->curr != &con->main)
				EMSG("%s is used out of code block", cmd);
			else
				con->curr = NULL;
		}
		else if (strcmp(cmd, "#plcuda_sanity_check") == 0)
		{
			Oid		func_oid = plcuda_lookup_helper(con, cmd, options,
													BOOLOID);
			if (OidIsValid(con->fn_sanity_check))
				EMSG("%s appeared twice", cmd);
			else
				con->fn_sanity_check = func_oid;
		}
		else if (strcmp(cmd, "#plcuda_include") == 0)
		{
			Oid		func_oid = plcuda_lookup_helper(con, cmd, options,
													TEXTOID);
			con->include_count++;
			if (OidIsValid(func_oid) && con->fcinfo)
				plcuda_code_include(con, func_oid);
		}
		else if (strcmp(cmd, "#plcuda_library") == 0)
		{
			if (list_length(options) != 1)
				EMSG("syntax error: %s", cmd);
			else
			{
				char	   *library = linitial(options);
				ListCell   *lc;

				foreach (lc, con->link_libs)
				{
					if (strcmp(library, lfirst(lc)) == 0)
						break;
				}
				if (!lc)
					con->link_libs = lappend(con->link_libs, library);
			}
		}
		else
		{
			EMSG("unknown command: %s", cmd);
		}
	}
}

/*
 * plcuda_add_extra_typeinfo - put referenced type declaration for convenient
 */
static void
__add_extra_rowtype_info(StringInfo source,
						 char *type_name, Oid type_relid)
{
	HeapTuple		reltup;
	HeapTuple		atttup;
	Form_pg_class	relForm;
	Form_pg_attribute attForm;
	AttrNumber		anum;

	reltup = SearchSysCache1(RELOID, ObjectIdGetDatum(type_relid));
	if (!HeapTupleIsValid(reltup))
		elog(ERROR, "cache lookup failed for relation %u", type_relid);
	relForm = (Form_pg_class) GETSTRUCT(reltup);

	appendStringInfo(
		source,
		"static __device__ kern_colmeta pg_%s_typeinfo[] %s = {\n",
		type_name, __attr_unused);

	for (anum=1; anum <= relForm->relnatts; anum++)
	{
		atttup = SearchSysCache2(ATTNUM,
								 ObjectIdGetDatum(type_relid),
								 Int16GetDatum(anum));
		if (!HeapTupleIsValid(atttup))
			elog(ERROR, "cache lookup failed for attribute %d of relation %u",
				 anum, type_relid);
		attForm = (Form_pg_attribute) GETSTRUCT(atttup);
		appendStringInfo(
			source,
			"    { %s, %d, %d, %d, %d, %u, %d, 0, 0 },\n",
			attForm->attbyval ? "true" : "false",
			typealign_get_width(attForm->attalign),
			attForm->attlen,
			attForm->attnum,
			attForm->attcacheoff,
			attForm->atttypid,
			attForm->atttypmod);
		ReleaseSysCache(atttup);
	}
	ReleaseSysCache(reltup);

	appendStringInfo(
		source,
		"};\n");
}

static void
plcuda_add_extra_typeinfo(StringInfo source, plcuda_code_context *con)
{
	ListCell   *lc1, *lc2;

	appendStringInfo(source, "/* ---- PG Type OIDs ---- */\n");
	pgstrom_codegen_typeoid_declarations(source);
	appendStringInfo(source, "\n/* ---- PG Type Properties ---- */\n");
	forboth (lc1, con->all_type_oids,
			 lc2, con->all_type_names)
	{
		Oid				type_oid = lfirst_oid(lc1);
		char		   *type_name = lfirst(lc2);
		HeapTuple		tup;
		Form_pg_type	typeForm;

		tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "cache lookup failed for type %u", type_oid);
		typeForm = (Form_pg_type) GETSTRUCT(tup);
		if (typeForm->typtype == TYPTYPE_COMPOSITE)
		{
			__add_extra_rowtype_info(source, type_name,
									 typeForm->typrelid);
		}
		else
		{
			appendStringInfo(
				source,
				"static __device__ kern_colmeta pg_%s_typeinfo %s\n"
				"    = { %s, %d, %d, 0, -1, %u, -1, 0, 0 };\n",
				type_name, __attr_unused,
				typeForm->typbyval ? "true" : "false",
				typealign_get_width(typeForm->typalign),
				typeForm->typlen,
				type_oid);
		}
		ReleaseSysCache(tup);
	}
	appendStringInfoChar(source, '\n');
}

/*
 * plcuda_make_flat_source
 */
static inline const char *
plcuda_get_type_label(Oid type_oid, int16 *p_typlen, bool *p_typbyval)
{
	const char *label = NULL;
	int16		typlen;
	bool		typbyval;

	get_typlenbyval(type_oid, &typlen, &typbyval);
	if (type_oid == REGGSTOREOID)
	{
		typlen	= -2;
		typbyval = false;
		label = "void *";			/* device pointer */
	}
	else if (!typbyval)
	{
		if (typlen == -1)
			label = "varlena *";	/* device pointer */
		else if (typlen > 0)
			label = "void *";		/* device pointer */
		else
			elog(ERROR, "unexpected type properties");
	}
	else if (type_oid == FLOAT4OID)
		label = "float";
	else if (type_oid == FLOAT8OID)
		label = "double";
	else if (typlen == sizeof(cl_char))
		label = "cl_char";
	else if (typlen == sizeof(cl_short))
		label = "cl_short";
	else if (typlen == sizeof(cl_int))
		label = "cl_int";
	else if (typlen == sizeof(cl_long))
		label = "cl_long";
	else
		elog(ERROR, "unexpected type properties");
	if (p_typlen)
		*p_typlen = typlen;
	if (p_typbyval)
		*p_typbyval = typbyval;
	return label;
}

static void
plcuda_make_flat_source(StringInfo source, plcuda_code_context *con)
{
	oidvector	   *proargtypes = con->proargtypes;
	int16			typlen;
	bool			typbyval;
	const char	   *label;
	int				i;

	appendStringInfo(
		source,
		"/* ----------------------------------------\n"
		" * PL/CUDA function (%s)\n"
		" * ----------------------------------------*/\n"
		"#define MAXIMUM_ALIGNOF %u\n"
		"#define NAMEDATALEN %u\n"
		"#define KERN_CONTEXT_VARLENA_BUFSZ 0\n"
		"#include \"cuda_common.h\"\n"
		"#include <cuda_runtime.h>\n"
		"\n",
		con->proname,
		MAXIMUM_ALIGNOF,
		NAMEDATALEN);
	plcuda_add_extra_typeinfo(source, con);
	if (con->decl.data)
		appendStringInfoString(source, con->decl.data);

	if (con->prorettype == REGGSTOREOID)
	{
		label = "cl_uint";
		typlen = sizeof(cl_int);
		typbyval = true;
	}
	else
	{
		label = plcuda_get_type_label(con->prorettype, &typlen, &typbyval);
	}
	appendStringInfo(
		source,
		"typedef %s PLCUDA_RESULT_TYPE;\n"
		"#define PLCUDA_RESULT_TYPBYVAL %d\n"
		"#define PLCUDA_RESULT_TYPLEN   %d\n"
		"#define PLCUDA_NUM_ARGS        %d\n"
		"#define PLCUDA_ARG_ISNULL(x)	(p_args[(x)] == NULL)\n"
		"#define PLCUDA_GET_ARGVAL(x,type) (PLCUDA_ARG_ISNULL(x) ? 0 : *((type *)p_args[(x)]))\n"
		"\n"
		"static PLCUDA_RESULT_TYPE plcuda_main(void *p_args[])\n"
		"{\n"
		"  %s retval = %s;\n",
		label, typlen, typbyval, con->proargtypes->dim1, label,
		strchr(label, '*') ? "NULL" : "0");

	for (i=0; i < proargtypes->dim1; i++)
	{
		Oid		type_oid = proargtypes->values[i];

		label = plcuda_get_type_label(type_oid, &typlen, &typbyval);
		if (typbyval)
			appendStringInfo(
				source,
				"  %s arg%d %s = PLCUDA_GET_ARGVAL(%d,%s);\n",
				label, i+1, __attr_unused, i, label);
		else
			appendStringInfo(
				source,
				"  %s arg%d %s = (%s)p_args[%d];\n",
				label, i+1, __attr_unused, label, i);
	}
	if (con->main.data)
		appendStringInfo(source, "{\n%s}\n", con->main.data);
	else
		appendStringInfoString(source, "exit(1);\n");	//NULL result
	appendStringInfo(source, "  return retval;\n}\n\n");

	/* merge PL/CUDA host template */
	appendStringInfoString(source, pgsql_host_plcuda_code);
}

/*
 * plcuda_build_program
 */
static void
plcuda_build_program(plcuda_code_context *con,
					 const char *name, StringInfo source)
{
	File		fdesc;
	FILE	   *filp;
	ssize_t		nbytes;
	int			status;
	ListCell   *lc;
	char		path[MAXPGPATH];
	StringInfoData cmd;
	StringInfoData log;

	initStringInfo(&cmd);
	initStringInfo(&log);
	/* write out source file */
	snprintf(path, sizeof(path), "%s.cu", name);
	fdesc = PathNameOpenFile(path, O_RDWR|O_CREAT|O_EXCL|PG_BINARY, 0600);
	nbytes = FileWrite(fdesc, source->data, source->len
#if PG_VERSION_NUM >= 100000
					   ,WAIT_EVENT_DATA_FILE_WRITE
#endif
		);
	if (nbytes != source->len)
		elog(ERROR, "could not write source file of PL/CUDA");
	FileClose(fdesc);

	/* make nvcc command line */
	appendStringInfo(
		&cmd,
		CUDA_BINARY_PATH "/nvcc "
		" --gpu-architecture=sm_%lu"
		" --default-stream=per-thread"
		" -I " PGSHAREDIR "/extension"
		" -O2 -std=c++11",
		devComputeCapability);
	if (plcuda_enable_debug)
		appendStringInfo(&cmd, " -g -G");
	foreach (lc, con->link_libs)
		appendStringInfo(&cmd, " -l%s", (char *)lfirst(lc));
	appendStringInfo(&cmd, " -o %s %s", name, path);

	/* kick nvcc compiler */
	if (plcuda_enable_debug)
		elog(NOTICE, "PL/CUDA build:\n%s", cmd.data);
	filp = OpenPipeStream(cmd.data, PG_BINARY_R);
	if (!filp)
		elog(ERROR, "could not kick nvcc compiler: %s", cmd.data);
	do {
		enlargeStringInfo(&log, 4096);
		nbytes = fread(log.data + log.len, 1, 4096, filp);
		if (nbytes < 0)
		{
			if (errno == EINTR)
				continue;
			elog(ERROR, "failed on fread: %m");
		}
		log.len += nbytes;
	} while (nbytes > 0);
	status = ClosePipeStream(filp);

	if (status != 0)
		elog(ERROR, "PL/CUDA compilation failed.\n%s", log.data);
	else if (log.len > 0)
		elog(NOTICE, "PL/CUDA compilation log.\n%s", log.data);

	pfree(log.data);
	pfree(cmd.data);
}

/*
 * plcuda_function_validator
 */
Datum
plcuda_function_validator(PG_FUNCTION_ARGS)
{
	Oid				func_oid = PG_GETARG_OID(0);
	HeapTuple		tuple;
	Form_pg_proc	proc;
	Datum			value;
	bool			isnull;
	char			prokind;
	int16			typlen;
	bool			typbyval;
	int				i;
	plcuda_code_context con;

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

	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	value = SysCacheGetAttr(PROCOID, tuple,
							Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "PL/CUDA source is missing");
	proc = (Form_pg_proc) GETSTRUCT(tuple);

	/* check result and arguments types */
	get_typlenbyval(proc->prorettype, &typlen, &typbyval);
	if (!typbyval && !(typlen > 0 || typlen==-1))
		elog(ERROR, "type %s is not supported to use in PL/CUDA",
			 format_type_be(proc->prorettype));
	for (i=0; i < proc->proargtypes.dim1; i++)
	{
		Oid		type_oid = proc->proargtypes.values[i];

		get_typlenbyval(type_oid, &typlen, &typbyval);
		if (!typbyval && !(typlen > 0 || typlen==-1))
			elog(ERROR, "type %s is not supported to use in PL/CUDA",
				 format_type_be(type_oid));
	}

	/* check argument (fix-len or varlena) */
	plcuda_init_code_context(&con, tuple, NULL, NULL);
	plcuda_expand_source(&con, TextDatumGetCString(value));
	if (con.emsg.len > 0)
		elog(ERROR, "failed on kernel source construction:%s", con.emsg.data);
	if (con.include_count > 0)
	{
		elog(NOTICE, "#plcuda_include changes PL/CUDA source on run-time, so validator does not try to compile on CREATE FUNCTION time");
	}
	else
	{
		StringInfoData source;
		File		tempFile;

		initStringInfo(&source);
		plcuda_make_flat_source(&source, &con);
		tempFile = OpenTemporaryFile(false);
		plcuda_build_program(&con, FilePathName(tempFile), &source);
		FileClose(tempFile);

		pfree(source.data);
	}
	ReleaseSysCache(tuple);

	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(plcuda_function_validator);

/*
 * plcuda_setup_arguments
 */
static void
plcuda_setup_arguments(plcuda_code_context *con)
{
	FunctionCallInfo fcinfo = con->fcinfo;
	oidvector  *argtypes = con->proargtypes;
	ssize_t		required;
	ssize_t		offset[FUNC_MAX_ARGS];
	char		name[sizeof(con->afname)];
	int			i, fdesc = -1;
	char	   *buffer = NULL;

	required = 0;
	memset(offset, 0, sizeof(offset));
	for (i=0; i < fcinfo->nargs; i++)
	{
		Oid		type_oid = argtypes->values[i];
		int16	typlen;
		bool	typbyval;

		offset[i] = required;
		if (fcinfo->argnull[i])
		{
			con->prog_args[i] = "__null__";
			continue;
		}
		if (type_oid == REGGSTOREOID)
		{
			Oid		ftable_oid = DatumGetObjectId(fcinfo->arg[i]);
			Datum	handle;
			StringInfoData buf;
			const unsigned char *src;
			size_t	i, len;

			handle = DirectFunctionCall1(pgstrom_gstore_export_ipchandle,
										 ObjectIdGetDatum(ftable_oid));
			src = (unsigned char *)VARDATA_ANY(handle);
			len = VARSIZE_ANY_EXHDR(handle);
			initStringInfo(&buf);
			appendStringInfo(&buf, "g:");
			for (i=0; i < len; i++)
			{
				enlargeStringInfo(&buf, 2);
				appendStringInfo(&buf, "%02x", (unsigned int)src[i]);
			}
			con->prog_args[i] = buf.data;
			continue;	/* passed by IPC_mhandle */
		}
		get_typlenbyval(type_oid, &typlen, &typbyval);
		if (typbyval)
		{
			con->prog_args[i] = psprintf("v:%lx", fcinfo->arg[i]);
		}
		else if (typlen > 0)
		{
			con->prog_args[i] = psprintf("r:%lx", required);
			required += MAXALIGN(typlen);
		}
		else if (typlen == -1)
		{
			con->prog_args[i] = psprintf("r:%lx", required);
			required += MAXALIGN(toast_raw_datum_size(fcinfo->arg[i]));
		}
		else
			elog(ERROR, "Data type is not suitable for PL/CUDA: %s",
                 format_type_be(type_oid));
	}
	if (required == 0)
		return;		/* no argument buffer is needed */

	do {
		snprintf(name, sizeof(name), "/.plcuda_%u_argbuf.%u.dat",
				 fcinfo->flinfo->fn_oid, (unsigned int)random());
		fdesc = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fdesc < 0)
		{
			if (errno == EEXIST)
				continue;
			elog(ERROR, "failed on shm_open('%s'): %m", name);
		}
	} while (fdesc < 0);

	PG_TRY();
	{
		if (ftruncate(fdesc, required))
			elog(ERROR, "failed on ftruncate: %m");

		buffer = mmap(NULL, required,
					  PROT_READ | PROT_WRITE,
					  MAP_SHARED,
					  fdesc, 0);
		if (buffer == MAP_FAILED)
			elog(ERROR, "failed on mmap('%s'): %m", name);

		for (i=0; i < fcinfo->nargs; i++)
		{
			Oid		type_oid = argtypes->values[i];
			int16	typlen;
			bool	typbyval;

			if (fcinfo->argnull[i])
				continue;
			if (type_oid == REGGSTOREOID)
				continue;
			get_typlenbyval(type_oid, &typlen, &typbyval);
			if (typbyval)
				continue;
			if (typlen > 0)
			{
				memcpy(buffer + offset[i],
					   DatumGetPointer(fcinfo->arg[i]),
					   typlen);
			}
			else
			{
				struct varlena *datum = (struct varlena *)fcinfo->arg[i];

				Assert(typlen == -1);
				if (VARATT_IS_EXTENDED(datum))
					datum = heap_tuple_untoast_attr(datum);
				memcpy(buffer + offset[i], datum, VARSIZE(datum));
			}
		}
	}
	PG_CATCH();
	{
		if (buffer && munmap(buffer, required) != 0)
			elog(WARNING, "failed on munmap('%s'): %m", con->afname);
		if (shm_unlink(name))
			elog(WARNING, "failed on shm_unlink('%s'): %m", name);
		close(fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();
	if (munmap(buffer, required) != 0)
		elog(WARNING, "failed on munmap('%s'): %m", name);
	if (close(fdesc))
		elog(WARNING, "failed on close(2): %m");
	strcpy(con->afname, name);
}

/*
 * plcuda_setup_result_buffer
 */
static int
plcuda_setup_result_buffer(plcuda_code_context *con)
{
	FunctionCallInfo fcinfo = con->fcinfo;
	int16		typlen;
	bool		typbyval;
	size_t		required;
	char		name[sizeof(con->rfname)];
	int			fdesc = -1;

	get_typlenbyval(con->prorettype, &typlen, &typbyval);
	required = Max(BLCKSZ, typlen);

	/* create a new shared segment */
	do {
		snprintf(name, sizeof(name), "/.plcuda_%u_result.%u.dat",
				 fcinfo->flinfo->fn_oid, (unsigned int)random());
		fdesc = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fdesc < 0)
		{
			if (errno == EEXIST)
				continue;
			elog(ERROR, "failed on shm_open('%s'): %m", name);
		}
	} while (fdesc < 0);

    PG_TRY();
    {
        if (ftruncate(fdesc, required))
			elog(ERROR, "failed on ftruncate: %m");
	}
	PG_CATCH();
	{
		if (shm_unlink(name))
			elog(WARNING, "failed on shm_unlink('%s'): %m", name);
		close(fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();
	strcpy(con->rfname, name);
	return fdesc;
}

/*
 * plcuda_exec_child_program
 */
static void
plcuda_exec_child_program(const char *command, char *cmd_argv[])
{
	DIR	   *dir;
	struct dirent *dent;

	/*
	 * For security reason, close all the file-descriptors except for stdXXX
	 */
	dir = opendir("/proc/self/fd");
	if (!dir)
	{
		fprintf(stderr, "failed on opendir('/proc/self/fd'): %m\n");
		_exit(2);
	}
	while ((dent = readdir(dir)) != NULL)
	{
		const char *pos = dent->d_name;
		int			fdesc;

		while (isdigit(*pos))
			pos++;
		if (*pos == '\0')
		{
			fdesc = atoi(dent->d_name);

			if (fdesc > 2)
				fcntl(fdesc, F_SETFD, O_CLOEXEC);
		}
	}
	closedir(dir);

	/* kick PL/CUDA program */
	execv(command, cmd_argv);

	fprintf(stderr, "failed on execv('%s', ...): %m\n", command);
	_exit(2);
}

/*
 * plcuda_wait_child_program
 */
static void
plcuda_sigchld_handler(SIGNAL_ARGS)
{
	SetLatch(MyLatch);
}

static bool
plcuda_wait_child_program(pid_t child)
{
	pqsigfunc	sigchld_saved = pqsignal(SIGCHLD, plcuda_sigchld_handler);
	int			status;
	bool		isnull;

	/* wait for completion of the child process */
	PG_TRY();
	{
		for (;;)
		{
			pid_t	rv;

			CHECK_FOR_INTERRUPTS();
			rv = waitpid(child, &status, WNOHANG);
			if (rv > 0)
			{
				Assert(rv == child);
				if (WIFEXITED(status) || WIFSIGNALED(status))
					break;
			}
			else if (rv < 0)
			{
				if (errno == EINTR)
					continue;
				elog(ERROR, "failed on waitpid(2): %m");
			}
			(void) WaitLatch(MyLatch,
							 WL_LATCH_SET |
							 WL_TIMEOUT |
							 WL_POSTMASTER_DEATH,
							 5000L,
							 PG_WAIT_EXTENSION);
			ResetLatch(MyLatch);
		}
	}
	PG_CATCH();
	{
		kill(child, SIGKILL);
		pqsignal(SIGCHLD, sigchld_saved);
		PG_RE_THROW();
	}
	PG_END_TRY();
	pqsignal(SIGCHLD, sigchld_saved);

	if (WIFSIGNALED(status))
		elog(ERROR, "PL/CUDA script was terminated by signal: %d",
			 WTERMSIG(status));
	if (WEXITSTATUS(status) == 0)
		isnull = false;
	else if (WEXITSTATUS(status) == 1)
		isnull = true;
	else
		elog(ERROR, "PL/CUDA script was terminated abnormally (code: %d)",
			 WEXITSTATUS(status));
	return isnull;
}

/*
 * plcuda_exec_cuda_program
 */
static Datum
plcuda_exec_cuda_program(char *command, plcuda_code_context *con,
						 int rbuf_fdesc, bool *p_isnull)
{
	oidvector  *proargtypes = con->proargtypes;
	char	   *cmd_argv[FUNC_MAX_ARGS + 20];
	pid_t		child;
	int			i, j=0;
	Datum		result = 0;
	bool		isnull;

	//XXX put env, nvprof or ..., if any

	cmd_argv[j++] = command;
	if (con->afname[0] != '\0')
	{
		cmd_argv[j++] = "-a";
		cmd_argv[j++] = con->afname;
	}
	if (con->rfname[0] != '\0')
	{
		cmd_argv[j++] = "-r";
		cmd_argv[j++] = con->rfname;
	}
	cmd_argv[j++] = "--";
	for (i=0; i < proargtypes->dim1; i++)
		cmd_argv[j++] = con->prog_args[i];
	cmd_argv[j++] = NULL;

	/* shows command line if debug mode */
	if (plcuda_enable_debug)
	{
		StringInfoData	temp;

		initStringInfo(&temp);
		appendStringInfo(&temp, "%s", command);
		for (i=1; cmd_argv[i] != NULL; i++)
			appendStringInfo(&temp, " %s", cmd_argv[i]);
		elog(NOTICE, "PL/CUDA: %s", temp.data);
		pfree(temp.data);
	}
	/* fork child */
	child = fork();
	if (child == 0)
		plcuda_exec_child_program(command, cmd_argv);
	else if (child > 0)
		isnull = plcuda_wait_child_program(child);
	else
		elog(ERROR, "failed on fork(2): %m");

	/* get result */
	if (!isnull)
	{
		int16		typlen;
		bool		typbyval;
		struct stat	stbuf;
		void	   *buffer;

		get_typlenbyval(con->prorettype, &typlen, &typbyval);
		if (fstat(rbuf_fdesc, &stbuf) != 0)
			elog(ERROR, "failed on stat('%s'): %m", con->rfname);
		buffer = mmap(NULL, stbuf.st_size,
					  PROT_READ, MAP_SHARED,
					  rbuf_fdesc, 0);
		if (buffer == MAP_FAILED)
			elog(ERROR, "failed on mmap: %m");
		PG_TRY();
		{
			MemoryContext	oldcxt
				= MemoryContextSwitchTo(con->results_memcxt);
			if (typbyval)
			{
				Assert(typlen <= sizeof(Datum));
				memcpy(&result, buffer, typlen);
			}
			else if (typlen > 0)
			{
				char   *temp = palloc(typlen);

				memcpy(temp, buffer, typlen);
				result = PointerGetDatum(temp);
			}
			else if (typlen == -1)
			{
				size_t	len = VARSIZE_ANY(buffer);
				void   *temp = palloc(len);

				memcpy(temp, buffer, len);
				result = PointerGetDatum(temp);
			}
			else
				elog(ERROR, "unexpected type attribute");
			MemoryContextSwitchTo(oldcxt);
		}
		PG_CATCH();
		{
			if (munmap(buffer, stbuf.st_size))
				elog(WARNING, "failed on munmap: %m");
			PG_RE_THROW();
		}
		PG_END_TRY();

		if (munmap(buffer, stbuf.st_size))
			elog(WARNING, "failed on munmap: %m");
	}
	*p_isnull = isnull;
	return result;
}

/*
 * plcuda_scalar_function_handler
 */
static Datum
plcuda_scalar_function_handler(FunctionCallInfo fcinfo,
							   MemoryContext results_memcxt)
{
	FmgrInfo   *flinfo = fcinfo->flinfo;
	HeapTuple	tuple;
	plcuda_code_context con;
	Oid			fn_oid = fcinfo->flinfo->fn_oid;
	int			rbuf_fdesc = -1;
	char		hexsum[33];
	char	   *command;
	struct stat	stbuf;
	StringInfoData source;
	Datum		result;
	Datum		value;
	bool		isnull;

	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(flinfo->fn_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", flinfo->fn_oid);
	value = SysCacheGetAttr(PROCOID, tuple,
							Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "PL/CUDA source is missing");
	plcuda_init_code_context(&con, tuple, fcinfo, results_memcxt);
	plcuda_expand_source(&con, TextDatumGetCString(value));
	if (con.emsg.len > 0)
		elog(ERROR, "failed on kernel source construction:%s", con.emsg.data);

	/* sanity check */
	if (OidIsValid(con.fn_sanity_check))
	{
		FunctionCallInfoData __fcinfo;
		FmgrInfo	__flinfo;

		fmgr_info(con.fn_sanity_check, &__flinfo);
		InitFunctionCallInfoData(__fcinfo, &__flinfo,
								 fcinfo->nargs,
								 fcinfo->fncollation,
								 NULL, NULL);
		memcpy(__fcinfo.arg, fcinfo->arg,
			   sizeof(Datum) * fcinfo->nargs);
		memcpy(__fcinfo.argnull, fcinfo->argnull,
			   sizeof(bool) * fcinfo->nargs);
		result = FunctionCallInvoke(&__fcinfo);
		if (__fcinfo.isnull || DatumGetBool(result))
			elog(ERROR, "PL/CUDA sanity check failed by %s",
				 format_procedure(con.fn_sanity_check));
	}
	initStringInfo(&source);
	plcuda_make_flat_source(&source, &con);
	if (!pg_md5_hash(source.data, source.len, hexsum))
		elog(ERROR, "out of memory");
	command = psprintf("base/%s/%s_plcuda_%u%s_%s_cc%ld",
					   PG_TEMP_FILES_DIR,
					   PG_TEMP_FILE_PREFIX,
					   fn_oid,
					   (plcuda_enable_debug ? "g" : ""),
					   hexsum,
					   devComputeCapability);
	/* lookup PL/CUDA binary */
	if (stat(command, &stbuf) != 0)
	{
		if (errno != ENOENT)
			elog(ERROR, "failed on stat('%s'): %m", command);
		plcuda_build_program(&con, command, &source);
	}

	PG_TRY();
	{
		/* setup arguments */
		plcuda_setup_arguments(&con);
		/* setup result buffer */
		rbuf_fdesc = plcuda_setup_result_buffer(&con);
		/* kick PL/CUDA program */
		result = plcuda_exec_cuda_program(command, &con,
										  rbuf_fdesc, &fcinfo->isnull);
	}
	PG_CATCH();
	{
		if (con.afname[0] != '\0' && !plcuda_enable_debug)
			shm_unlink(con.afname);
		if (con.rfname[0] != '\0' && !plcuda_enable_debug)
			shm_unlink(con.rfname);
		close(rbuf_fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* cleanup */
	if (con.afname[0] != '\0' && !plcuda_enable_debug)
		shm_unlink(con.afname);
	if (con.rfname[0] != '\0' && !plcuda_enable_debug)
		shm_unlink(con.rfname);
	close(rbuf_fdesc);
	ReleaseSysCache(tuple);

	return result;
}

/*
 * plcudaSetFuncContext - for SET RETURNING FUNCTION
 */
typedef struct {
	TypeFuncClass fn_class;
	ArrayType  *results;
	int16		elemlen;
	bool		elembyval;
	char		elemalign;
	cl_int		nitems;
	char	   *curr_pos;
	char	   *tail_pos;
	Datum	   *tup_values;
	bool	   *tup_isnull;
} plcudaSetFuncContext;

/*
 * plcuda_setfunc_firstcall
 */
static plcudaSetFuncContext *
plcuda_setfunc_firstcall(FunctionCallInfo fcinfo,
						 FuncCallContext *fn_cxt,
						 Datum results_datum)
{
	MemoryContext	oldcxt;
	ArrayType	   *results;
	Oid				fn_rettype;
	TupleDesc		fn_tupdesc;
	plcudaSetFuncContext *setfcxt;

	oldcxt = MemoryContextSwitchTo(fn_cxt->multi_call_memory_ctx);
	setfcxt = palloc0(sizeof(plcudaSetFuncContext));
	/* save properties of the array */
	results = DatumGetArrayTypeP(results_datum);
	get_typlenbyvalalign(ARR_ELEMTYPE(results),
						 &setfcxt->elemlen,
						 &setfcxt->elembyval,
						 &setfcxt->elemalign);
	setfcxt->results  = results;
	setfcxt->curr_pos = ARR_DATA_PTR(results);
	setfcxt->tail_pos = (char *)results + VARSIZE(results);

	setfcxt->fn_class = get_call_result_type(fcinfo,
											 &fn_rettype,
											 &fn_tupdesc);
	if (setfcxt->fn_class == TYPEFUNC_SCALAR ||
		setfcxt->fn_class == TYPEFUNC_COMPOSITE)
	{
		if (ARR_ELEMTYPE(results) != fn_rettype)
			elog(ERROR, "PL/CUDA returned wrong type: %s, not %s",
				 format_type_be(results->elemtype),
				 format_type_be(fn_rettype));
		if (ARR_NDIM(results) != 1 ||
			ARR_LBOUND(results)[0] != 0)
			elog(ERROR, "PL/CUDA logic made wrong data array");
		setfcxt->nitems = ARR_DIMS(results)[0];
	}
	else if (setfcxt->fn_class == TYPEFUNC_RECORD)
	{
		if (ARR_NDIM(results) == 1)
		{
			if (ARR_LBOUND(results)[0] != 0)
				elog(ERROR, "PL/CUDA logic made wrong data array");
			fn_tupdesc = CreateTemplateTupleDesc(1, false);
			TupleDescInitEntry(fn_tupdesc, (AttrNumber) 1, "values",
							   ARR_ELEMTYPE(results), -1, 0);
			setfcxt->nitems = ARR_DIMS(results)[0];
			setfcxt->tup_values = palloc(sizeof(Datum));
			setfcxt->tup_isnull = palloc(sizeof(bool));
		}
		else if (ARR_NDIM(results) == 2)
		{
			int		i, nattrs = ARR_DIMS(results)[0];

			if (ARR_LBOUND(results)[0] != 0 ||
				ARR_LBOUND(results)[1] != 0)
				elog(ERROR, "PL/CUDA logic made wrong data array");
			fn_tupdesc = CreateTemplateTupleDesc(nattrs, false);
			for (i=1; i <= nattrs; i++)
				TupleDescInitEntry(fn_tupdesc, (AttrNumber) i,
								   psprintf("v%d", i),
								   ARR_ELEMTYPE(results), -1, 0);
			setfcxt->nitems = ARR_DIMS(results)[1];
			setfcxt->tup_values = palloc(sizeof(Datum) * nattrs);
			setfcxt->tup_isnull = palloc(sizeof(bool) * nattrs);
		}
		else
			elog(ERROR, "PL/CUDA logic made wrong data array");
		fn_cxt->tuple_desc = BlessTupleDesc(fn_tupdesc);
	}
	else
	{
		elog(ERROR, "unexpected PL/CUDA function result class");
	}
	MemoryContextSwitchTo(oldcxt);

	return setfcxt;
}



/*
 * plcuda_setfunc_getnext
 */
static Datum
plcuda_setfunc_getnext(FuncCallContext *fn_cxt,
					   plcudaSetFuncContext *setfcxt,
					   bool *p_isnull)
{
	ArrayType  *results = setfcxt->results;
	bits8	   *nullmap = ARR_NULLBITMAP(results);
	size_t		index = fn_cxt->call_cntr;
	Datum		datum;

	if (setfcxt->fn_class == TYPEFUNC_SCALAR ||
		setfcxt->fn_class == TYPEFUNC_COMPOSITE)
	{
		Assert(ARR_NDIM(results) == 1);
		if (nullmap && att_isnull(index, nullmap))
		{
			*p_isnull = true;
			return 0;
		}
		if (setfcxt->curr_pos >= setfcxt->tail_pos)
			elog(ERROR, "PL/CUDA: corruption of the results");
		setfcxt->curr_pos = (char *)
			att_align_nominal(setfcxt->curr_pos,
							  setfcxt->elemalign);
		datum = fetch_att(setfcxt->curr_pos,
						  setfcxt->elembyval,
						  setfcxt->elemlen);
		if (setfcxt->elemlen > 0)
			setfcxt->curr_pos += setfcxt->elemlen;
		else if (setfcxt->elemlen == -1)
			setfcxt->curr_pos += VARSIZE_ANY(datum);
		else
			elog(ERROR, "PL/CUDA: results has unknown data type");
	}
	else if (setfcxt->fn_class == TYPEFUNC_RECORD)
	{
		TupleDesc	tupdesc = fn_cxt->tuple_desc;
		int			j, natts = tupdesc->natts;
		size_t		index = fn_cxt->call_cntr * natts;
		HeapTuple	tuple;

		memset(setfcxt->tup_isnull, 0, sizeof(bool) * natts);
		memset(setfcxt->tup_values, 0, sizeof(Datum) * natts);
		for (j=0; j < natts; j++)
		{
			if (nullmap && att_isnull(index+j, nullmap))
			{
				setfcxt->tup_isnull[j] = true;
				continue;
			}
			if (setfcxt->curr_pos >= setfcxt->tail_pos)
				elog(ERROR, "PL/CUDA: result is out of range");
			setfcxt->curr_pos = (char *)
				att_align_nominal(setfcxt->curr_pos,
								  setfcxt->elemalign);
			datum = fetch_att(setfcxt->curr_pos,
							  setfcxt->elembyval,
							  setfcxt->elemlen);
			setfcxt->tup_values[j] = datum;
			if (setfcxt->elemlen > 0)
				setfcxt->curr_pos += setfcxt->elemlen;
			else if (setfcxt->elemlen == -1)
				setfcxt->curr_pos += VARSIZE_ANY(datum);
			else
				elog(ERROR, "unexpected PL/CUDA function result type");
		}
		tuple = heap_form_tuple(fn_cxt->tuple_desc,
								setfcxt->tup_values,
								setfcxt->tup_isnull);
		datum = HeapTupleGetDatum(tuple);
	}
	else
	{
		elog(ERROR, "unexpected PL/CUDA function result class");
	}
	*p_isnull = false;
	return datum;
}

/*
 * plcuda_function_handler
 */
Datum
plcuda_function_handler(PG_FUNCTION_ARGS)
{
	FmgrInfo   *flinfo = fcinfo->flinfo;
	FuncCallContext	*fn_cxt;
	plcudaSetFuncContext *setfcxt;
	bool		isnull;
	Datum		datum;

	if (!flinfo->fn_retset)
		return plcuda_scalar_function_handler(fcinfo,
											  CurrentMemoryContext);
	if (SRF_IS_FIRSTCALL())
	{
		fn_cxt = SRF_FIRSTCALL_INIT();
		datum = plcuda_scalar_function_handler(fcinfo,
											   fn_cxt->multi_call_memory_ctx);
		if (fcinfo->isnull)
			SRF_RETURN_DONE(fn_cxt);
		fn_cxt->user_fctx = plcuda_setfunc_firstcall(fcinfo, fn_cxt, datum);
	}
	fn_cxt = SRF_PERCALL_SETUP();
	setfcxt = fn_cxt->user_fctx;
	if (fn_cxt->call_cntr >= setfcxt->nitems)
		SRF_RETURN_DONE(fn_cxt);
	datum = plcuda_setfunc_getnext(fn_cxt, setfcxt, &isnull);
	if (isnull)
		SRF_RETURN_NEXT_NULL(fn_cxt);
	SRF_RETURN_NEXT(fn_cxt, datum);
}
PG_FUNCTION_INFO_V1(plcuda_function_handler);

/*
 * pgstrom_init_plcuda
 */
void
pgstrom_init_plcuda(void)
{
	DefineCustomBoolVariable("pl_cuda.enable_debug",
							 "Enables debugging stuff of PL/CUDA",
							 NULL,
							 &plcuda_enable_debug,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}

/*
 * SQL support functions
 */
#define get_table_desc(table_oid)				\
	getObjectDescriptionOids(RelationRelationId,(table_oid))

Datum
pgsql_table_attr_numbers_by_names(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	ArrayType  *column_names = PG_GETARG_ARRAYTYPE_P(1);
	int			i = 0, nitems;
	Datum		value;
	bool		isnull;
	ArrayIterator iter;
	int2vector *result;

	if (ARR_NDIM(column_names) != 1 ||
		ARR_ELEMTYPE(column_names) != TEXTOID)
		elog(ERROR, "column names must be a vector of text");
	nitems = ARR_DIMS(column_names)[0];
	/* int2vector */
	result = palloc0(offsetof(int2vector, values[nitems]));
	SET_VARSIZE(result, offsetof(int2vector, values[nitems]));
	result->ndim = 1;
	result->dataoffset = 0;
	result->elemtype = INT2OID;
	result->dim1 = nitems;
	result->lbound1 = ARR_LBOUND(column_names)[0];

	iter = array_create_iterator(column_names, 0, NULL);
	while (array_iterate(iter, &value, &isnull))
	{
		char	   *temp = TextDatumGetCString(value);
		AttrNumber	anum = get_attnum(table_oid, temp);

		if (anum == InvalidAttrNumber)
			elog(ERROR, "column '%s' of %s was not found",
				 temp, get_table_desc(table_oid));
		Assert(i < nitems);
		result->values[i++] = anum;
	}
	array_free_iterator(iter);
	Assert(i == nitems);

	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(pgsql_table_attr_numbers_by_names);

Datum
pgsql_table_attr_number_by_name(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	char	   *column_name = TextDatumGetCString(PG_GETARG_DATUM(1));
	AttrNumber	attnum;

	attnum = get_attnum(table_oid, column_name);
	if (attnum == InvalidAttrNumber)
		elog(ERROR, "column '%s' of %s was not found",
			 column_name,
			 getObjectDescriptionOids(RelationRelationId, table_oid));
	PG_RETURN_INT16(attnum);
}
PG_FUNCTION_INFO_V1(pgsql_table_attr_number_by_name);

Datum
pgsql_table_attr_types_by_names(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	ArrayType  *column_names = PG_GETARG_ARRAYTYPE_P(1);
	int			i = 0, nitems;
	Datum		value;
	bool		isnull;
	ArrayIterator iter;
	oidvector  *result;

	if (ARR_NDIM(column_names) != 1 ||
		ARR_ELEMTYPE(column_names) != TEXTOID)
		elog(ERROR, "column names must be a vector of text");
	nitems = ARR_DIMS(column_names)[0];

	/* oidvector */
	result = palloc0(offsetof(oidvector, values[nitems]));
	SET_VARSIZE(result, offsetof(oidvector, values[nitems]));
	result->ndim = 1;
	result->dataoffset = 0;
	result->elemtype = OIDOID;
	result->dim1 = nitems;
	result->lbound1 = ARR_LBOUND(column_names)[0];

	iter = array_create_iterator(column_names, 0, NULL);
	while (array_iterate(iter, &value, &isnull))
	{
		char	   *temp = TextDatumGetCString(value);
		HeapTuple	tup;

		tup = SearchSysCacheAttName(table_oid, temp);
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "column '%s' of %s was not found",
				 temp, get_table_desc(table_oid));
		Assert(i < nitems);
		result->values[i++] = ((Form_pg_attribute) GETSTRUCT(tup))->atttypid;
		ReleaseSysCache(tup);
	}
	array_free_iterator(iter);
	Assert(i == nitems);

	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(pgsql_table_attr_types_by_names);

Datum
pgsql_table_attr_type_by_name(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	char	   *column_name = TextDatumGetCString(PG_GETARG_DATUM(1));
	HeapTuple	tup;
	Oid			type_oid;

	tup = SearchSysCacheAttName(table_oid, column_name);
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "column '%s' of %s was not found",
			 column_name, get_table_desc(table_oid));
	type_oid = ((Form_pg_attribute) GETSTRUCT(tup))->atttypid;
	ReleaseSysCache(tup);

	PG_RETURN_OID(type_oid);
}
PG_FUNCTION_INFO_V1(pgsql_table_attr_type_by_name);

Datum
pgsql_check_attrs_of_types(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	ArrayType  *attr_names = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType  *type_names = PG_GETARG_ARRAYTYPE_P(2);
	Datum		value1, value2;
	bool		isnull1, isnull2;
	ArrayIterator aiter;
	ArrayIterator titer;
	bool		result = true;

	if (ARR_NDIM(attr_names) != 1 || ARR_ELEMTYPE(attr_names) != TEXTOID)
		elog(ERROR, "column names must be a vector of text");
	if (ARR_NDIM(type_names) != 1 || ARR_ELEMTYPE(type_names) != OIDOID)
		elog(ERROR, "types must be vector of regtype");
	if (ARR_DIMS(attr_names)[0] != ARR_DIMS(type_names)[0])
		elog(ERROR, "number of columns and types are mismatch");

	aiter = array_create_iterator(attr_names, 0, NULL);
	titer = array_create_iterator(type_names, 0, NULL);
	while (array_iterate(aiter, &value1, &isnull1) &&
		   array_iterate(titer, &value2, &isnull2))
	{
		char	   *temp = TextDatumGetCString(value1);
		Oid			type_oid = DatumGetObjectId(value2);
		HeapTuple	tup;

		Assert(!isnull1 && !isnull2);
		tup = SearchSysCacheAttName(table_oid, temp);
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "column '%s' of %s was not found",
				 temp, get_table_desc(table_oid));
		if (type_oid != ((Form_pg_attribute) GETSTRUCT(tup))->atttypid)
			result = false;
		ReleaseSysCache(tup);
	}
	array_free_iterator(aiter);
	array_free_iterator(titer);

	PG_RETURN_BOOL(result);
}
PG_FUNCTION_INFO_V1(pgsql_check_attrs_of_types);

Datum
pgsql_check_attrs_of_type(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	ArrayType  *attr_names = PG_GETARG_ARRAYTYPE_P(1);
	Oid			type_oid = PG_GETARG_OID(2);
	Datum		value;
	bool		isnull;
	ArrayIterator iter;
	bool		result = true;

	if (ARR_NDIM(attr_names) != 1 || ARR_ELEMTYPE(attr_names) != TEXTOID)
		elog(ERROR, "column names must be a vector of text");

	iter = array_create_iterator(attr_names, 0, NULL);
	while (array_iterate(iter, &value, &isnull))
	{
		char	   *temp = TextDatumGetCString(value);
		HeapTuple	tup;

		Assert(!isnull);
		tup = SearchSysCacheAttName(table_oid, temp);
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "column '%s' of %s was not found",
				 temp, get_table_desc(table_oid));
		if (type_oid != ((Form_pg_attribute) GETSTRUCT(tup))->atttypid)
			result = false;
		ReleaseSysCache(tup);
	}
	array_free_iterator(iter);

	PG_RETURN_BOOL(result);
}
PG_FUNCTION_INFO_V1(pgsql_check_attrs_of_type);

Datum
pgsql_check_attr_of_type(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	char	   *column_name = TextDatumGetCString(PG_GETARG_DATUM(1));
	Oid			type_oid = PG_GETARG_OID(2);
	HeapTuple	tup;
	bool		result = true;

	tup = SearchSysCacheAttName(table_oid, column_name);
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "column '%s' of %s was not found",
			 column_name, get_table_desc(table_oid));
	if (type_oid != ((Form_pg_attribute) GETSTRUCT(tup))->atttypid)
		result = false;
	ReleaseSysCache(tup);

	PG_RETURN_BOOL(result);
}
PG_FUNCTION_INFO_V1(pgsql_check_attr_of_type);
