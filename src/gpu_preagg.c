/*
 * gpu_groupby.c
 *
 * Aggregation and Group-By with GPU acceleration
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* static variables */
static create_upper_paths_hook_type	create_upper_paths_next = NULL;
static CustomPathMethods	gpupreagg_path_methods;
static CustomScanMethods	gpupreagg_plan_methods;
static CustomExecMethods	gpupreagg_exec_methods;
static CustomPathMethods	dpupreagg_path_methods;
static CustomScanMethods	dpupreagg_plan_methods;
static CustomExecMethods	dpupreagg_exec_methods;
static bool					pgstrom_enable_dpupreagg = false;
static bool					pgstrom_enable_partitionwise_dpupreagg = false;
static bool					pgstrom_enable_gpupreagg = false;
static bool					pgstrom_enable_partitionwise_gpupreagg = false;
static bool					pgstrom_enable_numeric_aggfuncs;
int							pgstrom_hll_register_bits;

/*
 * List of supported aggregate functions
 */
typedef struct
{
	/* aggregate function can be preprocessed */
	const char *aggfn_signature;
	/*
	 * A pair of final/partial function will generate same result.
	 * Its prefix indicates the schema that stores these functions.
	 * c: pg_catalog ... the system default
	 * s: pgstrom    ... PG-Strom's special ones
	 */
	const char *finalfn_signature;
	const char *partfn_signature;
	int			partfn_action;	/* any of KAGG_ACTION__* */
	bool		numeric_aware;	/* ignored, if !enable_numeric_aggfuncs */
} aggfunc_catalog_t;

static aggfunc_catalog_t	aggfunc_catalog_array[] = {
	/* COUNT(*) = SUM(NROWS()) */
	{"count()",
	 "s:fcount(int8)",
	 "s:nrows()",
	 KAGG_ACTION__NROWS_ANY, false
	},
	/* COUNT(X) = SUM(NROWS(X)) */
	{"count(any)",
	 "s:fcount(int8)",
	 "s:nrows(any)",
	 KAGG_ACTION__NROWS_COND, false
	},
	/*
	 * MIN(X) = MIN(PMIN(X))
	 */
	{"min(int1)",
	 "s:min_i1(bytea)",
	 "s:pmin(int4)",
	 KAGG_ACTION__PMIN_INT32, false
	},
	{"min(int2)",
	 "s:min_i2(bytea)",
	 "s:pmin(int4)",
	 KAGG_ACTION__PMIN_INT32, false
	},
	{"min(int4)",
	 "s:min_i4(bytea)",
	 "s:pmin(int4)",
	 KAGG_ACTION__PMIN_INT32, false
	},
	{"min(int8)",
	 "s:min_i8(bytea)",
	 "s:pmin(int8)",
	 KAGG_ACTION__PMIN_INT64, false
	},
	{"min(float2)",
	 "s:min_f2(bytea)",
	 "s:pmin(float8)",
	 KAGG_ACTION__PMIN_FP64, false
	},
	{"min(float4)",
	 "s:min_f4(bytea)",
	 "s:pmin(float8)",
	 KAGG_ACTION__PMIN_FP64, false
	},
	{"min(float8)",
	 "s:min_f8(bytea)",
	 "s:pmin(float8)",
	 KAGG_ACTION__PMIN_FP64, false
	},
	{"min(numeric)",
	 "s:min_num(bytea)",
	 "s:pmin(float8)",
	 KAGG_ACTION__PMIN_FP64, true
	},
	{"min(money)",
	 "s:min_cash(bytea)",
	 "s:pmin(money)",
	 KAGG_ACTION__PMIN_INT64, false
	},
	{"min(date)",
	 "s:min_date(bytea)",
	 "s:pmin(date)",
	 KAGG_ACTION__PMIN_INT32, false
	},
	{"min(time)",
	 "s:min_time(bytea)",
	 "s:pmin(time)",
	 KAGG_ACTION__PMIN_INT64, false
	},
	{"min(timestamp)",
	 "s:min_ts(bytea)",
	 "s:pmin(timestamp)",
	 KAGG_ACTION__PMIN_INT64, false
	},
	{"min(timestamptz)",
	 "s:min_tstz(bytea)",
	 "s:pmin(timestamptz)",
	 KAGG_ACTION__PMIN_INT64, false
	},
	/*
	 * MAX(X) = MAX(PMAX(X))
	 */
	{"max(int1)",
	 "s:max_i1(bytea)",
	 "s:pmax(int4)",
	 KAGG_ACTION__PMAX_INT32, false
	},
	{"max(int2)",
	 "s:max_i2(bytea)",
	 "s:pmax(int4)",
	 KAGG_ACTION__PMAX_INT32, false
	},
	{"max(int4)",
	 "s:max_i4(bytea)",
	 "s:pmax(int4)",
	 KAGG_ACTION__PMAX_INT32, false
	},
	{"max(int8)",
	 "s:max_i8(bytea)",
	 "s:pmax(int8)",
	 KAGG_ACTION__PMAX_INT64, false
	},
	{"max(float2)",
	 "s:max_f2(bytea)",
	 "s:pmax(float8)",
	 KAGG_ACTION__PMAX_FP64, false
	},
	{"max(float4)",
	 "s:max_f4(bytea)",
	 "s:pmax(float8)",
	 KAGG_ACTION__PMAX_FP64, false
	},
	{"max(float8)",
	 "s:max_f8(bytea)",
	 "s:pmax(float8)",
	 KAGG_ACTION__PMAX_FP64, false
	},
	{"max(numeric)",
	 "s:max_num(bytea)",
	 "s:pmax(float8)",
	 KAGG_ACTION__PMAX_FP64, true
	},
	{"max(money)",
	 "s:max_cash(bytea)",
	 "s:pmax(money)",
	 KAGG_ACTION__PMAX_INT64, false
	},
	{"max(date)",
	 "s:max_date(bytea)",
	 "s:pmax(date)",
	 KAGG_ACTION__PMAX_INT32, false
	},
	{"max(time)",
	 "s:max_time(bytea)",
	 "s:pmax(time)",
	 KAGG_ACTION__PMAX_INT64, false
	},
	{"max(timestamp)",
	 "s:max_ts(bytea)",
	 "s:pmax(timestamp)",
	 KAGG_ACTION__PMAX_INT64, false
	},
	{"max(timestamptz)",
	 "s:max_tstz(bytea)",
	 "s:pmax(timestamptz)",
	 KAGG_ACTION__PMAX_INT64, false
	},
	/*
	 * SUM(X) = SUM(PSUM(X))
	 */
	{"sum(int1)",
	 "s:sum_int(bytea)",
	 "s:psum(int8)",
	 KAGG_ACTION__PSUM_INT,  false
	},
	{"sum(int2)",
	 "s:sum_int(bytea)",
	 "s:psum(int8)",
	 KAGG_ACTION__PSUM_INT,  false
	},
	{"sum(int4)",
	 "s:sum_int(bytea)",
	 "s:psum(int8)",
	 KAGG_ACTION__PSUM_INT,  false
	},
	{"sum(int8)",
	 "s:sum_int_num(bytea)",
	 "s:psum(int8)",
	 KAGG_ACTION__PSUM_INT,  false
	},
	{"sum(float2)",
	 "s:sum_fp64(bytea)",
	 "s:psum(float8)",
	 KAGG_ACTION__PSUM_FP, false
	},
	{"sum(float4)",
	 "s:sum_fp32(bytea)",
	 "s:psum(float8)",
	 KAGG_ACTION__PSUM_FP, false
	},
	{"sum(float8)",
	 "s:sum_fp64(bytea)",
	 "s:psum(float8)",
	 KAGG_ACTION__PSUM_FP, false
	},
	{"sum(numeric)",
	 "s:sum_fp_num(bytea)",
	 "s:psum(float8)",
	 KAGG_ACTION__PSUM_FP, true
	},
	{"sum(money)",
	 "s:sum_cash(bytea)",
	 "s:psum(money)",
	 KAGG_ACTION__PSUM_INT,  false
	},
	/*
	 * AVG(X) = EX_AVG(NROWS(X), PSUM(X))
	 */
	{"avg(int1)",
	 "s:avg_int(bytea)",
	 "s:pavg(int8)",
	 KAGG_ACTION__PAVG_INT, false
	},
	{"avg(int2)",
	 "s:avg_int(bytea)",
	 "s:pavg(int8)",
	 KAGG_ACTION__PAVG_INT, false
	},
	{"avg(int4)",
	 "s:avg_int(bytea)",
	 "s:pavg(int8)",
	 KAGG_ACTION__PAVG_INT, false
	},
	{"avg(int8)",
	 "s:avg_int(bytea)",
	 "s:pavg(int8)",
	 KAGG_ACTION__PAVG_INT, false
	},
	{"avg(float2)",
	 "s:avg_fp(bytea)",
	 "s:pavg(float8)",
	 KAGG_ACTION__PAVG_FP, false
	},
	{"avg(float4)",
	 "s:avg_fp(bytea)",
	 "s:pavg(float8)",
	 KAGG_ACTION__PAVG_FP, false
	},
	{"avg(float8)",
	 "s:avg_fp(bytea)",
	 "s:pavg(float8)",
	 KAGG_ACTION__PAVG_FP, false
	},
	{"avg(numeric)",
	 "s:avg_num(bytea)",
	 "s:pavg(float8)",
	 KAGG_ACTION__PAVG_FP, true
	},
	/*
	 * STDDEV(X) = EX_STDDEV_SAMP(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{"stddev(int1)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev(int2)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev(int4)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev(int8)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev(float2)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev(float4)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev(float8)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev(numeric)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, true
	},
	/*
	 * STDDEV_SAMP(X) = EX_STDDEV_SAMP(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{"stddev_samp(int1)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_samp(int2)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_samp(int4)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_samp(int8)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_samp(float2)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_samp(float4)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_samp(float8)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_samp(numeric)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, true
	},
	/*
	 * STDDEV_POP(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{"stddev_pop(int1)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_pop(int2)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_pop(int4)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_pop(int8)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_pop(float2)",
	 "s:stddev_popf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_pop(float4)",
	 "s:stddev_popf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_pop(float8)",
	 "s:stddev_popf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"stddev_pop(numeric)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, true
	},
	/*
	 * VARIANCE(X) = VAR_SAMP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{"variance(int1)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"variance(int2)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"variance(int4)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"variance(int8)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"variance(float2)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"variance(float4)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"variance(float8)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"variance(numeric)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, true
	},
	/*
	 * VAR_SAMP(X) = VAR_SAMP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{"var_samp(int1)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_samp(int2)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_samp(int4)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_samp(int8)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_samp(float2)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_samp(float4)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_samp(float8)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_samp(numeric)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, true
	},
	/*
	 * VAR_POP(X)  = VAR_POP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{"var_pop(int1)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_pop(int2)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_pop(int4)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false},
	{"var_pop(int8)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_pop(float2)",
	 "s:var_popf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_pop(float4)",
	 "s:var_popf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_pop(float8)",
	 "s:var_popf(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, false
	},
	{"var_pop(numeric)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 KAGG_ACTION__STDDEV, true
	},
	/*
	 * CORR(X,Y) = PGSTROM.CORR(NROWS(X,Y),
	 *                          PCOV_X(X,Y),  PCOV_Y(X,Y)
	 *                          PCOV_X2(X,Y), PCOV_Y2(X,Y),
	 *                          PCOV_XY(X,Y))
	 */
	{"corr(float8,float8)",
	 "s:covar_samp(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"covar_samp(float8,float8)",
	 "s:covar_samp(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"covar_pop(float8,float8)",
	 "s:covar_pop(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	/*
	 * Aggregation to support least squares method
	 *
	 * That takes PSUM_X, PSUM_Y, PSUM_X2, PSUM_Y2, PSUM_XY according
	 * to the function
	 */
	{"regr_avgx(float8,float8)",
	 "s:regr_avgx(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"regr_avgy(float8,float8)",
	 "s:regr_avgy(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"regr_count(float8,float8)",
	 "s:regr_count(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"regr_intercept(float8,float8)",
	 "s:regr_intercept(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"regr_r2(float8,float8)",
	 "s:regr_r2(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"regr_slope(float8,float8)",
	 "s:regr_slope(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"regr_sxx(float8,float8)",
	 "s:regr_sxx(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"regr_sxy(float8,float8)",
	 "s:regr_sxy(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{"regr_syy(float8,float8)",
	 "s:regr_syy(bytea)",
	 "s:pcovar(float8,float8)",
	 KAGG_ACTION__COVAR, false
	},
	{ NULL, NULL, NULL, -1, false },
};

/*
 * aggfunc_catalog_entry; hashed catalog entry
 */
typedef struct
{
	Oid		aggfn_oid;
	Oid		final_func_oid;
	Oid		partial_func_oid;
	Oid		partial_func_rettype;
	int		partial_func_nargs;
	int		partial_func_action;
	int		partial_func_bufsz;
	bool	numeric_aware;
	bool	is_valid_entry;
} aggfunc_catalog_entry;

static HTAB	   *aggfunc_catalog_htable = NULL;

static void
aggfunc_catalog_htable_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	hash_destroy(aggfunc_catalog_htable);
	aggfunc_catalog_htable = NULL;
}

static Oid
__aggfunc_resolve_func_signature(const char *signature)
{
	char	   *fn_name = alloca(strlen(signature));
	Oid			fn_namespace;
	oidvector  *fn_argtypes;
	int			fn_nargs = 0;
	Oid			fn_oid;
	Oid			type_oid;
	char	   *base, *tok, *pos;

	if (strncmp(signature, "c:", 2) == 0)
		fn_namespace = PG_CATALOG_NAMESPACE;
	else if (strncmp(signature, "s:", 2) == 0)
		fn_namespace = get_namespace_oid("pgstrom", false);
	else
		elog(ERROR, "wrong function signature: %s", signature);

	strcpy(fn_name, signature + 2);
	base = strchr(fn_name, '(');
	if (!base)
		elog(ERROR, "wrong function signature: %s", signature);
	*base++ = '\0';
	pos = strchr(base, ')');
	if (!pos)
		elog(ERROR, "wrong function signature: %s", signature);
	*pos = '\0';

	fn_argtypes = alloca(offsetof(oidvector, values[80]));
	fn_argtypes->ndim = 1;
	fn_argtypes->dataoffset = 0;
	fn_argtypes->elemtype = OIDOID;
	fn_argtypes->dim1 = 0;
	fn_argtypes->lbound1 = 0;
	for (tok = strtok_r(base, ",", &pos);
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &pos))
	{
		type_oid = GetSysCacheOid2(TYPENAMENSP,
								   Anum_pg_type_oid,
								   CStringGetDatum(tok),
								   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
		if (!OidIsValid(type_oid))
			elog(ERROR, "cache lookup failed for type '%s'", tok);
		fn_argtypes->values[fn_nargs++] = type_oid;
	}
	fn_argtypes->dim1 = fn_nargs;
	SET_VARSIZE(fn_argtypes, offsetof(oidvector, values[fn_nargs]));

	fn_oid = GetSysCacheOid3(PROCNAMEARGSNSP,
							 Anum_pg_proc_oid,
							 CStringGetDatum(fn_name),
							 PointerGetDatum(fn_argtypes),
							 ObjectIdGetDatum(fn_namespace));
	if (!OidIsValid(fn_oid))
		elog(ERROR, "Catalog corruption? '%s' was not found",
			 funcname_signature_string(fn_name,
									   fn_argtypes->dim1,
									   NIL,
									   fn_argtypes->values));
	return fn_oid;
}

static void
__aggfunc_resolve_partial_func(aggfunc_catalog_entry *entry,
							   const char *partfn_signature,
							   int partfn_action)
{
	Oid		func_oid = __aggfunc_resolve_func_signature(partfn_signature);
	Oid		type_oid;
	int		func_nargs;
	int		partfn_bufsz;

	switch (partfn_action)
	{
		case KAGG_ACTION__NROWS_ANY:
			func_nargs = 0;
			type_oid = INT8OID;
			partfn_bufsz = sizeof(int64_t);
			break;
		case KAGG_ACTION__NROWS_COND:
			func_nargs = 1;
			type_oid = INT8OID;
			partfn_bufsz = sizeof(int64_t);
			break;
		case KAGG_ACTION__PMIN_INT32:
		case KAGG_ACTION__PMIN_INT64:
		case KAGG_ACTION__PMAX_INT32:
		case KAGG_ACTION__PMAX_INT64:
			func_nargs = 1;
            type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__pminmax_int64_packed);
			break;

		case KAGG_ACTION__PMIN_FP64:
		case KAGG_ACTION__PMAX_FP64:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__pminmax_fp64_packed);
			break;

		case KAGG_ACTION__PAVG_INT:
		case KAGG_ACTION__PSUM_INT:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__psum_int_packed);
			break;

		case KAGG_ACTION__PAVG_FP:
		case KAGG_ACTION__PSUM_FP:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__psum_fp_packed);
			break;
			
		case KAGG_ACTION__STDDEV:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__stddev_packed);
			break;
		case KAGG_ACTION__COVAR:
			func_nargs = 2;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__covar_packed);
			break;
		default:
			elog(ERROR, "Catalog corruption? unknown action: %d", partfn_action);
			break;
	}
	entry->partial_func_oid = func_oid;
	entry->partial_func_rettype = get_func_rettype(func_oid);
	entry->partial_func_nargs = get_func_nargs(func_oid);
	entry->partial_func_action = partfn_action;
	entry->partial_func_bufsz  = partfn_bufsz;

	if (entry->partial_func_rettype != type_oid ||
		entry->partial_func_nargs != func_nargs ||
		entry->partial_func_bufsz != MAXALIGN(partfn_bufsz))
		elog(ERROR, "Catalog curruption? partial function mismatch: %s",
			 partfn_signature);
}

static void
__aggfunc_resolve_final_func(aggfunc_catalog_entry *entry,
							 const char *finalfn_signature,
							 Oid agg_rettype)
{
	Oid			func_oid = __aggfunc_resolve_func_signature(finalfn_signature);
	HeapTuple	htup;
	Form_pg_proc proc;

	if (!SearchSysCacheExists1(AGGFNOID, ObjectIdGetDatum(func_oid)) ||
		get_func_rettype(func_oid) != agg_rettype)
		elog(ERROR, "Catalog corruption? final function mismatch: %s",
			 format_procedure(func_oid));
	htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	proc = (Form_pg_proc) GETSTRUCT(htup);
	if (proc->pronargs != 1 ||
		proc->proargtypes.dim1 != 1 ||
		proc->proargtypes.values[0] != entry->partial_func_rettype)
		elog(ERROR, "Catalog corruption? final function mismatch: %s",
			 format_procedure(func_oid));
	ReleaseSysCache(htup);

	entry->final_func_oid = func_oid;
}

static const aggfunc_catalog_entry *
aggfunc_catalog_lookup_by_oid(Oid aggfn_oid)
{
	aggfunc_catalog_entry *entry;
	bool		found;

	/* fast path by the hashtable */
	if (!aggfunc_catalog_htable)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(aggfunc_catalog_entry);
		hctl.hcxt = CacheMemoryContext;
		aggfunc_catalog_htable = hash_create("XPU GroupBy Catalog Hash",
											 256,
											 &hctl,
											 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}
	entry = hash_search(aggfunc_catalog_htable,
						&aggfn_oid,
						HASH_ENTER,
						&found);
	if (!found)
	{
		Form_pg_proc proc;
		HeapTuple htup;

		entry->is_valid_entry = false;
		PG_TRY();
		{
			htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(aggfn_oid));
			if (!HeapTupleIsValid(htup))
				elog(ERROR, "cache lookup failed for function %u", aggfn_oid);
			proc = (Form_pg_proc) GETSTRUCT(htup);
			if (proc->pronamespace == PG_CATALOG_NAMESPACE &&
				proc->pronargs <= 2)
			{
				char	buf[3*NAMEDATALEN+100];
				int		off;

				off = sprintf(buf, "%s(", NameStr(proc->proname));
				for (int j=0; j < proc->pronargs; j++)
				{
					Oid		type_oid = proc->proargtypes.values[j];
					char   *type_name = get_type_name(type_oid, false);

					off += sprintf(buf + off, "%s%s",
								   (j>0 ? "," : ""),
								   type_name);
				}
				off += sprintf(buf + off, ")");

				for (int i=0; aggfunc_catalog_array[i].aggfn_signature != NULL; i++)
				{
					const aggfunc_catalog_t *cat = &aggfunc_catalog_array[i];

					if (strcmp(buf, cat->aggfn_signature) == 0)
					{
						__aggfunc_resolve_partial_func(entry,
													   cat->partfn_signature,
													   cat->partfn_action);
						__aggfunc_resolve_final_func(entry,
													 cat->finalfn_signature,
													 proc->prorettype);
						entry->numeric_aware = cat->numeric_aware;
						entry->is_valid_entry = true;
						break;
					}
				}
			}
			ReleaseSysCache(htup);
		}
		PG_CATCH();
		{
			hash_search(aggfunc_catalog_htable, &aggfn_oid, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	if (!entry->is_valid_entry)
		return NULL;
	if (entry->numeric_aware && !pgstrom_enable_numeric_aggfuncs)
		return NULL;
	return entry;
}

/*
 * xpugroupby_build_path_context
 */
typedef struct
{
	bool			device_executable;
	PlannerInfo	   *root;
	RelOptInfo	   *group_rel;
	RelOptInfo	   *input_rel;
	ParamPathInfo  *param_info;
	double			num_groups;
	bool			try_parallel;
	PathTarget	   *target_upper;
	PathTarget	   *target_partial;
	PathTarget	   *target_final;
	AggClauseCosts	final_clause_costs;
	pgstromPlanInfo *pp_info;
	List		   *inner_paths_list;
	List		   *inner_target_list;
	List		   *groupby_keys;
	List		   *groupby_keys_refno;
	Node		   *havingQual;
} xpugroupby_build_path_context;

/*
 * make_expr_typecast - constructor of type cast
 */
static Expr *
make_expr_typecast(Expr *expr, Oid target_type)
{
	Oid		source_type = exprType((Node *) expr);
	HeapTuple htup;
	Form_pg_cast cast;

	if (target_type == source_type ||
		target_type == ANYOID)
		return expr;

	htup = SearchSysCache2(CASTSOURCETARGET,
						   ObjectIdGetDatum(source_type),
						   ObjectIdGetDatum(target_type));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for cast (%s -> %s)",
			 format_type_be(source_type),
			 format_type_be(target_type));
	cast = (Form_pg_cast) GETSTRUCT(htup);
	if (cast->castmethod == COERCION_METHOD_BINARY)
	{
		RelabelType    *relabel = makeNode(RelabelType);

		relabel->arg = expr;
		relabel->resulttype = target_type;
		relabel->resulttypmod = exprTypmod((Node *) expr);
		relabel->resultcollid = exprCollation((Node *) expr);
		relabel->relabelformat = COERCE_IMPLICIT_CAST;
		relabel->location = -1;

		expr = (Expr *) relabel;
	}
	else if (cast->castmethod == COERCION_METHOD_FUNCTION)
	{
		Assert(OidIsValid(cast->castfunc));
		expr = (Expr *)makeFuncExpr(cast->castfunc,
									target_type,
									list_make1(expr),
									InvalidOid,		/* always right? */
									exprCollation((Node *) expr),
									COERCE_IMPLICIT_CAST);
	}
	else
	{
		elog(ERROR, "cast-method '%c' is not supported in the kernel mode",
			 cast->castmethod);
	}
	ReleaseSysCache(htup);

	return expr;
}

/*
 * make_alternative_aggref
 *
 * It makes an alternative final aggregate function towards the supplied
 * Aggref, and append its arguments on the target_partial/target_device.
 */
static Node *
make_alternative_aggref(xpugroupby_build_path_context *con, Aggref *aggref)
{
	const aggfunc_catalog_entry *aggfn_cat;
	PathTarget *target_partial = con->target_partial;
	pgstromPlanInfo *pp_info = con->pp_info;
	List	   *partfn_args = NIL;
	Expr	   *partfn;
	Aggref	   *aggref_alt;
	Oid			func_oid;
	HeapTuple	htup;
	Form_pg_proc proc;
	Form_pg_aggregate agg;
	ListCell   *lc;
	int			j;

	if (aggref->aggorder != NIL || aggref->aggdistinct != NIL)
	{
		elog(DEBUG2, "Aggregate with ORDER BY/DISTINCT is not supported: %s",
			 nodeToString(aggref));
		return NULL;
	}
	if (AGGKIND_IS_ORDERED_SET(aggref->aggkind))
	{
		elog(DEBUG2, "ORDERED SET Aggregation is not supported: %s",
			 nodeToString(aggref));
		return NULL;
	}

	/*
	 * Lookup properties of aggregate function
	 */
	aggfn_cat = aggfunc_catalog_lookup_by_oid(aggref->aggfnoid);
	if (!aggfn_cat)
	{
		elog(DEBUG2, "Aggregate function '%s' is not device executable",
			 format_procedure(aggref->aggfnoid));
		return NULL;
	}
	/* sanity checks */
	Assert(aggref->aggkind == AGGKIND_NORMAL &&
		   !aggref->aggvariadic);

	/*
	 * Build partial-aggregate function
	 */
	htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(aggfn_cat->partial_func_oid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for function %u",
			 aggfn_cat->partial_func_oid);
	proc = (Form_pg_proc) GETSTRUCT(htup);
	Assert(list_length(aggref->args) == proc->pronargs);
	j = 0;
	foreach (lc, aggref->args)
	{
		TargetEntry *tle = lfirst(lc);
		Expr   *expr = tle->expr;
		Oid		type_oid = exprType((Node *)expr);
		Oid		dest_oid = proc->proargtypes.values[j++];

		if (type_oid != dest_oid)
			expr = make_expr_typecast(expr, dest_oid);
		if (!pgstrom_xpu_expression(expr,
									pp_info->xpu_task_flags,
									pp_info->scan_relid,
									con->inner_target_list,
									NULL))
		{
			elog(DEBUG2, "Partial aggregate argument is not executable: %s",
				 nodeToString(expr));
			return NULL;
		}
		partfn_args = lappend(partfn_args, expr);
	}
	ReleaseSysCache(htup);

	partfn = (Expr *)makeFuncExpr(aggfn_cat->partial_func_oid,
								  aggfn_cat->partial_func_rettype,
								  partfn_args,
								  aggref->aggcollid,
								  aggref->inputcollid,
								  COERCE_EXPLICIT_CALL);
	/* see add_new_column_to_pathtarget */
	if (!list_member(target_partial->exprs, partfn))
	{
		add_column_to_pathtarget(target_partial, partfn, 0);
		pp_info->groupby_actions = lappend_int(pp_info->groupby_actions,
											   aggfn_cat->partial_func_action);
		pp_info->groupby_prepfn_bufsz += aggfn_cat->partial_func_bufsz;
	}

	/*
	 * Build final-aggregate function
	 */
	func_oid = aggfn_cat->final_func_oid;
	htup = SearchSysCache1(AGGFNOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for pg_aggregate %u", func_oid);
	agg = (Form_pg_aggregate) GETSTRUCT(htup);

	aggref_alt = makeNode(Aggref);
	aggref_alt->aggfnoid      = func_oid;
	aggref_alt->aggtype       = aggref->aggtype;
	aggref_alt->aggcollid     = aggref->aggcollid;
	aggref_alt->inputcollid   = aggref->inputcollid;
	aggref_alt->aggtranstype  = agg->aggtranstype;
	aggref_alt->aggargtypes   = list_make1_oid(exprType((Node *)partfn));
	aggref_alt->aggdirectargs = NIL;	/* see sanity checks */
	aggref_alt->args          = list_make1(makeTargetEntry(partfn, 1, NULL, false));
	aggref_alt->aggorder      = NIL;  /* see sanity check */
	aggref_alt->aggdistinct   = NIL;  /* see sanity check */
	aggref_alt->aggfilter     = NULL; /* processed in partial-function */
	aggref_alt->aggstar       = false;
	aggref_alt->aggvariadic   = false;
	aggref_alt->aggkind       = AGGKIND_NORMAL;   /* see sanity check */
	aggref_alt->agglevelsup   = 0;
	aggref_alt->aggsplit      = AGGSPLIT_SIMPLE;
	aggref_alt->aggno         = aggref->aggno;
	aggref_alt->aggtransno    = aggref->aggno;
	aggref_alt->location      = aggref->location;
	/*
	 * MEMO: nodeAgg.c creates AggStatePerTransData for each aggtransno (that is
	 * unique ID of transition state in the Agg). This is a kind of optimization
	 * for the case when multiple aggregate function has identical transition state.
	 * However, its impact is not large for GpuPreAgg because most of reduction
	 * works are already executed at the xPU device side.
	 * So, we simply assign aggref->aggno (unique ID within the Agg node) to
	 * construct transition state for each alternative aggregate function.
	 *
	 * See the issue #614 to reproduce the problem in the future version.
	 */

	/*
	 * Update the cost factor
	 */
	if (OidIsValid(agg->aggtransfn))
		add_function_cost(con->root,
						  agg->aggtransfn,
						  NULL,
						  &con->final_clause_costs.transCost);
	if (OidIsValid(agg->aggfinalfn))
		add_function_cost(con->root,
						  agg->aggfinalfn,
						  NULL,
						  &con->final_clause_costs.finalCost);
	ReleaseSysCache(htup);

	return (Node *)aggref_alt;
}

static Node *
replace_expression_by_altfunc(Node *node, xpugroupby_build_path_context *con)
{
	PathTarget *target_input = con->input_rel->reltarget;
	Node	   *aggfn;
	ListCell   *lc;

	if (!node)
		return NULL;
	/* aggregate function? */
	if (IsA(node, Aggref))
	{
		aggfn = make_alternative_aggref(con, (Aggref *)node);
		if (!aggfn)
			con->device_executable = false;
		return aggfn;
	}
	/* grouping key? */
	foreach (lc, con->groupby_keys)
	{
		Expr   *key = lfirst(lc);

		if (equal(node, key))
			return copyObject(node);
	}
	/*
	 * Elsewhere, non-grouping-key columns if GROUP BY <primary key> is used,
	 * because it is equivalent to GROUP BY <all the columns>.
	 * Also, the device code don't need to understand its data type, so we
	 * have no device type checks here.
	 */
	foreach (lc, target_input->exprs)
	{
		Expr   *expr = lfirst(lc);

		if (equal(node, expr))
		{
			/*
			 * this expression shall be attached onto the target-partial later,
			 * however, it don't need to be added to the target-final, because
			 * this expression is consumed within HAVING quals, thus not exists
			 * on the final Aggregate results.
			 */
			con->groupby_keys = lappend(con->groupby_keys, expr);
			con->groupby_keys_refno = lappend_int(con->groupby_keys_refno, 0);
			return copyObject(node);
		}
	}

	if (IsA(node, Var) || IsA(node, PlaceHolderVar))
	{
		elog(ERROR, "Bug? referenced variable is grouping-key nor its dependent key: %s",
			 nodeToString(node));
	}
	return expression_tree_mutator(node, replace_expression_by_altfunc, con);
}

static bool
xpugroupby_build_path_target(xpugroupby_build_path_context *con)
{
	PlannerInfo	   *root = con->root;
	Query		   *parse = root->parse;
	pgstromPlanInfo *pp_info = con->pp_info;
	PathTarget	   *target_upper = con->target_upper;
	Node		   *havingQual = NULL;
	ListCell	   *lc1, *lc2;
	int				i = 0;

	/*
	 * Pick up grouping-keys and aggregate-functions to be replaced by
	 * a pair of final-aggregate and partial-function.
	 */
	foreach (lc1, target_upper->exprs)
	{
		Expr   *expr = lfirst(lc1);
		Index	sortgroupref = get_pathtarget_sortgroupref(target_upper, i++);

		if (sortgroupref && parse->groupClause &&
			get_sortgroupref_clause_noerr(sortgroupref,
										  parse->groupClause) != NULL)
		{
			/* Grouping Key */
			devtype_info *dtype;
			Oid		type_oid = exprType((Node *)expr);
			Oid		coll_oid;

			dtype = pgstrom_devtype_lookup(type_oid);
			if (!dtype || !dtype->type_hashfunc)
			{
				elog(DEBUG2, "GROUP BY contains unsupported type (%s): %s",
					 format_type_be(type_oid),
					 nodeToString(expr));
				return false;
			}
			coll_oid = exprCollation((Node *)expr);
			if (devtype_lookup_equal_func(dtype, coll_oid) == NULL)
			{
				elog(DEBUG2, "GROUP BY contains unsupported device type (%s): %s",
					 format_type_be(type_oid),
					 nodeToString(expr));
				return false;
			}
			/* grouping-key must be device executable. */
			if (!pgstrom_xpu_expression(expr,
										pp_info->xpu_task_flags,
										pp_info->scan_relid,
										con->inner_target_list,
										NULL))
			{
				elog(DEBUG2, "Grouping-key must be device executable: %s",
					 nodeToString(expr));
				return false;
			}
			add_column_to_pathtarget(con->target_final, expr, sortgroupref);
			/* to be attached to target-partial later */
			con->groupby_keys = lappend(con->groupby_keys, expr);
			con->groupby_keys_refno = lappend_int(con->groupby_keys_refno,
												  sortgroupref);
		}
		else if (IsA(expr, Aggref))
		{
			Expr   *altfn;

			altfn = (Expr *)make_alternative_aggref(con, (Aggref *)expr);
			if (!altfn)
			{
				elog(DEBUG2, "No alternative aggregation: %s",
					 nodeToString(expr));
				return false;
			}
			if (exprType((Node *)expr) != exprType((Node *)altfn))
			{
				elog(ERROR, "Bug? XpuGroupBy catalog is not consistent: %s --> %s",
					 nodeToString(expr),
					 nodeToString(altfn));
			}
			add_column_to_pathtarget(con->target_final, altfn, 0);
		}
		else
		{
			elog(DEBUG2, "unexpected expression on the upper-tlist: %s",
				 nodeToString(expr));
			return false;
		}
	}

	/*
	 * HAVING clause
	 */
	if (parse->havingQual)
	{
		havingQual = replace_expression_by_altfunc(parse->havingQual, con);
		if (!con->device_executable)
		{
			elog(DEBUG2, "unable to replace HAVING to alternative aggregation: %s",
				 nodeToString(parse->havingQual));
			return false;
		}
	}
	con->havingQual = havingQual;

	/*
	 * Due to data alignment on the tuple on the kds_final, grouping-keys must
	 * be located after the aggregate functions.
	 */
	forboth (lc1, con->groupby_keys,
			 lc2, con->groupby_keys_refno)
	{
		Expr   *key = lfirst(lc1);
		Index	keyref = lfirst_int(lc2);

		add_column_to_pathtarget(con->target_partial, key, keyref);
		pp_info->groupby_actions = lappend_int(pp_info->groupby_actions,
											   keyref == 0
											   ? KAGG_ACTION__VREF_NOKEY
											   : KAGG_ACTION__VREF);
	}
	set_pathtarget_cost_width(root, con->target_final);
	set_pathtarget_cost_width(root, con->target_partial);

	return true;
}

/*
 * try_add_final_groupby_paths
 */
static void
try_add_final_groupby_paths(xpugroupby_build_path_context *con,
							Path *part_path)
{
	Query	   *parse = con->root->parse;
	Path	   *agg_path;
	Path	   *dummy_path;
	double		hashTableSz;

	if (!parse->groupClause)
	{
		agg_path = (Path *)create_agg_path(con->root,
										   con->group_rel,
										   part_path,
										   con->target_final,
										   AGG_PLAIN,
										   AGGSPLIT_SIMPLE,
										   parse->groupClause,
										   (List *)con->havingQual,
										   &con->final_clause_costs,
										   con->num_groups);
		dummy_path = pgstrom_create_dummy_path(con->root, agg_path);
		add_path(con->group_rel, dummy_path);
	}
	else
	{
		Assert(grouping_is_hashable(parse->groupClause));
		hashTableSz = estimate_hashagg_tablesize(con->root,
												 part_path,
												 &con->final_clause_costs,
												 con->num_groups);
		if (hashTableSz <= (double)work_mem * 1024.0)
		{
			agg_path = (Path *)create_agg_path(con->root,
											   con->group_rel,
											   part_path,
											   con->target_final,
											   AGG_HASHED,
											   AGGSPLIT_SIMPLE,
											   parse->groupClause,
											   (List *)con->havingQual,
											   &con->final_clause_costs,
											   con->num_groups);
			dummy_path = pgstrom_create_dummy_path(con->root, agg_path);
			add_path(con->group_rel, dummy_path);
		}
	}
}

/*
 * __buildXpuPreAggCustomPath
 */
static CustomPath *
__buildXpuPreAggCustomPath(xpugroupby_build_path_context *con)
{
	Query	   *parse = con->root->parse;
	CustomPath *cpath = makeNode(CustomPath);
	PathTarget *target_partial = con->target_partial;
	pgstromPlanInfo *pp_info = con->pp_info;
	double		input_nrows = PP_INFO_NUM_ROWS(pp_info);
	double		num_group_keys;
	double		xpu_ratio;
	Cost		xpu_operator_cost;
	Cost		xpu_tuple_cost;
	Cost		startup_cost = 0.0;
	Cost		run_cost = 0.0;
	const CustomPathMethods *xpu_cpath_methods;

	/*
	 * Parameters related to devices
	 */
	if ((pp_info->xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU)
	{
		xpu_operator_cost = pgstrom_gpu_operator_cost;
		xpu_tuple_cost    = pgstrom_gpu_tuple_cost;
		xpu_ratio         = pgstrom_gpu_operator_ratio();
		xpu_cpath_methods = &gpupreagg_path_methods;
	}
	else if ((pp_info->xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU)
	{
		xpu_operator_cost = pgstrom_dpu_operator_cost;
		xpu_tuple_cost    = pgstrom_dpu_tuple_cost;
		xpu_ratio         = pgstrom_dpu_operator_ratio();
		xpu_cpath_methods = &dpupreagg_path_methods;
	}
	else
	{
		elog(ERROR, "Bug? unexpected task_kind: %08x", pp_info->xpu_task_flags);
	}
	pp_info->xpu_task_flags &= ~DEVTASK__MASK;
	pp_info->xpu_task_flags |= DEVTASK__PREAGG;

	/* No tuples shall be generated until child JOIN/SCAN path completion */
	startup_cost = (PP_INFO_STARTUP_COST(pp_info) +
					PP_INFO_RUN_COST(pp_info));
	/* Cost estimation for grouping */
	num_group_keys = list_length(parse->groupClause);
	startup_cost += (xpu_operator_cost *
					 num_group_keys *
					 input_nrows);
	/* Cost estimation for aggregate function */
	startup_cost += (target_partial->cost.per_tuple * input_nrows +
					 target_partial->cost.startup) * xpu_ratio;
	/* Cost estimation to fetch results */
	run_cost = xpu_tuple_cost * con->num_groups;

	cpath->path.pathtype         = T_CustomScan;
	cpath->path.parent           = con->input_rel;
	cpath->path.pathtarget       = con->target_partial;
	cpath->path.param_info       = con->param_info;
	cpath->path.parallel_aware   = con->try_parallel;
	cpath->path.parallel_safe    = con->input_rel->consider_parallel;
	cpath->path.parallel_workers = pp_info->parallel_nworkers;
	cpath->path.rows             = con->num_groups;
	cpath->path.startup_cost     = startup_cost;
	cpath->path.total_cost       = startup_cost + run_cost;
	cpath->path.pathkeys         = NIL;
	cpath->custom_paths          = con->inner_paths_list;
	cpath->custom_private        = list_make1(pp_info);
	cpath->methods               = xpu_cpath_methods;

	return cpath;
}

/*
 * __tryAddXpuPreAggCustomPath
 */
static void
__tryAddXpuPreAggCustomPath(PlannerInfo *root,
							RelOptInfo *input_rel,
							RelOptInfo *group_rel,
							GroupPathExtraData *gp_extra,
							uint32_t xpu_task_flags,
							bool try_parallel_path,
							List *op_leaf_list)
{
	xpugroupby_build_path_context con;
	Query	   *parse = root->parse;
	List	   *preagg_cpath_list = NIL;
	ListCell   *lc;
	Path	   *part_path;
	int			parallel_nworkers = 0;
	double		total_nrows = 0.0;

	memset(&con, 0, sizeof(con));
	foreach (lc, op_leaf_list)
	{
		pgstromOuterPartitionLeafInfo *op_leaf = lfirst(lc);
		pgstromPlanInfo *pp_info = op_leaf->pp_info;
		Path	   *__outerpath = op_leaf->pseudo_outer_path;
		double		num_groups = 1.0;
		List	   *inner_target_list = NIL;
		ListCell   *cell;
		CustomPath *cpath;

		/* estimate number of groups */
		if (parse->groupClause)
		{
			List   *groupExprs;

			groupExprs = get_sortgrouplist_exprs(parse->groupClause,
												 gp_extra->targetList);
			num_groups = estimate_num_groups(root, groupExprs,
											 PP_INFO_NUM_ROWS(pp_info),
											 NULL, NULL);
		}
		/* setup inner_target_list */
		foreach (cell, op_leaf->inner_paths_list)
		{
			Path	   *i_path = lfirst(cell);
			inner_target_list = lappend(inner_target_list, i_path->pathtarget);
		}
		/* setup context */
		memset(&con, 0, sizeof(con));
		con.device_executable = true;
		con.root           = root;
		con.group_rel      = group_rel;
		con.input_rel      = __outerpath->parent;
		con.param_info     = __outerpath->param_info;
		con.num_groups     = num_groups;
		con.try_parallel   = try_parallel_path;
		con.target_upper   = root->upper_targets[UPPERREL_GROUP_AGG];
		con.target_partial = create_empty_pathtarget();
		con.target_final   = create_empty_pathtarget();
		con.pp_info        = pp_info;
		con.inner_paths_list = op_leaf->inner_paths_list;
		con.inner_target_list = inner_target_list;
		/* construction of the target-list for each level */
		if (!xpugroupby_build_path_target(&con))
			return;
		cpath = __buildXpuPreAggCustomPath(&con);

		parallel_nworkers += cpath->path.parallel_workers;
        total_nrows       += cpath->path.rows;

		preagg_cpath_list = lappend(preagg_cpath_list, cpath);
	}

	if (list_length(preagg_cpath_list) > 1)
	{
		part_path = (Path *)
			create_append_path(root,
							   input_rel,
							   (try_parallel_path ? NIL : preagg_cpath_list),
							   (try_parallel_path ? preagg_cpath_list : NIL),
							   NIL,
							   NULL,
							   (try_parallel_path ? parallel_nworkers : 0),
							   try_parallel_path,
							   total_nrows);
		part_path->pathtarget = con.target_partial;
	}
	else if (list_length(preagg_cpath_list) == 1)
	{
		part_path = linitial(preagg_cpath_list);
	}
	else
	{
		return;
	}

	/* put Gather path if parallel-aware */
	if (try_parallel_path)
	{
		part_path = (Path *)
			create_gather_path(root,
							   group_rel,
							   part_path,
							   part_path->pathtarget,
							   NULL,
							   &total_nrows);
	}
	/* try add final groupby path */
	try_add_final_groupby_paths(&con, part_path);
}

static void
__xpuPreAggAddCustomPathCommon(PlannerInfo *root,
							   RelOptInfo *input_rel,
							   RelOptInfo *group_rel,
							   void *extra,
							   uint32_t xpu_task_flags,
							   bool consider_partition)
{
	Query	   *parse = root->parse;

	/* quick bailout if not supported */
	if (parse->groupingSets != NIL ||
		!grouping_is_hashable(parse->groupClause))
	{
		elog(DEBUG2, "GROUP BY clause is not supported form");
		return;
	}

	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		List   *op_leaf_list;

		/*
		 * Try simple XpuPreAgg Path
		 */
		op_leaf_list = buildOuterJoinPlanInfo(root,
											  input_rel,
											  xpu_task_flags,
											  (try_parallel > 0),
											  false);
		if (op_leaf_list != NIL)
		{
			Assert(list_length(op_leaf_list) == 1);
			__tryAddXpuPreAggCustomPath(root,
										input_rel,
										group_rel,
										extra,
										xpu_task_flags,
										(try_parallel > 0),
										op_leaf_list);
		}

		/*
		 * Try partition-wise XpuPreAgg Path
		 */
		if (consider_partition)
		{
			op_leaf_list = buildOuterJoinPlanInfo(root,
												  input_rel,
												  xpu_task_flags,
												  (try_parallel > 0),
												  true);
			if (op_leaf_list != NIL)
			{
				__tryAddXpuPreAggCustomPath(root,
											input_rel,
											group_rel,
											extra,
											xpu_task_flags,
											(try_parallel > 0),
											op_leaf_list);
			}
		}
	}
}

/*
 * XpuPreAggAddCustomPath
 */
static void
XpuPreAggAddCustomPath(PlannerInfo *root,
					   UpperRelationKind stage,
					   RelOptInfo *input_rel,
					   RelOptInfo *group_rel,
					   void *extra)
{
	if (create_upper_paths_next)
		create_upper_paths_next(root,
								stage,
								input_rel,
								group_rel,
								extra);
	if (stage != UPPERREL_GROUP_AGG)
		return;
	if (pgstrom_enabled())
	{
		if (pgstrom_enable_gpupreagg)
			__xpuPreAggAddCustomPathCommon(root,
										   input_rel,
										   group_rel,
										   extra,
										   TASK_KIND__GPUPREAGG,
										   pgstrom_enable_partitionwise_gpupreagg);
		if (pgstrom_enable_dpupreagg)
			__xpuPreAggAddCustomPathCommon(root,
										   input_rel,
										   group_rel,
										   extra,
										   TASK_KIND__DPUPREAGG,
										   pgstrom_enable_partitionwise_dpupreagg);
	}
}

/*
 * PlanGpuPreAggPath
 */
static Plan *
PlanGpuPreAggPath(PlannerInfo *root,
				  RelOptInfo *joinrel,
				  CustomPath *cpath,
				  List *tlist,
				  List *clauses,
				  List *custom_plans)
{
	pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
	CustomScan	   *cscan;

	cscan = PlanXpuJoinPathCommon(root,
								  joinrel,
								  cpath,
								  tlist,
								  custom_plans,
								  pp_info,
								  &gpupreagg_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);
	return &cscan->scan.plan;
}

/*
 * PlanDpuPreAggPath
 */
static Plan *
PlanDpuPreAggPath(PlannerInfo *root,
				  RelOptInfo *joinrel,
				  CustomPath *cpath,
				  List *tlist,
				  List *clauses,
				  List *custom_plans)
{
	pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
	CustomScan	   *cscan;

	cscan = PlanXpuJoinPathCommon(root,
								  joinrel,
								  cpath,
								  tlist,
								  custom_plans,
								  pp_info,
								  &dpupreagg_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);
	return &cscan->scan.plan;
}

/*
 * CreateGpuPreAggScanState
 */
static Node *
CreateGpuPreAggScanState(CustomScan *cscan)
{
	pgstromTaskState *pts;
	pgstromPlanInfo  *pp_info = deform_pgstrom_plan_info(cscan);
	int		num_rels = list_length(cscan->custom_plans);

	Assert(cscan->methods == &gpupreagg_plan_methods);
	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &gpupreagg_exec_methods;
	pts->xpu_task_flags = pp_info->xpu_task_flags;
	pts->pp_info = pp_info;
	Assert((pts->xpu_task_flags & TASK_KIND__MASK) == TASK_KIND__GPUPREAGG &&
		   pp_info->num_rels == num_rels);
	pts->num_rels = num_rels;

	return (Node *)pts;
}

/*
 * CreateDpuPreAggScanState
 */
static Node *
CreateDpuPreAggScanState(CustomScan *cscan)
{
	pgstromTaskState *pts;
	pgstromPlanInfo  *pp_info = deform_pgstrom_plan_info(cscan);
	int			num_rels = list_length(cscan->custom_plans);

	Assert(cscan->methods == &dpupreagg_plan_methods);
	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &dpupreagg_exec_methods;
	pts->xpu_task_flags = pp_info->xpu_task_flags;
	pts->pp_info = pp_info;
	Assert((pts->xpu_task_flags & TASK_KIND__MASK) == TASK_KIND__DPUPREAGG &&
		   pp_info->num_rels == num_rels);
	pts->num_rels = num_rels;

	return (Node *)pts;
}

/*
 * ExecFallbackCpuPreAgg
 */
bool
ExecFallbackCpuPreAgg(pgstromTaskState *pts, HeapTuple tuple)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("CPU Fallback of GpuPreAgg is not implemented yet"),
			 errhint("'pg_strom.enable_gpupreagg' configuration can turn off GpuPreAgg only"),
			 errdetail("GpuPreAgg often detects uncontinuable errors during update of the final aggregation buffer. Even if we re-run the source data chunk, the final aggregation buffer is already polluted, so we have no reasonable way to recover right now.")));
	return false;
}

/*
 * Entrypoint of GpuPreAgg
 */
void
pgstrom_init_gpu_preagg(void)
{
	/* turn on/off gpu_groupby */
	DefineCustomBoolVariable("pg_strom.enable_gpupreagg",
							 "Enables the use of GPU-PreAgg",
							 NULL,
							 &pgstrom_enable_gpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.enable_numeric_aggfuncs */
	DefineCustomBoolVariable("pg_strom.enable_numeric_aggfuncs",
							 "Enable aggregate functions on numeric type",
							 NULL,
							 &pgstrom_enable_numeric_aggfuncs,
							 true,
							 PGC_USERSET,
							 GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.enable_partitionwise_gpugroupby */
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_gpupreagg",
							 "Enabled Enables partition wise GPU-PreAgg",
							 NULL,
							 &pgstrom_enable_partitionwise_gpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.hll_registers_bits */
	DefineCustomIntVariable("pg_strom.hll_registers_bits",
							"Accuracy of HyperLogLog COUNT(distinct ...) estimation",
							NULL,
							&pgstrom_hll_register_bits,
							9,
							4,
							15,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/* initialization of path method table */
	memset(&gpupreagg_path_methods, 0, sizeof(CustomPathMethods));
	gpupreagg_path_methods.CustomName          = "GpuPreAgg";
	gpupreagg_path_methods.PlanCustomPath      = PlanGpuPreAggPath;

	/* initialization of plan method table */
	memset(&gpupreagg_plan_methods, 0, sizeof(CustomScanMethods));
	gpupreagg_plan_methods.CustomName          = "GpuPreAgg";
	gpupreagg_plan_methods.CreateCustomScanState = CreateGpuPreAggScanState;
	RegisterCustomScanMethods(&gpupreagg_plan_methods);

	/* initialization of exec method table */
	memset(&gpupreagg_exec_methods, 0, sizeof(CustomExecMethods));
	gpupreagg_exec_methods.CustomName          = "GpuPreAgg";
	gpupreagg_exec_methods.BeginCustomScan     = pgstromExecInitTaskState;
	gpupreagg_exec_methods.ExecCustomScan      = pgstromExecTaskState;
	gpupreagg_exec_methods.EndCustomScan       = pgstromExecEndTaskState;
	gpupreagg_exec_methods.ReScanCustomScan    = pgstromExecResetTaskState;
	gpupreagg_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
	gpupreagg_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
	gpupreagg_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
	gpupreagg_exec_methods.ShutdownCustomScan  = pgstromSharedStateShutdownDSM;
	gpupreagg_exec_methods.ExplainCustomScan   = pgstromExplainTaskState;
	/* hook registration */
	if (!create_upper_paths_next)
	{
		create_upper_paths_next = create_upper_paths_hook;
		create_upper_paths_hook = XpuPreAggAddCustomPath;
		CacheRegisterSyscacheCallback(PROCOID, aggfunc_catalog_htable_invalidator, 0);
	}
}

/*
 * pgstrom_init_dpu_preagg
 */
void
pgstrom_init_dpu_preagg(void)
{
	/* turn on/off gpu_groupby */
	DefineCustomBoolVariable("pg_strom.enable_dpupreagg",
							 "Enables the use of DPU PreAgg",
							 NULL,
							 &pgstrom_enable_dpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.enable_partitionwise_dpupreagg */
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_dpupreagg",
							 "Enabled Enables partition wise DpuPreAgg",
							 NULL,
							 &pgstrom_enable_partitionwise_dpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* initialization of path method table */
	memset(&dpupreagg_path_methods, 0, sizeof(CustomPathMethods));
	dpupreagg_path_methods.CustomName     = "DpuPreAgg";
	dpupreagg_path_methods.PlanCustomPath = PlanDpuPreAggPath;

    /* initialization of plan method table */
    memset(&dpupreagg_plan_methods, 0, sizeof(CustomScanMethods));
    dpupreagg_plan_methods.CustomName     = "DpuPreAgg";
    dpupreagg_plan_methods.CreateCustomScanState = CreateDpuPreAggScanState;
    RegisterCustomScanMethods(&dpupreagg_plan_methods);

    /* initialization of exec method table */
    memset(&dpupreagg_exec_methods, 0, sizeof(CustomExecMethods));
    dpupreagg_exec_methods.CustomName          = "GpuPreAgg";
    dpupreagg_exec_methods.BeginCustomScan     = pgstromExecInitTaskState;
    dpupreagg_exec_methods.ExecCustomScan      = pgstromExecTaskState;
    dpupreagg_exec_methods.EndCustomScan       = pgstromExecEndTaskState;
    dpupreagg_exec_methods.ReScanCustomScan    = pgstromExecResetTaskState;
    dpupreagg_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
    dpupreagg_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
    dpupreagg_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
    dpupreagg_exec_methods.ShutdownCustomScan  = pgstromSharedStateShutdownDSM;
    dpupreagg_exec_methods.ExplainCustomScan   = pgstromExplainTaskState;
    /* hook registration */
	if (!create_upper_paths_next)
	{
		create_upper_paths_next = create_upper_paths_hook;
		create_upper_paths_hook = XpuPreAggAddCustomPath;
		CacheRegisterSyscacheCallback(PROCOID, aggfunc_catalog_htable_invalidator, 0);
	}
}
