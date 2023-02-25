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
static create_upper_paths_hook_type	create_upper_paths_next;
static CustomPathMethods	gpugroupby_path_methods;
static CustomScanMethods	gpugroupby_scan_methods;
static CustomExecMethods	gpugroupby_exec_methods;
static bool		pgstrom_enable_gpugroupby;
static bool		pgstrom_enable_partitionwise_gpugroupby;
static bool		pgstrom_enable_numeric_aggfuncs;
int				pgstrom_hll_register_bits;

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
	const char *finalfn_name;
	Oid			finalfn_argtype;
	const char *partfn_name;
	int			partfn_nargs;
	Oid			partfn_argtypes[8];
	int			partfn_argexprs[8];
	uint32_t	extra_flags;
	bool		numeric_aware;	/* ignored, if !enable_numeric_aggfuncs */
	
} aggfunc_catalog_t;

#define ALTFUNC_EXPR_NROWS_ONE      100	/* NROWS() */
#define ALTFUNC_EXPR_NROWS			101	/* NROWS(X) */
#define ALTFUNC_EXPR_PMIN			102	/* PMIN(X) */
#define ALTFUNC_EXPR_PMAX			103	/* PMAX(X) */
#define ALTFUNC_EXPR_PSUM			104	/* PSUM(X) */
#define ALTFUNC_EXPR_PSUM_X2		105	/* PSUM_X2(X) = PSUM(X^2) */
#define ALTFUNC_EXPR_PCOV_X			106	/* PCOV_X(X,Y) */
#define ALTFUNC_EXPR_PCOV_Y			107	/* PCOV_Y(X,Y) */
#define ALTFUNC_EXPR_PCOV_X2		108	/* PCOV_X2(X,Y) */
#define ALTFUNC_EXPR_PCOV_Y2		109	/* PCOV_Y2(X,Y) */
#define ALTFUNC_EXPR_PCOV_XY		110	/* PCOV_XY(X,Y) */
#define ALTFUNC_EXPR_HLL_HASH		111	/* HLL_HASH(X) */

static aggfunc_catalog_t	aggfunc_catalog_array[] = {
	/* COUNT(*) = SUM(NROWS()) */
	{ "count()",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS_ONE},
	  0, false
	},
	/* COUNT(X) = SUM(NROWS(X)) */
	{ "count(*)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS},
	  0, false
	},
	/* AVG(X) = EX_AVG(NROWS(X), PSUM(X)) */
	{ "avg(int1)",
	  "s:favg", INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "avg(int2)",
	  "s:favg", INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "avg(int4)",
	  "s:favg", INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "avg(int8)",
	  "s:favg", INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "avg(float2)",
	  "s:favg", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "avg(float4)",
	  "s:favg", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "avg(float8)",
	  "s:favg", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "avg(numeric)",
	  "s:favg", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, true
	},
	/*
	 * SUM(X) = SUM(PSUM(X))
	 */
	{ "sum(int1)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "sum(int2)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "sum(int4)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "sum(int8)",
	  "c:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "sum(float2)",
	  "c:sum", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "sum(float4)",
	  "c:sum", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "sum(float8)",
	  "c:sum", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false
	},
	{ "sum(numeric)",
	  "s:fsum", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, true
	},
	{ "sum(money)",
	  "c:sum", CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false
	},
	/*
	 * MAX(X) = MAX(PMAX(X))
	 */
	{ "max(int1)",
	  "s:fmax", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(int2)",
	  "s:fmax", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(int4)",
	  "s:fmax", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(int8)",
	  "s:fmax", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(float2)",
	  "c:max", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(float4)",
	  "c:max", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(float8)",
	  "c:max", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(numeric)",
	  "s:fmax", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(money)",
	  "c:max", CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false
	},
	{ "max(date)",
	  "c:max", DATEOID,
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(time)",
	  "c:max", TIMEOID,
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(timestamp)",
	  "c:max", TIMESTAMPOID,
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(timestamptz)",
	  "c:max", TIMESTAMPTZOID,
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	/*
	 * MIN(X) = MIN(PMIN(X))
	 */
	{ "min(int1)",
	  "s:fmin", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(int2)",
	  "s:fmin", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(int4)",
	  "s:fmin", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(int8)",
	  "s:fmin", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(float2)",
	  "c:min", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(float4)",
	  "c:min", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(float8)",
	  "c:min", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(numeric)",
	  "s:fmin", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(money)",
	  "c:min", CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(date)",
	  "c:min", DATEOID,
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(time)",
	  "c:min", TIMEOID,
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(timestamp)",
	  "c:min", TIMESTAMPOID,
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	{ "min(timestamptz)",
	  "c:min", TIMESTAMPTZOID,
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false
	},
	/*
	 * STDDEV(X) = EX_STDDEV_SAMP(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{ "stddev(int1)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev(int2)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev(int4)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev(int8)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev(float2)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev(float4)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev(float8)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev(numeric)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},

	/*
	 * STDDEV_SAMP(X) = EX_STDDEV_SAMP(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{ "stddev_samp(int1)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_samp(int2)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_samp(int4)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_samp(int8)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_samp(float2)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_samp(float4)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_samp(float8)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_samp(numeric)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	/*
	 * STDDEV_POP(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{ "stddev_pop(int1)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_pop(int2)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_pop(int4)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_pop(int8)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_pop(float2)",
	  "s:stddev_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_pop(float4)",
	  "s:stddev_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_pop(float8)",
	  "s:stddev_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false
	},
	{ "stddev_pop(numeric)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	/*
	 * VARIANCE(X) = VAR_SAMP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{ "variance(int1)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "variance(int2)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "variance(int4)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "variance(int8)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "variance(float2)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "variance(float4)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "variance(float8)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "variance(numeric)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, true
	},

	/*
	 * VAR_SAMP(X) = VAR_SAMP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{ "var_samp(int1)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_samp(int2)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_samp(int4)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_samp(int8)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_samp(float2)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_samp(float4)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_samp(float8)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_samp(numeric)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, true
	},

	/*
	 * VAR_POP(X)  = VAR_POP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{ "var_pop(int1)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_pop(int2)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_pop(int4)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_pop(int8)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_pop(float2)",
	  "s:var_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_pop(float4)",
	  "s:var_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_pop(float8)",
	  "s:var_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false
	},
	{ "var_pop(numeric)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, true
	},
	/*
	 * CORR(X,Y) = PGSTROM.CORR(NROWS(X,Y),
	 *                          PCOV_X(X,Y),  PCOV_Y(X,Y)
	 *                          PCOV_X2(X,Y), PCOV_Y2(X,Y),
	 *                          PCOV_XY(X,Y))
	 */
	{ "corr(float8,float8)",
	  "s:covar_samp", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "covar_pop(float8,float8)",
	  "s:covar_pop", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "covar_samp(float8,float8)",
	  "s:covar_samp", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	/*
	 * Aggregation to support least squares method
	 *
	 * That takes PSUM_X, PSUM_Y, PSUM_X2, PSUM_Y2, PSUM_XY according
	 * to the function
	 */
	{ "regr_avgx(float8,float8)",
	  "s:regr_avgx", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "regr_avgy(float8,float8)",
	  "s:regr_avgy", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "regr_count(float8,float8)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS},
	  0, false
	},
	{ "regr_intercept(float8,float8)",
	  "s:regr_intercept", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "regr_r2(float8,float8)",
	  "s:regr_r2", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "regr_slope(float8,float8)",
	  "s:regr_slope", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "regr_sxx(float8,float8)",
	  "s:regr_sxx", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "regr_sxy(float8,float8)",
	  "s:regr_sxy", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	{ "regr_syy(float8,float8)",
	  "s:regr_syy", FLOAT8ARRAYOID,
	  "s:pcovar", 6, {INT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  0, false
	},
	/* end of the catalog */
	{ NULL, NULL, InvalidOid, NULL, 0, {}, {}, 0, false },
};












/*
 * gpugroupby_add_custompath
 */
static void
gpugroupby_add_custompath(PlannerInfo *root,
						  UpperRelationKind stage,
						  RelOptInfo *input_rel,
						  RelOptInfo *output_rel,
						  void *extra)
{

	if (create_upper_paths_next)
		create_upper_paths_next(root,
								stage,
								input_rel,
								output_rel,
								extra);
	if (stage != UPPERREL_GROUP_AGG)
		return;
	if (!pgstrom_enabled || !pgstrom_enable_gpugroupby)
		return;
	



	

}

/*
 * PlanGpuGroupByPath
 */
static Plan *
PlanGpuGroupByPath(PlannerInfo *root,
				   RelOptInfo *rel,
				   struct CustomPath *best_path,
				   List *tlist,
				   List *clauses,
				   List *custom_plans)
{
	return NULL;
}

/*
 * CreateGpuGroupByScanState
 */
static Node *
CreateGpuGroupByScanState(CustomScan *cscan)
{
	pgstromTaskState *pts;
	int		num_rels = list_length(cscan->custom_plans);

	Assert(cscan->methods == &gpugroupby_plan_methods);
	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &gpugroupby_exec_methods;
	pts->task_kind = TASK_KIND__GPUGROUPBY;
	pts->pp_info = deform_pgstrom_plan_info(cscan);
	Assert(pts->pp_info->task_kind == pts->task_kind &&
		   pts->pp_info->num_rels == num_rels);
	pts->num_rels = num_rels;

	return (Node *)pts;
}

/*
 * ExecGpuGroupBy
 */
static TupleTableSlot *
ExecGpuGroupBy(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *)node;

	return pgstromExecTaskState(pts);
}

/*
 * ExecFallbackCpuGroupBy
 */
void
ExecFallbackCpuGroupBy(pgstromTaskState *pts, HeapTuple tuple)
{
	elog(ERROR, "ExecFallbackCpuGroupBy implemented");
}

/*
 * Entrypoint of GpuPreAgg
 */
void
pgstrom_init_gpu_groupby(void)
{
	/* turn on/off gpu_groupby */
	DefineCustomBoolVariable("pg_strom.enable_gpugroupby",
							 "Enables the use of GPU Group-By",
							 NULL,
							 &pgstrom_enable_gpugroupby,
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
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_gpugroupby",
							 "Enabled Enables partition wise GpuGroupBy",
							 NULL,
							 &pgstrom_enable_partitionwise_gpugroupby,
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
	memset(&gpugroupby_path_methods, 0, sizeof(CustomPathMethods));
	gpugroupby_path_methods.CustomName          = "GpuGroupBy";
	gpugroupby_path_methods.PlanCustomPath      = PlanGpuGroupByPath;

	/* initialization of plan method table */
	memset(&gpugroupby_scan_methods, 0, sizeof(CustomScanMethods));
	gpugroupby_scan_methods.CustomName          = "GpuGroupBy";
	gpugroupby_scan_methods.CreateCustomScanState = CreateGpuGroupByScanState;
	RegisterCustomScanMethods(&gpugroupby_scan_methods);

	/* initialization of exec method table */
	memset(&gpugroupby_exec_methods, 0, sizeof(CustomExecMethods));
	gpugroupby_exec_methods.CustomName          = "GpuGroupBy";
	gpugroupby_exec_methods.BeginCustomScan     = pgstromExecInitTaskState;
	gpugroupby_exec_methods.ExecCustomScan      = ExecGpuGroupBy;
	gpugroupby_exec_methods.EndCustomScan       = pgstromExecEndTaskState;
	gpugroupby_exec_methods.ReScanCustomScan    = pgstromExecResetTaskState;
	gpugroupby_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
	gpugroupby_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
	gpugroupby_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
	gpugroupby_exec_methods.ShutdownCustomScan  = pgstromSharedStateShutdownDSM;
	gpugroupby_exec_methods.ExplainCustomScan   = pgstromExplainTaskState;
	/* hook registration */
	create_upper_paths_next = create_upper_paths_hook;
	create_upper_paths_hook = gpugroupby_add_custompath;
}

