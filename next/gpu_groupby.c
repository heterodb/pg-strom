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
static CustomScanMethods	gpugroupby_plan_methods;
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

#define ALTFUNC_EXPR_NROWS			-101	/* NROWS(X) */
#define ALTFUNC_EXPR_PMIN			-102	/* PMIN(X) */
#define ALTFUNC_EXPR_PMAX			-103	/* PMAX(X) */
#define ALTFUNC_EXPR_PSUM			-104	/* PSUM(X) */
#define ALTFUNC_EXPR_PSUM_X2		-105	/* PSUM_X2(X) = PSUM(X^2) */
#define ALTFUNC_EXPR_PCOV_X			-106	/* PCOV_X(X,Y) */
#define ALTFUNC_EXPR_PCOV_Y			-107	/* PCOV_Y(X,Y) */
#define ALTFUNC_EXPR_PCOV_X2		-108	/* PCOV_X2(X,Y) */
#define ALTFUNC_EXPR_PCOV_Y2		-109	/* PCOV_Y2(X,Y) */
#define ALTFUNC_EXPR_PCOV_XY		-110	/* PCOV_XY(X,Y) */
#define ALTFUNC_EXPR_HLL_HASH		-111	/* HLL_HASH(X) */

static aggfunc_catalog_t	aggfunc_catalog_array[] = {
	/* COUNT(*) = SUM(NROWS()) */
	{ "count()",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS},
	  0, false,
	},
	/* COUNT(X) = SUM(NROWS(X)) */
	{ "count(any)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS},
	  0, false,
	},
	/* AVG(X) = EX_AVG(NROWS(X), PSUM(X)) */
	{ "avg(int1)",
	  "s:favg", INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "avg(int2)",
	  "s:favg", INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "avg(int4)",
	  "s:favg", INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "avg(int8)",
	  "s:favg", INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "avg(float2)",
	  "s:favg", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "avg(float4)",
	  "s:favg", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "avg(float8)",
	  "s:favg", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "avg(numeric)",
	  "s:favg", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  0, true,
	},
	/*
	 * SUM(X) = SUM(PSUM(X))
	 */
	{ "sum(int1)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "sum(int2)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "sum(int4)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "sum(int8)",
	  "c:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "sum(float2)",
	  "c:sum", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "sum(float4)",
	  "c:sum", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "sum(float8)",
	  "c:sum", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	{ "sum(numeric)",
	  "s:fsum_numeric", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  0, true,
 	},
	{ "sum(money)",
	  "c:sum", CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PSUM},
	  0, false,
	},
	/*
	 * MAX(X) = MAX(PMAX(X))
	 */
	{ "max(int1)",
	  "s:fmax", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(int2)",
	  "s:fmax", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(int4)",
	  "s:fmax", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(int8)",
	  "s:fmax", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(float2)",
	  "c:max", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(float4)",
	  "c:max", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(float8)",
	  "c:max", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(numeric)",
	  "s:fmax", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
	},
	{ "max(money)",
	  "c:max", CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX},
	  0, false,
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
	  0, false,
	},
	{ "min(int2)",
	  "s:fmin", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(int4)",
	  "s:fmin", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(int8)",
	  "s:fmin", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(float2)",
	  "c:min", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(float4)",
	  "c:min", FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(float8)",
	  "c:min", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(numeric)",
	  "s:fmin", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(money)",
	  "c:min", CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(date)",
	  "c:min", DATEOID,
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(time)",
	  "c:min", TIMEOID,
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(timestamp)",
	  "c:min", TIMESTAMPOID,
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	{ "min(timestamptz)",
	  "c:min", TIMESTAMPTZOID,
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMIN},
	  0, false,
	},
	/*
	 * STDDEV(X) = EX_STDDEV_SAMP(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{ "stddev(int1)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev(int2)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev(int4)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev(int8)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev(float2)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev(float4)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev(float8)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev(numeric)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
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
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_samp(int2)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_samp(int4)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_samp(int8)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_samp(float2)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_samp(float4)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_samp(float8)",
	  "s:stddev_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_samp(numeric)",
	  "s:stddev_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
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
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_pop(int2)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_pop(int4)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_pop(int8)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_pop(float2)",
	  "s:stddev_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_pop(float4)",
	  "s:stddev_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_pop(float8)",
	  "s:stddev_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  0, false,
	},
	{ "stddev_pop(numeric)",
	  "s:stddev_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
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
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "variance(int2)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "variance(int4)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "variance(int8)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "variance(float2)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "variance(float4)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "variance(float8)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "variance(numeric)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, true,
	},

	/*
	 * VAR_SAMP(X) = VAR_SAMP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{ "var_samp(int1)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_samp(int2)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_samp(int4)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_samp(int8)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_samp(float2)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_samp(float4)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_samp(float8)",
	  "s:var_sampf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_samp(numeric)",
	  "s:var_samp", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, true,
	},

	/*
	 * VAR_POP(X)  = VAR_POP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{ "var_pop(int1)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_pop(int2)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_pop(int4)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_pop(int8)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_pop(float2)",
	  "s:var_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_pop(float4)",
	  "s:var_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_pop(float8)",
	  "s:var_popf", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, false,
	},
	{ "var_pop(numeric)",
	  "s:var_pop", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID,FLOAT8OID,FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
      0, true,
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
	  0, false,
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
	  0, false,
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
	  0, false,
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
	  0, false,
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
	  0, false,
	},
	{ "regr_count(float8,float8)",
	  "s:sum", INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS},
	  0, false,
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
	  0, false,
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
	  0, false,
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
	  0, false,
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
	  0, false,
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
	  0, false,
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
	  0, false,
	},
};

static const aggfunc_catalog_t *
aggfunc_lookup_by_oid(Oid aggfn_oid)
{
	Form_pg_proc	proc;
	HeapTuple		htup;

	htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(aggfn_oid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for function %u", aggfn_oid);
	proc = (Form_pg_proc) GETSTRUCT(htup);
	if (proc->pronamespace == PG_CATALOG_NAMESPACE &&
		proc->pronargs <= 2)
	{
		char	signature[3 * NAMEDATALEN + 10];
		int		off;

		off = sprintf(signature, "%s(", NameStr(proc->proname));
		for (int j=0; j < proc->pronargs; j++)
		{
			Oid		type_oid = proc->proargtypes.values[j];
			char   *type_name = get_type_name(type_oid, false);

			off += sprintf(signature + off, "%s%s", (j>0 ? "," : ""), type_name);
		}
		off += sprintf(signature + off, ")");
		
		for (int i=0; i < lengthof(aggfunc_catalog_array); i++)
		{
			const aggfunc_catalog_t *cat = &aggfunc_catalog_array[i];

			if (strcmp(signature, cat ->aggfn_signature) == 0)
			{
				/* Is NUMERIC with xPU GroupBy acceptable? */
				if (cat->numeric_aware && !pgstrom_enable_numeric_aggfuncs)
					continue;
				/* all ok */
				ReleaseSysCache(htup);
				return cat;
			}
		}
	}
	ReleaseSysCache(htup);
	return NULL;
}

/*
 * xpugroupby_build_path_context
 */
typedef struct
{
	bool		device_executable;
	PlannerInfo	   *root;
	RelOptInfo	   *group_rel;
	double			num_groups;
	Path		   *input_path;
	PathTarget	   *target_upper;
	PathTarget	   *target_partial;
	PathTarget	   *target_final;
	AggClauseCosts	final_clause_costs;
	pgstromPlanInfo *pp_prev;
	List		   *input_rels_tlist;
	List		   *inner_paths_list;
	Node		   *havingQual;
	uint32_t		task_kind;
	const CustomPathMethods *custom_path_methods;
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

	if (source_type == target_type)
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
 * make_expr_conditional - constructor of expression conditional
 */
static Expr *
make_expr_conditional(Expr *expr, Expr *filter, bool zero_if_unmatched)
{
	Oid			expr_typeoid = exprType((Node *)expr);
	int32		expr_typemod = exprTypmod((Node *)expr);
	Oid			expr_collid = exprCollation((Node *)expr);
	CaseWhen   *case_when;
	CaseExpr   *case_expr;
	Expr	   *defresult;

	if (!filter)
		return expr;
	if (zero_if_unmatched)
	{
		int16   typlen;
		bool    typbyval;

		get_typlenbyval(expr_typeoid, &typlen, &typbyval);
		defresult = (Expr *) makeConst(expr_typeoid,
									   expr_typemod,
									   expr_collid,
									   (int) typlen,
									   (Datum) 0,
									   false,
									   typbyval);
	}
	else
	{
		defresult = (Expr *) makeNullConst(expr_typeoid,
										   expr_typemod,
										   expr_collid);
	}
	/* in case when the 'filter' is matched */
	case_when = makeNode(CaseWhen);
	case_when->expr = filter;
	case_when->result = expr;
	case_when->location = -1;

	/* case body */
	case_expr = makeNode(CaseExpr);
	case_expr->casetype = exprType((Node *) expr);
	case_expr->arg = NULL;
	case_expr->args = list_make1(case_when);
	case_expr->defresult = defresult;
	case_expr->location = -1;

	return (Expr *) case_expr;
}

/*
 * make_altfunc_simple_expr - constructor of simple function call
 */
static FuncExpr *
make_altfunc_simple_expr(const char *func_name, Expr *func_arg)
{
	Oid			namespace_oid = get_namespace_oid("pgstrom", false);
	oidvector  *func_argtypes;
	HeapTuple	htup;
	Form_pg_proc proc;
	FuncExpr   *fn_expr;

	if (func_arg)
	{
		Oid		type_oid = exprType((Node *)func_arg);
		func_argtypes = buildoidvector(&type_oid, 1);
	}
	else
	{
		func_argtypes = buildoidvector(NULL, 0);
	}

	/* find an alternative partial function */
	htup = SearchSysCache3(PROCNAMEARGSNSP,
						   PointerGetDatum(func_name),
                           PointerGetDatum(func_argtypes),
                           ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "alternative function '%s' not found", func_name);
	proc = (Form_pg_proc) GETSTRUCT(htup);
	fn_expr = makeFuncExpr(proc->oid,
						   proc->prorettype,
						   func_arg ? list_make1(func_arg) : NIL,
						   InvalidOid,
						   InvalidOid,
						   COERCE_EXPLICIT_CALL);
	ReleaseSysCache(htup);

	return fn_expr;
}

/*
 * make_altfunc_nrows_expr - constructor for ALTFUNC_EXPR_NROWS
 */
static FuncExpr *
make_altfunc_nrows_expr(Aggref *aggref)
{
	List	   *nrows_args = NIL;
	ListCell   *lc;
	Expr	   *expr;

	foreach (lc, aggref->args)
	{
		TargetEntry *tle = lfirst(lc);
		NullTest	*ntest = makeNode(NullTest);

		Assert(IsA(tle, TargetEntry));
		ntest->arg = copyObject(tle->expr);
		ntest->nulltesttype = IS_NOT_NULL;
		ntest->argisrow = false;

		nrows_args = lappend(nrows_args, ntest);
	}

	if (aggref->aggfilter)
	{
		Expr   *filter = aggref->aggfilter;

		Assert(exprType((Node *)filter) == BOOLOID);
		nrows_args = lappend(nrows_args, copyObject(filter));
	}

	if (nrows_args == NIL)
		expr = NULL;
	else if (list_length(nrows_args) == 1)
		expr = linitial(nrows_args);
	else
		expr = make_andclause(nrows_args);

	return make_altfunc_simple_expr("nrows", expr);
}

/*
 * make_altfunc_minmax_expr - constructor of ALTFUNC_EXPR_PMIN/PMAX
 */
static FuncExpr *
make_altfunc_minmax_expr(Aggref *aggref, const char *func_name,
						 Oid pminmax_typeoid)
{
	TargetEntry *tle;
	Expr   *expr;

	Assert(list_length(aggref->args) == 1);
	tle = linitial(aggref->args);
	Assert(IsA(tle, TargetEntry));
	/* cast to pminmax_typeoid, if mismatch */
	expr = make_expr_typecast(tle->expr, pminmax_typeoid);
	/* make conditional if aggref has any filter */
	expr = make_expr_conditional(expr, aggref->aggfilter, false);

	return make_altfunc_simple_expr(func_name, expr);
}

/*
 * make_altfunc_psum_expr - constructor of ALTFUNC_EXPR_PSUM/PSUM_X2
 */
static FuncExpr *
make_altfunc_psum_expr(Aggref *aggref, const char *func_name,
					   Oid psum_typeoid)
{
	TargetEntry *tle;
    Expr   *expr;

	Assert(list_length(aggref->args) == 1);
	tle = linitial(aggref->args);
	Assert(IsA(tle, TargetEntry));
	/* cast to psum_typeoid, if mismatch */
	expr = make_expr_typecast(tle->expr, psum_typeoid);
	/* make conditional if aggref has any filter */
	expr = make_expr_conditional(expr, aggref->aggfilter, true);

	return make_altfunc_simple_expr(func_name, expr);
}

/*
 * make_altfunc_pcov_xy - constructor of a co-variance arguments
 */
static FuncExpr *
make_altfunc_pcov_xy(Aggref *aggref, const char *func_name)
{
	Oid			namespace_oid = get_namespace_oid("pgstrom", false);
	oidvector  *func_argtypes;
    Oid			func_argtypes_oid[3];
    Oid			func_oid;
    TargetEntry *tle_x;
    TargetEntry *tle_y;
    Expr	   *filter;

	Assert(list_length(aggref->args) == 2);
	tle_x = linitial(aggref->args);
	tle_y = lsecond(aggref->args);
	if (exprType((Node *)tle_x->expr) != FLOAT8OID ||
		exprType((Node *)tle_y->expr) != FLOAT8OID)
		elog(ERROR, "Bug? unexpected argument type for co-variance");

	/* lookup pcov_XXX functions */
	func_argtypes_oid[0] = BOOLOID;
	func_argtypes_oid[1] = FLOAT8OID;
	func_argtypes_oid[2] = FLOAT8OID;
	func_argtypes = buildoidvector(func_argtypes_oid, 3);
	func_oid = GetSysCacheOid3(PROCNAMEARGSNSP,
							   Anum_pg_proc_oid,
							   CStringGetDatum(func_name),
							   PointerGetDatum(func_argtypes),
							   ObjectIdGetDatum(namespace_oid));
	if (!OidIsValid(func_oid))
		elog(ERROR, "cache lookup failed for function '%s'", func_name);
	/* filter if any */
	if (aggref->aggfilter)
	{
		filter = aggref->aggfilter;
	}
    else
	{
        filter = (Expr *)makeBoolConst(true, false);
	}
	return makeFuncExpr(func_oid,
						FLOAT8OID,
						list_make3(filter,
								   tle_x->expr,
								   tle_y->expr),
						InvalidOid,
						InvalidOid,
						COERCE_EXPLICIT_CALL);
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
	const aggfunc_catalog_t *aggfn_cat;
	List	   *partfn_args = NIL;
	Expr	   *partfn;
	Aggref	   *aggref_alt;
	Oid			namespace_oid;
	Oid			func_oid;
	const char *func_name;
	oidvector  *func_argtypes;
	HeapTuple	htup;
	Form_pg_proc proc;
	Form_pg_aggregate agg;


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
	aggfn_cat = aggfunc_lookup_by_oid(aggref->aggfnoid);
	if (!aggfn_cat)
	{
		elog(DEBUG2, "Aggregate function '%s' is not device executable",
			 format_procedure(aggref->aggfnoid));
		return NULL;
	}
	/* sanity checks */
	Assert(aggref->aggkind == AGGKIND_NORMAL &&
		   !aggref->aggvariadic &&
		   list_length(aggref->args) <= 2);
	/*
	 * construct arguments list of the partial aggregation
	 */
	for (int i=0; i < aggfn_cat->partfn_nargs; i++)
	{
		Oid			argtype = aggfn_cat->partfn_argtypes[i];
		int			action  = aggfn_cat->partfn_argexprs[i];
		FuncExpr   *pfunc;
		ListCell   *lc;

		switch (action)
		{
			case ALTFUNC_EXPR_NROWS:
				pfunc = make_altfunc_nrows_expr(aggref);
				break;
			case ALTFUNC_EXPR_PMIN:
				pfunc = make_altfunc_minmax_expr(aggref, "pmin", argtype);
				break;
			case ALTFUNC_EXPR_PMAX:
				pfunc = make_altfunc_minmax_expr(aggref, "pmax", argtype);
				break;
			case ALTFUNC_EXPR_PSUM:
				pfunc = make_altfunc_psum_expr(aggref, "psum", argtype);
				break;
			case ALTFUNC_EXPR_PSUM_X2:
				pfunc = make_altfunc_psum_expr(aggref, "psum_x2", argtype);
				break;
			case ALTFUNC_EXPR_PCOV_X:
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_x");
				break;
			case ALTFUNC_EXPR_PCOV_Y:
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_y");
				break;
			case ALTFUNC_EXPR_PCOV_X2:
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_x2");
				break;
			case ALTFUNC_EXPR_PCOV_Y2:
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_y2");
				break;
			case ALTFUNC_EXPR_PCOV_XY:
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_xy");
				break;
			default:
				elog(ERROR, "Bug? XPU GroupBy catalog is corrupted?");
				break;
		}
		/* actuall supported? */
		if (!pfunc)
			return NULL;
		/* device executable? */
		foreach (lc, pfunc->args)
		{
			Expr   *expr = lfirst(lc);

			if (!pgstrom_xpu_expression(expr,
										con->task_kind,
										con->input_rels_tlist,
										NULL))
			{
				elog(DEBUG2, "Partial aggregate argument is not executable: %s",
					 nodeToString(expr));
				return NULL;
			}
		}
		/* append to the argument list */
		partfn_args = lappend(partfn_args, (Expr *)pfunc);
	}

	/*
	 * Lookup the partial function that generate partial state of the aggregate
	 * function, or varref if internal state of aggregate is identical.
	 */
	if (strcmp(aggfn_cat->partfn_name, "varref") == 0)
	{
		Assert(list_length(partfn_args) == 1);
		partfn = linitial(partfn_args);
	}
	else
	{
		if (strncmp(aggfn_cat->partfn_name, "c:", 2) == 0)
			namespace_oid = PG_CATALOG_NAMESPACE;
		else if (strncmp(aggfn_cat->partfn_name, "s:", 2) == 0)
			namespace_oid = get_namespace_oid("pgstrom", false);
		else
			elog(ERROR, "Bug? corrupted aggregate function catalog");

		func_name = aggfn_cat->partfn_name + 2;
		func_argtypes = buildoidvector(aggfn_cat->partfn_argtypes,
									   aggfn_cat->partfn_nargs);
		htup = SearchSysCache3(PROCNAMEARGSNSP,
							   PointerGetDatum(func_name),
                               PointerGetDatum(func_argtypes),
                               ObjectIdGetDatum(namespace_oid));
		if (!HeapTupleIsValid(htup))
			elog(ERROR, "cache lookup failed for function %s", func_name);
		proc = (Form_pg_proc) GETSTRUCT(htup);
		partfn = (Expr *)makeFuncExpr(proc->oid,
									  proc->prorettype,
									  partfn_args,
									  InvalidOid,
									  InvalidOid,
									  COERCE_EXPLICIT_CALL);
		ReleaseSysCache(htup);
	}
	/* add partial function if unique */
	add_new_column_to_pathtarget(con->target_partial, partfn);

	/*
	 * Construction of the final Aggref instead of the original one
	 */
	if (strncmp(aggfn_cat->finalfn_name, "c:", 2) == 0)
		namespace_oid = PG_CATALOG_NAMESPACE;
	else if (strncmp(aggfn_cat->finalfn_name, "s:", 2) == 0)
		namespace_oid = get_namespace_oid("pgstrom", false);
	else
		elog(ERROR, "Bug? corrupted aggregate function catalog");

	func_name = aggfn_cat->finalfn_name + 2;
	func_argtypes = buildoidvector(&aggfn_cat->finalfn_argtype, 1);
	func_oid = GetSysCacheOid3(PROCNAMEARGSNSP,
							   Anum_pg_proc_oid,
							   CStringGetDatum(func_name),
							   PointerGetDatum(func_argtypes),
							   ObjectIdGetDatum(namespace_oid));
	if (!OidIsValid(func_oid))
		elog(ERROR, "cache lookup failed for function %s", func_name);
	Assert(aggref->aggtype == get_func_rettype(func_oid));

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
	aggref_alt->aggtransno    = aggref->aggtransno;
	aggref_alt->location      = aggref->location;

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
	PathTarget *target_input = con->input_path->pathtarget;
	Node	   *aggfn;
	ListCell   *lc;

	if (!node)
		return NULL;
	if (IsA(node, Aggref))
	{
		aggfn = make_alternative_aggref(con, (Aggref *)node);
		if (!aggfn)
			con->device_executable = false;
		return aggfn;
	}

	foreach (lc, target_input->exprs)
	{
		Expr   *expr = lfirst(lc);

		if (equal(node, expr))
		{
			add_new_column_to_pathtarget(con->target_partial, copyObject(expr));
			return copyObject(node);
		}
	}
	if (IsA(node, Var) || IsA(node, PlaceHolderVar))
		elog(ERROR, "Bug? referenced variable is grouping-key nor its dependent key: %s",
			 nodeToString(node));
	return expression_tree_mutator(node, replace_expression_by_altfunc, con);
}

static bool
xpugroupby_build_path_target(xpugroupby_build_path_context *con)
{
	PlannerInfo *root = con->root;
	Query	   *parse = root->parse;
	PathTarget *target_upper = con->target_upper;
	Node	   *havingQual = NULL;
	ListCell   *lc;
	int			i;

	/*
	 * Pick up Grouping-Keys and Aggregate-Functions
	 */
	i = 0;
	foreach (lc, target_upper->exprs)
	{
		Expr   *expr = lfirst(lc);
		Index	sortgroupref = get_pathtarget_sortgroupref(target_upper, i);

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
			/*
			 * grouping-key must be device executable.
			 */
			if (!pgstrom_xpu_expression(expr,
										con->task_kind,
										con->input_rels_tlist,
										NULL))
			{
				elog(DEBUG2, "Grouping-key must be device executable: %s",
					 nodeToString(expr));
				return false;
			}
			/* add grouping-key */
			add_column_to_pathtarget(con->target_partial, expr, sortgroupref);
			add_column_to_pathtarget(con->target_final, expr, sortgroupref);
		}
		else
		{
			/* Aggregation */
			Expr   *altfn;

			altfn = (Expr *)replace_expression_by_altfunc((Node *)expr, con);
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
		i++;
	}

	/*
	 * HAVING clause
	 */
	if (parse->havingQual)
	{
		havingQual = replace_expression_by_altfunc(parse->havingQual, con);
		if (!havingQual)
		{
			elog(DEBUG2, "unable to replace HAVING to alternative aggregation: %s",
				 nodeToString(parse->havingQual));
			return false;
		}
	}
	con->havingQual = havingQual;

	set_pathtarget_cost_width(root, con->target_final);
	set_pathtarget_cost_width(root, con->target_partial);

	return true;
}

/*
 * prepend_partial_groupby_custompath
 */
static Path *
prepend_partial_groupby_custompath(xpugroupby_build_path_context *con)
{
	Query	   *parse = con->root->parse;
	CustomPath *cpath = makeNode(CustomPath);
	Path	   *input_path = con->input_path;
	PathTarget *target_partial = con->target_partial;
	pgstromPlanInfo *pp_prev = con->pp_prev;
	pgstromPlanInfo *pp_info;
	double		num_group_keys;
	double		xpu_ratio;
	Cost		xpu_operator_cost;
	Cost		xpu_tuple_cost;
	Cost		startup_cost = 0.0;
	Cost		run_cost = 0.0;
	Cost		final_cost = 0.0;

	/*
	 * Parameters related to devices
	 */
	if ((con->task_kind & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU)
	{
		xpu_operator_cost = pgstrom_gpu_operator_cost;
		xpu_tuple_cost    = pgstrom_gpu_tuple_cost;
		xpu_ratio         = pgstrom_gpu_operator_ratio();
	}
	else if ((con->task_kind & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU)
	{
		xpu_operator_cost = pgstrom_dpu_operator_cost;
        xpu_tuple_cost    = pgstrom_dpu_tuple_cost;
		xpu_ratio         = pgstrom_dpu_operator_ratio();
	}
	else
	{
		elog(ERROR, "Bug? unexpected task_kind: %08x", con->task_kind);
	}
	startup_cost = input_path->startup_cost;
	run_cost = (input_path->total_cost -
				input_path->startup_cost - pp_prev->final_cost);
	/* Cost estimation for grouping */
	num_group_keys = list_length(parse->groupClause);
	startup_cost += (xpu_operator_cost *
					 num_group_keys *
					 input_path->rows);
	/* Cost estimation for aggregate function */
	startup_cost += (target_partial->cost.per_tuple * input_path->rows +
					 target_partial->cost.startup) * xpu_ratio;
	/* Cost estimation to fetch results */
	final_cost = xpu_tuple_cost * con->num_groups;
	if (input_path->parallel_workers > 0)
		final_cost *= (0.5 + (double)input_path->parallel_workers);

	pp_info = pmemdup(pp_prev, offsetof(pgstromPlanInfo,
										inners[pp_prev->num_rels]));
	pp_info->final_cost = final_cost;

	cpath->path.pathtype         = T_CustomScan;
	cpath->path.parent           = input_path->parent;
	cpath->path.pathtarget       = con->target_partial;
	cpath->path.param_info       = input_path->param_info;
	cpath->path.parallel_safe    = input_path->parallel_safe;
	cpath->path.parallel_aware   = input_path->parallel_aware;
	cpath->path.parallel_workers = input_path->parallel_workers;
	cpath->path.rows             = con->num_groups;
	cpath->path.startup_cost     = startup_cost;
	cpath->path.total_cost       = startup_cost + run_cost + final_cost;
	cpath->path.pathkeys         = NIL;
	cpath->custom_paths          = con->inner_paths_list;
	cpath->custom_private        = list_make1(pp_info);
	cpath->methods               = con->custom_path_methods;

	return &cpath->path;
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
		add_path(con->group_rel, agg_path);
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
			add_path(con->group_rel, agg_path);
		}
	}
}

static void
__xpugroupby_add_custompath(PlannerInfo *root,
							Path *input_path,
							RelOptInfo *group_rel,
							void *extra,
							bool try_parallel,
							double num_groups,
							uint32_t task_kind,
							const CustomPathMethods *custom_path_methods)
{
	xpugroupby_build_path_context con;
	Path	   *part_path;

	/* setup context */
	memset(&con, 0, sizeof(con));
	con.device_executable = true;
	con.root           = root;
	con.group_rel      = group_rel;
	con.num_groups     = num_groups;
	con.input_path     = input_path;
	con.target_upper   = root->upper_targets[UPPERREL_GROUP_AGG];
	con.target_partial = create_empty_pathtarget();
    con.target_final   = create_empty_pathtarget();
	con.task_kind      = task_kind;
	con.custom_path_methods = custom_path_methods;
	extract_input_path_params(input_path,
							  NULL,
							  &con.pp_prev,
							  &con.input_rels_tlist,
							  &con.inner_paths_list);
	/* construction of the target-list for each level */
	if (!xpugroupby_build_path_target(&con))
		return;

	/* build partial groupby custom-path */
	part_path = prepend_partial_groupby_custompath(&con);

	/* prepend Gather if parallel-aware path */
	if (try_parallel)
	{
		if (part_path->parallel_aware &&
			part_path->parallel_workers > 0)
		{
			double	total_groups = (part_path->rows *
									part_path->parallel_workers);
			part_path = (Path *)create_gather_path(root,
												   group_rel,
												   part_path,
												   con.target_partial,
												   NULL,
												   &total_groups);
		}
		else
		{
			/* unable to inject parallel paths */
			return;
		}
	}
	/* try add final groupby path */
	try_add_final_groupby_paths(&con, part_path);
}

void
xpugroupby_add_custompath(PlannerInfo *root,
						  RelOptInfo *input_rel,
						  RelOptInfo *group_rel,
						  void *extra,
						  uint32_t task_kind,
						  const CustomPathMethods *custom_path_methods)
{
	Query	   *parse = root->parse;
	Path	   *input_path;
	double		num_groups = 1.0;
	ListCell   *lc;

	/* fetch num groups from the standard paths */
	if (parse->groupClause)
	{
		if (parse->groupingSets != NIL ||
			!grouping_is_hashable(parse->groupClause))
		{
			elog(DEBUG2, "GROUP BY clause is not supported form");
			return;
		}

		foreach (lc, group_rel->pathlist)
		{
			Path   *i_path = lfirst(lc);

			if (IsA(i_path, Agg))
			{
				num_groups = ((AggPath *)i_path)->numGroups;
				break;
			}
		}
		if (!lc)
			return;		/* unable to determine the num_groups */
	}
	
	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		if (IS_SIMPLE_REL(input_rel))
		{
			input_path = (Path *)buildXpuScanPath(root,
												  input_rel,
												  (try_parallel > 0),
												  false,
												  true,
												  task_kind);
		}
		else
		{
			input_path = (Path *)custom_path_find_cheapest(root,
														   input_rel,
														   (try_parallel > 0),
														   task_kind);
		}

		if (input_path)
			__xpugroupby_add_custompath(root,
										input_path,
										group_rel,
										extra,
										num_groups,
										(try_parallel > 0),
										task_kind,
										custom_path_methods);
	}
}

/*
 * gpugroupby_add_custompath
 */
static void
gpugroupby_add_custompath(PlannerInfo *root,
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
	if (!pgstrom_enabled || !pgstrom_enable_gpugroupby)
		return;
	/* add custom-paths */
	xpugroupby_add_custompath(root,
							  input_rel,
							  group_rel,
							  extra,
							  TASK_KIND__GPUGROUPBY,
							  &gpugroupby_path_methods);
}

/*
 * PlanGpuGroupByPath
 */
static Plan *
PlanGpuGroupByPath(PlannerInfo *root,
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
								  &gpugroupby_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);
	return &cscan->scan.plan;
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
	memset(&gpugroupby_plan_methods, 0, sizeof(CustomScanMethods));
	gpugroupby_plan_methods.CustomName          = "GpuGroupBy";
	gpugroupby_plan_methods.CreateCustomScanState = CreateGpuGroupByScanState;
	RegisterCustomScanMethods(&gpugroupby_plan_methods);

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

