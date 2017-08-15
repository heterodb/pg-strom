--
-- functions for GpuPreAgg
--

-- NROWS()
CREATE FUNCTION pgstrom.nrows()
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.nrows(bool)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.sum(int8)
(
  sfunc = pg_catalog.int8pl,
  stype = int8,
  initcond = 0,
  parallel = safe
);

-- AVG()
CREATE FUNCTION pgstrom.pavg(int8,int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','pgstrom_partial_avg_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pavg(int8,float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','pgstrom_partial_avg_float8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_accum(int8[], int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_int8_accum'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_final(int8[])
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_int8_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_accum(float8[], float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_float8_accum'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_final(float8[])
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_float8_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_numeric_final(float8[])
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_numeric_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.favg(int8[])
(
  sfunc = pgstrom.favg_accum,
  stype = int8[],
  finalfunc = pgstrom.favg_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.favg(float8[])
(
  sfunc = pgstrom.favg_accum,
  stype = float8[],
  finalfunc = pgstrom.favg_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.favg_numeric(float8[])
(
  sfunc = pgstrom.favg_accum,
  stype = float8[],
  finalfunc = pgstrom.favg_numeric_final,
  parallel = safe
);

-- PMIN()/PMAX()
CREATE FUNCTION pgstrom.pmin(int2)
  RETURNS int2
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(int2)
  RETURNS int2
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmin(int4)
  RETURNS int4
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(int4)
  RETURNS int4
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmin(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmin(float4)
  RETURNS float4
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(float4)
  RETURNS float4
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmin(float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.fmin_numeric(float8)
(
  sfunc = pg_catalog.float8smaller,
  stype = float8,
  finalfunc = pg_catalog.numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.fmax_numeric(float8)
(
  sfunc = pg_catalog.float8larger,
  stype = float8,
  finalfunc = pg_catalog.numeric,
  parallel = safe
);

CREATE FUNCTION pgstrom.pmin(money)
  RETURNS money
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(money)
  RETURNS money
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmin(time)
  RETURNS time
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(time)
  RETURNS time
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmin(timestamp)
  RETURNS timestamp
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(timestamp)
  RETURNS timestamp
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmin(timestamp with time zone)
  RETURNS timestamp with time zone
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(timestamp with time zone)
  RETURNS timestamp with time zone
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

-- PSUM()/PSUM_X2()
CREATE FUNCTION pgstrom.psum(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_sum_any'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.psum(float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_sum_any'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.psum(float4)
  RETURNS float4
  AS 'MODULE_PATHNAME', 'pgstrom_partial_sum_any'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.psum(money)
  RETURNS money
  AS 'MODULE_PATHNAME', 'pgstrom_partial_sum_any'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.psum_x2(float4)
  RETURNS float4
  AS 'MODULE_PATHNAME', 'pgstrom_partial_sum_x2_float4'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.psum_x2(float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_sum_x2_float8'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.fsum_numeric(float8)
(
  sfunc = pg_catalog.float8pl,
  stype = float8,
  finalfunc = pg_catalog.numeric,
  parallel = safe
);

-- PCOV_*
CREATE FUNCTION pgstrom.pcov_x(bool,float8,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_cov_x'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pcov_y(bool,float8,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_cov_y'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pcov_x2(bool,float8,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_cov_x2'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pcov_y2(bool,float8,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_cov_y2'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pcov_xy(bool,float8,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_partial_cov_xy'
  LANGUAGE C STRICT PARALLEL SAFE;

-- STDDEV/STDDEV_POP/STDDEV_SAMP
-- VARIANCE/VAR_POP/VAR_SAM
CREATE FUNCTION pgstrom.pvariance(int8,float8,float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_partial_variance_float8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8_stddev_samp_numeric(float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','pgstrom_float8_stddev_samp_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8_stddev_pop_numeric(float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','pgstrom_float8_stddev_pop_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8_var_samp_numeric(float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','pgstrom_float8_var_samp_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8_var_pop_numeric(float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','pgstrom_float8_var_pop_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.stddev(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  finalfunc = pg_catalog.float8_stddev_samp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_numeric(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  finalfunc = pgstrom.float8_stddev_samp_numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_pop(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pg_catalog.float8_stddev_pop,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_pop_numeric(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pgstrom.float8_stddev_pop_numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_samp(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pg_catalog.float8_stddev_samp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_samp_numeric(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pgstrom.float8_stddev_samp_numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.variance(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pg_catalog.float8_var_samp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.variance_numeric(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pgstrom.float8_var_samp_numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_pop(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pg_catalog.float8_var_pop,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_pop_numeric(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pgstrom.float8_var_pop_numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_samp(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pg_catalog.float8_var_samp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_samp_numeric(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = pgstrom.float8_var_samp_numeric,
  parallel = safe
);

-- CORR/COVAR_POP/COVAR_SAMP
-- REGR_*
CREATE FUNCTION pgstrom.pcovar(int8,float8,float8,float8,float8,float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_partial_covariance_float8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.corr(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_corr,
  parallel = safe
);

CREATE AGGREGATE pgstrom.covar_pop(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_covar_pop,
  parallel = safe
);

CREATE AGGREGATE pgstrom.covar_samp(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_covar_samp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_avgx(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_avgx,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_avgy(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_avgy,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_intercept(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_intercept,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_r2(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_r2,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_slope(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_slope,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_sxx(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_sxx,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_sxy(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_sxy,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_syy(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_syy,
  parallel = safe
);
