--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

--
-- pg_strom installation queries
--
CREATE TYPE __pgstrom_device_info AS (
  id		int4,
  property	text,
  value		text
);
CREATE FUNCTION pgstrom_device_info()
  RETURNS SETOF __pgstrom_device_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_scoreboard_info AS (
  attribute	text,
  value		text
);
CREATE FUNCTION pgstrom_scoreboard_info()
  RETURNS SETOF __pgstrom_scoreboard_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_program_info AS (
  addr			int8,
  length		int8,
  active		bool,
  status		text,
  crc32			int4,
  flags			int4,
  kern_define   text,
  kern_source	text,
  kern_binary	bytea,
  error_msg		text,
  backends		text
);
CREATE FUNCTION pgstrom_program_info()
  RETURNS SETOF __pgstrom_program_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

--
-- functions for GpuPreAgg
--

-- Definition of NROWS(...)
CREATE FUNCTION pgstrom.nrows()
  RETURNS int4
  AS 'MODULE_PATHNAME', 'gpupreagg_partial_nrows'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.nrows(bool)
  RETURNS int4
  AS 'MODULE_PATHNAME', 'gpupreagg_partial_nrows'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.nrows(bool, bool)
  RETURNS int4
  AS 'MODULE_PATHNAME', 'gpupreagg_partial_nrows'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.nrows(bool, bool, bool)
  RETURNS int4
  AS 'MODULE_PATHNAME', 'gpupreagg_partial_nrows'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.nrows(bool, bool, bool, bool)
  RETURNS int4
  AS 'MODULE_PATHNAME', 'gpupreagg_partial_nrows'
  LANGUAGE C CALLED ON NULL INPUT;

--
-- Alternative aggregate function for count(int4)
--
CREATE AGGREGATE pgstrom.count(int4)
(
  sfunc = pg_catalog.int4_sum,
  stype = int8,
  initcond = 0
);

-- Definition of Partial MAX
CREATE FUNCTION pgstrom.pmax(int2)
  RETURNS int2
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(int4)
  RETURNS int4
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(float4)
  RETURNS float4
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(numeric)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(money)
  RETURNS money
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(date)
  RETURNS date
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(time)
  RETURNS time
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(timestamp)
  RETURNS timestamp
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmax(timestamptz)
  RETURNS timestamptz
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;

-- Definition of Partial MIN
CREATE FUNCTION pgstrom.pmin(int2)
  RETURNS int2
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(int4)
  RETURNS int4
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(float4)
  RETURNS float4
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(numeric)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(money)
  RETURNS money
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(date)
  RETURNS date
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(time)
  RETURNS time
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(timestamp)
  RETURNS timestamp
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.pmin(timestamptz)
  RETURNS timestamptz
  AS 'MODULE_PATHNAME', 'gpupreagg_pseudo_expr'
  LANGUAGE C STRICT;

-- Definition of Partial SUM
CREATE FUNCTION pgstrom.psum(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'gpupreagg_psum_int'
  LANGUAGE C CALLED ON NULL INPUT;
CREATE FUNCTION pgstrom.psum(float4)
  RETURNS float4
  AS 'MODULE_PATHNAME', 'gpupreagg_psum_float4'
  LANGUAGE C CALLED ON NULL INPUT;
CREATE FUNCTION pgstrom.psum(float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_psum_float8'
  LANGUAGE C CALLED ON NULL INPUT;
CREATE FUNCTION pgstrom.psum_x2(float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_psum_x2_float'
  LANGUAGE C CALLED ON NULL INPUT;
CREATE FUNCTION pgstrom.psum(numeric)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'gpupreagg_psum_numeric'
  LANGUAGE C CALLED ON NULL INPUT;
CREATE FUNCTION pgstrom.psum_x2(numeric)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'gpupreagg_psum_x2_numeric'
  LANGUAGE C CALLED ON NULL INPUT;
CREATE FUNCTION pgstrom.psum(money)
  RETURNS money
  AS 'MODULE_PATHNAME', 'gpupreagg_psum_money'
  LANGUAGE C CALLED ON NULL INPUT;

  
-- Definition of Partial SUM for covariance/least square method (only float8)
CREATE FUNCTION pgstrom.pcov_x(bool, float8, float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_corr_psum_x'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.pcov_y(bool, float8, float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_corr_psum_y'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.pcov_x2(bool, float8, float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_corr_psum_x2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.pcov_y2(bool, float8, float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_corr_psum_y2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.pcov_xy(bool, float8, float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'gpupreagg_corr_psum_xy'
  LANGUAGE C CALLED ON NULL INPUT;


--
-- Partial aggregate function for int2/int4 data types
--
CREATE FUNCTION pgstrom.avg_int8_accum(int8[], int4, int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME', 'pgstrom_avg_int8_accum'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.avg(int4, int8)
(
  sfunc = pgstrom.avg_int8_accum,
  stype = int8[],
  finalfunc = pg_catalog.int8_avg,
  initcond = '{0,0}'
);

CREATE FUNCTION pgstrom.sum_int8_accum(int8, int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_sum_int8_accum'
  LANGUAGE C CALLED ON NULL INPUT;;

CREATE AGGREGATE pgstrom.sum(int8)
(
  sfunc = pgstrom.sum_int8_accum,
  stype = int8
);

--
-- Partial aggregates for int8 / numeric data type
--
CREATE FUNCTION pgstrom.int8_avg_accum(internal, int4, int8)
  RETURNS internal
  AS 'MODULE_PATHNAME', 'pgstrom_int8_avg_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.numeric_avg_accum(internal, int4, numeric)
  RETURNS internal
  AS 'MODULE_PATHNAME', 'pgstrom_numeric_avg_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.numeric_avg_final(internal)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_numeric_avg_final'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.avg_int8(int4, int8)
(
  sfunc = pgstrom.int8_avg_accum,
  stype = internal,
  finalfunc = pgstrom.numeric_avg_final
);

CREATE AGGREGATE pgstrom.avg_numeric(int4, numeric)
(
  sfunc = pgstrom.numeric_avg_accum,
  stype = internal,
  finalfunc = pgstrom.numeric_avg_final
);

--
-- Partial aggregates for real/float data type
--
CREATE FUNCTION pgstrom.sum_float8_accum(float8[], int4, float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_sum_float8_accum'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.avg(int4, float8)
(
  sfunc = pgstrom.sum_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_avg,
  initcond = "{0,0,0}"
);

--
-- PreAgg functions for standard diviation / variance
--
CREATE FUNCTION pgstrom.variance_float8_accum(float8[], int4, float8, float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_variance_float8_accum'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.stddev(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_stddev_samp,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.stddev_samp(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_stddev_samp,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.stddev_pop(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_stddev_pop,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.variance(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_var_samp,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.var_samp(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_var_samp,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.var_pop(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_var_pop,
  initcond = '{0,0,0}'
);

CREATE FUNCTION pgstrom.numeric_var_accum(internal, int4, numeric, numeric)
  RETURNS internal
  AS 'MODULE_PATHNAME', 'pgstrom_numeric_var_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.numeric_var_samp(internal)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_numeric_var_samp'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.numeric_stddev_samp(internal)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_numeric_stddev_samp'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.numeric_var_pop(internal)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_numeric_var_pop'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.numeric_stddev_pop(internal)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_numeric_stddev_pop'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.stddev(int4, numeric, numeric)
(
  sfunc = pgstrom.numeric_var_accum,
  stype = internal,
  finalfunc = pgstrom.numeric_stddev_samp
);

CREATE AGGREGATE pgstrom.stddev_samp(int4, numeric, numeric)
(
  sfunc = pgstrom.numeric_var_accum,
  stype = internal,
  finalfunc = pgstrom.numeric_stddev_samp
);

CREATE AGGREGATE pgstrom.stddev_pop(int4, numeric, numeric)
(
  sfunc = pgstrom.numeric_var_accum,
  stype = internal,
  finalfunc = pgstrom.numeric_stddev_pop
);

CREATE AGGREGATE pgstrom.variance(int4, numeric, numeric)
(
  sfunc = pgstrom.numeric_var_accum,
  stype = internal,
  finalfunc = pgstrom.numeric_var_samp
);

CREATE AGGREGATE pgstrom.var_samp(int4, numeric, numeric)
(
  sfunc = pgstrom.numeric_var_accum,
  stype = internal,
  finalfunc = pgstrom.numeric_var_samp
);

CREATE AGGREGATE pgstrom.var_pop(int4, numeric, numeric)
(
  sfunc = pgstrom.numeric_var_accum,
  stype = internal,
  finalfunc = pgstrom.numeric_var_pop
);

--
-- PreAgg functions for covariance/least square method (with float8)
--
CREATE FUNCTION pgstrom.covariance_float8_accum(float8[], int4, float8, float8,
                                                float8, float8, float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_covariance_float8_accum'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.corr(int4, float8, float8,
                              float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_corr,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.covar_pop(int4, float8, float8,
                                   float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_covar_pop,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.covar_samp(int4, float8, float8,
                                    float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_covar_samp,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.regr_avgx(int4, float8, float8,
                                   float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_regr_avgx,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.regr_avgy(int4, float8, float8,
                                   float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_regr_avgy,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.regr_count(int4)
(
  sfunc = pg_catalog.int84pl,
  stype = int8,
  initcond = '0'
);

CREATE AGGREGATE pgstrom.regr_intercept(int4, float8, float8,
                                        float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_regr_intercept,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.regr_r2(int4, float8, float8,
                                 float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_regr_r2,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.regr_slope(int4, float8, float8,
                                    float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_regr_slope,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.regr_sxx(int4, float8, float8,
                                  float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_regr_sxx,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.regr_sxy(int4, float8, float8,
                                  float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_regr_sxy,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.regr_syy(int4, float8, float8,
                                  float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_regr_syy,
  initcond = '{0,0,0,0,0,0}'
);

--
-- Functions/Languages to support PL/CUDA
--
CREATE FUNCTION pgstrom.plcuda_function_validator(oid)
  RETURNS void
  AS 'MODULE_PATHNAME','plcuda_function_validator'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.plcuda_function_handler()
  RETURNS language_handler
  AS 'MODULE_PATHNAME','plcuda_function_handler'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.plcuda_function_source(regproc)
  RETURNS text
  AS 'MODULE_PATHNAME','plcuda_function_source'
  LANGUAGE C STRICT;

CREATE LANGUAGE plcuda
  HANDLER pgstrom.plcuda_function_handler
  VALIDATOR pgstrom.plcuda_function_validator;
COMMENT ON LANGUAGE plcuda IS 'PL/CUDA procedural language';

--
-- Matrix like 2D-Array type support
--
CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic int2[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic int4[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic int8[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic real[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic float[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

-- varbit as matrix of int4[]
CREATE FUNCTION pgstrom.array_matrix_accum_varbit(internal, bit)
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum_varbit'
  LANGUAGE C CALLED ON NULL INPUT;

-- type case varbit <--> int4[]
CREATE FUNCTION pgstrom.varbit_to_int4_array(bit)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','varbit_to_int4_array'
  LANGUAGE C STRICT;

CREATE CAST (bit AS int4[])
  WITH FUNCTION pgstrom.varbit_to_int4_array(bit)
  AS ASSIGNMENT;

CREATE FUNCTION pgstrom.int4_array_to_varbit(int4[])
  RETURNS bit
  AS 'MODULE_PATHNAME','int4_array_to_varbit'
  LANGUAGE C STRICT;

CREATE CAST (int4[] AS bit)
  WITH FUNCTION pgstrom.int4_array_to_varbit(int4[])
  AS ASSIGNMENT;

CREATE FUNCTION pgstrom.array_matrix_final_int2(internal)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_final_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_int4(internal)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_final_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_int8(internal)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_final_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_float4(internal)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_final_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_float8(internal)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_final_float8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE AGGREGATE pg_catalog.array_matrix(variadic int2[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_int2
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic int4[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_int4
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic int8[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_int8
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic float4[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_float4
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic float8[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_float8
);

CREATE AGGREGATE pg_catalog.array_matrix(bit)
(
  sfunc = pgstrom.array_matrix_accum_varbit,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_int4
);

CREATE FUNCTION pg_catalog.array_matrix_validation(anyarray)
  RETURNS bool
  AS 'MODULE_PATHNAME','array_matrix_validation'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.array_matrix_height(anyarray)
  RETURNS int
  AS 'MODULE_PATHNAME','array_matrix_height'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.array_matrix_width(anyarray)
  RETURNS int
  AS 'MODULE_PATHNAME','array_matrix_width'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.array_matrix_rawsize(regtype,int,int)
  RETURNS bigint
  AS 'MODULE_PATHNAME','array_matrix_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.matrix_unnest(anyarray)
  RETURNS SETOF record
  AS 'MODULE_PATHNAME','array_matrix_unnest'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int2[], int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int2'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int4[], int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int8[], int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float4[], float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_float4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float8[], float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_float8'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int2, int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int2t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int2[], int2)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int2b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int4, int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int4t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int4[], int4)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int4b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int8, int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int8t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int8[], int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int8b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float4, float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_float4t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float4[], float4)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_float4b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float8, float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_float8t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float8[], float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_float8b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int2[], int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int2'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int4[], int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int8[], int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float4[], float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_float4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float8[], float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_float8'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int2, int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int2l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int2[], int2)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int2r'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int4, int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int4l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int4[], int4)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int4r'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int8, int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int8l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int8[], int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int8r'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float4, float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_float4l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float4[], float4)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_float4r'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float8, float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_float8l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float8[], float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_float8r'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.array_matrix_rbind_accum(internal, anyarray)
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_rbind_accum'
  LANGUAGE C CALLED ON NULL INPUT;;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_int2(internal)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_int4(internal)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_int8(internal)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_float4(internal)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_float8(internal)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_float8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE AGGREGATE pg_catalog.rbind(int2[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_int2
);

CREATE AGGREGATE pg_catalog.rbind(int4[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_int4
);

CREATE AGGREGATE pg_catalog.rbind(int8[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_int8
);

CREATE AGGREGATE pg_catalog.rbind(float4[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_float4
);

CREATE AGGREGATE pg_catalog.rbind(float8[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_float8
);


CREATE FUNCTION pgstrom.array_matrix_cbind_accum(internal, anyarray)
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_cbind_accum'
  LANGUAGE C CALLED ON NULL INPUT;;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_int2(internal)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_int4(internal)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_int8(internal)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_float4(internal)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_float8(internal)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_float8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE AGGREGATE pg_catalog.cbind(int2[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_int2
);

CREATE AGGREGATE pg_catalog.cbind(int4[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_int4
);

CREATE AGGREGATE pg_catalog.cbind(int8[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_int8
);

CREATE AGGREGATE pg_catalog.cbind(float4[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_float4
);

CREATE AGGREGATE pg_catalog.cbind(float8[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_float8
);

CREATE FUNCTION pg_catalog.transpose(int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_int2'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_int4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_float4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_float8'
  LANGUAGE C STRICT;

--
-- Type re-interpretation routines
--
CREATE FUNCTION pg_catalog.float4_as_int4(float4)
  RETURNS int4
  AS 'MODULE_PATHNAME','float4_as_int4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.int4_as_float4(int4)
  RETURNS float4
  AS 'MODULE_PATHNAME','int4_as_float4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.float8_as_int8(float8)
  RETURNS int8
  AS 'MODULE_PATHNAME','float8_as_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.int8_as_float8(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME','int8_as_float8'
  LANGUAGE C STRICT;
