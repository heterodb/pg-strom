--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

--
-- Support routines for ctid reference with gpu projection
--
CREATE FUNCTION pgstrom.cast_tid_to_int8(tid)
  RETURNS bigint
  AS 'MODULE_PATHNAME', 'pgstrom_cast_tid_to_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.cast_int8_to_tid(bigint)
  RETURNS tid
  AS 'MODULE_PATHNAME', 'pgstrom_cast_int8_to_tid'
  LANGUAGE C STRICT;

CREATE CAST (tid AS bigint)
  WITH FUNCTION pgstrom.cast_tid_to_int8(tid)
  AS IMPLICIT;

CREATE CAST (bigint AS tid)
  WITH FUNCTION pgstrom.cast_int8_to_tid(bigint)
  AS IMPLICIT;

--
-- pg_strom installation queries
--
CREATE TYPE pgstrom.__pgstrom_dma_buffer_info AS (
  seg_id    int4,
  revision  int4,
  mclass    int4,
  actives   int4,
  frees     int4
);
CREATE FUNCTION pg_catalog.pgstrom_dma_buffer_info()
  RETURNS SETOF pgstrom.__pgstrom_dma_buffer_info
  AS 'MODULE_PATHNAME','pgstrom_dma_buffer_info'
  LANGUAGE C STRICT;

-- for debug
CREATE FUNCTION pg_catalog.pgstrom_dma_buffer_alloc(bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_dma_buffer_alloc'
  LANGUAGE C STRICT;
-- for debug
CREATE FUNCTION pg_catalog.pgstrom_dma_buffer_free(bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_dma_buffer_free'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_device_info AS (
  id		int4,
  property	text,
  value		text
);
CREATE FUNCTION pgstrom_device_info()
  RETURNS SETOF __pgstrom_device_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

--CREATE TYPE __pgstrom_program_info AS (
--  addr			int8,
--  length		int8,
--  active		bool,
--  status		text,
--  crc32			int4,
--  flags			int4,
--  kern_define   text,
--  kern_source	text,
--  kern_binary	bytea,
--  error_msg		text,
--  backends		text
--);
--CREATE FUNCTION pgstrom_program_info()
--  RETURNS SETOF __pgstrom_program_info
--  AS 'MODULE_PATHNAME'
--  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_iomap_buffer_info AS (
  gpuid			int,
  paddr			int8,
  length		int8,
  state			text
);

CREATE FUNCTION pgstrom_iomap_buffer_info()
  RETURNS SETOF __pgstrom_iomap_buffer_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

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

CREATE FUNCTION pgstrom.pavg(int8,numeric)
  RETURNS numeric[]
  AS 'MODULE_PATHNAME','pgstrom_partial_avg_numeric'
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

CREATE FUNCTION pgstrom.favg_accum(numeric[], numeric[])
  RETURNS numeric[]
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_numeric_accum'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_final(numeric[])
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

CREATE AGGREGATE pgstrom.favg(numeric[])
(
  sfunc = pgstrom.favg_accum,
  stype = numeric[],
  finalfunc = pgstrom.favg_final,
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

CREATE FUNCTION pgstrom.pmin(numeric)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(numeric)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_partial_max_any'
  LANGUAGE C STRICT PARALLEL SAFE;

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

CREATE FUNCTION pgstrom.psum(numeric)
  RETURNS numeric
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

CREATE FUNCTION pgstrom.psum_x2(numeric)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_partial_sum_x2_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

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

CREATE AGGREGATE pgstrom.stddev(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  finalfunc = float8_stddev_samp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_pop(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_stddev_pop,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_samp(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_stddev_samp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.variance(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_var_samp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_pop(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_var_pop,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_samp(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_var_samp,
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
CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic bool[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

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

-- final functions of array_matrix
CREATE FUNCTION pgstrom.array_matrix_final_bool(internal)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_final_bool'
  LANGUAGE C CALLED ON NULL INPUT;

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

CREATE AGGREGATE pg_catalog.array_matrix(variadic bool[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_bool
);

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

CREATE FUNCTION pg_catalog.type_len(regtype)
  RETURNS bigint
  AS 'MODULE_PATHNAME','postgresql_type_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.composite_type_rawsize(VARIADIC int[])
  RETURNS bigint
  AS 'MODULE_PATHNAME','composite_type_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.composite_type_rawsize(VARIADIC bigint[])
  RETURNS bigint
  AS 'MODULE_PATHNAME','composite_type_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.matrix_unnest(anyarray)
  RETURNS SETOF record
  AS 'MODULE_PATHNAME','array_matrix_unnest'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(bool[], bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_bool'
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

CREATE FUNCTION pg_catalog.rbind(bool, bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_boolt'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(bool[], bool)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_boolb'
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

CREATE FUNCTION pg_catalog.cbind(bool[], bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_bool'
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

CREATE FUNCTION pg_catalog.cbind(bool, bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_booll'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(bool[], bool)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_boolr'
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

CREATE FUNCTION pgstrom.array_matrix_rbind_final_bool(internal)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_bool'
  LANGUAGE C CALLED ON NULL INPUT;

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

CREATE AGGREGATE pg_catalog.rbind(bool[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_bool
);

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

CREATE FUNCTION pgstrom.array_matrix_cbind_final_bool(internal)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_bool'
  LANGUAGE C CALLED ON NULL INPUT;

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

CREATE AGGREGATE pg_catalog.cbind(bool[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_bool
);

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

CREATE FUNCTION pg_catalog.transpose(bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_bool'
  LANGUAGE C STRICT;

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
