--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

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
  RETURN bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.nrows(bool)
  RETURN bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.nrows(bool,bool)
  RETURN bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.nrows(bool,bool,bool)
  RETURN bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.nrows(bool,bool,bool,bool)
  RETURN bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT;

-- AVG()
CREATE FUNCTION pgstrom.pavg(int8,int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','pgstrom_partial_avg_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.pavg(int8,float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','pgstrom_partial_avg_float8'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.pavg(int8,numeric)
  RETURNS numeric[]
  AS 'MODULE_PATHNAME','pgstrom_partial_avg_numeric'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.favg_accum(int8[], int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_int8_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.favg(int8[])
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.favg_accum(float8[], float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_float8_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.favg(float8[])
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_float8'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.favg_accum(numeric[], numeric[])
  RETURNS numeric[]
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_numeric_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.favg(numeric[])
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_final_avg_numeric'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.favg(int8[])
(
  sfunc = pgstrom.favg_,
  stype = int8[],
  finalfunc = pgstrom.favg_final
);

CREATE AGGREGATE pgstrom.favg(float8[])
(
  sfunc = pgstrom.favg_,
  stype = float8[],
  finalfunc = pgstrom.
);

CREATE AGGREGATE pgstrom.favg(numeric[])
(
  sfunc = pgstrom.favg,
  stype = numeric[],
  finalfunc = pgstrom.
);

-- MIN()/MAX()
CREATE FUNCTION pgstrom.pmin(anyelement)
  RETURNS anyelement
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.pmax(anyelement)
  RETURNS anyelement
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min'
  LANGUAGE C STRICT;

-- SUM()


CREATE AGGREGATE pgstrom.sum(int8)
(
  sfunc = pg_catalog.int8pl,
  stype = int8,
  initcond = 0
);

-- STDDEV/STDDEV_POP/STDDEV_SAMP
-- VARIANCE/VAR_POP/VAR_SAM
CREATE FUNCTION pgstrom.pvariance(int8,float8,float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_partial_variance_float8'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.stddev(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  finalfunc = float8_stddev_samp
);

CREATE AGGREGATE pgstrom.stddev_pop(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_stddev_pop
);

CREATE AGGREGATE pgstrom.stddev_samp(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_stddev_samp
);

CREATE AGGREGATE pgstrom.variance(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_var_samp
);

CREATE AGGREGATE pgstrom.var_pop(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_var_pop
);

CREATE AGGREGATE pgstrom.var_samp(float8[])
(
  sfunc = pg_catalog.float8_combine,
  stype = float8[],
  initcond = "{0,0,0}",
  finalfunc = float8_var_samp
);

-- CORR/COVAR_POP/COVAR_SAMP
-- REGR_*
CREATE FUNCTION pgstrom.pcovar(int8,float8,float8,float8,float8,float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_partial_covar_float8'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.corr(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_corr
);

CREATE AGGREGATE pgstrom.covar_pop(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_covar_pop
);

CREATE AGGREGATE pgstrom.covar_samp(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_covar_samp
);

CREATE AGGREGATE pgstrom.regr_avgx(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_avgx
);

CREATE AGGREGATE pgstrom.regr_avgy(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_avgy
);

CREATE AGGREGATE pgstrom.regr_intercept(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_intercept
);

CREATE AGGREGATE pgstrom.regr_r2(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_r2
);

CREATE AGGREGATE pgstrom.regr_slope(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_slope
);

CREATE AGGREGATE pgstrom.regr_sxx(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_sxx
);

CREATE AGGREGATE pgstrom.regr_sxy(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_sxy
);

CREATE AGGREGATE pgstrom.regr_syy(float8[])
(
  sfunc = pg_catalog.float8_regr_combine,
  stype = float8[],
  initcond = "{0,0,0,0,0,0}",
  finalfunc = pg_catalog.float8_regr_syy
);








--
-- Functions/Languages to support PL/CUDA
--
--CREATE FUNCTION pgstrom.plcuda_function_validator(oid)
--  RETURNS void
--  AS 'MODULE_PATHNAME','plcuda_function_validator'
--  LANGUAGE C STRICT;

--CREATE FUNCTION pgstrom.plcuda_function_handler()
--  RETURNS language_handler
--  AS 'MODULE_PATHNAME','plcuda_function_handler'
--  LANGUAGE C STRICT;

--CREATE FUNCTION pgstrom.plcuda_function_source(regproc)
--  RETURNS text
--  AS 'MODULE_PATHNAME','plcuda_function_source'
--  LANGUAGE C STRICT;

--CREATE LANGUAGE plcuda
--  HANDLER pgstrom.plcuda_function_handler
--  VALIDATOR pgstrom.plcuda_function_validator;
--COMMENT ON LANGUAGE plcuda IS 'PL/CUDA procedural language';

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
