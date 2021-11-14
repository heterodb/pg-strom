--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

-- ================================================================
--
-- PG-Strom System Functions, Views and others
--
-- ================================================================

--- Query GitHash of the binary module
CREATE FUNCTION pgstrom.githash()
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_githash'
  LANGUAGE C STRICT;

-- Query commercial license
CREATE FUNCTION pgstrom.license_query()
  RETURNS json
  AS 'MODULE_PATHNAME','pgstrom_license_query'
  LANGUAGE C STRICT;

-- System view for device information
CREATE TYPE pgstrom.__pgstrom_device_info AS (
  device_nr     int,
  aindex        int,
  attribute     text,
  value         text
);
CREATE FUNCTION pgstrom.pgstrom_device_info()
  RETURNS SETOF pgstrom.__pgstrom_device_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;
CREATE VIEW pgstrom.device_info AS
  SELECT * FROM pgstrom.pgstrom_device_info();

-- Create a shell type with particular type-oid
CREATE FUNCTION pgstrom.define_shell_type(name,oid,regnamespace='public')
  RETURNS oid
  AS 'MODULE_PATHNAME','pgstrom_define_shell_type'
  LANGUAGE C STRICT VOLATILE;

-- ================================================================
--
-- Arrow_Fdw functions
--
-- ================================================================
CREATE FUNCTION pgstrom.arrow_fdw_handler()
  RETURNS fdw_handler
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_handler'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.arrow_fdw_validator(text[],oid)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_validator'
  LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER arrow_fdw
  HANDLER   pgstrom.arrow_fdw_handler
  VALIDATOR pgstrom.arrow_fdw_validator;

CREATE SERVER arrow_fdw
  FOREIGN DATA WRAPPER arrow_fdw;

CREATE OR REPLACE FUNCTION pgstrom.arrow_fdw_precheck_schema()
  RETURNS event_trigger
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_precheck_schema'
  LANGUAGE C STRICT;

CREATE EVENT TRIGGER pgstrom_arrow_fdw_precheck_schema
    ON ddl_command_end
  WHEN tag IN ('CREATE FOREIGN TABLE',
               'ALTER FOREIGN TABLE')
EXECUTE PROCEDURE pgstrom.arrow_fdw_precheck_schema();

CREATE OR REPLACE FUNCTION pgstrom.arrow_fdw_truncate(regclass)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_truncate'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.arrow_fdw_import_file(text,	    -- relname
                                              text,	    -- filename
                                              text = null)  -- schema
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_import_file'
  LANGUAGE C;

-- deprecated at v3.1
CREATE OR REPLACE FUNCTION
pgstrom.arrow_fdw_export_cupy(regclass, text[] = null, int = null)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_export_cupy'
  LANGUAGE C CALLED ON NULL INPUT;

-- deprecated at v3.1
CREATE OR REPLACE FUNCTION
pgstrom.arrow_fdw_export_cupy_pinned(regclass, text[] = null, int = null)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_export_cupy_pinned'
  LANGUAGE C CALLED ON NULL INPUT;

-- deprecated at v3.1
CREATE OR REPLACE FUNCTION
pgstrom.arrow_fdw_unpin_gpu_buffer(text)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_unpin_gpu_buffer'
  LANGUAGE C STRICT;

-- deprecated at v3.1
CREATE OR REPLACE FUNCTION
pgstrom.arrow_fdw_put_gpu_buffer(text)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_put_gpu_buffer'
  LANGUAGE C STRICT;

-- ================================================================
--
-- GPU Cache Functions
--
-- ================================================================
CREATE FUNCTION pgstrom.gpucache_sync_trigger()
  RETURNS trigger
  AS 'MODULE_PATHNAME','pgstrom_gpucache_sync_trigger'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.gpucache_apply_redo(regclass)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gpucache_apply_redo'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.gpucache_compaction(regclass)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gpucache_compaction'
  LANGUAGE C STRICT;

CREATE TYPE pgstrom.__pgstrom_gpucache_info_t AS (
  database_oid			oid,
  database_name			text,
  table_oid				oid,
  table_name			text,
  signature				int8,
  refcnt				int4,
  corrupted				bool,
  gpu_main_sz			int8,
  gpu_extra_sz			int8,
  redo_write_ts			timestamptz,
  redo_write_nitems		int8,
  redo_write_pos		int8,
  redo_read_nitems		int8,
  redo_read_pos			int8,
  redo_sync_pos			int8,
  config_options		text
);
CREATE FUNCTION pgstrom.__pgstrom_gpucache_info()
  RETURNS SETOF pgstrom.__pgstrom_gpucache_info_t
  AS 'MODULE_PATHNAME','pgstrom_gpucache_info'
  LANGUAGE C STRICT;
CREATE VIEW pgstrom.gpucache_info AS
  SELECT * FROM pgstrom.__pgstrom_gpucache_info();

-- ==================================================================
--
-- Partial / Alternative aggregate functions for GpuPreAgg
--
-- ==================================================================

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

CREATE AGGREGATE pgstrom.fmin_int2(int4)
(
  sfunc = pg_catalog.int4smaller,
  stype = int4,
  finalfunc = pg_catalog.int2,
  parallel = safe
);

CREATE AGGREGATE pgstrom.fmax_int2(int4)
(
  sfunc = pg_catalog.int4larger,
  stype = int4,
  finalfunc = pg_catalog.int2,
  parallel = safe
);

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

CREATE FUNCTION pgstrom.pmin(date)
  RETURNS date
  AS 'MODULE_PATHNAME', 'pgstrom_partial_min_any'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pmax(date)
  RETURNS date
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
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_float8_stddev_samp_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8_stddev_pop_numeric(float8[])
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_float8_stddev_pop_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8_var_samp_numeric(float8[])
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_float8_var_samp_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8_var_pop_numeric(float8[])
  RETURNS numeric
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

-- ==================================================================
--
-- float2 - half-precision floating point data support
--
-- ==================================================================
SELECT pgstrom.define_shell_type('float2',421,'pg_catalog');
-- instead of CREATE TYPE pg_catalog.float2;

CREATE FUNCTION pgstrom.float2_in(cstring)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_in'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2_out(float2)
  RETURNS cstring
  AS 'MODULE_PATHNAME','pgstrom_float2_out'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2_recv(internal)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_recv'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2_send(float2)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_float2_send'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE TYPE pg_catalog.float2
(
  input =  pgstrom.float2_in,
  output = pgstrom.float2_out,
  receive = pgstrom.float2_recv,
  send = pgstrom.float2_send,
  like = pg_catalog.int2
);
--
-- float2 cast definitions
--
CREATE FUNCTION pgstrom.float4(float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_to_float4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float8(float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float2_to_float8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int2(float2)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_float2_to_int2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int4(float2)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_float2_to_int4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int8(float2)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_float2_to_int8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.numeric(float2)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_float2_to_numeric'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2(float4)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float4_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2(float8)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float8_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2(int2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int2_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2(int4)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int4_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2(int8)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int8_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2(numeric)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_numeric_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE CAST (float2 AS float4)
  WITH FUNCTION pgstrom.float4(float2)
  AS IMPLICIT;
CREATE CAST (float2 AS float8)
  WITH FUNCTION pgstrom.float8(float2)
  AS IMPLICIT;
CREATE CAST (float2 AS int2)
  WITH FUNCTION pgstrom.int2(float2)
  AS ASSIGNMENT;
CREATE CAST (float2 AS int4)
  WITH FUNCTION pgstrom.int4(float2)
  AS ASSIGNMENT;
CREATE CAST (float2 AS int8)
  WITH FUNCTION pgstrom.int8(float2)
  AS ASSIGNMENT;
CREATE CAST (float2 AS numeric)
  WITH FUNCTION pgstrom.numeric(float2)
  AS ASSIGNMENT;

CREATE CAST (float4 AS float2)
  WITH FUNCTION pgstrom.float2(float4)
  AS ASSIGNMENT;
CREATE CAST (float8 AS float2)
  WITH FUNCTION pgstrom.float2(float8)
  AS ASSIGNMENT;
CREATE CAST (int2 AS float2)
  WITH FUNCTION pgstrom.float2(int2)
  AS ASSIGNMENT;
CREATE CAST (int4 AS float2)
  WITH FUNCTION pgstrom.float2(int4)
  AS ASSIGNMENT;
CREATE CAST (int8 AS float2)
  WITH FUNCTION pgstrom.float2(int8)
  AS ASSIGNMENT;
CREATE CAST (numeric AS float2)
  WITH FUNCTION pgstrom.float2(numeric)
  AS ASSIGNMENT;
--
-- float2 comparison operators
--
CREATE FUNCTION pgstrom.float2_eq(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_ne(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_lt(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_le(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_gt(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_ge(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_cmp(float2,float2)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float2_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_larger(float2,float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_larger'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_smaller(float2,float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_smaller'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_hash(float2)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float2_hash'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_eq(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_ne(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_lt(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_le(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_gt(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_ge(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_cmp(float4,float2)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float42_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float82_eq(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_ne(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_lt(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_le(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_gt(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_ge(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_cmp(float8,float2)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float82_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float24_eq(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_ne(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_lt(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_le(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_gt(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_ge(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_cmp(float2,float4)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float24_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float28_eq(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_ne(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_lt(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_le(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_gt(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_ge(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_cmp(float2,float8)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float28_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float2_eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float2_ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float2_lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float2_le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float2_gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float2_ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float42_eq,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float42_ne,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float42_lt,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float42_le,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float42_gt,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float42_ge,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float82_eq,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float82_ne,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float82_lt,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float82_le,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float82_gt,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float82_ge,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <
);

CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float24_eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float24_ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float24_lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float24_le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float24_gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float24_ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <=, NEGATOR = <
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float28_eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float28_ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float28_lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float28_le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float28_gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float28_ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <=, NEGATOR = <
);

--
-- float2 unary operator
--
CREATE FUNCTION pgstrom.float2_up(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_up'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_um(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_um'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.abs(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_abs'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float2_up,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float2_um,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.@ (
  PROCEDURE = pg_catalog.abs,
  RIGHTARG = pg_catalog.float2
);

--
-- float2 arithmetic operators
--
CREATE FUNCTION pgstrom.float2_pl(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_mi(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_mul(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_div(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_pl(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_mi(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_mul(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_div(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_pl(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_mi(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_mul(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_div(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_pl(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_mi(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_mul(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_div(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_pl(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_mi(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_mul(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_div(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float2_pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float2_mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float2_mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float2_div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float24_pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float24_mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float24_mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float24_div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float28_pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float28_mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float28_mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float28_div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float42_pl,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float42_mi,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float42_mul,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float42_div,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float82_pl,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float82_mi,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float82_mul,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float82_div,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);

--
-- float2 misc operators
--
CREATE FUNCTION pgstrom.cash_mul_flt2(money,float2)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_cash_mul_flt2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.flt2_mul_cash(float2,money)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_flt2_mul_cash'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.cash_div_flt2(money,float2)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_cash_div_flt2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.cash_mul_flt2,
  LEFTARG = pg_catalog.money,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.flt2_mul_cash,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.money
);

CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.cash_div_flt2,
  LEFTARG = pg_catalog.money,
  RIGHTARG = pg_catalog.float2
);

CREATE FUNCTION pg_catalog.as_int8(float8)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_float8_as_int8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_int4(float4)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_float4_as_int4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_int2(float2)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_float2_as_int2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_float8(int8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_int8_as_float8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_float4(int4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_int4_as_float4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_float2(int2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int2_as_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

--
-- float2 aggregate functions
--
CREATE FUNCTION pgstrom.float2_accum(float8[], float2)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','pgstrom_float2_accum'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_sum(float8, float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float2_sum'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE AGGREGATE pg_catalog.avg(float2) (
  sfunc = pgstrom.float2_accum,
  stype = float8[],
  finalfunc = float8_avg,
  combinefunc = float8_combine,
  initcond = "{0,0,0}"
);

CREATE AGGREGATE pg_catalog.sum(float2) (
  sfunc = pgstrom.float2_sum,
  stype = float8
);

CREATE AGGREGATE pg_catalog.max(float2) (
  sfunc = pgstrom.float2_larger,
  stype = float2
);

CREATE AGGREGATE pg_catalog.min(float2) (
  sfunc = pgstrom.float2_smaller,
  stype = float2
);

CREATE AGGREGATE pg_catalog.var_pop(float2) (
  sfunc = pgstrom.float2_accum,
  stype = float8[],
  finalfunc = float8_var_pop,
  combinefunc = float8_combine,
  initcond = "{0,0,0}"
);

CREATE AGGREGATE pg_catalog.var_samp(float2) (
  sfunc = pgstrom.float2_accum,
  stype = float8[],
  finalfunc = float8_var_samp,
  combinefunc = float8_combine,
  initcond = "{0,0,0}"
);

CREATE AGGREGATE pg_catalog.variance(float2) (
  sfunc = pgstrom.float2_accum,
  stype = float8[],
  finalfunc = float8_var_samp,
  combinefunc = float8_combine,
  initcond = "{0,0,0}"
);

CREATE AGGREGATE pg_catalog.stddev_pop(float2) (
  sfunc = pgstrom.float2_accum,
  stype = float8[],
  finalfunc = float8_stddev_pop,
  combinefunc = float8_combine,
  initcond = "{0,0,0}"
);

CREATE AGGREGATE pg_catalog.stddev_samp(float2) (
  sfunc = pgstrom.float2_accum,
  stype = float8[],
  finalfunc = float8_stddev_samp,
  combinefunc = float8_combine,
  initcond = "{0,0,0}"
);

CREATE AGGREGATE pg_catalog.stddev(float2) (
  sfunc = pgstrom.float2_accum,
  stype = float8[],
  finalfunc = float8_stddev_samp,
  combinefunc = float8_combine,
  initcond = "{0,0,0}"
);

--
-- float2 index support
--
CREATE OPERATOR CLASS pg_catalog.float2_ops
  default for type pg_catalog.float2
  using btree family pg_catalog.float_ops as
  operator 1 <  (float2, float2) for search,
  operator 2 <= (float2, float2) for search,
  operator 3 =  (float2, float2) for search,
  operator 4 >= (float2, float2) for search,
  operator 5 >  (float2, float2) for search,
  function 1 (float2, float2) pgstrom.float2_cmp(float2, float2);

CREATE OPERATOR CLASS pg_catalog.float2_ops
  default for type pg_catalog.float2
  using hash family pg_catalog.float_ops as
  function 1 (float2) pgstrom.float2_hash(float2);

-- ==================================================================
--
-- int1(tinyint) - 8bit width integer data support
--
-- ==================================================================
SELECT pgstrom.define_shell_type('int1',606,'pg_catalog');
--CREATE TYPE pg_catalog.int1;

CREATE FUNCTION pgstrom.int1in(cstring)
  RETURNS pg_catalog.int1
  AS 'MODULE_PATHNAME','pgstrom_int1in'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1out(pg_catalog.int1)
  RETURNS cstring
  AS 'MODULE_PATHNAME','pgstrom_int1out'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1recv(internal)
  RETURNS pg_catalog.int1
  AS 'MODULE_PATHNAME','pgstrom_int1recv'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1send(pg_catalog.int1)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_int1send'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE TYPE pg_catalog.int1
(
  input = pgstrom.int1in,
  output = pgstrom.int1out,
  receive = pgstrom.int1recv,
  send = pgstrom.int1send,
  like = pg_catalog.bool,
  category = 'N'
);

--
-- Type Cast Definitions
--
CREATE FUNCTION pgstrom.int2(int1)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_int1_to_int2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int4(int1)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_int1_to_int4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int8(int1)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_int1_to_int8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2(int1)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int1_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float4(int1)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_int1_to_float4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8(int1)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_int1_to_float8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.numeric(int1)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_int1_to_numeric'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1(int2)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int2_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1(int4)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int4_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1(int8)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int8_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1(float2)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_float2_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1(float4)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_float4_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1(float8)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_float8_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1(numeric)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_numeric_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE CAST (int1 AS int2)
  WITH FUNCTION pgstrom.int2(int1)
  AS IMPLICIT;
CREATE CAST (int1 AS int4)
  WITH FUNCTION pgstrom.int4(int1)
  AS IMPLICIT;
CREATE CAST (int1 AS int8)
  WITH FUNCTION pgstrom.int8(int1)
  AS IMPLICIT;
CREATE CAST (int1 AS float2)
  WITH FUNCTION pgstrom.float2(int1)
  AS IMPLICIT;
CREATE CAST (int1 AS float4)
  WITH FUNCTION pgstrom.float4(int1)
  AS IMPLICIT;
CREATE CAST (int1 AS float8)
  WITH FUNCTION pgstrom.float8(int1)
  AS IMPLICIT;
CREATE CAST (int1 AS numeric)
  WITH FUNCTION pgstrom.numeric(int1)
  AS IMPLICIT;

CREATE CAST (int2 AS int1)
  WITH FUNCTION pgstrom.int1(int2)
  AS ASSIGNMENT;
CREATE CAST (int4 AS int1)
  WITH FUNCTION pgstrom.int1(int4)
  AS ASSIGNMENT;
CREATE CAST (int8 AS int1)
  WITH FUNCTION pgstrom.int1(int8)
  AS ASSIGNMENT;
CREATE CAST (float2 AS int1)
  WITH FUNCTION pgstrom.int1(float2)
  AS ASSIGNMENT;
CREATE CAST (float4 AS int1)
  WITH FUNCTION pgstrom.int1(float4)
  AS ASSIGNMENT;
CREATE CAST (float8 AS int1)
  WITH FUNCTION pgstrom.int1(float8)
  AS ASSIGNMENT;
CREATE CAST (numeric AS int1)
  WITH FUNCTION pgstrom.int1(numeric)
  AS ASSIGNMENT;

---
--- tinyint comparison functions
---
CREATE FUNCTION pgstrom.int1eq(int1,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1ne(int1,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1lt(int1,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1le(int1,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1gt(int1,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1ge(int1,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.btint1cmp(int1,int1)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_btint1cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1larger(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1larger'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1smaller(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1smaller'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1hash(int1)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int1hash'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int12eq(int1,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12ne(int1,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12lt(int1,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12le(int1,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12gt(int1,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12ge(int1,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.btint12cmp(int1,smallint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_btint12cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int14eq(int1,int4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14ne(int1,int4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14lt(int1,int4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14le(int1,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14gt(int1,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14ge(int1,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.btint14cmp(int1,int)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_btint14cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int18eq(int1,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18ne(int1,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18lt(int1,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18le(int1,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18gt(int1,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18ge(int1,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.btint18cmp(int1,bigint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_btint18cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int21eq(smallint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21ne(smallint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21lt(smallint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21le(smallint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21gt(smallint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21ge(smallint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.btint21cmp(smallint,int1)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_btint21cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int41eq(int,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41ne(int,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41lt(int,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41le(int,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41gt(int,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41ge(int,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.btint41cmp(int,int1)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_btint41cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int81eq(bigint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81ne(bigint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81lt(bigint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81le(bigint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81gt(bigint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81ge(bigint,int1)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.btint81cmp(bigint,int1)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_btint81cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

-- <int1> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int1eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int1ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int1lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int1le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int1gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int1ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int1> OPER <smallint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int12eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int12ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int12lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int12le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int12gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int12ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int1> OPER <int>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int14eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int14ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int14lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int14le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int14gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int14ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int1> OPER <bigint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int18eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int18ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int18lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int18le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int18gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int18ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <smallint> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int21eq,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int21ne,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int21lt,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int21le,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int21gt,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int21ge,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int41eq,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int41ne,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int41lt,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int41le,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int41gt,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int41ge,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <
);

-- <bigint> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int81eq,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int81ne,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int81lt,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int81le,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int81gt,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int81ge,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <
);

--
-- tinyint unary operators
--
CREATE FUNCTION pgstrom.int1up(int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1up'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1um(int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1um'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1abs(int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1abs'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.abs(int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1abs'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int1up,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int1um,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.@ (
  PROCEDURE = pgstrom.int1abs,
  RIGHTARG = pg_catalog.int1
);

---
--- tinyint arithmetic operators
---
CREATE FUNCTION pgstrom.int1pl(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1mi(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1mul(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1div(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1mod(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1mod'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12pl(int1,smallint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int12pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int12mi(int1,smallint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int12mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int12mul(int1,smallint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int12mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int12div(int1,smallint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int12div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14pl(int1,integer)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int14pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int14mi(int1,integer)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int14mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int14mul(int1,integer)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int14mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int14div(int1,integer)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int14div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18pl(int1,bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int18pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int18mi(int1,bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int18mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int18mul(int1,bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int18mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int18div(int1,bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int18div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21pl(smallint,int1)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int21pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int21mi(smallint,int1)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int21mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int21mul(smallint,int1)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int21mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int21div(smallint,int1)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int21div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41pl(integer,int1)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int41pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int41mi(integer,int1)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int41mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int41mul(integer,int1)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int41mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int41div(integer,int1)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int41div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81pl(bigint,int1)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int81pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int81mi(bigint,int1)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int81mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int81mul(bigint,int1)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int81mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int81div(bigint,int1)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int81div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int1pl,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int1mi,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int1mul,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int1div,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.% (
  PROCEDURE = pgstrom.int1mod,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int12pl,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int12mi,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int12mul,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int12div,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int14pl,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int14mi,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int14mul,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int14div,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int18pl,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int18mi,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int18mul,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int18div,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int21pl,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int21mi,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int21mul,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int21div,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int41pl,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int41mi,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int41mul,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int41div,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int81pl,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int81mi,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int81mul,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int81div,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1
);

---
--- tinyint bit operations
---
CREATE FUNCTION pgstrom.int1and(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1and'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1or(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1or'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1xor(int1,int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1xor'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1not(int1)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1not'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1shl(int1,integer)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1shl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1shr(int1,integer)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_int1shr'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.& (
  PROCEDURE = pgstrom.int1and,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.| (
  PROCEDURE = pgstrom.int1or,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.# (
  PROCEDURE = pgstrom.int1xor,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.~ (
  PROCEDURE = pgstrom.int1not,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.<< (
  PROCEDURE = pgstrom.int1shl,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer
);
CREATE OPERATOR pg_catalog.>> (
  PROCEDURE = pgstrom.int1shr,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer
);

---
--- Misc functions
---
CREATE FUNCTION pgstrom.cash_mul_int1(money,int1)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_cash_mul_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1_mul_cash(int1,money)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_int1_mul_cash'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.cash_div_int1(money,int1)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_cash_div_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.cash_mul_int1,
  LEFTARG = pg_catalog.money,
  RIGHTARG = pg_catalog.int1
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int1_mul_cash,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.money
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.cash_div_int1,
  LEFTARG = pg_catalog.money,
  RIGHTARG = pg_catalog.int1
);

---
--- tinyint aggregate functions
---
CREATE FUNCTION pgstrom.int1_sum(bigint, pg_catalog.int1)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int1_sum'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_avg_accum(bigint[], pg_catalog.int1)
  RETURNS bigint[]
  AS 'MODULE_PATHNAME','pgstrom_int1_avg_accum'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_avg_accum_inv(bigint[], pg_catalog.int1)
  RETURNS bigint[]
  AS 'MODULE_PATHNAME','pgstrom_int1_avg_accum_inv'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_var_accum(internal, pg_catalog.int1)
  RETURNS internal
  AS 'MODULE_PATHNAME','pgstrom_int1_var_accum'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_var_accum_inv(internal, pg_catalog.int1)
  RETURNS internal
  AS 'MODULE_PATHNAME','pgstrom_int1_var_accum_inv'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE AGGREGATE pg_catalog.sum(int1)
(
  sfunc = pgstrom.int1_sum,
  stype = bigint
);

CREATE AGGREGATE pg_catalog.max(int1)
(
 sfunc = pgstrom.int1larger,
 stype = int1
);

CREATE AGGREGATE pg_catalog.min(int1)
(
 sfunc = pgstrom.int1smaller,
 stype = int1
);

CREATE AGGREGATE pg_catalog.avg(int1)
(
  sfunc = pgstrom.int1_avg_accum,
  stype = bigint[],
  finalfunc = int8_avg,
  initcond = "{0,0}",
  combinefunc = int4_avg_combine,
  msfunc = pgstrom.int1_avg_accum,
  minvfunc = pgstrom.int1_avg_accum_inv,
  mfinalfunc = int8_avg,
  mstype = bigint[],
  minitcond = "{0,0}",
  parallel = safe
);

CREATE AGGREGATE pg_catalog.variance(int1)
(
  sfunc = pgstrom.int1_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_var_samp,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.int1_var_accum,
  minvfunc = pgstrom.int1_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_var_samp,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.var_samp(int1)
(
  sfunc = pgstrom.int1_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_var_samp,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.int1_var_accum,
  minvfunc = pgstrom.int1_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_var_samp,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.var_pop(int1)
(
  sfunc = pgstrom.int1_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_var_pop,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.int1_var_accum,
  minvfunc = pgstrom.int1_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_var_pop,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.stddev(int1)
(
  sfunc = pgstrom.int1_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_stddev_samp,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.int1_var_accum,
  minvfunc = pgstrom.int1_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_stddev_samp,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.stddev_samp(int1)
(
  sfunc = pgstrom.int1_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_stddev_samp,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.int1_var_accum,
  minvfunc = pgstrom.int1_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_stddev_samp,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.stddev_pop(int1)
(
  sfunc = pgstrom.int1_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_stddev_pop,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.int1_var_accum,
  minvfunc = pgstrom.int1_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_stddev_pop,
  parallel = safe
);

---
--- tinyint index support
---
CREATE OPERATOR CLASS pg_catalog.int1_ops
  default for type pg_catalog.int1
  using btree family pg_catalog.integer_ops as
  operator 1 <  (int1,int1) for search,
  operator 2 <= (int1,int1) for search,
  operator 3 =  (int1,int1) for search,
  operator 4 >= (int1,int1) for search,
  operator 5 >  (int1,int1) for search,
  function 1 (int1,int1) pgstrom.btint1cmp(int1,int1);

CREATE OPERATOR CLASS pg_catalog.int1_ops
  default for type pg_catalog.int1
  using hash family pg_catalog.integer_ops as
  function 1 (int1) pgstrom.int1hash(int1);

-- ==================================================================
--
-- PG-Strom regression test support functions
--
-- ==================================================================

-- dummy regression test revision
-- it is very old timestamp; shall not be matched
-- without valid configuration.
CREATE OR REPLACE FUNCTION
pgstrom.regression_testdb_revision()
RETURNS int
AS 'SELECT 0'
LANGUAGE 'sql';


CREATE OR REPLACE FUNCTION pgstrom.random_setseed(int)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_random_setseed'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.random_int(float=0.0,      -- NULL ratio (%)
                                   bigint=null,    -- lower bound
                                   bigint=null)    -- upper bound
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_random_int'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_float(float=0.0,
                                     float=null,
                                     float=null)
  RETURNS float
  AS 'MODULE_PATHNAME','pgstrom_random_float'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_date(float=0.0,
                                    date=null,
                                    date=null)
  RETURNS date
  AS 'MODULE_PATHNAME','pgstrom_random_date'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_time(float=0.0,
                                    time=null,
                                    time=null)
  RETURNS time
  AS 'MODULE_PATHNAME','pgstrom_random_time'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_timetz(float=0.0,
                                      time=null,
                                      time=null)
  RETURNS timetz
  AS 'MODULE_PATHNAME','pgstrom_random_timetz'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_timestamp(float=0.0,
                                         timestamp=null,
                                         timestamp=null)
  RETURNS timestamp
  AS 'MODULE_PATHNAME','pgstrom_random_timestamp'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_macaddr(float=0.0,
                                       macaddr=null,
                                       macaddr=null)
  RETURNS macaddr
  AS 'MODULE_PATHNAME','pgstrom_random_macaddr'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_inet(float=0.0,
                                    inet=null)
  RETURNS inet
  AS 'MODULE_PATHNAME','pgstrom_random_inet'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_text(float=0.0,
                                    text=null)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_random_text'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_text_len(float=0.0,
                                        int=null)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_random_text_length'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_int4range(float=0.0,
                                         int=null,
                                         int=null)
  RETURNS int4range
  AS 'MODULE_PATHNAME','pgstrom_random_int4range'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_int8range(float=0.0,
                                         bigint=null,
                                         bigint=null)
  RETURNS int8range
  AS 'MODULE_PATHNAME','pgstrom_random_int8range'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_tsrange(float=0.0,
                                       timestamp=null,
                                       timestamp=null)
  RETURNS tsrange
  AS 'MODULE_PATHNAME','pgstrom_random_tsrange'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_tstzrange(float=0.0,
                                         timestamptz=null,
                                         timestamptz=null)
  RETURNS tstzrange
  AS 'MODULE_PATHNAME','pgstrom_random_tstzrange'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.random_daterange(float=0.0,
                                         date=null,
                                         date=null)
  RETURNS daterange
  AS 'MODULE_PATHNAME','pgstrom_random_daterange'
  LANGUAGE C CALLED ON NULL INPUT;
