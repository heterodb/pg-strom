--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

--
-- Functions for device properties
--
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

CREATE FUNCTION public.gpu_device_name(int = 0)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_gpu_device_name'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_global_memsize(int = 0)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gpu_global_memsize'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_max_blocksize(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_max_blocksize'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_warp_size(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_warp_size'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_max_shared_memory_perblock(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_max_shared_memory_perblock'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_num_registers_perblock(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_num_registers_perblock'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_num_multiptocessors(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_num_multiptocessors'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_num_cuda_cores(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_num_cuda_cores'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_cc_major(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_cc_major'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_cc_minor(int = 0)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gpu_cc_minor'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gpu_pci_id(int = 0)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_gpu_pci_id'
  LANGUAGE C STRICT;

--
-- Functions for system internal state
--
CREATE TYPE pgstrom.__pgstrom_device_preserved_meminfo AS (
  device_nr int4,
  handle    bytea,
  owner     regrole,
  length    int8,
  ctime     timestamp with time zone
);
CREATE FUNCTION pgstrom.pgstrom_device_preserved_meminfo()
  RETURNS SETOF pgstrom.__pgstrom_device_preserved_meminfo
  AS 'MODULE_PATHNAME'
  LANGUAGE C VOLATILE;
CREATE VIEW pgstrom.device_preserved_meminfo
  AS SELECT * FROM pgstrom.pgstrom_device_preserved_meminfo();

/*
--
-- Functions/Languages to support PL/CUDA
--
-- FEATURES DEPRECATED AT V2.3
--
CREATE FUNCTION pgstrom.plcuda_function_validator(oid)
  RETURNS void
  AS 'MODULE_PATHNAME','plcuda_function_validator'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.plcuda_function_handler()
  RETURNS language_handler
  AS 'MODULE_PATHNAME','plcuda_function_handler'
  LANGUAGE C STRICT;

CREATE LANGUAGE plcuda
  HANDLER pgstrom.plcuda_function_handler
  VALIDATOR pgstrom.plcuda_function_validator;
COMMENT ON LANGUAGE plcuda IS 'PL/CUDA procedural language';

CREATE FUNCTION pg_catalog.attnums_of(regclass,text[])
  RETURNS smallint[]
  AS 'MODULE_PATHNAME','pgsql_table_attr_numbers_by_names'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.attnum_of(regclass,text)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgsql_table_attr_number_by_name'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.atttypes_of(regclass,text[])
  RETURNS regtype[]
  AS 'MODULE_PATHNAME','pgsql_table_attr_types_by_names'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.atttype_of(regclass,text)
  RETURNS regtype
  AS 'MODULE_PATHNAME','pgsql_table_attr_type_by_name'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.attrs_types_check(regclass,text[],regtype[])
  RETURNS bool
  AS 'MODULE_PATHNAME','pgsql_check_attrs_of_types'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.attrs_type_check(regclass,text[],regtype)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgsql_check_attrs_of_type'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.attr_type_check(regclass,text,regtype)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgsql_check_attr_of_type'
  LANGUAGE C STRICT;
*/

--
-- Handlers for arrow_fdw extension
--
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

/*
--
-- Handlers for gstore_fdw extension
--
-- FEATURES DEPRECATED AT v2.3
--
CREATE FUNCTION pgstrom.gstore_fdw_handler()
  RETURNS fdw_handler
  AS  'MODULE_PATHNAME','pgstrom_gstore_fdw_handler'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.gstore_fdw_validator(text[],oid)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_validator'
  LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER gstore_fdw
  HANDLER   pgstrom.gstore_fdw_handler
  VALIDATOR pgstrom.gstore_fdw_validator;

CREATE SERVER gstore_fdw
  FOREIGN DATA WRAPPER gstore_fdw;

CREATE TYPE public.reggstore;
CREATE FUNCTION pgstrom.reggstore_in(cstring)
  RETURNS reggstore
  AS 'MODULE_PATHNAME','pgstrom_reggstore_in'
  LANGUAGE C STRICT IMMUTABLE;
CREATE FUNCTION pgstrom.reggstore_out(reggstore)
  RETURNS cstring
  AS 'MODULE_PATHNAME','pgstrom_reggstore_out'
  LANGUAGE C STRICT IMMUTABLE;
CREATE FUNCTION pgstrom.reggstore_recv(internal)
  RETURNS reggstore
  AS 'MODULE_PATHNAME','pgstrom_reggstore_recv'
  LANGUAGE C STRICT IMMUTABLE;
CREATE FUNCTION pgstrom.reggstore_send(reggstore)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_reggstore_send'
  LANGUAGE C STRICT IMMUTABLE;
CREATE TYPE public.reggstore
(
  input = pgstrom.reggstore_in,
  output = pgstrom.reggstore_out,
  receive = pgstrom.reggstore_recv,
  send = pgstrom.reggstore_send,
  like = pg_catalog.oid
);

CREATE CAST (reggstore AS oid)
  WITHOUT FUNCTION AS IMPLICIT;
CREATE CAST (oid AS reggstore)
  WITHOUT FUNCTION AS IMPLICIT;
CREATE CAST (reggstore AS integer)
  WITHOUT FUNCTION AS ASSIGNMENT;
CREATE CAST (reggstore AS bigint)
  WITH FUNCTION pg_catalog.int8(oid) AS ASSIGNMENT;
CREATE CAST (integer AS reggstore)
  WITHOUT FUNCTION AS IMPLICIT;
CREATE CAST (smallint AS reggstore)
  WITH FUNCTION pg_catalog.int4(smallint) AS IMPLICIT;
CREATE CAST (bigint AS reggstore)
  WITH FUNCTION oid(bigint) AS IMPLICIT;

CREATE FUNCTION public.gstore_fdw_format(reggstore)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_format'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_nitems(reggstore)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_nitems'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_nattrs(reggstore)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_nattrs'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_rawsize(reggstore)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_export_ipchandle(reggstore)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_gstore_export_ipchandle'
  LANGUAGE C;

CREATE TYPE pgstrom.__gstore_fdw_chunk_info AS (
  database_oid	oid,
  table_oid		oid,
  revision		int,
  xmin			xid,
  xmax			xid,
  pinning		int,
  format		text,
  rawsize		bigint,
  nitems		bigint
);
CREATE FUNCTION pgstrom.gstore_fdw_chunk_info()
  RETURNS SETOF pgstrom.__gstore_fdw_chunk_info
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_chunk_info'
  LANGUAGE C VOLATILE;
CREATE VIEW pgstrom.gstore_fdw_chunk_info AS
  SELECT * FROM pgstrom.gstore_fdw_chunk_info();

CREATE FUNCTION public.lo_import_gpu(int, bytea, bigint, bigint, oid=0)
  RETURNS oid
  AS 'MODULE_PATHNAME','pgstrom_lo_import_gpu'
  LANGUAGE C STRICT VOLATILE;

CREATE FUNCTION public.lo_export_gpu(oid, int, bytea, bigint, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_lo_export_gpu'
  LANGUAGE C STRICT VOLATILE;
*/

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

--
-- Function to query commercial license
--
CREATE FUNCTION pgstrom.license_query()
  RETURNS json
  AS 'MODULE_PATHNAME','pgstrom_license_query'
  LANGUAGE C STRICT;

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

/*
--
-- 2D-array like matrix type support routines
--
-- FEATURES DEPRECATED AT v2.3
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
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.array_matrix_width(anyarray)
  RETURNS int
  AS 'MODULE_PATHNAME','array_matrix_width'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.matrix_unnest(anyarray)
  RETURNS SETOF record
  AS 'MODULE_PATHNAME','array_matrix_unnest'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(bool[], bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_bool'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(int2[], int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(int4[], int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(int8[], int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(float4[], float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(float8[], float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_float8'
  LANGUAGE C CALLED ON NULL INPUT;

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
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(int2[], int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(int4[], int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(int8[], int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(float4[], float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(float8[], float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_float8'
  LANGUAGE C CALLED ON NULL INPUT;

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
*/
-- ==================================================================
--
-- float2 - half-precision floating point data support
--
-- ==================================================================
CREATE FUNCTION pgstrom.define_shell_type(name,oid,regnamespace='public')
  RETURNS oid
  AS 'MODULE_PATHNAME','pgstrom_define_shell_type'
  LANGUAGE C STRICT VOLATILE;
SELECT pgstrom.define_shell_type('float2',421,'pg_catalog');
--CREATE TYPE pg_catalog.float2;

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
-- Type Cast Definitions
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
  AS IMPLICIT;
CREATE CAST (float8 AS float2)
  WITH FUNCTION pgstrom.float2(float8)
  AS IMPLICIT;
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
-- comparison operators
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
-- Unary operator
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
-- Arithmetic operators
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
-- Misc operators
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
-- aggregate functions
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
-- Index Support
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
-- SQL functions to support PG-Strom regression test
--
-- ==================================================================
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

-- dummy regression test revision (very old)
CREATE OR REPLACE FUNCTION
pgstrom.regression_testdb_revision()
RETURNS int
AS 'SELECT 0'
LANGUAGE 'sql';
