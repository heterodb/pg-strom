--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

--
-- pg_strom installation queries
--
CREATE TYPE __pgstrom_shmem_info AS (
  zone    int4,
  size    text,
  active  int8,
  free    int8
);
CREATE FUNCTION pgstrom_shmem_info()
  RETURNS SETOF __pgstrom_shmem_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_shmem_active_info AS (
  zone      int4,
  address   int8,
  size      int4,
  owner     int4,
  location  text,
  broken    bool,
  overrun   bool
);
CREATE FUNCTION pgstrom_shmem_active_info()
  RETURNS SETOF __pgstrom_shmem_active_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_shmem_slab_info AS (
  address   int8,
  slabname  text,
  owner		int4,
  location	text,
  active    bool,
  broken	bool
);
CREATE FUNCTION pgstrom_shmem_slab_info()
  RETURNS SETOF __pgstrom_shmem_slab_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_opencl_device_info AS (
  dnum      int4,
  pnum		int4,
  property	text,
  value		text
);
CREATE FUNCTION pgstrom_opencl_device_info()
  RETURNS SETOF __pgstrom_opencl_device_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_opencl_program_info AS (
  key  		text,
  refcnt	int4,
  state		text,
  crc		text,
  flags		int4,
  length	int4,
  source	text,
  errmsg	text
);
CREATE FUNCTION pgstrom_opencl_program_info()
  RETURNS SETOF __pgstrom_opencl_program_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_mqueue_info AS (
  mqueue	text,
  owner		int4,
  state     text,
  refcnt	int4
);
CREATE FUNCTION pgstrom_mqueue_info()
  RETURNS SETOF __pgstrom_mqueue_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_shmem_alloc(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_shmem_alloc_func'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_shmem_free(int8)
  RETURNS bool
  AS 'MODULE_PATHNAME', 'pgstrom_shmem_free_func'
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

-- Definition of Partial SUM (returns 64bit value)
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
  
-- Definition of Partial SUM for covariance (only float8)
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
-- definition of trans/final functioin of alternative aggregates
--
CREATE FUNCTION pgstrom.avg_int8_accum(int8[], int4, int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME', 'pgstrom_avg_int8_accum'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.sum_int8_accum(int8[], int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME', 'pgstrom_sum_int8_accum'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.sum_int8_final(int8[])
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_sum_int8_final'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.avg(int4, int8)
(
  sfunc = pgstrom.avg_int8_accum,
  stype = int8[],
  finalfunc = pg_catalog.int8_avg,
  initcond = '{0,0}'
);

CREATE AGGREGATE pgstrom.sum(int8)
(
  sfunc = pgstrom.sum_int8_accum,
  stype = int8[],
  finalfunc = pgstrom.sum_int8_final,
  initcond = '{0,0}'
);

CREATE FUNCTION pgstrom.avg_numeric_accum(internal, int4, int8)
  RETURNS internal
  AS 'MODULE_PATHNAME', 'pgstrom_avg_numeric_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE AGGREGATE pgstrom.avg_numeric(int4, int8)
(
  sfunc = pgstrom.avg_numeric_accum,
  stype = internal,
  finalfunc = pg_catalog.numeric_avg
);

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


