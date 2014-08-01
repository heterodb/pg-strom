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

CREATE TYPE __pgstrom_tcache_info AS (
  datoid    oid,
  reloid    oid,
  cached    int2vector,
  refcnt    int4,
  state     text,
  lwlock    text
);
CREATE FUNCTION pgstrom_tcache_info()
  RETURNS SETOF __pgstrom_tcache_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_tcache_node_info AS (
  type      text,
  addr      int8,
  l_node    int8,
  r_node    int8,
  l_depth   int4,
  r_depth   int4,
  refcnt    int4,
  nrows     int4,
  usage     int8,
  length    int8,
  blkno_min int4,
  blkno_max int4
);
CREATE FUNCTION pgstrom_tcache_node_info(regclass)
  RETURNS SETOF __pgstrom_tcache_node_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE FUNCTION public.pgstrom_tcache_synchronizer()
  RETURNS trigger
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
CREATE FUNCTION pgstrom.sum_int8_accum(int8[], int4, int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME', 'pgstrom_sum_int8_accum'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.sum_int8_final(int8[])
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_sum_int8_final'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.ex_avg(int4, int8)
(
  sfunc = pgstrom.accum_int8,
  stype = int8[],
  finalfunc = pg_catalog.int8_avg,
  initcond = '{0,0}'
);

CREATE AGGREGATE pgstrom.ex_sum(int8)
(
  sfunc = pgstrom.accum_int8,
  stype = int8[],
  finalfunc = pgstrom.sum_int8_final,
  initcond = '{0,0}'
);

CREATE FUNCTION pgstrom.sum_float8_accum(float8[], int4, float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME', 'pgstrom_sum_float8_accum'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.ex_avg(int4, float8)
(
  sfunc = pgstrom.sum_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_avg,
  initcond = '{0,0,0}'
);

CREATE FUNCTION pgstrom.variance_float8_accum(float8[], int4, float8, float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_variance_float8_accum'
  LANGUAGE C STRICT;




CREATE AGGREGATE pgstrom.ex_stddev(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_stddev,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.ex_stddev_samp(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_stddev_samp,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.ex_stddev_pop(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_stddev_pop,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.ex_variance(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_var_samp,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.ex_var_samp(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_var_samp,
  initcond = '{0,0,0}'
);

CREATE AGGREGATE pgstrom.ex_var_pop(int4, float8, float8)
(
  sfunc = pgstrom.variance_float8_accum,
  stype = float8[],
  finalfunc = pg_catalog.float8_var_pop,
  initcond = '{0,0,0}'
);

CREATE FUNCTION pgstrom.covariance_float8_accum(float8[], int4, float8, float8)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_covariance_float8_accum'
  LANGUAGE C STRICT;

CREATE AGGREGATE pgstrom.ex_corr(int4, float8, float8, float8,
                                       float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = float8_corr,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.ex_covar_pop(int4, float8, float8, float8,
                                       float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = float8_covar_pop,
  initcond = '{0,0,0,0,0,0}'
);

CREATE AGGREGATE pgstrom.ex_covar_samp(int4, float8, float8, float8,
                                       float8, float8, float8)
(
  sfunc = pgstrom.covariance_float8_accum,
  stype = float8[],
  finalfunc = float8_covar_samp,
  initcond = '{0,0,0,0,0,0}'
);
