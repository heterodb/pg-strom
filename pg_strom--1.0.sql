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
