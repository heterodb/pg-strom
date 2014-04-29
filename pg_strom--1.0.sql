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

CREATE FUNCTION pgstrom_shmem_alloc(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_shmem_alloc_func'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_shmem_free(int8)
  RETURNS bool
  AS 'MODULE_PATHNAME', 'pgstrom_shmem_free_func'
  LANGUAGE C STRICT;
