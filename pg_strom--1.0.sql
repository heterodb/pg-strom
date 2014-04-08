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

CREATE FUNCTION pgstrom_shmem_alloc(int8)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_shmem_alloc_func'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_shmem_free(int8)
  RETURNS bool
  AS 'MODULE_PATHNAME', 'pgstrom_shmem_free_func'
  LANGUAGE C STRICT;
