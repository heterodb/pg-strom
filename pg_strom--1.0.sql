--
-- pg_strom installation queries
--
CREATE TYPE __pgstrom_shmem_block_info AS (
  zone    int4,
  size    text,
  active  int8,
  free    int8
);
CREATE FUNCTION pgstrom_shmem_block_info()
  RETURNS SETOF __pgstrom_shmem_block_info
  AS 'MODULE_PATHNAME'
  LANGUAGE C STRICT;

CREATE TYPE __pgstrom_shmem_context_info AS (
  name        text,
  owner       int4,
  usage       int8,
  alloc       int8,
  num_chunks  int8,
  md5sum      text
);
CREATE FUNCTION pgstrom_shmem_context_info()
  RETURNS SETOF __pgstrom_shmem_context_info
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
