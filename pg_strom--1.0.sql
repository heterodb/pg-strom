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
  tracked   bool,
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

CREATE FUNCTION pgstrom_create_queue()
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_create_queue_func'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_close_queue(int8)
  RETURNS bool
  AS 'MODULE_PATHNAME', 'pgstrom_close_queue_func'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_create_testmsg(int8,int4,text)
  RETURNS int8
  AS 'MODULE_PATHNAME', 'pgstrom_create_testmsg_func'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_enqueue_testmsg(int8)
  RETURNS bool
  AS 'MODULE_PATHNAME', 'pgstrom_enqueue_testmsg_func'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_dequeue_testmsg(int8)
  RETURNS bool
  AS 'MODULE_PATHNAME', 'pgstrom_dequeue_testmsg_func'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom_release_testmsg(int8)
  RETURNS bool
  AS 'MODULE_PATHNAME', 'pgstrom_release_testmsg_func'
  LANGUAGE C STRICT;
