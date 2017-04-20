--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

--
-- Support routines for ctid reference with gpu projection
--
CREATE FUNCTION pgstrom.cast_tid_to_int8(tid)
  RETURNS bigint
  AS 'MODULE_PATHNAME', 'pgstrom_cast_tid_to_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.cast_int8_to_tid(bigint)
  RETURNS tid
  AS 'MODULE_PATHNAME', 'pgstrom_cast_int8_to_tid'
  LANGUAGE C STRICT;

CREATE CAST (tid AS bigint)
  WITH FUNCTION pgstrom.cast_tid_to_int8(tid)
  AS IMPLICIT;

CREATE CAST (bigint AS tid)
  WITH FUNCTION pgstrom.cast_int8_to_tid(bigint)
  AS IMPLICIT;

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
-- Functions/Languages to support PL/CUDA
--
CREATE FUNCTION pgstrom.plcuda_function_validator(oid)
  RETURNS void
  AS 'MODULE_PATHNAME','plcuda_function_validator'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.plcuda_function_handler()
  RETURNS language_handler
  AS 'MODULE_PATHNAME','plcuda_function_handler'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.plcuda_function_source(regproc)
  RETURNS text
  AS 'MODULE_PATHNAME','plcuda_function_source'
  LANGUAGE C STRICT;

CREATE LANGUAGE plcuda
  HANDLER pgstrom.plcuda_function_handler
  VALIDATOR pgstrom.plcuda_function_validator;
COMMENT ON LANGUAGE plcuda IS 'PL/CUDA procedural language';

CREATE FUNCTION pgstrom.pltext_function_validator(oid)
  RETURNS void
  AS 'MODULE_PATHNAME','pltext_function_validator'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.pltext_function_handler()
  RETURNS language_handler
  AS 'MODULE_PATHNAME','pltext_function_handler'
  LANGUAGE C STRICT;

CREATE LANGUAGE pltext
  HANDLER pgstrom.pltext_function_handler
  VALIDATOR pgstrom.pltext_function_validator;
COMMENT ON LANGUAGE pltext IS 'PL/Text contents holder';

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
