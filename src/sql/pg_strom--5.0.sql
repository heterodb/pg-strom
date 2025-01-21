--
-- Schema to deploy PG-Strom objects
--
CREATE SCHEMA IF NOT EXISTS pgstrom;

--
-- MEMO: Some DDL command may internally create a pseudo object
-- to avoid dependency problems, but does not allow to qualify
-- the referenced object by namespace.
-- For example, CREATE OPERATOR '=' can take a negator operator '<>'
-- before the creation of the '<>', and internally creates the '<>'
-- operator in the default namespace but it is not an intended
-- behavior. So, we switch the default namespace to pg_catalog here.
--
SET search_path = 'pg_catalog';

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
CREATE TYPE pgstrom.__gpu_device_info AS (
  gpu_id        int,
  att_name      text,
  att_value     text,
  att_desc      text
);
CREATE FUNCTION pgstrom.gpu_device_info()
  RETURNS SETOF pgstrom.__gpu_device_info
  AS 'MODULE_PATHNAME','pgstrom_gpu_device_info'
  LANGUAGE C STRICT;
CREATE VIEW pgstrom.gpu_device_info AS
  SELECT * FROM pgstrom.gpu_device_info();

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

CREATE FUNCTION pgstrom.arrow_fdw_import_file(text,	    -- relname
                                              text,	    -- filename
                                              text = null)  -- schema
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_import_file'
  LANGUAGE C;

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
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_gpucache_apply_redo'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.gpucache_compaction(regclass)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_gpucache_compaction'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.gpucache_recovery(regclass)
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_gpucache_recovery'
  LANGUAGE C STRICT;

CREATE TYPE pgstrom.__pgstrom_gpucache_info_t AS (
    database_oid        oid,
    database_name       text,
    table_oid           oid,
    table_name          text,
    signature           int8,
    phase               text,
    rowid_num_used      int8,
    rowid_num_free      int8,
    gpu_main_sz         int8,
    gpu_main_nitems     int8,
    gpu_extra_sz        int8,
    gpu_extra_usage     int8,
    gpu_extra_dead      int8,
    redo_write_ts       timestamptz,
    redo_write_nitems   int8,
    redo_write_pos      int8,
    redo_read_nitems    int8,
    redo_read_pos       int8,
    redo_sync_pos       int8,
    config_options      text
);

CREATE FUNCTION pgstrom.__pgstrom_gpucache_info()
  RETURNS SETOF pgstrom.__pgstrom_gpucache_info_t
  AS 'MODULE_PATHNAME','pgstrom_gpucache_info'
  LANGUAGE C STRICT;
CREATE VIEW pgstrom.gpucache_info AS
  SELECT * FROM pgstrom.__pgstrom_gpucache_info();

-- ==================================================================
--
-- float2 - half-precision floating point data support
--
-- ==================================================================
CREATE TYPE pg_catalog.float2;

CREATE FUNCTION pgstrom.float2in(cstring)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2in'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2out(float2)
  RETURNS cstring
  AS 'MODULE_PATHNAME','pgstrom_float2out'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2recv(internal)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2recv'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2send(float2)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_float2send'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE TYPE pg_catalog.float2
(
  input =  pgstrom.float2in,
  output = pgstrom.float2out,
  receive = pgstrom.float2recv,
  send = pgstrom.float2send,
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
CREATE FUNCTION pgstrom.float2eq(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2ne(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2lt(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2le(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2gt(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2ge(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2cmp(float2,float2)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float2cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2larger(float2,float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2larger'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2smaller(float2,float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2smaller'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2hash(float2)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float2hash'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42eq(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42ne(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42lt(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42le(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42gt(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42ge(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42cmp(float4,float2)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float42cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float82eq(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82ne(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82lt(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82le(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82gt(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82ge(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82cmp(float8,float2)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float82cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float24eq(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24ne(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24lt(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24le(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24gt(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24ge(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24cmp(float2,float4)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float24cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float28eq(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28ne(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28lt(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28le(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28gt(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28ge(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28cmp(float2,float8)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_float28cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float2eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float2ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float2lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float2le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float2gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float2ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float42eq,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float42ne,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float42lt,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float42le,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float42gt,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float42ge,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float82eq,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float82ne,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float82lt,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float82le,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float82gt,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float82ge,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);

CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float24eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float24ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float24lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float24le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float24gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float24ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float28eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float28ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float28lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float28le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float28gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float28ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);

--
-- float2 unary operator
--
CREATE FUNCTION pgstrom.float2up(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2up'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2um(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2um'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.abs(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2abs'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float2up,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float2um,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.@ (
  PROCEDURE = pg_catalog.abs,
  RIGHTARG = pg_catalog.float2
);

--
-- float2 arithmetic operators
--
CREATE FUNCTION pgstrom.float2pl(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2mi(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2mul(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2div(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24pl(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24mi(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24mul(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24div(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28pl(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28mi(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28mul(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28div(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42pl(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42mi(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42mul(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42div(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82pl(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82mi(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82mul(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82div(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float2pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float2mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float2mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float2div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float24pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float24mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float24mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float24div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float28pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float28mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float28mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float28div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float42pl,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float42mi,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float42mul,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float42div,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float82pl,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float82mi,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float82mul,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float82div,
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
  sfunc = pgstrom.float2larger,
  stype = float2
);

CREATE AGGREGATE pg_catalog.min(float2) (
  sfunc = pgstrom.float2smaller,
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
  function 1 (float2, float2) pgstrom.float2cmp(float2, float2);

CREATE OPERATOR CLASS pg_catalog.float2_ops
  default for type pg_catalog.float2
  using hash family pg_catalog.float_ops as
  operator 1 = (float2, float2) for search,
  function 1 (float2) pgstrom.float2hash(float2);

-- ==================================================================
--
-- int1(tinyint) - 8bit width integer data support
--
-- ==================================================================
CREATE TYPE pg_catalog.int1;

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
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int1ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int1lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int1le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int1gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int1ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);

-- <int1> OPER <smallint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int12eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int12ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int12lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int12le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int12gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int12ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);

-- <int1> OPER <int>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int14eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int14ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int14lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int14le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int14gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int14ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);

-- <int1> OPER <bigint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int18eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int18ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int18lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int18le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int18gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int18ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);

-- <smallint> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int21eq,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int21ne,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int21lt,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int21le,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int21gt,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int21ge,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);

-- <int> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int41eq,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int41ne,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int41lt,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int41le,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int41gt,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int41ge,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
);

-- <bigint> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int81eq,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>,
  RESTRICT = pg_catalog.eqsel,
  JOIN = pg_catalog.eqjoinsel,
  HASHES, MERGES
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int81ne,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =,
  RESTRICT = pg_catalog.neqsel,
  JOIN = pg_catalog.neqjoinsel
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int81lt,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=,
  RESTRICT = pg_catalog.scalarltsel,
  JOIN = pg_catalog.scalarltjoinsel
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int81le,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >,
  RESTRICT = pg_catalog.scalarlesel,
  JOIN = pg_catalog.scalarlejoinsel
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int81gt,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=,
  RESTRICT = pg_catalog.scalargtsel,
  JOIN = pg_catalog.scalargtjoinsel
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int81ge,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <,
  RESTRICT = pg_catalog.scalargesel,
  JOIN = pg_catalog.scalargejoinsel
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
  operator 1 = (int1,int1) for search,
  function 1 (int1) pgstrom.int1hash(int1);

-- ==================================================================
--
-- Partial / Alternative aggregate functions for GpuGroupBy
--
-- ==================================================================

--
-- NROWS()
--
CREATE FUNCTION pgstrom.nrows()
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.nrows("any")
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_partial_nrows'
  LANGUAGE C STRICT PARALLEL SAFE;


CREATE AGGREGATE pgstrom.fcount(bigint)
(
  sfunc = pg_catalog.int8pl,
  stype = bigint,
  initcond = "0",
  parallel = safe
);

--
-- PMIN(X)
--
CREATE FUNCTION pgstrom.pmin(int1)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(int2)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(int4)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(int8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(float2)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(float4)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(float8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(money)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(date)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(time)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(timestamp)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(timestamptz)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

--
-- PMAX(X)
--
CREATE FUNCTION pgstrom.pmax(int1)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(int2)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(int4)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(int8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(float2)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(float4)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(float8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(money)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(date)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(time)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(timestamp)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(timestamptz)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

---
--- Final MIN(X)/MAX(X) functions
---
CREATE FUNCTION pgstrom.fmin_trans_int64(bytea,bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fmin_trans_int64'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_trans_fp64(bytea,bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fmin_trans_fp64'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_trans_int64(bytea,bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fmax_trans_int64'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_trans_fp64(bytea,bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fmax_trans_fp64'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;


CREATE FUNCTION pgstrom.fminmax_final_int8(bytea)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_int16(bytea)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int16'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_int32(bytea)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int32'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_int64(bytea)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_fp16(bytea)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp16'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_fp32(bytea)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp32'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_fp64(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_numeric(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fminmax_final_money(bytea)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fminmax_final_date(bytea)
  RETURNS date
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int32'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fminmax_final_time(bytea)
  RETURNS time
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fminmax_final_timestamp(bytea)
  RETURNS timestamp
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fminmax_final_timestamptz(bytea)
  RETURNS timestamptz
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

-- alternative MIN(X) for each supported type
CREATE AGGREGATE pgstrom.min_i1(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_int8,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_i2(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_int16,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_i4(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_int32,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_i8(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_int64,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_f2(bytea)
(
  sfunc = pgstrom.fmin_trans_fp64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_fp16,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_f4(bytea)
(
  sfunc = pgstrom.fmin_trans_fp64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_fp32,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_f8(bytea)
(
  sfunc = pgstrom.fmin_trans_fp64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_fp64,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_num(bytea)
(
  sfunc = pgstrom.fmin_trans_fp64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_cash(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_money,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_date(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_date,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_time(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_time,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_ts(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_timestamp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.min_tstz(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_timestamptz,
  parallel = safe
);

-- alternative MAX(X) for each supported type
CREATE AGGREGATE pgstrom.max_i1(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_int8,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_i2(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_int16,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_i4(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_int32,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_i8(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_int64,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_f2(bytea)
(
  sfunc = pgstrom.fmax_trans_fp64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_fp16,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_f4(bytea)
(
  sfunc = pgstrom.fmax_trans_fp64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_fp32,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_f8(bytea)
(
  sfunc = pgstrom.fmax_trans_fp64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_fp64,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_num(bytea)
(
  sfunc = pgstrom.fmax_trans_fp64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_cash(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_money,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_date(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_date,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_time(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_time,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_ts(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_timestamp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.max_tstz(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_timestamptz,
  parallel = safe
);

---
--- SUM(X)
---
CREATE FUNCTION pgstrom.psum(int8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_int'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.psum(float8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_fp'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.psum(money)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_cash'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_trans_int(bytea, bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fsum_trans_int'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_trans_fp(bytea, bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fsum_trans_fp'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_final_int(bytea)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_int'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_final_int_as_numeric(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_int_as_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_final_int_as_cash(bytea)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_int_as_cash'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_final_fp32(bytea)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_fp32'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_final_fp64(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_final_fp_as_numeric(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_fp64_as_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

-- SUM(int1/int2/int4) --> bigint
CREATE AGGREGATE pgstrom.sum_int(bytea)
(
  sfunc = pgstrom.fsum_trans_int,
  stype = bytea,
  finalfunc = pgstrom.fsum_final_int,
  parallel = safe
);
-- SUM(int8) --> numeric (DEPRECATED))
CREATE AGGREGATE pgstrom.sum_int_num(bytea)
(
  sfunc = pgstrom.fsum_trans_int,
  stype = bytea,
  finalfunc = pgstrom.fsum_final_int_as_numeric,
  parallel = safe
);
-- SUM(float2/float4) --> float4
CREATE AGGREGATE pgstrom.sum_fp32(bytea)
(
  sfunc = pgstrom.fsum_trans_fp,
  stype = bytea,
  finalfunc = pgstrom.fsum_final_fp32,
  parallel = safe
);
-- SUM(float8) --> float8
CREATE AGGREGATE pgstrom.sum_fp64(bytea)
(
  sfunc = pgstrom.fsum_trans_fp,
  stype = bytea,
  finalfunc = pgstrom.fsum_final_fp64,
  parallel = safe
);
-- SUM(numeric) --> numeric
CREATE AGGREGATE pgstrom.sum_fp_num(bytea)
(
  sfunc = pgstrom.fsum_trans_fp,
  stype = bytea,
  finalfunc = pgstrom.fsum_final_fp_as_numeric,
  parallel = safe
);
-- SUM(money) --> money
CREATE AGGREGATE pgstrom.sum_cash(bytea)
(
  sfunc = pgstrom.fsum_trans_int,
  stype = bytea,
  finalfunc = pgstrom.fsum_final_int_as_cash,
  parallel = safe
);

---
--- AVG(X)
---
CREATE FUNCTION pgstrom.pavg(int8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_int'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pavg(float8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_fp'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_trans_int(bytea, bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fsum_trans_int'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_trans_fp(bytea, bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fsum_trans_fp'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_final_int(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_favg_final_int'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_final_fp(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_favg_final_fp'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_final_num(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_favg_final_num'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.avg_int(bytea)
(
  sfunc = pgstrom.favg_trans_int,
  stype = bytea,
  finalfunc = pgstrom.favg_final_int,
  parallel = safe
);

CREATE AGGREGATE pgstrom.avg_fp(bytea)
(
  sfunc = pgstrom.favg_trans_fp,
  stype = bytea,
  finalfunc = pgstrom.favg_final_fp,
  parallel = safe
);

CREATE AGGREGATE pgstrom.avg_num(bytea)
(
  sfunc = pgstrom.favg_trans_fp,
  stype = bytea,
  finalfunc = pgstrom.favg_final_num,
  parallel = safe
);

---
--- STDDEV/VARIANCE
---
CREATE FUNCTION pgstrom.pvariance(float8)
  RETURNS bytea
  AS 'MODULE_PATHNAME', 'pgstrom_partial_variance'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.stddev_trans(bytea,bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME', 'pgstrom_stddev_trans'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.stddev_samp_final(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_stddev_samp_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.stddev_sampf_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_stddev_sampf_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.stddev_pop_final(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_stddev_pop_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.stddev_popf_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_stddev_popf_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.var_samp_final(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_var_samp_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.var_sampf_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_var_sampf_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.var_pop_final(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME', 'pgstrom_var_pop_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.var_popf_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME', 'pgstrom_var_popf_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.stddev_samp(bytea)
(
  sfunc = pgstrom.stddev_trans,
  stype = bytea,
  finalfunc = pgstrom.stddev_samp_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_sampf(bytea)
(
  sfunc = pgstrom.stddev_trans,
  stype = bytea,
  finalfunc = pgstrom.stddev_sampf_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_pop(bytea)
(
  sfunc = pgstrom.stddev_trans,
  stype = bytea,
  finalfunc = pgstrom.stddev_pop_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.stddev_popf(bytea)
(
  sfunc = pgstrom.stddev_trans,
  stype = bytea,
  finalfunc = pgstrom.stddev_popf_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_samp(bytea)
(
  sfunc = pgstrom.stddev_trans,
  stype = bytea,
  finalfunc = pgstrom.var_samp_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_sampf(bytea)
(
  sfunc = pgstrom.stddev_trans,
  stype = bytea,
  finalfunc = pgstrom.var_sampf_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_pop(bytea)
(
  sfunc = pgstrom.stddev_trans,
  stype = bytea,
  finalfunc = pgstrom.var_pop_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.var_popf(bytea)
(
  sfunc = pgstrom.stddev_trans,
  stype = bytea,
  finalfunc = pgstrom.var_popf_final,
  parallel = safe
);

---
--- COVAR/REGR_*
---
CREATE FUNCTION pgstrom.pcovar(float8,float8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_covar'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.covar_accum(bytea,bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_covar_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.covar_samp_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_covar_samp_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.covar_pop_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_covar_pop_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.covar_samp(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.covar_samp_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.covar_pop(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.covar_pop_final,
  parallel = safe
);

CREATE FUNCTION pgstrom.regr_avgx_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_avgx_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.regr_avgy_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_avgy_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.regr_count_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_count_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.regr_intercept_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_intercept_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.regr_r2_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_r2_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.regr_slope_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_slope_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.regr_sxx_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_sxx_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.regr_sxy_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_sxy_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.regr_syy_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_regr_syy_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.regr_avgx(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_avgx_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_avgy(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_avgy_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_count(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_count_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_intercept(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_intercept_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_r2(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_r2_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_slope(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_slope_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_sxx(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_sxx_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_sxy(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_sxy_final,
  parallel = safe
);

CREATE AGGREGATE pgstrom.regr_syy(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.regr_syy_final,
  parallel = safe
);

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
RETURNS text
AS 'SELECT ''unknown'''
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

--
-- Reset GUC parameters
--
RESET search_path;
