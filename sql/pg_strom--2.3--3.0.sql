--
-- Handlers of gstore_fdw extension (renew at v3.0)
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

CREATE OR REPLACE FUNCTION pgstrom.gstore_fdw_post_creation()
  RETURNS event_trigger
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_post_creation'
  LANGUAGE C STRICT;
CREATE EVENT TRIGGER pgstrom_gstore_fdw_post_creation
  ON ddl_command_end
WHEN tag IN ('CREATE FOREIGN TABLE')
EXECUTE PROCEDURE pgstrom.gstore_fdw_post_creation();

CREATE FUNCTION public.gstore_fdw_apply_redo(regclass)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_apply_redo'
  LANGUAGE C STRICT;

CREATE FUNCTION public.gstore_fdw_compaction(regclass, bool = false)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_compaction'
  LANGUAGE C STRICT;

SELECT pgstrom.define_shell_type('gstore_fdw_sysattr',6116,'pgstrom');
CREATE FUNCTION pgstrom.gstore_fdw_sysattr_in(cstring)
  RETURNS pgstrom.gstore_fdw_sysattr
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_sysattr_in'
  LANGUAGE C STRICT IMMUTABLE;
CREATE FUNCTION pgstrom.gstore_fdw_sysattr_out(pgstrom.gstore_fdw_sysattr)
  RETURNS cstring
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_sysattr_out'
  LANGUAGE C STRICT IMMUTABLE;
CREATE TYPE pgstrom.gstore_fdw_sysattr
(
  INPUT = pgstrom.gstore_fdw_sysattr_in,
  OUTPUT = pgstrom.gstore_fdw_sysattr_out,
  INTERNALLENGTH = 12,
  ALIGNMENT = int4
);

CREATE FUNCTION pgstrom.gstore_fdw_replication_base(regclass,int)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_replication_base'
  LANGUAGE C STRICT;
CREATE FUNCTION pgstrom.gstore_fdw_replication_redo(regclass, bigint,
													float = 5.0,     -- 5.0sec
													bigint = 64,     -- 64kB
													bigint = 131072) -- 128MB
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_gstore_fdw_replication_redo'
  LANGUAGE C STRICT;

---
--- arrow_fdw special import function
---
CREATE FUNCTION pgstrom.arrow_fdw_import_file(text,         -- relname
                                              text,	        -- filename
                                              text = null)  -- schema
  RETURNS void
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_import_file'
  LANGUAGE C;

---
--- tinyint (8bit-integer) support functions
---
SELECT pgstrom.define_shell_type('tinyint',606,'pg_catalog');
--CREATE TYPE pg_catalog.float2;

CREATE FUNCTION pgstrom.int1in(cstring)
  RETURNS pg_catalog.tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1in'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1out(pg_catalog.tinyint)
  RETURNS cstring
  AS 'MODULE_PATHNAME','pgstrom_int1out'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1recv(internal)
  RETURNS pg_catalog.tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1recv'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1send(pg_catalog.tinyint)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_int1send'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE TYPE pg_catalog.tinyint
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
CREATE FUNCTION pgstrom.int2(tinyint)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_int1_to_int2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int4(tinyint)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_int1_to_int4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int8(tinyint)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_int1_to_int8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2(tinyint)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int1_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float4(tinyint)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_int1_to_float4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8(tinyint)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_int1_to_float8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.numeric(tinyint)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_int1_to_numeric'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint(int2)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int2_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint(int4)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int4_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint(int8)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int8_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint(float2)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_float2_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint(float4)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_float4_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint(float8)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_float8_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint(numeric)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_numeric_to_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE CAST (tinyint AS int2)
  WITH FUNCTION pgstrom.int2(tinyint)
  AS IMPLICIT;
CREATE CAST (tinyint AS int4)
  WITH FUNCTION pgstrom.int4(tinyint)
  AS IMPLICIT;
CREATE CAST (tinyint AS int8)
  WITH FUNCTION pgstrom.int8(tinyint)
  AS IMPLICIT;
CREATE CAST (tinyint AS float2)
  WITH FUNCTION pgstrom.float2(tinyint)
  AS IMPLICIT;
CREATE CAST (tinyint AS float4)
  WITH FUNCTION pgstrom.float4(tinyint)
  AS IMPLICIT;
CREATE CAST (tinyint AS float8)
  WITH FUNCTION pgstrom.float8(tinyint)
  AS IMPLICIT;

CREATE CAST (int2 AS tinyint)
  WITH FUNCTION pgstrom.tinyint(int2)
  AS IMPLICIT;
CREATE CAST (int4 AS tinyint)
  WITH FUNCTION pgstrom.tinyint(int4)
  AS IMPLICIT;
CREATE CAST (int8 AS tinyint)
  WITH FUNCTION pgstrom.tinyint(int8)
  AS IMPLICIT;
CREATE CAST (float2 AS tinyint)
  WITH FUNCTION pgstrom.tinyint(float2)
  AS IMPLICIT;
CREATE CAST (float4 AS tinyint)
  WITH FUNCTION pgstrom.tinyint(float4)
  AS IMPLICIT;
CREATE CAST (float8 AS tinyint)
  WITH FUNCTION pgstrom.tinyint(float8)
  AS IMPLICIT;

---
--- Comparison functions
---
CREATE FUNCTION pgstrom.int1_eq(tinyint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_ne(tinyint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_lt(tinyint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_le(tinyint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_gt(tinyint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_ge(tinyint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int1_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_cmp(tinyint,tinyint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int1_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_larger(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_larger'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_smaller(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_smaller'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_hash(tinyint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int1_hash'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int12_eq(tinyint,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12_ne(tinyint,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12_lt(tinyint,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12_le(tinyint,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12_gt(tinyint,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12_ge(tinyint,smallint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int12_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12_cmp(tinyint,smallint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int12_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int14_eq(tinyint,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14_ne(tinyint,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14_lt(tinyint,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14_le(tinyint,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14_gt(tinyint,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14_ge(tinyint,int)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int14_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14_cmp(tinyint,int)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int14_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int18_eq(tinyint,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18_ne(tinyint,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18_lt(tinyint,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18_le(tinyint,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18_gt(tinyint,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18_ge(tinyint,bigint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int18_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18_cmp(tinyint,bigint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int18_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int21_eq(smallint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21_ne(smallint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21_lt(smallint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21_le(smallint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21_gt(smallint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21_ge(smallint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int21_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21_cmp(smallint,tinyint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int21_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int41_eq(int,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41_ne(int,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41_lt(int,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41_le(int,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41_gt(int,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41_ge(int,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int41_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41_cmp(int,tinyint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int41_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.int81_eq(bigint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81_ne(bigint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81_lt(bigint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81_le(bigint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81_gt(bigint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81_ge(bigint,tinyint)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_int81_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81_cmp(bigint,tinyint)
  RETURNS int
  AS 'MODULE_PATHNAME','pgstrom_int81_cmp'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

-- <tinyint> OPER <tinyint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int1_eq,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int1_ne,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int1_lt,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int1_le,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int1_gt,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int1_ge,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <tinyint> OPER <smallint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int12_eq,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int12_ne,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int12_lt,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int12_le,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int12_gt,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int12_ge,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <tinyint> OPER <int>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int14_eq,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int14_ne,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int14_lt,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int14_le,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int14_gt,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int14_ge,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer,
  COMMUTATOR = <=, NEGATOR = <
);

-- <tinyint> OPER <bigint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int18_eq,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int18_ne,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int18_lt,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int18_le,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int18_gt,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int18_ge,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <smallint> OPER <tinyint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int21_eq,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int21_ne,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int21_lt,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int21_le,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int21_gt,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int21_ge,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int> OPER <tinyint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int41_eq,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int41_ne,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int41_lt,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int41_le,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int41_gt,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int41_ge,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <bigint> OPER <tinyint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int81_eq,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int81_ne,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int81_lt,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int81_le,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int81_gt,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int81_ge,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint,
  COMMUTATOR = <=, NEGATOR = <
);

--
-- unary operators
--
CREATE FUNCTION pgstrom.int1_up(tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_up'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_um(tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_um'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_abs(tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_abs'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int1_up,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int1_um,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.@ (
  PROCEDURE = pgstrom.int1_abs,
  RIGHTARG = pg_catalog.tinyint
);

---
--- Arithmetic operators
---
CREATE FUNCTION pgstrom.int1_pl(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1_mi(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1_mul(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1_div(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_int1_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int12_pl(tinyint,smallint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int12_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int12_mi(tinyint,smallint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int12_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int12_mul(tinyint,smallint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int12_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int12_div(tinyint,smallint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int12_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int14_pl(tinyint,integer)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int14_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int14_mi(tinyint,integer)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int14_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int14_mul(tinyint,integer)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int14_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int14_div(tinyint,integer)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int14_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int18_pl(tinyint,bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int18_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int18_mi(tinyint,bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int18_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int18_mul(tinyint,bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int18_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int18_div(tinyint,bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int18_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int21_pl(smallint,tinyint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int21_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int21_mi(smallint,tinyint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int21_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int21_mul(smallint,tinyint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int21_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int21_div(smallint,tinyint)
  RETURNS smallint
  AS 'MODULE_PATHNAME','pgstrom_int21_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int41_pl(integer,tinyint)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int41_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int41_mi(integer,tinyint)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int41_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int41_mul(integer,tinyint)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int41_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int41_div(integer,tinyint)
  RETURNS integer
  AS 'MODULE_PATHNAME','pgstrom_int41_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.int81_pl(bigint,tinyint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int81_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int81_mi(bigint,tinyint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int81_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int81_mul(bigint,tinyint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int81_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int81_div(bigint,tinyint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int81_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int1_pl,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int1_mi,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int1_mul,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int1_div,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int12_pl,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int12_mi,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int12_mul,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int12_div,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = smallint
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int14_pl,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int14_mi,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int14_mul,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int14_div,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int18_pl,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int18_mi,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int18_mul,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int18_div,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = bigint
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int21_pl,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int21_mi,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int21_mul,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int21_div,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.tinyint
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int41_pl,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int41_mi,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int41_mul,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int41_div,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.tinyint
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.int81_pl,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.int81_mi,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int81_mul,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.int81_div,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.tinyint
);

---
--- Bit operations
---
CREATE FUNCTION pgstrom.tinyint_and(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_tinyint_and'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.tinyint_or(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_tinyint_or'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.tinyint_xor(tinyint,tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_tinyint_xor'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.tinyint_not(tinyint)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_tinyint_not'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.tinyint_shl(tinyint,integer)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_tinyint_shl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.tinyint_shr(tinyint,integer)
  RETURNS tinyint
  AS 'MODULE_PATHNAME','pgstrom_tinyint_shr'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.& (
  PROCEDURE = pgstrom.tinyint_and,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.| (
  PROCEDURE = pgstrom.tinyint_or,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.# (
  PROCEDURE = pgstrom.tinyint_xor,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.~ (
  PROCEDURE = pgstrom.tinyint_not,
  LEFTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.<< (
  PROCEDURE = pgstrom.tinyint_shl,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer
);
CREATE OPERATOR pg_catalog.>> (
  PROCEDURE = pgstrom.tinyint_shr,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = integer
);

---
--- Misc functions
---
CREATE FUNCTION pgstrom.cash_mul_int1(money,tinyint)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_cash_mul_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.int1_mul_cash(tinyint,money)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_int1_mul_cash'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.cash_div_int1(money,tinyint)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_cash_div_int1'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.cash_mul_int1,
  LEFTARG = pg_catalog.money,
  RIGHTARG = pg_catalog.tinyint
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.int1_mul_cash,
  LEFTARG = pg_catalog.tinyint,
  RIGHTARG = pg_catalog.money
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.cash_div_int1,
  LEFTARG = pg_catalog.money,
  RIGHTARG = pg_catalog.tinyint
);

---
--- aggregate functions
---
CREATE FUNCTION pgstrom.tinyint_sum(bigint, pg_catalog.tinyint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_tinyint_sum'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint_avg_accum(bigint[], pg_catalog.tinyint)
  RETURNS bigint[]
  AS 'MODULE_PATHNAME','pgstrom_tinyint_avg_accum'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint_avg_accum_inv(bigint[], pg_catalog.tinyint)
  RETURNS bigint[]
  AS 'MODULE_PATHNAME','pgstrom_tinyint_avg_accum_inv'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint_var_accum(internal, pg_catalog.tinyint)
  RETURNS internal
  AS 'MODULE_PATHNAME','pgstrom_tinyint_var_accum'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.tinyint_var_accum_inv(internal, pg_catalog.tinyint)
  RETURNS internal
  AS 'MODULE_PATHNAME','pgstrom_tinyint_var_accum_inv'
  LANGUAGE C CALLED ON NULL INPUT IMMUTABLE PARALLEL SAFE;

CREATE AGGREGATE pg_catalog.sum(tinyint)
(
  sfunc = pgstrom.tinyint_sum,
  stype = bigint
);

CREATE AGGREGATE pg_catalog.max(tinyint)
(
 sfunc = pgstrom.int1_larger,
 stype = tinyint
);

CREATE AGGREGATE pg_catalog.min(tinyint)
(
 sfunc = pgstrom.int1_smaller,
 stype = tinyint
);

CREATE AGGREGATE pg_catalog.avg(tinyint)
(
  sfunc = pgstrom.tinyint_avg_accum,
  stype = bigint[],
  finalfunc = int8_avg,
  initcond = "{0,0}",
  combinefunc = int4_avg_combine,
  msfunc = pgstrom.tinyint_avg_accum,
  minvfunc = pgstrom.tinyint_avg_accum_inv,
  mfinalfunc = int8_avg,
  mstype = bigint[],
  minitcond = "{0,0}",
  parallel = safe
);

CREATE AGGREGATE pg_catalog.variance(tinyint)
(
  sfunc = pgstrom.tinyint_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_var_samp,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.tinyint_var_accum,
  minvfunc = pgstrom.tinyint_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_var_samp,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.var_samp(tinyint)
(
  sfunc = pgstrom.tinyint_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_var_samp,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.tinyint_var_accum,
  minvfunc = pgstrom.tinyint_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_var_samp,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.var_pop(tinyint)
(
  sfunc = pgstrom.tinyint_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_var_pop,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.tinyint_var_accum,
  minvfunc = pgstrom.tinyint_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_var_pop,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.stddev(tinyint)
(
  sfunc = pgstrom.tinyint_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_stddev_samp,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.tinyint_var_accum,
  minvfunc = pgstrom.tinyint_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_stddev_samp,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.stddev_samp(tinyint)
(
  sfunc = pgstrom.tinyint_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_stddev_samp,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.tinyint_var_accum,
  minvfunc = pgstrom.tinyint_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_stddev_samp,
  parallel = safe
);

CREATE AGGREGATE pg_catalog.stddev_pop(tinyint)
(
  sfunc = pgstrom.tinyint_var_accum,
  stype = internal,
  sspace = 48,
  finalfunc = numeric_poly_stddev_pop,
  combinefunc = numeric_poly_combine,
  serialfunc = numeric_poly_serialize,
  deserialfunc = numeric_poly_deserialize,
  msfunc = pgstrom.tinyint_var_accum,
  minvfunc = pgstrom.tinyint_var_accum_inv,
  mstype = internal,
  msspace = 48,
  mfinalfunc = numeric_poly_stddev_pop,
  parallel = safe
);

---
--- Index Support
---
CREATE OPERATOR CLASS pg_catalog.tinyint_ops
  default for type pg_catalog.tinyint
  using btree family pg_catalog.integer_ops as
  operator 1 <  (tinyint,tinyint) for search,
  operator 2 <= (tinyint,tinyint) for search,
  operator 3 =  (tinyint,tinyint) for search,
  operator 4 >= (tinyint,tinyint) for search,
  operator 5 >  (tinyint,tinyint) for search,
  function 1 (tinyint,tinyint) pgstrom.int1_cmp(tinyint,tinyint);

CREATE OPERATOR CLASS pg_catalog.tinyint_ops
  default for type pg_catalog.tinyint
  using hash family pg_catalog.integer_ops as
  function 1 (tinyint) pgstrom.int1_hash(tinyint);

---
--- Deprecated functions
---
DROP FUNCTION IF EXISTS public.gpu_device_name(int);
DROP FUNCTION IF EXISTS public.gpu_global_memsize(int);
DROP FUNCTION IF EXISTS public.gpu_max_blocksize(int);
DROP FUNCTION IF EXISTS public.gpu_warp_size(int);
DROP FUNCTION IF EXISTS public.gpu_max_shared_memory_perblock(int);
DROP FUNCTION IF EXISTS public.gpu_num_registers_perblock(int);
DROP FUNCTION IF EXISTS public.gpu_num_multiptocessors(int);
DROP FUNCTION IF EXISTS public.gpu_num_cuda_cores(int);
DROP FUNCTION IF EXISTS public.gpu_cc_major(int);
DROP FUNCTION IF EXISTS public.gpu_cc_minor(int);
DROP FUNCTION IF EXISTS public.gpu_pci_id(int);

DROP TYPE IF EXISTS pgstrom.__pgstrom_device_preserved_meminfo CASCADE;
