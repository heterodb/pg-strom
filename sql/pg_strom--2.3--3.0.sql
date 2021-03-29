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
SELECT pgstrom.define_shell_type('int1',606,'pg_catalog');
--CREATE TYPE pg_catalog.int1;

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
--- Comparison functions
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
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int1ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int1lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int1le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int1gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int1ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int1> OPER <smallint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int12eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int12ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int12lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int12le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int12gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int12ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = smallint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int1> OPER <int>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int14eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int14ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int14lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int14le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int14gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int14ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = integer,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int1> OPER <bigint>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int18eq,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int18ne,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int18lt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int18le,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int18gt,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int18ge,
  LEFTARG = pg_catalog.int1,
  RIGHTARG = bigint,
  COMMUTATOR = <=, NEGATOR = <
);

-- <smallint> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int21eq,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int21ne,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int21lt,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int21le,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int21gt,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int21ge,
  LEFTARG = smallint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <
);

-- <int> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int41eq,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int41ne,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int41lt,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int41le,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int41gt,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int41ge,
  LEFTARG = integer,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <
);

-- <bigint> OPER <int1>
CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.int81eq,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = =, NEGATOR = <>
);
CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.int81ne,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <>, NEGATOR = =
);
CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.int81lt,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >, NEGATOR = >=
);
CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.int81le,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = >=, NEGATOR = >
);
CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.int81gt,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <, NEGATOR = <=
);
CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.int81ge,
  LEFTARG = bigint,
  RIGHTARG = pg_catalog.int1,
  COMMUTATOR = <=, NEGATOR = <
);

--
-- unary operators
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
--- Arithmetic operators
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
--- Bit operations
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
--- aggregate functions
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
--- Index Support
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
  function 1 (int1) pgstrom.int1hash(int1);

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
