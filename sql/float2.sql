/*
 * float2 - half-precision floating point data support
 */
CREATE TYPE pg_catalog.float2;
CREATE FUNCTION pgstrom.float2_in(cstring)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_in'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2_out(float2)
  RETURNS cstring
  AS 'MODULE_PATHNAME','pgstrom_float2_out'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2_recv(internal)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_recv'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.float2_send(float2)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_float2_send'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE TYPE pg_catalog.float2
(
  input =  pgstrom.float2_in,
  output = pgstrom.float2_out,
  receive = pgstrom.float2_recv,
  send = pgstrom.float2_send,
  like = pg_catalog.int2
);

--
-- Type Cast Definitions
--
CREATE FUNCTION pgstrom.as_float4(float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_to_float4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_float8(float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float2_to_float8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_int2(float2)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_float2_to_int2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_int4(float2)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_float2_to_int4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_int8(float2)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_float2_to_int8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_numeric(float2)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_float2_to_numeric'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.as_float2(float4)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float4_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_float2(float8)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float8_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_float2(int2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int2_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_float2(int4)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int4_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_float2(int8)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int8_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
CREATE FUNCTION pgstrom.as_float2(numeric)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_numeric_to_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE CAST (float2 AS float4)
  WITH FUNCTION pgstrom.as_float4(float2)
  AS IMPLICIT;
CREATE CAST (float2 AS float8)
  WITH FUNCTION pgstrom.as_float8(float2)
  AS IMPLICIT;
CREATE CAST (float2 AS int2)
  WITH FUNCTION pgstrom.as_int2(float2)
  AS ASSIGNMENT;
CREATE CAST (float2 AS int4)
  WITH FUNCTION pgstrom.as_int4(float2)
  AS ASSIGNMENT;
CREATE CAST (float2 AS int8)
  WITH FUNCTION pgstrom.as_int8(float2)
  AS ASSIGNMENT;
CREATE CAST (float2 AS numeric)
  WITH FUNCTION pgstrom.as_numeric(float2)
  AS ASSIGNMENT;

CREATE CAST (float4 AS float2)
  WITH FUNCTION pgstrom.as_float2(float4)
  AS IMPLICIT;
CREATE CAST (float8 AS float2)
  WITH FUNCTION pgstrom.as_float2(float8)
  AS IMPLICIT;
CREATE CAST (int2 AS float2)
  WITH FUNCTION pgstrom.as_float2(int2)
  AS ASSIGNMENT;
CREATE CAST (int4 AS float2)
  WITH FUNCTION pgstrom.as_float2(int4)
  AS ASSIGNMENT;
CREATE CAST (int8 AS float2)
  WITH FUNCTION pgstrom.as_float2(int8)
  AS ASSIGNMENT;
CREATE CAST (numeric AS float2)
  WITH FUNCTION pgstrom.as_float2(numeric)
  AS ASSIGNMENT;

--
-- comparison operators
--
CREATE FUNCTION pgstrom.float2_eq(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_ne(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_lt(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_le(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_gt(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_ge(float2,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float2_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_larger(float2,float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_larger'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_smaller(float2,float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_smaller'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float42_eq(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_ne(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_lt(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_le(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_gt(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_ge(float4,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float42_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float82_eq(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_ne(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_lt(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_le(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_gt(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_ge(float8,float2)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float82_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float24_eq(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_ne(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_lt(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_le(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_gt(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_ge(float2,float4)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float24_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION pgstrom.float28_eq(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_eq'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_ne(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_ne'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_lt(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_lt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_le(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_le'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_gt(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_gt'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_ge(float2,float8)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_float28_ge'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float2_eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float2_ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float2_lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float2_le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float2_gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float2_ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float42_eq,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float42_ne,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float42_lt,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float42_le,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float42_gt,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float42_ge,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float82_eq,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float82_ne,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float82_lt,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float82_le,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float82_gt,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float82_ge,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2,
  COMMUTATOR = <=, NEGATOR = <
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float24_eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float24_ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float24_lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float24_le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float24_gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float24_ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4,
  COMMUTATOR = <=, NEGATOR = <
);


CREATE OPERATOR pg_catalog.= (
  PROCEDURE = pgstrom.float28_eq,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = =, NEGATOR = <>
);

CREATE OPERATOR pg_catalog.<> (
  PROCEDURE = pgstrom.float28_ne,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <>, NEGATOR = =
);

CREATE OPERATOR pg_catalog.< (
  PROCEDURE = pgstrom.float28_lt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = >, NEGATOR = >=
);

CREATE OPERATOR pg_catalog.<= (
  PROCEDURE = pgstrom.float28_le,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = >=, NEGATOR = >
);

CREATE OPERATOR pg_catalog.> (
  PROCEDURE = pgstrom.float28_gt,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <, NEGATOR = <=
);

CREATE OPERATOR pg_catalog.>= (
  PROCEDURE = pgstrom.float28_ge,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8,
  COMMUTATOR = <=, NEGATOR = <
);

--
-- Unary operator
--
CREATE FUNCTION pgstrom.float2_up(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_up'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_um(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_um'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_abs(float2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_float2_abs'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float2_up,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float2_um,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.@ (
  PROCEDURE = pgstrom.float2_abs,
  RIGHTARG = pg_catalog.float2
);

--
-- Arithmetic operators
--
CREATE FUNCTION pgstrom.float2_pl(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_mi(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_mul(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_div(float2,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float2_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_pl(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_mi(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_mul(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float24_div(float2,float4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float24_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_pl(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_mi(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_mul(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float28_div(float2,float8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float28_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_pl(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_mi(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_mul(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float42_div(float4,float2)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_float42_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_pl(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82_pl'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_mi(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82_mi'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_mul(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82_mul'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float82_div(float8,float2)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_float82_div'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float2_pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float2_mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float2_mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float2_div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float24_pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float24_mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float24_mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float24_div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float4
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float28_pl,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float28_mi,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float28_mul,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float28_div,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.float8
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float42_pl,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float42_mi,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float42_mul,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float42_div,
  LEFTARG = pg_catalog.float4,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.+ (
  PROCEDURE = pgstrom.float82_pl,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.- (
  PROCEDURE = pgstrom.float82_mi,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float82_mul,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);
CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.float82_div,
  LEFTARG = pg_catalog.float8,
  RIGHTARG = pg_catalog.float2
);

--
-- Misc operators
--
CREATE FUNCTION pgstrom.cash_mul_float2(money,float2)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_cash_mul_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_mul_cash(float2,money)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_float2_mul_cash'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pgstrom.cash_div_float2(money,float2)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_cash_div_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.cash_mul_float2,
  LEFTARG = pg_catalog.money,
  RIGHTARG = pg_catalog.float2
);

CREATE OPERATOR pg_catalog.* (
  PROCEDURE = pgstrom.float2_mul_cash,
  LEFTARG = pg_catalog.float2,
  RIGHTARG = pg_catalog.money
);

CREATE OPERATOR pg_catalog./ (
  PROCEDURE = pgstrom.cash_div_float2,
  LEFTARG = pg_catalog.money,
  RIGHTARG = pg_catalog.float2
);

CREATE FUNCTION pg_catalog.as_int8(float8)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_float8_as_int8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_int4(float4)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_float4_as_int4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_int2(float2)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_float2_as_int2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_float8(int8)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_int8_as_float8'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_float4(int4)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_int4_as_float4'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION pg_catalog.as_float2(int2)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_int2_as_float2'
  LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

--
-- aggregate functions
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
  sfunc = pgstrom.float2_larger,
  stype = float2
);

CREATE AGGREGATE pg_catalog.min(float2) (
  sfunc = pgstrom.float2_smaller,
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
