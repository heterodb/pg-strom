--
-- Matrix like 2D-Array type support
--
CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic bool[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic int2[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic int4[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic int8[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic real[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_accum(internal, variadic float[])
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum'
  LANGUAGE C CALLED ON NULL INPUT;

-- varbit as matrix of int4[]
CREATE FUNCTION pgstrom.array_matrix_accum_varbit(internal, bit)
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_accum_varbit'
  LANGUAGE C CALLED ON NULL INPUT;

-- type case varbit <--> int4[]
CREATE FUNCTION pgstrom.varbit_to_int4_array(bit)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','varbit_to_int4_array'
  LANGUAGE C STRICT;

CREATE CAST (bit AS int4[])
  WITH FUNCTION pgstrom.varbit_to_int4_array(bit)
  AS ASSIGNMENT;

CREATE FUNCTION pgstrom.int4_array_to_varbit(int4[])
  RETURNS bit
  AS 'MODULE_PATHNAME','int4_array_to_varbit'
  LANGUAGE C STRICT;

CREATE CAST (int4[] AS bit)
  WITH FUNCTION pgstrom.int4_array_to_varbit(int4[])
  AS ASSIGNMENT;

-- final functions of array_matrix
CREATE FUNCTION pgstrom.array_matrix_final_bool(internal)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_final_bool'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_int2(internal)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_final_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_int4(internal)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_final_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_int8(internal)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_final_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_float4(internal)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_final_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_final_float8(internal)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_final_float8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE AGGREGATE pg_catalog.array_matrix(variadic bool[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_bool
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic int2[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_int2
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic int4[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_int4
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic int8[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_int8
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic float4[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_float4
);

CREATE AGGREGATE pg_catalog.array_matrix(variadic float8[])
(
  sfunc = pgstrom.array_matrix_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_float8
);

CREATE AGGREGATE pg_catalog.array_matrix(bit)
(
  sfunc = pgstrom.array_matrix_accum_varbit,
  stype = internal,
  finalfunc = pgstrom.array_matrix_final_int4
);

CREATE FUNCTION pg_catalog.array_matrix_validation(anyarray)
  RETURNS bool
  AS 'MODULE_PATHNAME','array_matrix_validation'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.array_matrix_height(anyarray)
  RETURNS int
  AS 'MODULE_PATHNAME','array_matrix_height'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.array_matrix_width(anyarray)
  RETURNS int
  AS 'MODULE_PATHNAME','array_matrix_width'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.array_vector_rawsize(regtype,int)
  RETURNS bigint
  AS 'MODULE_PATHNAME','array_matrix_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.array_matrix_rawsize(regtype,int,int)
  RETURNS bigint
  AS 'MODULE_PATHNAME','array_matrix_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.array_cube_rawsize(regtype,int,int,int)
  RETURNS bigint
  AS 'MODULE_PATHNAME','array_matrix_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.type_len(regtype)
  RETURNS bigint
  AS 'MODULE_PATHNAME','postgresql_type_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.composite_type_rawsize(VARIADIC int[])
  RETURNS bigint
  AS 'MODULE_PATHNAME','composite_type_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.composite_type_rawsize(VARIADIC bigint[])
  RETURNS bigint
  AS 'MODULE_PATHNAME','composite_type_rawsize'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.matrix_unnest(anyarray)
  RETURNS SETOF record
  AS 'MODULE_PATHNAME','array_matrix_unnest'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(bool[], bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_bool'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(int2[], int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(int4[], int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(int8[], int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(float4[], float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(float8[], float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_float8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.rbind(bool, bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_boolt'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(bool[], bool)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_boolb'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int2, int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int2t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int2[], int2)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int2b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int4, int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int4t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int4[], int4)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int4b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int8, int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int8t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(int8[], int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_int8b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float4, float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_float4t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float4[], float4)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_float4b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float8, float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_float8t'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.rbind(float8[], float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_scalar_float8b'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(bool[], bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_bool'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(int2[], int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(int4[], int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(int8[], int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(float4[], float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(float8[], float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_float8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pg_catalog.cbind(bool, bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_booll'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(bool[], bool)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_boolr'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int2, int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int2l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int2[], int2)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int2r'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int4, int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int4l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int4[], int4)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int4r'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int8, int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int8l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(int8[], int8)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_int8r'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float4, float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_float4l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float4[], float4)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_float4r'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float8, float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_float8l'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.cbind(float8[], float8)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_scalar_float8r'
  LANGUAGE C STRICT;

CREATE FUNCTION pgstrom.array_matrix_rbind_accum(internal, anyarray)
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_rbind_accum'
  LANGUAGE C CALLED ON NULL INPUT;;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_bool(internal)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_bool'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_int2(internal)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_int4(internal)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_int8(internal)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_float4(internal)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_rbind_final_float8(internal)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_rbind_final_float8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE AGGREGATE pg_catalog.rbind(bool[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_bool
);

CREATE AGGREGATE pg_catalog.rbind(int2[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_int2
);

CREATE AGGREGATE pg_catalog.rbind(int4[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_int4
);

CREATE AGGREGATE pg_catalog.rbind(int8[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_int8
);

CREATE AGGREGATE pg_catalog.rbind(float4[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_float4
);

CREATE AGGREGATE pg_catalog.rbind(float8[])
(
  sfunc = pgstrom.array_matrix_rbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_rbind_final_float8
);


CREATE FUNCTION pgstrom.array_matrix_cbind_accum(internal, anyarray)
  RETURNS internal
  AS 'MODULE_PATHNAME','array_matrix_cbind_accum'
  LANGUAGE C CALLED ON NULL INPUT;;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_bool(internal)
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_bool'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_int2(internal)
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_int2'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_int4(internal)
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_int4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_int8(internal)
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_int8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_float4(internal)
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_float4'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE FUNCTION pgstrom.array_matrix_cbind_final_float8(internal)
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_cbind_final_float8'
  LANGUAGE C CALLED ON NULL INPUT;

CREATE AGGREGATE pg_catalog.cbind(bool[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_bool
);

CREATE AGGREGATE pg_catalog.cbind(int2[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_int2
);

CREATE AGGREGATE pg_catalog.cbind(int4[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_int4
);

CREATE AGGREGATE pg_catalog.cbind(int8[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_int8
);

CREATE AGGREGATE pg_catalog.cbind(float4[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_float4
);

CREATE AGGREGATE pg_catalog.cbind(float8[])
(
  sfunc = pgstrom.array_matrix_cbind_accum,
  stype = internal,
  finalfunc = pgstrom.array_matrix_cbind_final_float8
);

CREATE FUNCTION pg_catalog.transpose(bool[])
  RETURNS bool[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_bool'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(int2[])
  RETURNS int2[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_int2'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(int4[])
  RETURNS int4[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_int4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(int8[])
  RETURNS int8[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_int8'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(float4[])
  RETURNS float4[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_float4'
  LANGUAGE C STRICT;

CREATE FUNCTION pg_catalog.transpose(float8[])
  RETURNS float8[]
  AS 'MODULE_PATHNAME','array_matrix_transpose_float8'
  LANGUAGE C STRICT;
