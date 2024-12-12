---
--- PG-Strom v5.1 -> v5.3 (minor changes)
---

---
--- A function to check arrow_fdw's pattern option
--- related to the issue #834
---
CREATE FUNCTION pgstrom.arrow_fdw_check_pattern(text, text)
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_check_pattern'
  LANGUAGE C STRICT;

---
--- Functions to support 128bit fixed-point numeric aggregation
--- related to the issue #806
---
CREATE FUNCTION pgstrom.psum(numeric)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.pavg(numeric)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_trans_numeric(bytea, bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fsum_trans_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_final_numeric(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_final_numeric(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_favg_final_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.sum_numeric(bytea)
(
  sfunc = pgstrom.fsum_trans_numeric,
  stype = bytea,
  finalfunc = pgstrom.fsum_final_numeric,
  parallel = safe
);

CREATE AGGREGATE pgstrom.avg_numeric(bytea)
(
  sfunc = pgstrom.fsum_trans_numeric,
  stype = bytea,
  finalfunc = pgstrom.favg_final_numeric,
  parallel = safe
);

---
--- Functions to support 128bit SUM(int8) AVG(int8) aggregation related to
--- the issue #860
---

-- SUM(int8) --> numeric
CREATE FUNCTION pgstrom.psum64(int8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.sum_int64(bytea)
(
  sfunc = pgstrom.fsum_trans_numeric,
  stype = bytea,
  finalfunc = pgstrom.fsum_final_numeric,
  parallel = safe
);
-- AVG(int8) --> numeric
CREATE FUNCTION pgstrom.pavg64(int8)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_sum_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.avg_int64(bytea)
(
  sfunc = pgstrom.fsum_trans_numeric,
  stype = bytea,
  finalfunc = pgstrom.favg_final_numeric,
  parallel = safe
);

---
--- BUGFIX: corr(X,Y) has been defined incorrectly.
---
CREATE FUNCTION pgstrom.correlation_final(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_correlation_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE AGGREGATE pgstrom.corr(bytea)
(
  sfunc = pgstrom.covar_accum,
  stype = bytea,
  finalfunc = pgstrom.correlation_final,
  parallel = safe
);

---
--- Device Hash Functions
---
CREATE FUNCTION pgstrom.bool_devhash(pg_catalog.bool, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_bool_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.int1_devhash(pg_catalog.int1, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int1_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.int2_devhash(pg_catalog.int2, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int2_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.int4_devhash(pg_catalog.int4, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int4_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.int8_devhash(pg_catalog.int8, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_int8_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float2_devhash(pg_catalog.float2, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_float2_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float4_devhash(pg_catalog.float4, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_float4_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.float8_devhash(pg_catalog.float8, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_float8_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.numeric_devhash(pg_catalog.numeric, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_numeric_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.bytea_devhash(pg_catalog.bytea, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_bytea_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.text_devhash(pg_catalog.text, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_text_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.bpchar_devhash(pg_catalog.bpchar, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_bpchar_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.date_devhash(pg_catalog.date, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_date_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.time_devhash(pg_catalog.time, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_time_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.timetz_devhash(pg_catalog.timetz, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_timetz_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.timestamp_devhash(pg_catalog.timestamp, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_timestamp_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.timestamptz_devhash(pg_catalog.timestamptz, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_timestamptz_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;
CREATE FUNCTION pgstrom.interval_devhash(pg_catalog.interval, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_interval_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.money_devhash(pg_catalog.money, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_money_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.uuid_devhash(pg_catalog.uuid, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_uuid_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.macaddr_devhash(pg_catalog.macaddr, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_macaddr_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.inet_devhash(pg_catalog.inet, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_inet_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.jsonb_devhash(pg_catalog.jsonb, bigint)
  RETURNS bigint
  AS 'MODULE_PATHNAME','pgstrom_jsonb_devhash'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;
