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
--- A set-returning function to get custom-metadata of Arrow_Fdw tables
--- related to the issue #863
---
CREATE TYPE pgstrom.__arrow_fdw_metadata_info AS (
  relid     regclass,
  filename  text,
  field     text,
  key       text,
  value     text
);
CREATE FUNCTION pgstrom.arrow_fdw_metadata_info(regclass)
  RETURNS SETOF pgstrom.__arrow_fdw_metadata_info
  AS 'MODULE_PATHNAME','pgstrom_arrow_fdw_metadata_info'
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
