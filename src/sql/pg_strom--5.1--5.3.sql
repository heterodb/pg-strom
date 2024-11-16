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
