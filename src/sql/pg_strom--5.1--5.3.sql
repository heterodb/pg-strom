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

---
--- Functions to support full-aggregates pushdown under Gather (#745)
---

--- MIN(X) final projection functions
CREATE FUNCTION pgstrom.fmin_i1(bytea)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_i2(bytea)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_i4(bytea)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_i8(bytea)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_f2(bytea)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp16'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_f4(bytea)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp32'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_f8(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_num(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_cash(bytea)
  RETURNS cash
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_date(bytea)
  RETURNS date
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int32'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_time(bytea)
  RETURNS time
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_ts(bytea)
  RETURNS timestamp
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmin_tstz(bytea)
  RETURNS timestamptz
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

--- MAX(X) final projection functions
CREATE FUNCTION pgstrom.fmax_i1(bytea)
  RETURNS int1
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_i2(bytea)
  RETURNS int2
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_i4(bytea)
  RETURNS int4
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_i8(bytea)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int8'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_f2(bytea)
  RETURNS float2
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp16'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_f4(bytea)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp32'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_f8(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_num(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_cash(bytea)
  RETURNS cash
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_date(bytea)
  RETURNS date
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int32'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_time(bytea)
  RETURNS time
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_ts(bytea)
  RETURNS timestamp
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fmax_tstz(bytea)
  RETURNS timestamptz
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_int64'
  LANGUAGE C STRICT PARALLEL SAFE;

---
--- SUM(X)
---
CREATE FUNCTION pgstrom.fsum_int(bytea)
  RETURNS int8
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_int'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_int64(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_int_as_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_fp32(bytea)
  RETURNS float4
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_fp32'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_fp64(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_fp64'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_numeric(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fsum_cach(bytea)
  RETURNS money
  AS 'MODULE_PATHNAME','pgstrom_fsum_final_int_as_cash'
  LANGUAGE C STRICT PARALLEL SAFE;

---
--- AVG(X)
---
CREATE FUNCTION pgstrom.favg_int(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_favg_final_int'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_int64(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_favg_final_int'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_fp(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_favg_final_fp'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.favg_numeric(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_favg_final_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

---
--- STDDEV(X)
---
CREATE FUNCTION pgstrom.fstddev_samp(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_stddev_samp_final'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fstddev_sampf(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_stddev_sampf_final'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fstddev_pop(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_stddev_pop_final'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fstddev_popf(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_stddev_popf_final'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fvar_samp(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_var_samp_final'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fvar_sampf(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_var_sampf_final'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fvar_pop(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_var_pop_final'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fvar_popf(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_var_popf_final'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;

---
--- CO-RELATION(X,Y)
---
CREATE FUNCTION pgstrom.fcorr(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_correlation_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fcovar_samp(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_covar_samp_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fcovar_pop(bytea)
  RETURNS float8
  AS 'MODULE_PATHNAME','pgstrom_covar_pop_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_avgx(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_avgx_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_avgy(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_avgy_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_count(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_count_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_intercept(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_intercept_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_r2(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_r2_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_slope(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_slope_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_sxx(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_sxx_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_sxy(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_sxy_final'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION pgstrom.fregr_syy(bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_regr_syy_final'
  LANGUAGE C STRICT PARALLEL SAFE;
