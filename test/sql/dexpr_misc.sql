---
--- Test for misc device expressions
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dexpr_misc_temp CASCADE;
CREATE SCHEMA regtest_dexpr_misc_temp;
RESET client_min_messages;

SET search_path = regtest_dexpr_misc_temp,public;
CREATE TABLE regtest_data (
  id    int,
  a     numeric,
  b     numeric,
  c     numeric,
  d     numeric,
  memo  text
);
SELECT pgstrom.random_setseed(20190701);
INSERT INTO regtest_data (
  SELECT x, pgstrom.random_float(20,-100.0,100.0)::numeric(9,3),
            pgstrom.random_float(20,-100.0,100.0)::numeric(9,3),
            pgstrom.random_float(20,-100.0,100.0)::numeric(9,3),
            pgstrom.random_float(20,-100.0,100.0)::numeric(9,3),
            pgstrom.random_text_len(5, 48)
    FROM generate_series(1,6000) x
);

-- force to use GpuScan and disables to print source files
SET enable_seqscan = off;
SET pg_strom.debug_kernel_source = off;

-- test for COALESCE / GREATEST / LEAST
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, COALESCE(a, b, c, d) v1,
           GREATEST(a, b, c, d) v2,
           LEAST(a, b, c, d) v3,
           COALESCE(a+b,b+c,c+d) v4,
           GREATEST(a+b,b+c,c+d) v5,
           LEAST(a+b,b+c,c+d) v6
  INTO test01g
  FROM regtest_data
 WHERE id > 0;
SELECT id, COALESCE(a, b, c, d) v1,
           GREATEST(a, b, c, d) v2,
           LEAST(a, b, c, d) v3,
           COALESCE(a+b,b+c,c+d) v4,
           GREATEST(a+b,b+c,c+d) v5,
           LEAST(a+b,b+c,c+d) v6
  INTO test01g
  FROM regtest_data
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, COALESCE(a, b, c, d) v1,
           GREATEST(a, b, c, d) v2,
           LEAST(a, b, c, d) v3,
           COALESCE(a+b,b+c,c+d) v4,
           GREATEST(a+b,b+c,c+d) v5,
           LEAST(a+b,b+c,c+d) v6
  INTO test01p
  FROM regtest_data
 WHERE id > 0;
(SELECT * FROM test01g EXCEPT SELECT * FROM test01p) ORDER BY id;
(SELECT * FROM test01p EXCEPT SELECT * FROM test01g) ORDER BY id;

SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, COALESCE(a::float, b::float, -1.0, d::float / 0.0) v1
  INTO test02g
  FROM regtest_data
 WHERE id > 0;
SELECT id, COALESCE(a::float, b::float, -1.0, d::float / 0.0) v1
  INTO test02g
  FROM regtest_data
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, COALESCE(a::float, b::float, -1.0, d::float / 0.0) v1
  INTO test02p
  FROM regtest_data
 WHERE id > 0;
SELECT p.id, p.v1, g.v1
  FROM test02g g, test02p p
 WHERE p.id = g.id AND abs(p.v1 - g.v1) > 0.001;

-- test for BoolExpr
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, not a > b v1,
           (a + b > c + d or a - b < c - d) and memo like '%abc%' v2,
           (a + d > b + c and b < d) or memo like '%xyz%' v3
  INTO test10g
  FROM regtest_data
 WHERE id > 0;
SELECT id, not a > b v1,
           (a + b > c + d or a - b < c - d) and memo like '%abc%' v2,
           (a + d > b + c and b < d) or memo like '%xyz%' v3
  INTO test10g
  FROM regtest_data
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, not a > b v1,
           (a + b > c + d or a - b < c - d) and memo like '%abc%' v2,
           (a + d > b + c and b < d) or memo like '%xyz%' v3
  INTO test10p
  FROM regtest_data
 WHERE id > 0;
(SELECT * FROM test10g EXCEPT SELECT * FROM test10p) ORDER BY id;
(SELECT * FROM test10p EXCEPT SELECT * FROM test10g) ORDER BY id;

-- test for BooleanTest / NullTest
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, a > b IS TRUE v1,
           c > d IS FALSE v2,
           a > c IS NOT TRUE v3,
           b > d IS NOT FALSE v4,
           a > d IS UNKNOWN v5,
           b > c IS UNKNOWN v6,
           b IS NULL v7,
           c IS NOT NULL v8
  INTO test20g
  FROM regtest_data
 WHERE id > 0;
SELECT id, a > b IS TRUE v1,
           c > d IS FALSE v2,
           a > c IS NOT TRUE v3,
           b > d IS NOT FALSE v4,
           a > d IS UNKNOWN v5,
           b > c IS UNKNOWN v6,
           b IS NULL v7,
           c IS NOT NULL v8
  INTO test20g
  FROM regtest_data
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, a > b IS TRUE v1,
           c > d IS FALSE v2,
           a > c IS NOT TRUE v3,
           b > d IS NOT FALSE v4,
           a > d IS UNKNOWN v5,
           b > c IS UNKNOWN v6,
           b IS NULL v7,
           c IS NOT NULL v8
  INTO test20p
  FROM regtest_data
 WHERE id > 0;
(SELECT * FROM test20g EXCEPT SELECT * FROM test20p) ORDER BY id;
(SELECT * FROM test20p EXCEPT SELECT * FROM test20g) ORDER BY id;

-- test for CASE ... WHEN
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, CASE id % 4
           WHEN 0 THEN 'hoge'
           WHEN 1 THEN substring(memo, (id % 32) / 4, 6)
           WHEN 2 THEN substring(memo, 1, 6)
           END v1,
           CASE id % 5
           WHEN 0 THEN 'monu'
           WHEN 2 THEN substring(memo, 4, 6)
           WHEN 4 THEN substring(memo, (id % 32) / 4, 6)
           ELSE        'piyo'
           END v2
  INTO test30g
  FROM regtest_data
 WHERE id > 0;
SELECT id, CASE id % 4
           WHEN 0 THEN 'hoge'
           WHEN 1 THEN substring(memo,(id % 32) / 4, 6)
           WHEN 2 THEN substring(memo, 1, 6)
           END v1,
           CASE id % 5
           WHEN 0 THEN 'monu'
           WHEN 2 THEN substring(memo, 4, 6)
           WHEN 4 THEN substring(memo, (id % 32) / 4, 6)
           ELSE        'piyo'
           END v2
  INTO test30g
  FROM regtest_data
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, CASE id % 4
           WHEN 0 THEN 'hoge'
           WHEN 1 THEN substring(memo, (id % 32) / 4, 6)
           WHEN 2 THEN substring(memo, 1, 6)
           END v1,
           CASE id % 5
           WHEN 0 THEN 'monu'
           WHEN 2 THEN substring(memo, 4, 6)
           WHEN 4 THEN substring(memo,(id % 32) / 4, 6)
           ELSE        'piyo'
           END v2
  INTO test30p
  FROM regtest_data
 WHERE id > 0;
(SELECT * FROM test30g EXCEPT SELECT * FROM test30p) ORDER BY id;
(SELECT * FROM test30p EXCEPT SELECT * FROM test30g) ORDER BY id;

SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, CASE WHEN memo like '%aa%' THEN 'aaa'
                WHEN memo like '%bb%' THEN 'bbb'
                WHEN memo like '%cc%' THEN 'ccc'
                WHEN memo like '%dd%' THEN 'ddd'
                WHEN memo like '%ee%' THEN 'eee'
            END v1,
           CASE WHEN id % 100 != 0
                THEN a::real / (id % 100)::real
                ELSE -1.0
            END v2,
           CASE id % 71
           WHEN 0 THEN -1.0
           ELSE b::real / (id % 71)::real
            END v3
  INTO test31g
  FROM regtest_data
 WHERE id > 0;
SELECT id, CASE WHEN memo like '%aa%' THEN 'aaa'
                WHEN memo like '%bb%' THEN 'bbb'
                WHEN memo like '%cc%' THEN 'ccc'
                WHEN memo like '%dd%' THEN 'ddd'
                WHEN memo like '%ee%' THEN 'eee'
            END v1,
           CASE WHEN id % 100 != 0
                THEN a::real / (id % 100)::real
                ELSE -1.0
            END v2,
           CASE id % 71
           WHEN 0 THEN -1.0
           ELSE b::real / (id % 71)::real
            END v3
  INTO test31g
  FROM regtest_data
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, CASE WHEN memo like '%aa%' THEN 'aaa'
                WHEN memo like '%bb%' THEN 'bbb'
                WHEN memo like '%cc%' THEN 'ccc'
                WHEN memo like '%dd%' THEN 'ddd'
                WHEN memo like '%ee%' THEN 'eee'
            END v1,
           CASE WHEN id % 100 != 0
                THEN a::real / (id % 100)::real
                ELSE -1.0
            END v2,
           CASE id % 71
           WHEN 0 THEN -1.0
           ELSE b::real / (id % 71)::real
            END v3
  INTO test31p
  FROM regtest_data
 WHERE id > 0;
(SELECT * FROM test31g EXCEPT SELECT * FROM test31p) ORDER BY id;
(SELECT * FROM test31p EXCEPT SELECT * FROM test31g) ORDER BY id;

-- cleanup
SET client_min_messages = error;
DROP SCHEMA regtest_dexpr_misc_temp CASCADE;
