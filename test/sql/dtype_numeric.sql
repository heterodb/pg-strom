---
--- Micro test cases for numeric operators / functions
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dtype_numeric_temp CASCADE;
CREATE SCHEMA regtest_dtype_numeric_temp;
RESET client_min_messages;

SET search_path = regtest_dtype_numeric_temp,public;
CREATE TABLE rt_numeric (
  id    int,
  a     int2,
  b     int4,
  c     int8,
  d     float2,
  e     float4,
  f     float8,
  x     numeric,
  y     numeric(12,3),
  z     numeric(12,3)
);
SELECT pgstrom.random_setseed(20190611);
INSERT INTO rt_numeric (
  SELECT x, pgstrom.random_int(1,   -20000,   20000),
            pgstrom.random_int(1,  -200000,  200000),
            pgstrom.random_int(1, -2000000, 2000000),
            pgstrom.random_float(1,   -3200.0,   3200.0),
            pgstrom.random_float(1,  -32000.0,  32000.0),
            pgstrom.random_float(1, -320000.0, 320000.0),
            pgstrom.random_float(1,    -20000,    20000)::numeric,
            pgstrom.random_int(1, -2000000000,
                                   2000000000)::numeric / 1000::numeric,
            pgstrom.random_int(1, -2000000000,
                                   2000000000)::numeric / 1000::numeric
    FROM generate_series(1,3000) x);
VACUUM ANALYZE;

-- force to use GpuScan, instead of SeqScan
SET enable_seqscan = off;
-- not to print kernel source code
SET pg_strom.debug_kernel_source = off;

-- cast operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a::numeric, b::numeric, c::numeric,
           d::numeric, e::numeric, f::numeric
  INTO test01g
  FROM rt_numeric
 WHERE id > 0;
SELECT id, a::numeric, b::numeric, c::numeric,
           d::numeric, e::numeric, f::numeric
  INTO test01g
  FROM rt_numeric
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, a::numeric, b::numeric, c::numeric,
           d::numeric, e::numeric, f::numeric
  INTO test01p
  FROM rt_numeric
 WHERE id > 0;

SELECT g.*, p.*
  FROM test01g g JOIN test01p p ON g.id = p.id
 WHERE @(g.a - p.a) > 0
    OR @(g.b - p.b) > 0
    OR @(g.c - p.c) > 0
    OR @(g.d - p.d) > 0.5
    OR @(g.e - p.e) > 0.1
    OR @(g.f - p.f) > 0.01;

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, x::int2 i2, y::int4 i4, z::int8 i8,
           x::float2 f2, y::float4 f4, z::float8 f8
  INTO test02g
  FROM rt_numeric
 WHERE id > 0;
SELECT id, x::int2 i2, y::int4 i4, z::int8 i8,
           x::float2 f2, y::float4 f4, z::float8 f8
  INTO test02g
  FROM rt_numeric
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, x::int2 i2, y::int4 i4, z::int8 i8,
           x::float2 f2, y::float4 f4, z::float8 f8
  INTO test02p
  FROM rt_numeric
 WHERE id > 0;

SELECT g.*, p.*
  FROM test02g g JOIN test02p p ON g.id = p.id
 WHERE g.i2 <> p.i2
    OR g.i4 <> p.i4
    OR g.i8 <> p.i8
    OR @(g.f2 - p.f2) > 0.01
    OR @(g.f4 - p.f4) > 0.01
    OR @(g.f8 - p.f8) > 0.01;

-- numeric operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, x+y v1, y-z v2, x*z v3, +x v4, -y v5, @z v6
  INTO test03g
  FROM rt_numeric
 WHERE id > 0;
SELECT id, x+y v1, y-z v2, x*z v3, +x v4, -y v5, @z v6
  INTO test03g
  FROM rt_numeric
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, x+y v1, y-z v2, x*z v3, +x v4, -y v5, @z v6
  INTO test03p
  FROM rt_numeric
 WHERE id > 0;

(SELECT * FROM test03g EXCEPT SELECT * FROM test03p) ORDER BY id;
(SELECT * FROM test03p EXCEPT SELECT * FROM test03g) ORDER BY id;

-- numeric comparison
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, x = y eq, x <> y ne, x > y gt, x >= z ge, y < z lt, y <= x le
  INTO test04g
  FROM rt_numeric
 WHERE id > 0;
SELECT id, x = y eq, x <> y ne, x > y gt, x >= z ge, y < z lt, y <= x le
  INTO test04g
  FROM rt_numeric
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, x = y eq, x <> y ne, x > y gt, x >= z ge, y < z lt, y <= x le
  INTO test04p
  FROM rt_numeric
 WHERE id > 0;

(SELECT * FROM test04g EXCEPT SELECT * FROM test04p) ORDER BY id;
(SELECT * FROM test04p EXCEPT SELECT * FROM test04g) ORDER BY id;
