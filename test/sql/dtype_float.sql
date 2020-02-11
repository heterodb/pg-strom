---
--- Micro test cases for floating-point operators / functions
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dtype_float_temp CASCADE;
CREATE SCHEMA regtest_dtype_float_temp;
RESET client_min_messages;

SET search_path = regtest_dtype_float_temp,public;
CREATE TABLE rt_float (
  id   int,
  a    float2,
  b    float2,
  c    float4,
  d    float4,
  e    float8,
  f    float8,
  x    int2,
  y    int4,
  z    int8
);
SELECT pgstrom.random_setseed(20190609);
INSERT INTO rt_float (
  SELECT x, pgstrom.random_float(1,     -3200.0,     3200.0),
            pgstrom.random_float(1,     -3200.0,     3200.0),
            pgstrom.random_float(1,  -8000000.0,  8000000.0),
            pgstrom.random_float(1,  -8000000.0,  8000000.0),
            pgstrom.random_float(1, -80000000.0, 80000000.0),
            pgstrom.random_float(1, -80000000.0, 80000000.0),
            pgstrom.random_int(0,  -3200,  3200),
            pgstrom.random_int(0, -32000, 32000),
            pgstrom.random_int(0, -32000, 32000)
    FROM generate_series(1,2000) x);
-- force to use GpuScan, instead of SeqScan
SET enable_seqscan = off;
-- not to print kernel source code
SET pg_strom.debug_kernel_source = off;

-- cast operators
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, (c/400.0)::float2 c, (e/4000.0)::float2 e,
       x::float2, y::float2, z::float2
  INTO test01g
  FROM rt_float
 WHERE x BETWEEN -1800 AND 1800;
SELECT id, (c/400.0)::float2 c, (e/4000.0)::float2 e,
           x::float2, y::float2, z::float2
  INTO test01g
  FROM rt_float
 WHERE x BETWEEN -1800 AND 1800;
SET pg_strom.enabled = off;
SELECT id, (c/400.0)::float2 c, (e/4000.0)::float2 e,
           x::float2, y::float2, z::float2
  INTO test01p
  FROM rt_float
 WHERE x BETWEEN -1800 AND 1800;
(SELECT * FROM test01g EXCEPT ALL SELECT * FROM test01p) ORDER BY id;
(SELECT * FROM test01p EXCEPT ALL SELECT * FROM test01g) ORDER BY id;

SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, a::float4, e::float4, x::float4, y::float4, z::float4
  INTO test02g
  FROM rt_float
 WHERE x < 0;
SELECT id, a::float4, e::float4, x::float4, y::float4, z::float4
  INTO test02g
  FROM rt_float
 WHERE x < 0;
SET pg_strom.enabled = off;
SELECT id, a::float4, e::float4, x::float4, y::float4, z::float4
  INTO test02p
  FROM rt_float
 WHERE x < 0;
(SELECT * FROM test02g EXCEPT ALL SELECT * FROM test02p) ORDER BY id;
(SELECT * FROM test02p EXCEPT ALL SELECT * FROM test02g) ORDER BY id;

SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, a::float8, c::float8, x::float8, y::float8, z::float8
  INTO test03g
  FROM rt_float
 WHERE x > 0;
SELECT id, a::float8, c::float8, x::float8, y::float8, z::float8
  INTO test03g
  FROM rt_float
 WHERE x > 0;
SET pg_strom.enabled = off;
SELECT id, a::float8, c::float8, x::float8, y::float8, z::float8
  INTO test03p
  FROM rt_float
 WHERE x > 0;
(SELECT * FROM test03g EXCEPT ALL SELECT * FROM test03p) ORDER BY id;
(SELECT * FROM test03p EXCEPT ALL SELECT * FROM test03g) ORDER BY id;

-- '+' operators
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, a+b v1, a+c v2, a+e v3,
           c+b v4, c+d v5, c+e v6,
           e+b v7, e+d v8, e+f v9
  INTO test10g
  FROM rt_float
 WHERE y BETWEEN -16000 AND 16000;
SELECT id, a+b v1, a+c v2, a+e v3,
           c+b v4, c+d v5, c+e v6,
           e+b v7, e+d v8, e+f v9
  INTO test10g
  FROM rt_float
 WHERE y BETWEEN -16000 AND 16000;
SET pg_strom.enabled = off;
SELECT id, a+b v1, a+c v2, a+e v3,
           c+b v4, c+d v5, c+e v6,
           e+b v7, e+d v8, e+f v9
  INTO test10p
  FROM rt_float
 WHERE y BETWEEN -16000 AND 16000;
(SELECT * FROM test10g EXCEPT ALL SELECT * FROM test10p) ORDER BY id;
(SELECT * FROM test10p EXCEPT ALL SELECT * FROM test10g) ORDER BY id;

-- '-' operators
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, a-b v1, a-c v2, a-e v3,
           c-b v4, c-d v5, c-e v6,
           e-b v7, e-d v8, e-f v9
  INTO test11g
  FROM rt_float
 WHERE y < 0;
SELECT id, a-b v1, a-c v2, a-e v3,
           c-b v4, c-d v5, c-e v6,
           e-b v7, e-d v8, e-f v9
  INTO test11g
  FROM rt_float
 WHERE y < 0;
SET pg_strom.enabled = off;
SELECT id, a-b v1, a-c v2, a-e v3,
           c-b v4, c-d v5, c-e v6,
           e-b v7, e-d v8, e-f v9
  INTO test11p
  FROM rt_float
 WHERE y < 0;
(SELECT * FROM test11g EXCEPT ALL SELECT * FROM test11p) ORDER BY id;
(SELECT * FROM test11p EXCEPT ALL SELECT * FROM test11g) ORDER BY id;

-- '*' operators
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, a*b v1, a*c v2, a*e v3,
           c*b v4, c*d v5, c*e v6,
           e*b v7, e*d v8, e*f v9
  INTO test12g
  FROM rt_float
 WHERE y > 0;
SELECT id, a*b v1, a*c v2, a*e v3,
           c*b v4, c*d v5, c*e v6,
           e*b v7, e*d v8, e*f v9
  INTO test12g
  FROM rt_float
 WHERE y > 0;
SET pg_strom.enabled = off;
SELECT id, a*b v1, a*c v2, a*e v3,
           c*b v4, c*d v5, c*e v6,
           e*b v7, e*d v8, e*f v9
  INTO test12p
  FROM rt_float
 WHERE y > 0;
(SELECT * FROM test12g EXCEPT ALL SELECT * FROM test12p) ORDER BY id;
(SELECT * FROM test12p EXCEPT ALL SELECT * FROM test12g) ORDER BY id;

-- '/' operators
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, a/b v1, a/d v2, a/f v3,
           c/b v4, c/d v5, c/f v6,
           e/b v7, e/d v8, e/f v9
  INTO test13g
  FROM rt_float
 WHERE z BETWEEN -16000 AND 16000
   AND b != 0.0 AND d != 0.0 AND f != 0.0;
SELECT id, a/b v1, a/d v2, a/f v3,
           c/b v4, c/d v5, c/f v6,
           e/b v7, e/d v8, e/f v9
  INTO test13g
  FROM rt_float
 WHERE z BETWEEN -16000 AND 16000
   AND b != 0.0 AND d != 0.0 AND f != 0.0;
SET pg_strom.enabled = off;
SELECT id, a/b v1, a/d v2, a/f v3,
           c/b v4, c/d v5, c/f v6,
           e/b v7, e/d v8, e/f v9
  INTO test13p
  FROM rt_float
 WHERE z BETWEEN -16000 AND 16000
   AND b != 0.0 AND d != 0.0 AND f != 0.0;
(SELECT * FROM test13g EXCEPT ALL SELECT * FROM test13p) ORDER BY id;
(SELECT * FROM test13p EXCEPT ALL SELECT * FROM test13g) ORDER BY id;

-- unary operators ('+','-','@')
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, +a v1, -b v2, @(a+b) v3,
           +c v4, -d v5, @(c+d) v6,
           +e v7, -f v8, @(e+f) v9
  INTO test14g
  FROM rt_float
 WHERE z < 0;
SELECT id, +a v1, -b v2, @(a+b) v3,
           +c v4, -d v5, @(c+d) v6,
           +e v7, -f v8, @(e+f) v9
  INTO test14g
  FROM rt_float
 WHERE z < 0;
SET pg_strom.enabled = off;
SELECT id, +a v1, -b v2, @(a+b) v3,
           +c v4, -d v5, @(c+d) v6,
           +e v7, -f v8, @(e+f) v9
  INTO test14p
  FROM rt_float
 WHERE z < 0;
(SELECT * FROM test14g EXCEPT ALL SELECT * FROM test14p) ORDER BY id;
(SELECT * FROM test14p EXCEPT ALL SELECT * FROM test14g) ORDER BY id;

-- '='  : equal operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a=b v1, a=d v2, a=f v3,
           c=b v4, c=d v5, c=f v6,
           e=b v7, e=d v8, e=f v9
  INTO test20g
  FROM rt_float
 WHERE id % 6 = 0;
SELECT id, a=b v1, a=d v2, a=f v3,
           c=b v4, c=d v5, c=f v6,
           e=b v7, e=d v8, e=f v9
  INTO test20g
  FROM rt_float
 WHERE id % 6 = 0;
SET pg_strom.enabled = off;
SELECT id, a=b v1, a=d v2, a=f v3,
           c=b v4, c=d v5, c=f v6,
           e=b v7, e=d v8, e=f v9
  INTO test20p
  FROM rt_float
 WHERE id % 6 = 0;
(SELECT * FROM test20g EXCEPT ALL SELECT * FROM test20p) ORDER BY id;
(SELECT * FROM test20p EXCEPT ALL SELECT * FROM test20g) ORDER BY id;

-- '<>' : not equal operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a<>b v1, a<>d v2, a<>f v3,
           c<>b v4, c<>d v5, c<>f v6,
           e<>b v7, e<>d v8, e<>f v9
  INTO test21g
  FROM rt_float
 WHERE id % 6 = 1;
SELECT id, a<>b v1, a<>d v2, a<>f v3,
           c<>b v4, c<>d v5, c<>f v6,
           e<>b v7, e<>d v8, e<>f v9
  INTO test21g
  FROM rt_float
 WHERE id % 6 = 1;
SET pg_strom.enabled = off;
SELECT id, a<>b v1, a<>d v2, a<>f v3,
           c<>b v4, c<>d v5, c<>f v6,
           e<>b v7, e<>d v8, e<>f v9
  INTO test21p
  FROM rt_float
 WHERE id % 6 = 1;
(SELECT * FROM test21g EXCEPT ALL SELECT * FROM test21p) ORDER BY id;
(SELECT * FROM test21p EXCEPT ALL SELECT * FROM test21g) ORDER BY id;

-- '>'  : greater than operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a>b v1, a>d v2, a>f v3,
           c>b v4, c>d v5, c>f v6,
           e>b v7, e>d v8, e>f v9
  INTO test22g
  FROM rt_float
 WHERE id % 6 = 2;
SELECT id, a>b v1, a>d v2, a>f v3,
           c>b v4, c>d v5, c>f v6,
           e>b v7, e>d v8, e>f v9
  INTO test22g
  FROM rt_float
 WHERE id % 6 = 2;
SET pg_strom.enabled = off;
SELECT id, a>b v1, a>d v2, a>f v3,
           c>b v4, c>d v5, c>f v6,
           e>b v7, e>d v8, e>f v9
  INTO test22p
  FROM rt_float
 WHERE id % 6 = 2;
(SELECT * FROM test22g EXCEPT ALL SELECT * FROM test22p) ORDER BY id;
(SELECT * FROM test22p EXCEPT ALL SELECT * FROM test22g) ORDER BY id;

-- '<'  : less than operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a<b v1, a<d v2, a<f v3,
           c<b v4, c<d v5, c<f v6,
           e<b v7, e<d v8, e<f v9
  INTO test23g
  FROM rt_float
 WHERE id % 6 = 3;
SELECT id, a<b v1, a<d v2, a<f v3,
           c<b v4, c<d v5, c<f v6,
           e<b v7, e<d v8, e<f v9
  INTO test23g
  FROM rt_float
 WHERE id % 6 = 3;
SET pg_strom.enabled = off;
SELECT id, a<b v1, a<d v2, a<f v3,
           c<b v4, c<d v5, c<f v6,
           e<b v7, e<d v8, e<f v9
  INTO test23p
  FROM rt_float
 WHERE id % 6 = 3;
(SELECT * FROM test23g EXCEPT ALL SELECT * FROM test23p) ORDER BY id;
(SELECT * FROM test23p EXCEPT ALL SELECT * FROM test23g) ORDER BY id;

-- '>=' : equal operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a>=b v1, a>=d v2, a>=f v3,
           c>=b v4, c>=d v5, c>=f v6,
           e>=b v7, e>=d v8, e>=f v9
  INTO test24g
  FROM rt_float
 WHERE id % 6 = 4;
SELECT id, a>=b v1, a>=d v2, a>=f v3,
           c>=b v4, c>=d v5, c>=f v6,
           e>=b v7, e>=d v8, e>=f v9
  INTO test24g
  FROM rt_float
 WHERE id % 6 = 4;
SET pg_strom.enabled = off;
SELECT id, a>=b v1, a>=d v2, a>=f v3,
           c>=b v4, c>=d v5, c>=f v6,
           e>=b v7, e>=d v8, e>=f v9
  INTO test24p
  FROM rt_float
 WHERE id % 6 = 4;
(SELECT * FROM test24g EXCEPT ALL SELECT * FROM test24p) ORDER BY id;
(SELECT * FROM test24p EXCEPT ALL SELECT * FROM test24g) ORDER BY id;

-- '<=' : equal operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a<=b v1, a<=d v2, a<=f v3,
           c<=b v4, c<=d v5, c<=f v6,
           e<=b v7, e<=d v8, e<=f v9
  INTO test25g
  FROM rt_float
 WHERE id % 6 = 5;
SELECT id, a<=b v1, a<=d v2, a<=f v3,
           c<=b v4, c<=d v5, c<=f v6,
           e<=b v7, e<=d v8, e<=f v9
  INTO test25g
  FROM rt_float
 WHERE id % 6 = 5;
SET pg_strom.enabled = off;
SELECT id, a<=b v1, a<=d v2, a<=f v3,
           c<=b v4, c<=d v5, c<=f v6,
           e<=b v7, e<=d v8, e<=f v9
  INTO test25p
  FROM rt_float
 WHERE id % 6 = 5;
(SELECT * FROM test25g EXCEPT ALL SELECT * FROM test25p) ORDER BY id;
(SELECT * FROM test25p EXCEPT ALL SELECT * FROM test25g) ORDER BY id;

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA regtest_dtype_float_temp CASCADE;
