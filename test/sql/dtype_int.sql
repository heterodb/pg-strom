---
--- Micro test cases for integer operators / functions
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dtype_int_temp CASCADE;
CREATE SCHEMA regtest_dtype_int_temp;
RESET client_min_messages;

SET search_path = regtest_dtype_int_temp,public;
CREATE TABLE rt_int (
  id   int,
  a    int2,
  b    int2,
  c    int4,
  d    int4,
  e    int8,
  f    int8,
  x    float2,
  y    float4,
  z    float8
);
SELECT pgstrom.random_setseed(20190608);
INSERT INTO rt_int (
  SELECT x, pgstrom.random_int(1,  -3200,  3200),
            pgstrom.random_int(1,  -3200,  3200),
            pgstrom.random_int(1, -32000, 32000),
            pgstrom.random_int(1, -32000, 32000),
            pgstrom.random_int(1, -32000, 32000),
            pgstrom.random_int(1, -32000, 32000),
            pgstrom.random_float(0.5, -32000.0, 32000.0),
            pgstrom.random_float(0.5, -32000.0, 32000.0),
            pgstrom.random_float(0.5, -32000.0, 32000.0)
   FROM generate_series(1,2000) x);
VACUUM ANALYZE;

-- force to use GpuScan, instead of SeqScan
SET enable_seqscan = off;
-- not to print kernel source code
SET pg_strom.debug_kernel_source = off;

-- cast operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, c::smallint, e::smallint, x::smallint, y::smallint, z::smallint
  INTO test01g
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
SELECT id, c::smallint, e::smallint, x::smallint, y::smallint, z::smallint
  INTO test01g
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
SET pg_strom.enabled = off;
SELECT id, c::smallint, e::smallint, x::smallint, y::smallint, z::smallint
  INTO test01p
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
(SELECT * FROM test01g EXCEPT ALL SELECT * FROM test01p) order by id;
(SELECT * FROM test01p EXCEPT ALL SELECT * FROM test01g) order by id;

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a::int, e::int, x::int, y::int, z::int
  INTO test02g
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
SELECT id, a::int, e::int, x::int, y::int, z::int
  INTO test02g
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
SET pg_strom.enabled = off;
SELECT id, a::int, e::int, x::int, y::int, z::int
  INTO test02p
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
(SELECT * FROM test02g EXCEPT ALL SELECT * FROM test02p);
(SELECT * FROM test02p EXCEPT ALL SELECT * FROM test02g);

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a::bigint, c::bigint, x::bigint, y::bigint, z::bigint
  INTO test03g
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
SELECT id, a::bigint, c::bigint, x::bigint, y::bigint, z::bigint
  INTO test03g
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
SET pg_strom.enabled = off;
SELECT id, a::bigint, c::bigint, x::bigint, y::bigint, z::bigint
  INTO test03p
  FROM rt_int
 WHERE x between -15000.0 AND 15000.0;
(SELECT * FROM test03g EXCEPT ALL SELECT * FROM test03p);
(SELECT * FROM test03p EXCEPT ALL SELECT * FROM test03g);

-- '+' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a+b v1, a+c v2, b+e v3,
           c+b v4, c+d v5, c+e v6,
           e+a v7, e+d v8, e+f v9
  INTO test10g
  FROM rt_int
 WHERE x >= 0;
SELECT id, a+b v1, a+c v2, b+e v3,
           c+b v4, c+d v5, c+e v6,
           e+a v7, e+d v8, e+f v9
  INTO test10g
  FROM rt_int
 WHERE x >= 0;
SET pg_strom.enabled = off;
SELECT id, a+b v1, a+c v2, b+e v3,
           c+b v4, c+d v5, c+e v6,
           e+a v7, e+d v8, e+f v9
  INTO test10p
  FROM rt_int
 WHERE x >= 0;
(SELECT * FROM test10g EXCEPT ALL SELECT * FROM test10p);
(SELECT * FROM test10p EXCEPT ALL SELECT * FROM test10g);

-- '-' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a-b v1, a-c v2, b-e v3,
           c-b v4, c-d v5, c-e v6,
           e-a v7, e-d v8, e-f v9
  INTO test11g
  FROM rt_int
 WHERE x <= 0.0;
SELECT id, a-b v1, a-c v2, b-e v3,
           c-b v4, c-d v5, c-e v6,
           e-a v7, e-d v8, e-f v9
  INTO test11g
  FROM rt_int
 WHERE x <= 0.0;
SET pg_strom.enabled = off;
SELECT id, a-b v1, a-c v2, b-e v3,
           c-b v4, c-d v5, c-e v6,
           e-a v7, e-d v8, e-f v9
  INTO test11p
  FROM rt_int
 WHERE x <= 0.0;
(SELECT * FROM test11g EXCEPT ALL SELECT * FROM test11p);
(SELECT * FROM test11p EXCEPT ALL SELECT * FROM test11g);

-- '*' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a * (b % 10) v1, a*c v2, a*e v3,
           c*b v4, c*d v5, c*e v6,
           e*b v7, e*d v8, e*f v9
  INTO test12g
  FROM rt_int
 WHERE x BETWEEN -10000.0 AND 20000.0;
SELECT id, a * (b % 10) v1, a*c v2, a*e v3,
           c*b v4, c*d v5, c*e v6,
           e*b v7, e*d v8, e*f v9
  INTO test12g
  FROM rt_int
 WHERE x BETWEEN -10000.0 AND 20000.0;
SET pg_strom.enabled = off;
SELECT id, a * (b % 10) v1, a*c v2, a*e v3,
           c*b v4, c*d v5, c*e v6,
           e*b v7, e*d v8, e*f v9
  INTO test12p
  FROM rt_int
 WHERE x BETWEEN -10000.0 AND 20000.0;
(SELECT * FROM test12g EXCEPT ALL SELECT * FROM test12p);
(SELECT * FROM test12p EXCEPT ALL SELECT * FROM test12g);

-- '/' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a/(b%100) v1, a/(d%100) v2, a/(f%100) v3,
           c/(b%100) v4, c/(d%100) v5, c/(f%100) v6,
           e/(b%100) v7, e/(d%100) v8, e/(f%100) v9
  INTO test13g
  FROM rt_int
 WHERE x BETWEEN -20000.0 AND 10000.0
   AND (b%100) <> 0 AND (d%100) <> 0 AND (f%100) <> 0;
SELECT id, a/(b%100) v1, a/(d%100) v2, a/(f%100) v3,
           c/(b%100) v4, c/(d%100) v5, c/(f%100) v6,
           e/(b%100) v7, e/(d%100) v8, e/(f%100) v9
  INTO test13g
  FROM rt_int
 WHERE x BETWEEN -20000.0 AND 10000.0
   AND (b%100) <> 0 AND (d%100) <> 0 AND (f%100) <> 0;
SET pg_strom.enabled = off;
SELECT id, a/(b%100) v1, a/(d%100) v2, a/(f%100) v3,
           c/(b%100) v4, c/(d%100) v5, c/(f%100) v6,
           e/(b%100) v7, e/(d%100) v8, e/(f%100) v9
  INTO test13p
  FROM rt_int
 WHERE x BETWEEN -20000.0 AND 10000.0
   AND (b%100) <> 0 AND (d%100) <> 0 AND (f%100) <> 0;
(SELECT * FROM test13g EXCEPT ALL SELECT * FROM test13p);
(SELECT * FROM test13p EXCEPT ALL SELECT * FROM test13g);

-- '%' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a % 100::smallint v1, c % 1000::int v2, d % 1000::bigint v3
  INTO test14g
  FROM rt_int
 WHERE y BETWEEN -15000.0 AND 15000.0;
SELECT id, a % 100::smallint v1, c % 1000::int v2, d % 1000::bigint v3
  INTO test14g
  FROM rt_int
 WHERE y BETWEEN -15000.0 AND 15000.0;
SET pg_strom.enabled = off;
SELECT id, a % 100::smallint v1, c % 1000::int v2, d % 1000::bigint v3
  INTO test14p
  FROM rt_int
 WHERE y BETWEEN -15000.0 AND 15000.0;
(SELECT * FROM test14g EXCEPT ALL SELECT * FROM test14p);
(SELECT * FROM test14p EXCEPT ALL SELECT * FROM test14g);

-- unary operators ('+', '-' and '@')
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, +a v1, -b v2, @(a+b) v3,
           +c v4, -d v5, @(c+d) v6,
           +e v7, -f v8, @(e+f) v9
  INTO test15g
  FROM rt_int
 WHERE y BETWEEN -10000.0 AND 20000.0;
SELECT id, +a v1, -b v2, @(a+b) v3,
           +c v4, -d v5, @(c+d) v6,
           +e v7, -f v8, @(e+f) v9
  INTO test15g
  FROM rt_int
 WHERE y BETWEEN -10000.0 AND 20000.0;
SET pg_strom.enabled = off;
SELECT id, +a v1, -b v2, @(a+b) v3,
           +c v4, -d v5, @(c+d) v6,
           +e v7, -f v8, @(e+f) v9
  INTO test15p
  FROM rt_int
 WHERE y BETWEEN -10000.0 AND 20000.0;
(SELECT * FROM test15g EXCEPT ALL SELECT * FROM test15p);
(SELECT * FROM test15p EXCEPT ALL SELECT * FROM test15g);

-- '=' : equal operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a=b v1, a=d v2, a=f v3,
           c=b v4, c=d v5, c=f v6,
           e=b v7, e=d v8, e=f v9
  INTO test20g
  FROM rt_int
 WHERE id % 6 = 0;
SELECT id, a=b v1, a=d v2, a=f v3,
           c=b v4, c=d v5, c=f v6,
           e=b v7, e=d v8, e=f v9
  INTO test20g
  FROM rt_int
 WHERE id % 6 = 0;
SET pg_strom.enabled = off;
SELECT id, a=b v1, a=d v2, a=f v3,
           c=b v4, c=d v5, c=f v6,
           e=b v7, e=d v8, e=f v9
  INTO test20p
  FROM rt_int
 WHERE id % 6 = 0;
(SELECT * FROM test20g EXCEPT ALL SELECT * FROM test20p);
(SELECT * FROM test20p EXCEPT ALL SELECT * FROM test20g);

-- '<>' : not equal operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a<>b v1, a<>d v2, a<>f v3,
           c<>b v4, c<>d v5, c<>f v6,
           e<>b v7, e<>d v8, e<>f v9
  INTO test21g
  FROM rt_int
 WHERE id % 6 = 1;
SELECT id, a<>b v1, a<>d v2, a<>f v3,
           c<>b v4, c<>d v5, c<>f v6,
           e<>b v7, e<>d v8, e<>f v9
  INTO test21g
  FROM rt_int
 WHERE id % 6 = 1;
SET pg_strom.enabled = off;
SELECT id, a<>b v1, a<>d v2, a<>f v3,
           c<>b v4, c<>d v5, c<>f v6,
           e<>b v7, e<>d v8, e<>f v9
  INTO test21p
  FROM rt_int
 WHERE id % 6 = 1;
(SELECT * FROM test21g EXCEPT ALL SELECT * FROM test21p);
(SELECT * FROM test21p EXCEPT ALL SELECT * FROM test21g);

-- '>' : greater than operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a>b v1, a>d v2, a>f v3,
           c>b v4, c>d v5, c>f v6,
           e>b v7, e>d v8, e>f v9
  INTO test22g
  FROM rt_int
 WHERE id % 6 = 2;
SELECT id, a>b v1, a>d v2, a>f v3,
           c>b v4, c>d v5, c>f v6,
           e>b v7, e>d v8, e>f v9
  INTO test22g
  FROM rt_int
 WHERE id % 6 = 2;
SET pg_strom.enabled = off;
SELECT id, a>b v1, a>d v2, a>f v3,
           c>b v4, c>d v5, c>f v6,
           e>b v7, e>d v8, e>f v9
  INTO test22p
  FROM rt_int
 WHERE id % 6 = 2;
(SELECT * FROM test22g EXCEPT ALL SELECT * FROM test22p);
(SELECT * FROM test22p EXCEPT ALL SELECT * FROM test22g);

-- '<' : less than operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a<b v1, a<d v2, a<f v3,
           c<b v4, c<d v5, c<f v6,
           e<b v7, e<d v8, e<f v9
  INTO test23g
  FROM rt_int
 WHERE id % 6 = 3;
SELECT id, a<b v1, a<d v2, a<f v3,
           c<b v4, c<d v5, c<f v6,
           e<b v7, e<d v8, e<f v9
  INTO test23g
  FROM rt_int
 WHERE id % 6 = 3;
SET pg_strom.enabled = off;
SELECT id, a<b v1, a<d v2, a<f v3,
           c<b v4, c<d v5, c<f v6,
           e<b v7, e<d v8, e<f v9
  INTO test23p
  FROM rt_int
 WHERE id % 6 = 3;
(SELECT * FROM test23g EXCEPT ALL SELECT * FROM test23p);
(SELECT * FROM test23p EXCEPT ALL SELECT * FROM test23g);

-- '>=' : greater than or equal to operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a>=b v1, a>=d v2, a>=f v3,
           c>=b v4, c>=d v5, c>=f v6,
           e>=b v7, e>=d v8, e>=f v9
  INTO test24g
  FROM rt_int
 WHERE id % 6 = 4;
SELECT id, a>=b v1, a>=d v2, a>=f v3,
           c>=b v4, c>=d v5, c>=f v6,
           e>=b v7, e>=d v8, e>=f v9
  INTO test24g
  FROM rt_int
 WHERE id % 6 = 4;
SET pg_strom.enabled = off;
SELECT id, a>=b v1, a>=d v2, a>=f v3,
           c>=b v4, c>=d v5, c>=f v6,
           e>=b v7, e>=d v8, e>=f v9
  INTO test24p
  FROM rt_int
 WHERE id % 6 = 4;
(SELECT * FROM test24g EXCEPT ALL SELECT * FROM test24p);
(SELECT * FROM test24p EXCEPT ALL SELECT * FROM test24g);

-- '<=' : less than or equal to operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a<=b v1, a<=d v2, a<=f v3,
           c<=b v4, c<=d v5, c<=f v6,
           e<=b v7, e<=d v8, e<=f v9
  INTO test25g
  FROM rt_int
 WHERE id % 6 = 5;
SELECT id, a<=b v1, a<=d v2, a<=f v3,
           c<=b v4, c<=d v5, c<=f v6,
           e<=b v7, e<=d v8, e<=f v9
  INTO test25g
  FROM rt_int
 WHERE id % 6 = 5;
SET pg_strom.enabled = off;
SELECT id, a<=b v1, a<=d v2, a<=f v3,
           c<=b v4, c<=d v5, c<=f v6,
           e<=b v7, e<=d v8, e<=f v9
  INTO test25p
  FROM rt_int
 WHERE id % 6 = 5;
(SELECT * FROM test25g EXCEPT ALL SELECT * FROM test25p);
(SELECT * FROM test25p EXCEPT ALL SELECT * FROM test25g);

-- bitwise operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a&b v1, c&d v2, e&f v3,	-- bitwise AND
           a|b v4, c|d v5, e|f v6,	-- bitwise OR
           a#b v7, c#d v8, e#f v9,	-- bitwise XOR
           ~a v10, ~c v11, ~e v12	-- bitwise NOT
  INTO test30g
  FROM rt_int
 WHERE z BETWEEN -20000.0 AND 20000.0;
SELECT id, a&b v1, c&d v2, e&f v3,	-- bitwise AND
           a|b v4, c|d v5, e|f v6,	-- bitwise OR
           a#b v7, c#d v8, e#f v9,	-- bitwise XOR
           ~a v10, ~c v11, ~e v12	-- bitwise NOT
  INTO test30g
  FROM rt_int
 WHERE z BETWEEN -20000.0 AND 20000.0;
SET pg_strom.enabled = off;
SELECT id, a&b v1, c&d v2, e&f v3,	-- bitwise AND
           a|b v4, c|d v5, e|f v6,	-- bitwise OR
           a#b v7, c#d v8, e#f v9,	-- bitwise XOR
           ~a v10, ~c v11, ~e v12	-- bitwise NOT
  INTO test30p
  FROM rt_int
 WHERE z BETWEEN -20000.0 AND 20000.0;
(SELECT * FROM test30g EXCEPT ALL SELECT * FROM test30p);
(SELECT * FROM test30p EXCEPT ALL SELECT * FROM test30g);

-- bit shift
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, a >> (id%10) v1, c >> (id%12) v2, e >> (id%12) v3,
           b << (id%5)  v4, d << (id%7)  v5, f << (id%8)  v6
  INTO test31g
  FROM rt_int
 WHERE z > -10000.0;
SELECT id, a >> (id%10) v1, c >> (id%12) v2, e >> (id%12) v3,
           b << (id%5)  v4, d << (id%7)  v5, f << (id%8)  v6
  INTO test31g
  FROM rt_int
 WHERE z > -10000.0;
SET pg_strom.enabled = off;
SELECT id, a >> (id%10) v1, c >> (id%12) v2, e >> (id%12) v3,
           b << (id%5)  v4, d << (id%7)  v5, f << (id%8)  v6
  INTO test31p
  FROM rt_int
 WHERE z > -10000.0;
(SELECT * FROM test31g EXCEPT ALL SELECT * FROM test31p);
(SELECT * FROM test31p EXCEPT ALL SELECT * FROM test31g);

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA regtest_dtype_int_temp CASCADE;
