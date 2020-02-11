--
-- test for mathematical / trigonometric functions
--
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dfunc_math_temp CASCADE;
CREATE SCHEMA regtest_dfunc_math_temp;
RESET client_min_messages;

SET search_path = regtest_dfunc_math_temp,public;
CREATE TABLE rt_data (
  id   int,
  a    float8,   -- values between -1.0 and 1.0
  b    float8,   -- values between -10.0 and 10.0
  c    float8,   -- values between -100.0 and 100.0
  d    float8,   -- values between -1000.0 and 1000.0
  e    float2,
  f    float4,
  r    float8,   -- radian for trigonometric functions
  x    int2,
  y    int4,
  z    int8
);
SELECT pgstrom.random_setseed(20190610);
INSERT INTO rt_data (
  SELECT x, pgstrom.random_float(1, -1.0, 1.0),
            pgstrom.random_float(1, -10.0, 10.0),
            pgstrom.random_float(1, -100.0, 100.0),
            pgstrom.random_float(1, -1000.0, 1000.0),
            pgstrom.random_float(1, -1000.0, 1000.0),	-- float2
            pgstrom.random_float(1, -1000.0, 1000.0),	-- float4
			pgstrom.random_float(1, -2 * pi(), 2 * pi()), -- radian
            pgstrom.random_int(1, -32000, 32000),		-- int2
            pgstrom.random_int(1, -32000, 32000),		-- int4
            pgstrom.random_int(1, -32000, 32000)		-- int8
    FROM generate_series(1,2000) x);
VACUUM ANALYZE;

-- force to use GpuScan, instead of SeqScan
SET enable_seqscan = off;
-- not to print kernel source code
SET pg_strom.debug_kernel_source = off;

-- PG12 changed default of extra_float_digits, so it affects to number of
-- digits of float values.
SET extra_float_digits = 1;

-- absolute values
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, abs(e) f2, abs(f) f4, abs(a) f8, abs(x) i2, abs(y) i4, abs(z) i8
  INTO test01g
  FROM rt_data
 WHERE c BETWEEN -80.0 AND 80.0;
SELECT id, abs(e) f2, abs(f) f4, abs(a) f8, abs(x) i2, abs(y) i4, abs(z) i8
  INTO test01g
  FROM rt_data
 WHERE c BETWEEN -80.0 AND 80.0;
SET pg_strom.enabled = off;
SELECT id, abs(e) f2, abs(f) f4, abs(a) f8, abs(x) i2, abs(y) i4, abs(z) i8
  INTO test01p
  FROM rt_data
 WHERE c BETWEEN -80.0 AND 80.0;
(SELECT * FROM test01g EXCEPT ALL SELECT * FROM test01p) order by id;
(SELECT * FROM test01p EXCEPT ALL SELECT * FROM test01g) order by id;

-- mathmatical functions
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, round(b), dround(c),
           ceil(d), ceiling(d),
           floor(d), trunc(c), dtrunc(d)
  INTO test02g
  FROM rt_data
 WHERE c BETWEEN -40 AND 120;
SELECT id, round(b), dround(c),
           ceil(d), ceiling(d),
           floor(d), trunc(c), dtrunc(d)
  INTO test02g
  FROM rt_data
 WHERE c BETWEEN -40 AND 120;
SET pg_strom.enabled = off;
SELECT id, round(b), dround(c),
           ceil(d), ceiling(d),
           floor(d), trunc(c), dtrunc(d)
  INTO test02p
  FROM rt_data
 WHERE c BETWEEN -40 AND 120;
(SELECT * FROM test02g EXCEPT ALL SELECT * FROM test02p) order by id;
(SELECT * FROM test02p EXCEPT ALL SELECT * FROM test02g) order by id;

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, exp(a), dexp(b), ln(@c), dlog1(@d), log(@c), dlog10(@d),
           power(@b, a), pow(@c, a), dpow(@d, a)
  INTO test03g
  FROM rt_data
 WHERE c > 0;
SELECT id, exp(a), dexp(b), ln(@c), dlog1(@d), log(@c), dlog10(@d),
           power(@b, a), pow(@c, a), dpow(@d, a)
  INTO test03g
  FROM rt_data
 WHERE c > 0;
SET pg_strom.enabled = off;
SELECT id, exp(a), dexp(b), ln(@c), dlog1(@d), log(@c), dlog10(@d),
           power(@b, a), pow(@c, a), dpow(@d, a)
  INTO test03p
  FROM rt_data
 WHERE c > 0;

SELECT g.*, p.*
  FROM test03g g JOIN test03p p ON g.id = p.id
 WHERE @(g.exp    - p.exp)    > 0.000001
    OR @(g.dexp   - p.dexp)   > 0.000001
    OR @(g.ln     - p.ln)     > 0.000001
    OR @(g.dlog1  - p.dlog1)  > 0.000001
    OR @(g.log    - p.log)    > 0.000001
    OR @(g.dlog10 - p.dlog10) > 0.000001
    OR @(g.power  - p.power)  > 0.000001
    OR @(g.pow    - p.pow)    > 0.000001
    OR @(g.dpow   - p.dpow)   > 0.000001;

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, sign(a), sqrt(@c), dsqrt(@d), cbrt(d), dcbrt(d)
  INTO test04g
  FROM rt_data
 WHERE d < 0;
SELECT id, sign(a), sqrt(@c), dsqrt(@d), cbrt(d), dcbrt(d)
  INTO test04g
  FROM rt_data
 WHERE d < 0;
SET pg_strom.enabled = off;
SELECT id, sign(a), sqrt(@c), dsqrt(@d), cbrt(d), dcbrt(d)
  INTO test04p
  FROM rt_data
 WHERE d < 0;

SELECT g.*, p.*
  FROM test04g g JOIN test04p p ON g.id = p.id
 WHERE g.sign != p.sign
    OR @(g.sqrt  - p.sqrt)  > 0.000001
    OR @(g.dsqrt - p.dsqrt) > 0.000001
    OR @(g.cbrt  - p.cbrt)  > 0.000001
    OR @(g.dcbrt - p.dcbrt) > 0.000001;

-- trigonometric function
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, pi(), degrees(r), radians(d), cos(r), cot(r), sin(r), tan(r)
  INTO test05g
  FROM rt_data
 WHERE r > -360.0 AND r < 360.0;
SELECT id, pi(), degrees(r), radians(d), cos(r), cot(r), sin(r), tan(r)
  INTO test05g
  FROM rt_data
 WHERE r > -360.0 AND r < 360.0;
SET pg_strom.enabled = off;
SELECT id, pi(), degrees(r), radians(d), cos(r), cot(r), sin(r), tan(r)
  INTO test05p
  FROM rt_data
 WHERE r > -360.0 AND r < 360.0;

SELECT g.*, p.*
  FROM test05g g JOIN test05p p ON g.id = p.id
 WHERE g.pi != p.pi
    OR @(g.degrees - p.degrees) > 0.000001
    OR @(g.radians - p.radians) > 0.000001
    OR @(g.cos     - p.cos)     > 0.000001
    OR @(g.cot     - p.cot)     > 0.000001
    OR @(g.sin     - p.sin)     > 0.000001
    OR @(g.tan     - p.tan)     > 0.000001;

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, acos(a), asin(a), atan(b), atan2(d,f)
  INTO test06g
  FROM rt_data
 WHERE r > -360.0 AND r < 360.0;
SELECT id, acos(a), asin(a), atan(b), atan2(d,f)
  INTO test06g
  FROM rt_data
 WHERE r > -360.0 AND r < 360.0;
SET pg_strom.enabled = off;
SELECT id, acos(a), asin(a), atan(b), atan2(d,f)
  INTO test06p
  FROM rt_data
 WHERE r > -360.0 AND r < 360.0;

SELECT g.*, p.*
  FROM test06g g JOIN test06p p ON g.id = p.id
 WHERE @(g.acos  - p.acos)  > 0.000001
    OR @(g.asin  - p.asin)  > 0.000001
    OR @(g.atan  - p.atan)  > 0.000001
    OR @(g.atan2 - p.atan2) > 0.000001;

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA regtest_dfunc_math_temp CASCADE;
