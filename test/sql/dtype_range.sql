---
--- Micro test case for range type operators / functions
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dtype_range_temp CASCADE;
CREATE SCHEMA regtest_dtype_range_temp;
RESET client_min_messages;

SET search_path = regtest_dtype_range_temp,public;
SET enable_seqscan = off;

CREATE TABLE rt_int4range (
  id    int,
  v     int4,
  x     int4range,
  y     int4range
);

CREATE TABLE rt_int8range (
  id    int,
  v     int8,
  x     int8range,
  y     int8range
);

CREATE TABLE rt_tsrange (
  id    int,
  v     timestamp,
  x     tsrange,
  y     tsrange
);

CREATE TABLE rt_tstzrange (
  id    int,
  v     timestamptz,
  x     tstzrange,
  y     tstzrange
);

CREATE TABLE rt_daterange (
  id    int,
  v     date,
  x     daterange,
  y     daterange
);

INSERT INTO rt_int4range
    (SELECT x, pgstrom.random_int(1, 0, 1000),
	           pgstrom.random_int4range(1,   0,1000),
               pgstrom.random_int4range(1, 200,1200)
       FROM generate_series(1,1500) x);

INSERT INTO rt_int8range
    (SELECT x, pgstrom.random_int(1, 0, 1000),
               pgstrom.random_int8range(1,   0,1000),
               pgstrom.random_int8range(1, 200,1200)
       FROM generate_series(1,1500) x);

INSERT INTO rt_tsrange
    (SELECT x, pgstrom.random_timestamp(1),
               pgstrom.random_tsrange(1),
               pgstrom.random_tsrange(1)
       FROM generate_series(1,1500) x);

INSERT INTO rt_tstzrange
    (SELECT x, pgstrom.random_timestamp(1),
               pgstrom.random_tstzrange(1),
               pgstrom.random_tstzrange(1)
       FROM generate_series(1,1500) x);

INSERT INTO rt_daterange
    (SELECT x, pgstrom.random_date(1),
               pgstrom.random_daterange(1),
               pgstrom.random_daterange(1)
       FROM generate_series(1,1500) x);

---
--- operators for int4range
---
EXPLAIN (verbose, costs off)
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  FROM rt_int4range
 WHERE id > 0;

SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test01g
  FROM rt_int4range
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test01p
  FROM rt_int4range
 WHERE id > 0;

SELECT * FROM test01p p FULL OUTER JOIN test01g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 OR
       p.r7 != g.r7;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  FROM rt_int4range
 WHERE id > 0;

SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test02g
  FROM rt_int4range
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test02p
  FROM rt_int4range
 WHERE id > 0;
SELECT * FROM test02p p FULL OUTER JOIN test02g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  FROM rt_int4range
 WHERE id > 0;

SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test03g
  FROM rt_int4range
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test03p
  FROM rt_int4range
 WHERE id > 0;

SELECT * FROM test03p p FULL OUTER JOIN test03g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  FROM rt_int4range
 WHERE id > 0;

SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test04g
  FROM rt_int4range
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test04p
  FROM rt_int4range
 WHERE id > 0;

SELECT * FROM test04p p FULL OUTER JOIN test04g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  FROM rt_int4range
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test05g
  FROM rt_int4range
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SET pg_strom.enabled = off;
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test05p
  FROM rt_int4range
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT * FROM test05p p FULL OUTER JOIN test05g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 ;

---
--- operators for int8range
---
EXPLAIN (verbose, costs off)
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  FROM rt_int8range
 WHERE id > 0;

SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test11g
  FROM rt_int8range
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test11p
  FROM rt_int8range
 WHERE id > 0;

SELECT * FROM test11p p FULL OUTER JOIN test11g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 OR
       p.r7 != g.r7;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  FROM rt_int8range
 WHERE id > 0;

SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test12g
  FROM rt_int8range
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test12p
  FROM rt_int8range
 WHERE id > 0;
SELECT * FROM test12p p FULL OUTER JOIN test12g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  FROM rt_int8range
 WHERE id > 0;

SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test13g
  FROM rt_int8range
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test13p
  FROM rt_int8range
 WHERE id > 0;

SELECT * FROM test13p p FULL OUTER JOIN test13g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  FROM rt_int8range
 WHERE id > 0;

SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test14g
  FROM rt_int8range
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test14p
  FROM rt_int8range
 WHERE id > 0;

SELECT * FROM test14p p FULL OUTER JOIN test14g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  FROM rt_int8range
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test15g
  FROM rt_int8range
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SET pg_strom.enabled = off;
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test15p
  FROM rt_int8range
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT * FROM test15p p FULL OUTER JOIN test15g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 ;

---
--- operators for tsrange
---
EXPLAIN (verbose, costs off)
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  FROM rt_tsrange
 WHERE id > 0;

SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test21g
  FROM rt_tsrange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test21p
  FROM rt_tsrange
 WHERE id > 0;

SELECT * FROM test21p p FULL OUTER JOIN test21g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 OR
       p.r7 != g.r7;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  FROM rt_tsrange
 WHERE id > 0;

SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test22g
  FROM rt_tsrange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test22p
  FROM rt_tsrange
 WHERE id > 0;
SELECT * FROM test22p p FULL OUTER JOIN test22g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  FROM rt_tsrange
 WHERE id > 0;

SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test23g
  FROM rt_tsrange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test23p
  FROM rt_tsrange
 WHERE id > 0;

SELECT * FROM test23p p FULL OUTER JOIN test23g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  FROM rt_tsrange
 WHERE id > 0;

SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test24g
  FROM rt_tsrange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test24p
  FROM rt_tsrange
 WHERE id > 0;

SELECT * FROM test24p p FULL OUTER JOIN test24g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  FROM rt_tsrange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test25g
  FROM rt_tsrange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SET pg_strom.enabled = off;
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test25p
  FROM rt_tsrange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT * FROM test25p p FULL OUTER JOIN test25g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 ;

---
--- operators for tstzrange
---
EXPLAIN (verbose, costs off)
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  FROM rt_tstzrange
 WHERE id > 0;

SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test31g
  FROM rt_tstzrange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test31p
  FROM rt_tstzrange
 WHERE id > 0;

SELECT * FROM test31p p FULL OUTER JOIN test31g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 OR
       p.r7 != g.r7;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  FROM rt_tstzrange
 WHERE id > 0;

SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test32g
  FROM rt_tstzrange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test32p
  FROM rt_tstzrange
 WHERE id > 0;
SELECT * FROM test32p p FULL OUTER JOIN test32g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  FROM rt_tstzrange
 WHERE id > 0;

SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test33g
  FROM rt_tstzrange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test33p
  FROM rt_tstzrange
 WHERE id > 0;

SELECT * FROM test33p p FULL OUTER JOIN test33g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  FROM rt_tstzrange
 WHERE id > 0;

SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test34g
  FROM rt_tstzrange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test34p
  FROM rt_tstzrange
 WHERE id > 0;

SELECT * FROM test34p p FULL OUTER JOIN test34g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  FROM rt_tstzrange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test35g
  FROM rt_tstzrange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SET pg_strom.enabled = off;
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test35p
  FROM rt_tstzrange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT * FROM test35p p FULL OUTER JOIN test35g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 ;

---
--- operators for daterange
---
EXPLAIN (verbose, costs off)
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  FROM rt_daterange
 WHERE id > 0;

SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test41g
  FROM rt_daterange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x,
       lower(x) r1, upper(x) r2, isempty(x) r3,
       lower_inc(x) r4, upper_inc(x) r5, lower_inf(x) r6, upper_inf(x) r7
  INTO test41p
  FROM rt_daterange
 WHERE id > 0;

SELECT * FROM test41p p FULL OUTER JOIN test41g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 OR
       p.r7 != g.r7;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  FROM rt_daterange
 WHERE id > 0;

SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test42g
  FROM rt_daterange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x = y  r1,  x <> y r2,  x <  y r3,
	   y <= x r4,  x >  y r5,  y >= x r6
  INTO test42p
  FROM rt_daterange
 WHERE id > 0;
SELECT * FROM test42p p FULL OUTER JOIN test42g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 OR
       p.r6 != g.r6 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  FROM rt_daterange
 WHERE id > 0;

SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test43g
  FROM rt_daterange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, v, x, y,
       x @> y  r1,  y @> v  r2, y <@ x  r3,  v <@ x  r4, x && y  r5
  INTO test43p
  FROM rt_daterange
 WHERE id > 0;

SELECT * FROM test43p p FULL OUTER JOIN test43g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  FROM rt_daterange
 WHERE id > 0;

SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test44g
  FROM rt_daterange
 WHERE id > 0;

SET pg_strom.enabled = off;
SELECT id, x, y,
       x << y  r1,  y >> y  r2, x &< y r3, x &> y r4, x -|- y r5
  INTO test44p
  FROM rt_daterange
 WHERE id > 0;

SELECT * FROM test44p p FULL OUTER JOIN test44g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 OR
       p.r4 != g.r4 OR
       p.r5 != g.r5 ;

RESET pg_strom.enabled;
EXPLAIN (verbose, costs off)
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  FROM rt_daterange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test45g
  FROM rt_daterange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SET pg_strom.enabled = off;
SELECT id, x, y, x + y r1, x * y r2, x - y r3
  INTO test45p
  FROM rt_daterange
 WHERE upper(x) < upper(y)
   AND lower(x) < lower(y)
   AND upper(x) > lower(y);

SELECT * FROM test45p p FULL OUTER JOIN test45g g ON p.id = g.id
 WHERE p.r1 != g.r1 OR
       p.r2 != g.r2 OR
       p.r3 != g.r3 ;
