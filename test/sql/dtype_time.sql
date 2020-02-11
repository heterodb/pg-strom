---
--- Micro test cases for date & time operators / functions
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dtype_time_temp CASCADE;
CREATE SCHEMA regtest_dtype_time_temp;
RESET client_min_messages;

SET search_path = regtest_dtype_time_temp,public;
CREATE TABLE rt_datetime (
  id      int,
  d1      date,
  d2      date,
  t1      time,
  t2      time,
  t3      time,
  t4      time,
  tz1     timetz,
  tz2     timetz,
  tz3     timetz,
  tz4     timetz,
  ts1     timestamp,
  ts2     timestamp,
  ts3     timestamp,
  ts4     timestamp,
  tsz1    timestamptz,
  tsz2    timestamptz,
  tsz3    timestamptz,
  tsz4    timestamptz,
  iv1     interval,
  iv2     interval,
  ival    int
);
SELECT pgstrom.random_setseed(20190613);
INSERT INTO rt_datetime (
  SELECT x, pgstrom.random_date(1.0),
            pgstrom.random_date(1.0),
            pgstrom.random_time(1.0),
            pgstrom.random_time(1.0),
            pgstrom.random_time(1.0),
            pgstrom.random_time(1.0),
            pgstrom.random_timetz(1.0),
            pgstrom.random_timetz(1.0),
            pgstrom.random_timetz(1.0),
            pgstrom.random_timetz(1.0),
            pgstrom.random_timestamp(1.0),
            pgstrom.random_timestamp(1.0),
            pgstrom.random_timestamp(1.0),
            pgstrom.random_timestamp(1.0),
            pgstrom.random_timestamp(1.0),
            pgstrom.random_timestamp(1.0),
            pgstrom.random_timestamp(1.0),
            pgstrom.random_timestamp(1.0),
            pgstrom.random_timestamp(0.5) - pgstrom.random_timestamp(0.5),
            pgstrom.random_timestamp(0.5) - pgstrom.random_timestamp(0.5),
            pgstrom.random_int(1,-32000,32000)
    FROM generate_series(1,3000) x);
VACUUM ANALYZE;

-- force to use GpuScan, instead of SeqScan
SET enable_seqscan = off;
-- not to print kernel source code
SET pg_strom.debug_kernel_source = off;

-- type cast operators
SET timezone = 'Japan';

SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, d1::timestamp v1, d2::timestamptz v2, t1::timetz v3,
           ts1::date v4, ts2::time v5, ts1::timestamptz v6,
           tsz1::date v7, tsz1::time v8, tsz2::timetz v9
  INTO test01g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, d1::timestamp v1, d2::timestamptz v2, t1::timetz v3,
           ts1::date v4, ts2::time v5, ts1::timestamptz v6,
           tsz1::date v7, tsz1::time v8, tsz2::timetz v9
  INTO test01g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, d1::timestamp v1, d2::timestamptz v2, t1::timetz v3,
           ts1::date v4, ts2::time v5, ts1::timestamptz v6,
           tsz1::date v7, tsz1::time v8, tsz2::timetz v9
  INTO test01p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test01g EXCEPT SELECT * FROM test01p) ORDER BY id;
(SELECT * FROM test01p EXCEPT SELECT * FROM test01g) ORDER BY id;

-- type cast operators (different timezone)
SET timezone = 'CET';
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, d1::timestamp v1, d2::timestamptz v2, t1::timetz v3,
           ts1::date v4, ts2::time v5, ts1::timestamptz v6,
           tsz1::date v7, tsz1::time v8, tsz2::timetz v9
  INTO test02g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, d1::timestamp v1, d2::timestamptz v2, t1::timetz v3,
           ts1::date v4, ts2::time v5, ts1::timestamptz v6,
           tsz1::date v7, tsz1::time v8, tsz2::timetz v9
  INTO test02g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, d1::timestamp v1, d2::timestamptz v2, t1::timetz v3,
           ts1::date v4, ts2::time v5, ts1::timestamptz v6,
           tsz1::date v7, tsz1::time v8, tsz2::timetz v9
  INTO test02p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test02g EXCEPT SELECT * FROM test02p) ORDER BY id;
(SELECT * FROM test02p EXCEPT SELECT * FROM test02g) ORDER BY id;

-- 'date' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, d1 + ival v1, d2 - ival v2, d1 - d2 v3,
           d1 + t1 v4, d2 + tz1 v5, timestamptz(d1, tz2) v6
  INTO test10g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, d1 + ival v1, d2 - ival v2, d1 - d2 v3,
           d1 + t1 v4, d2 + tz1 v5, timestamptz(d1, tz2) v6
  INTO test10g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, d1 + ival v1, d2 - ival v2, d1 - d2 v3,
           d1 + t1 v4, d2 + tz1 v5, timestamptz(d1, tz2) v6
  INTO test10p
  FROM rt_datetime
 WHERE id > 0;
(SELECT * FROM test10g EXCEPT SELECT * FROM test10p) ORDER BY id;
(SELECT * FROM test10p EXCEPT SELECT * FROM test10g) ORDER BY id;

-- 'time' and 'timetz' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, t1 + d2 v1, t2 - t1 v2,
           tz1 + iv1 v3, tz2 - iv2 v4
  INTO test11g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, t1 + d2 v1, t2 - t1 v2,
           tz1 + iv1 v3, tz2 - iv2 v4
  INTO test11g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, t1 + d2 v1, t2 - t1 v2,
           tz1 + iv1 v3, tz2 - iv2 v4
  INTO test11p
  FROM rt_datetime
 WHERE id > 0;
(SELECT * FROM test11g EXCEPT SELECT * FROM test11p) ORDER BY id;
(SELECT * FROM test11p EXCEPT SELECT * FROM test11g) ORDER BY id;

-- 'timestamp' and 'timestamptz' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, ts1 - ts2 v1, tsz1 + iv1 v2, tsz2 - iv2 v3
  INTO test12g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, ts1 - ts2 v1, tsz1 + iv1 v2, tsz2 - iv2 v3
  INTO test12g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, ts1 - ts2 v1, tsz1 + iv1 v2, tsz2 - iv2 v3
  INTO test12p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test12g EXCEPT SELECT * FROM test12p) ORDER BY id;
(SELECT * FROM test12p EXCEPT SELECT * FROM test12g) ORDER BY id;

-- 'interval' operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, -iv1 v1, iv1 + iv2 v2, iv2 - iv1 v3
  INTO test13g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, -iv1 v1, iv1 + iv2 v2, iv2 - iv1 v3
  INTO test13g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, -iv1 v1, iv1 + iv2 v2, iv2 - iv1 v3
  INTO test13p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test13g EXCEPT SELECT * FROM test13p) ORDER BY id;
(SELECT * FROM test13p EXCEPT SELECT * FROM test13g) ORDER BY id;

-- 'date' <COMP> 'date'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, d1 =  d2 v1, d1 <> d2 v2, d1 <  d2 v3,
           d1 <= d2 v4, d1 >  d2 v5, d1 >= d2 v6
  INTO test20g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, d1 =  d2 v1, d1 <> d2 v2, d1 <  d2 v3,
           d1 <= d2 v4, d1 >  d2 v5, d1 >= d2 v6
  INTO test20g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, d1 =  d2 v1, d1 <> d2 v2, d1 <  d2 v3,
           d1 <= d2 v4, d1 >  d2 v5, d1 >= d2 v6
  INTO test20p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test20g EXCEPT SELECT * FROM test20p) ORDER BY id;
(SELECT * FROM test20p EXCEPT SELECT * FROM test20g) ORDER BY id;

-- 'date' <COMP> 'timestamp'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, d1 =  ts2 v1, d1 <> ts2 v2, d1 <  ts2 v3,
           d1 <= ts2 v4, d1 >  ts2 v5, d1 >= ts2 v6
  INTO test21g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, d1 =  ts2 v1, d1 <> ts2 v2, d1 <  ts2 v3,
           d1 <= ts2 v4, d1 >  ts2 v5, d1 >= ts2 v6
  INTO test21g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, d1 =  ts2 v1, d1 <> ts2 v2, d1 <  ts2 v3,
           d1 <= ts2 v4, d1 >  ts2 v5, d1 >= ts2 v6
  INTO test21p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test21g EXCEPT SELECT * FROM test21p) ORDER BY id;
(SELECT * FROM test21p EXCEPT SELECT * FROM test21g) ORDER BY id;

-- 'timestamp' <COMP> 'date'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, ts1 =  d2 v1, ts1 <> d2 v2, ts1 <  d2 v3,
           ts1 <= d2 v4, ts1 >  d2 v5, ts1 >= d2 v6
  INTO test22g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, ts1 =  d2 v1, ts1 <> d2 v2, ts1 <  d2 v3,
           ts1 <= d2 v4, ts1 >  d2 v5, ts1 >= d2 v6
  INTO test22g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, ts1 =  d2 v1, ts1 <> d2 v2, ts1 <  d2 v3,
           ts1 <= d2 v4, ts1 >  d2 v5, ts1 >= d2 v6
  INTO test22p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test22g EXCEPT SELECT * FROM test22p) ORDER BY id;
(SELECT * FROM test22p EXCEPT SELECT * FROM test22g) ORDER BY id;

-- 'date' <COMP> 'timestamptz'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, d1 =  tsz2 v1, d1 <> tsz2 v2, d1 <  tsz2 v3,
           d1 <= tsz2 v4, d1 >  tsz2 v5, d1 >= tsz2 v6
  INTO test23g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, d1 =  tsz2 v1, d1 <> tsz2 v2, d1 <  tsz2 v3,
           d1 <= tsz2 v4, d1 >  tsz2 v5, d1 >= tsz2 v6
  INTO test23g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, d1 =  tsz2 v1, d1 <> tsz2 v2, d1 <  tsz2 v3,
           d1 <= tsz2 v4, d1 >  tsz2 v5, d1 >= tsz2 v6
  INTO test23p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test23g EXCEPT SELECT * FROM test23p) ORDER BY id;
(SELECT * FROM test23p EXCEPT SELECT * FROM test23g) ORDER BY id;

-- 'timestamptz' <COMP> 'date'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, tsz1 =  d2 v1, tsz1 <> d2 v2, tsz1 <  d2 v3,
           tsz1 <= d2 v4, tsz1 >  d2 v5, tsz1 >= d2 v6
  INTO test24g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, tsz1 =  d2 v1, tsz1 <> d2 v2, tsz1 <  d2 v3,
           tsz1 <= d2 v4, tsz1 >  d2 v5, tsz1 >= d2 v6
  INTO test24g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, tsz1 =  d2 v1, tsz1 <> d2 v2, tsz1 <  d2 v3,
           tsz1 <= d2 v4, tsz1 >  d2 v5, tsz1 >= d2 v6
  INTO test24p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test24g EXCEPT SELECT * FROM test24p) ORDER BY id;
(SELECT * FROM test24p EXCEPT SELECT * FROM test24g) ORDER BY id;

-- 'time' <COMP> 'time'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, t1 =  t2 v1, t1 <> t2 v2, t1 <  t2 v3,
           t1 <= t2 v4, t1 >  t2 v5, t1 >= t2 v6
  INTO test25g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, t1 =  t2 v1, t1 <> t2 v2, t1 <  t2 v3,
           t1 <= t2 v4, t1 >  t2 v5, t1 >= t2 v6
  INTO test25g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, t1 =  t2 v1, t1 <> t2 v2, t1 <  t2 v3,
           t1 <= t2 v4, t1 >  t2 v5, t1 >= t2 v6
  INTO test25p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test25g EXCEPT SELECT * FROM test25p) ORDER BY id;
(SELECT * FROM test25p EXCEPT SELECT * FROM test25g) ORDER BY id;

-- 'timetz' <COMP> 'timetz'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, tz1 =  tz2 v1, tz1 <> tz2 v2, tz1 <  tz2 v3,
           tz1 <= tz2 v4, tz1 >  tz2 v5, tz1 >= tz2 v6
  INTO test26g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, tz1 =  tz2 v1, tz1 <> tz2 v2, tz1 <  tz2 v3,
           tz1 <= tz2 v4, tz1 >  tz2 v5, tz1 >= tz2 v6
  INTO test26g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, tz1 =  tz2 v1, tz1 <> tz2 v2, tz1 <  tz2 v3,
           tz1 <= tz2 v4, tz1 >  tz2 v5, tz1 >= tz2 v6
  INTO test26p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test26g EXCEPT SELECT * FROM test26p) ORDER BY id;
(SELECT * FROM test26p EXCEPT SELECT * FROM test26g) ORDER BY id;

-- 'timestamp' <COMP> 'timestamp'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, ts1 =  ts2 v1, ts1 <> ts2 v2, ts1 <  ts2 v3,
           ts1 <= ts2 v4, ts1 >  ts2 v5, ts1 >= ts2 v6
  INTO test27g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, ts1 =  ts2 v1, ts1 <> ts2 v2, ts1 <  ts2 v3,
           ts1 <= ts2 v4, ts1 >  ts2 v5, ts1 >= ts2 v6
  INTO test27g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, ts1 =  ts2 v1, ts1 <> ts2 v2, ts1 <  ts2 v3,
           ts1 <= ts2 v4, ts1 >  ts2 v5, ts1 >= ts2 v6
  INTO test27p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test27g EXCEPT SELECT * FROM test27p) ORDER BY id;
(SELECT * FROM test27p EXCEPT SELECT * FROM test27g) ORDER BY id;

-- 'timestamp' <COMP> 'date'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, ts1 =  d2 v1, ts1 <> d2 v2, ts1 <  d2 v3,
           ts1 <= d2 v4, ts1 >  d2 v5, ts1 >= d2 v6
  INTO test28g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, ts1 =  d2 v1, ts1 <> d2 v2, ts1 <  d2 v3,
           ts1 <= d2 v4, ts1 >  d2 v5, ts1 >= d2 v6
  INTO test28g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, ts1 =  d2 v1, ts1 <> d2 v2, ts1 <  d2 v3,
           ts1 <= d2 v4, ts1 >  d2 v5, ts1 >= d2 v6
  INTO test28p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test28g EXCEPT SELECT * FROM test28p) ORDER BY id;
(SELECT * FROM test28p EXCEPT SELECT * FROM test28g) ORDER BY id;

-- 'timestamptz' <COMP> 'timestamptz'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, tsz1 =  tsz2 v1, tsz1 <> tsz2 v2, tsz1 <  tsz2 v3,
           tsz1 <= tsz2 v4, tsz1 >  tsz2 v5, tsz1 >= tsz2 v6
  INTO test29g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, tsz1 =  tsz2 v1, tsz1 <> tsz2 v2, tsz1 <  tsz2 v3,
           tsz1 <= tsz2 v4, tsz1 >  tsz2 v5, tsz1 >= tsz2 v6
  INTO test29g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, tsz1 =  tsz2 v1, tsz1 <> tsz2 v2, tsz1 <  tsz2 v3,
           tsz1 <= tsz2 v4, tsz1 >  tsz2 v5, tsz1 >= tsz2 v6
  INTO test29p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test29g EXCEPT SELECT * FROM test29p) ORDER BY id;
(SELECT * FROM test29p EXCEPT SELECT * FROM test29g) ORDER BY id;

-- 'date' <COMP> 'timestamptz'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, d1 =  tsz2 v1, d1 <> tsz2 v2, d1 <  tsz2 v3,
           d1 <= tsz2 v4, d1 >  tsz2 v5, d1 >= tsz2 v6
  INTO test30g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, d1 =  tsz2 v1, d1 <> tsz2 v2, d1 <  tsz2 v3,
           d1 <= tsz2 v4, d1 >  tsz2 v5, d1 >= tsz2 v6
  INTO test30g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, d1 =  tsz2 v1, d1 <> tsz2 v2, d1 <  tsz2 v3,
           d1 <= tsz2 v4, d1 >  tsz2 v5, d1 >= tsz2 v6
  INTO test30p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test30g EXCEPT SELECT * FROM test30p) ORDER BY id;
(SELECT * FROM test30p EXCEPT SELECT * FROM test30g) ORDER BY id;

-- 'timestamptz' <COMP> 'date'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, tsz1 =  d2 v1, tsz1 <> d2 v2, tsz1 <  d2 v3,
           tsz1 <= d2 v4, tsz1 >  d2 v5, tsz1 >= d2 v6
  INTO test31g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, tsz1 =  d2 v1, tsz1 <> d2 v2, tsz1 <  d2 v3,
           tsz1 <= d2 v4, tsz1 >  d2 v5, tsz1 >= d2 v6
  INTO test31g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, tsz1 =  d2 v1, tsz1 <> d2 v2, tsz1 <  d2 v3,
           tsz1 <= d2 v4, tsz1 >  d2 v5, tsz1 >= d2 v6
  INTO test31p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test31g EXCEPT SELECT * FROM test31p) ORDER BY id;
(SELECT * FROM test31p EXCEPT SELECT * FROM test31g) ORDER BY id;

-- 'timestamp' <COMP> 'timestamptz'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, ts1 =  tsz2 v1, ts1 <> tsz2 v2, ts1 <  tsz2 v3,
           ts1 <= tsz2 v4, ts1 >  tsz2 v5, ts1 >= tsz2 v6
  INTO test32g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, ts1 =  tsz2 v1, ts1 <> tsz2 v2, ts1 <  tsz2 v3,
           ts1 <= tsz2 v4, ts1 >  tsz2 v5, ts1 >= tsz2 v6
  INTO test32g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, ts1 =  tsz2 v1, ts1 <> tsz2 v2, ts1 <  tsz2 v3,
           ts1 <= tsz2 v4, ts1 >  tsz2 v5, ts1 >= tsz2 v6
  INTO test32p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test32g EXCEPT SELECT * FROM test32p) ORDER BY id;
(SELECT * FROM test32p EXCEPT SELECT * FROM test32g) ORDER BY id;

-- 'timestamptz' <COMP> 'timestamp'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, tsz1 =  tsz2 v1, tsz1 <> tsz2 v2, tsz1 <  tsz2 v3,
           tsz1 <= tsz2 v4, tsz1 >  tsz2 v5, tsz1 >= tsz2 v6
  INTO test33g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, tsz1 =  tsz2 v1, tsz1 <> tsz2 v2, tsz1 <  tsz2 v3,
           tsz1 <= tsz2 v4, tsz1 >  tsz2 v5, tsz1 >= tsz2 v6
  INTO test33g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, tsz1 =  tsz2 v1, tsz1 <> tsz2 v2, tsz1 <  tsz2 v3,
           tsz1 <= tsz2 v4, tsz1 >  tsz2 v5, tsz1 >= tsz2 v6
  INTO test33p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test33g EXCEPT SELECT * FROM test33p) ORDER BY id;
(SELECT * FROM test33p EXCEPT SELECT * FROM test33g) ORDER BY id;

-- 'interval' <COMP> 'interval'
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, iv1 =  iv2 v1, iv1 <> iv2 v2, iv1 <  iv2 v3,
           iv1 <= iv2 v4, iv1 >  iv2 v5, iv1 >= iv2 v6
  INTO test34g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, iv1 =  iv2 v1, iv1 <> iv2 v2, iv1 <  iv2 v3,
           iv1 <= iv2 v4, iv1 >  iv2 v5, iv1 >= iv2 v6
  INTO test34g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, iv1 =  iv2 v1, iv1 <> iv2 v2, iv1 <  iv2 v3,
           iv1 <= iv2 v4, iv1 >  iv2 v5, iv1 >= iv2 v6
  INTO test34p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test34g EXCEPT SELECT * FROM test34p) ORDER BY id;
(SELECT * FROM test34p EXCEPT SELECT * FROM test34g) ORDER BY id;

-- test for overlaps functions
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, (t1,t2)     OVERLAPS (t3,t4)     v1,
           (t1,t3)     OVERLAPS (t2,t4)     v2,
           (tz1,tz2)   OVERLAPS (tz3,tz4)   v3,
           (tz1,tz3)   OVERLAPS (tz2,tz4)   v4,
           (ts1,ts2)   OVERLAPS (ts3,ts4)   v5,
           (ts1,ts3)   OVERLAPS (ts2,ts4)   v6,
           (tsz1,tsz2) OVERLAPS (tsz3,tsz4) v7,
           (tsz1,tsz3) OVERLAPS (tsz2,tsz4) v8
  INTO test40g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, (t1,t2)     OVERLAPS (t3,t4)     v1,
           (t1,t3)     OVERLAPS (t2,t4)     v2,
           (tz1,tz2)   OVERLAPS (tz3,tz4)   v3,
           (tz1,tz3)   OVERLAPS (tz2,tz4)   v4,
           (ts1,ts2)   OVERLAPS (ts3,ts4)   v5,
           (ts1,ts3)   OVERLAPS (ts2,ts4)   v6,
           (tsz1,tsz2) OVERLAPS (tsz3,tsz4) v7,
           (tsz1,tsz3) OVERLAPS (tsz2,tsz4) v8
  INTO test40g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, (t1,t2)     OVERLAPS (t3,t4)     v1,
           (t1,t3)     OVERLAPS (t2,t4)     v2,
           (tz1,tz2)   OVERLAPS (tz3,tz4)   v3,
           (tz1,tz3)   OVERLAPS (tz2,tz4)   v4,
           (ts1,ts2)   OVERLAPS (ts3,ts4)   v5,
           (ts1,ts3)   OVERLAPS (ts2,ts4)   v6,
           (tsz1,tsz2) OVERLAPS (tsz3,tsz4) v7,
           (tsz1,tsz3) OVERLAPS (tsz2,tsz4) v8
  INTO test40p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test40g EXCEPT SELECT * FROM test40p) ORDER BY id;
(SELECT * FROM test40p EXCEPT SELECT * FROM test40g) ORDER BY id;

-- extract() on 'time' type
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, EXTRACT (microseconds FROM t1) v1,
           EXTRACT (milliseconds FROM t2) v2,
           EXTRACT (second FROM t3) v3,
           EXTRACT (minute FROM t4) v4,
           EXTRACT (hour FROM t1) v5
  INTO test41g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, EXTRACT (microseconds FROM t1) v1,
           EXTRACT (milliseconds FROM t2) v2,
           EXTRACT (second FROM t3) v3,
           EXTRACT (minute FROM t4) v4,
           EXTRACT (hour FROM t1) v5
  INTO test41g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, EXTRACT (microseconds FROM t1) v1,
           EXTRACT (milliseconds FROM t2) v2,
           EXTRACT (second FROM t3) v3,
           EXTRACT (minute FROM t4) v4,
           EXTRACT (hour FROM t1) v5
  INTO test41p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test41g EXCEPT SELECT * FROM test41p) ORDER BY id;
(SELECT * FROM test41p EXCEPT SELECT * FROM test41g) ORDER BY id;

-- extract() on 'timetz' type
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, EXTRACT (microseconds FROM tz1) v1,
           EXTRACT (milliseconds FROM tz2) v2,
           EXTRACT (second FROM tz3) v3,
           EXTRACT (minute FROM tz4) v4,
           EXTRACT (hour FROM tz1) v5
  INTO test42g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, EXTRACT (microseconds FROM tz1) v1,
           EXTRACT (milliseconds FROM tz2) v2,
           EXTRACT (second FROM tz3) v3,
           EXTRACT (minute FROM tz4) v4,
           EXTRACT (hour FROM tz1) v5
  INTO test42g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, EXTRACT (microseconds FROM tz1) v1,
           EXTRACT (milliseconds FROM tz2) v2,
           EXTRACT (second FROM tz3) v3,
           EXTRACT (minute FROM tz4) v4,
           EXTRACT (hour FROM tz1) v5
  INTO test42p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test42g EXCEPT SELECT * FROM test42p) ORDER BY id;
(SELECT * FROM test42p EXCEPT SELECT * FROM test42g) ORDER BY id;

-- extract() on 'timestamp' type
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, EXTRACT (century         FROM ts1) v1,
           EXTRACT (day             FROM ts2) v2,
           EXTRACT (decade          FROM ts1) v3,
           EXTRACT (dow             FROM ts2) v4,
           EXTRACT (doy             FROM ts1) v5,
           EXTRACT (epoch           FROM ts2) v6,
           EXTRACT (hour            FROM ts1) v7,
           EXTRACT (isodow          FROM ts2) v8,
           EXTRACT (isoyear         FROM ts1) v9,
           EXTRACT (microseconds    FROM ts2) v10,
           EXTRACT (millennium      FROM ts1) v11,
           EXTRACT (milliseconds    FROM ts2) v12,
           EXTRACT (minute          FROM ts1) v13,
           EXTRACT (month           FROM ts2) v14,
           EXTRACT (quarter         FROM ts1) v15,
           EXTRACT (second          FROM ts2) v16,
           EXTRACT (week            FROM ts2) v17,
           EXTRACT (year            FROM ts1) v18
  INTO test43g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, EXTRACT (century         FROM ts1) v1,
           EXTRACT (day             FROM ts2) v2,
           EXTRACT (decade          FROM ts1) v3,
           EXTRACT (dow             FROM ts2) v4,
           EXTRACT (doy             FROM ts1) v5,
           EXTRACT (epoch           FROM ts2) v6,
           EXTRACT (hour            FROM ts1) v7,
           EXTRACT (isodow          FROM ts2) v8,
           EXTRACT (isoyear         FROM ts1) v9,
           EXTRACT (microseconds    FROM ts2) v10,
           EXTRACT (millennium      FROM ts1) v11,
           EXTRACT (milliseconds    FROM ts2) v12,
           EXTRACT (minute          FROM ts1) v13,
           EXTRACT (month           FROM ts2) v14,
           EXTRACT (quarter         FROM ts1) v15,
           EXTRACT (second          FROM ts2) v16,
           EXTRACT (week            FROM ts2) v17,
           EXTRACT (year            FROM ts1) v18
  INTO test43g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, EXTRACT (century         FROM ts1) v1,
           EXTRACT (day             FROM ts2) v2,
           EXTRACT (decade          FROM ts1) v3,
           EXTRACT (dow             FROM ts2) v4,
           EXTRACT (doy             FROM ts1) v5,
           EXTRACT (epoch           FROM ts2) v6,
           EXTRACT (hour            FROM ts1) v7,
           EXTRACT (isodow          FROM ts2) v8,
           EXTRACT (isoyear         FROM ts1) v9,
           EXTRACT (microseconds    FROM ts2) v10,
           EXTRACT (millennium      FROM ts1) v11,
           EXTRACT (milliseconds    FROM ts2) v12,
           EXTRACT (minute          FROM ts1) v13,
           EXTRACT (month           FROM ts2) v14,
           EXTRACT (quarter         FROM ts1) v15,
           EXTRACT (second          FROM ts2) v16,
           EXTRACT (week            FROM ts2) v17,
           EXTRACT (year            FROM ts1) v18
  INTO test43p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test43g EXCEPT SELECT * FROM test43p) ORDER BY id;
(SELECT * FROM test43p EXCEPT SELECT * FROM test43g) ORDER BY id;

-- extract() on 'timestamptz' type
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, EXTRACT (century         FROM tsz1) v1,
           EXTRACT (day             FROM tsz2) v2,
           EXTRACT (decade          FROM tsz1) v3,
           EXTRACT (dow             FROM tsz2) v4,
           EXTRACT (doy             FROM tsz1) v5,
           EXTRACT (epoch           FROM tsz2) v6,
           EXTRACT (hour            FROM tsz1) v7,
           EXTRACT (isodow          FROM tsz2) v8,
           EXTRACT (isoyear         FROM tsz1) v9,
           EXTRACT (microseconds    FROM tsz2) v10,
           EXTRACT (millennium      FROM tsz1) v11,
           EXTRACT (milliseconds    FROM tsz2) v12,
           EXTRACT (minute          FROM tsz1) v13,
           EXTRACT (month           FROM tsz2) v14,
           EXTRACT (quarter         FROM tsz1) v15,
           EXTRACT (second          FROM tsz2) v16,
           EXTRACT (timezone        FROM tsz1) v17,
           EXTRACT (timezone_hour   FROM tsz2) v18,
           EXTRACT (timezone_minute FROM tsz1) v19,
           EXTRACT (week            FROM tsz2) v20,
           EXTRACT (year            FROM tsz1) v21
  INTO test44g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, EXTRACT (century         FROM tsz1) v1,
           EXTRACT (day             FROM tsz2) v2,
           EXTRACT (decade          FROM tsz1) v3,
           EXTRACT (dow             FROM tsz2) v4,
           EXTRACT (doy             FROM tsz1) v5,
           EXTRACT (epoch           FROM tsz2) v6,
           EXTRACT (hour            FROM tsz1) v7,
           EXTRACT (isodow          FROM tsz2) v8,
           EXTRACT (isoyear         FROM tsz1) v9,
           EXTRACT (microseconds    FROM tsz2) v10,
           EXTRACT (millennium      FROM tsz1) v11,
           EXTRACT (milliseconds    FROM tsz2) v12,
           EXTRACT (minute          FROM tsz1) v13,
           EXTRACT (month           FROM tsz2) v14,
           EXTRACT (quarter         FROM tsz1) v15,
           EXTRACT (second          FROM tsz2) v16,
           EXTRACT (timezone        FROM tsz1) v17,
           EXTRACT (timezone_hour   FROM tsz2) v18,
           EXTRACT (timezone_minute FROM tsz1) v19,
           EXTRACT (week            FROM tsz2) v20,
           EXTRACT (year            FROM tsz1) v21
  INTO test44g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, EXTRACT (century         FROM tsz1) v1,
           EXTRACT (day             FROM tsz2) v2,
           EXTRACT (decade          FROM tsz1) v3,
           EXTRACT (dow             FROM tsz2) v4,
           EXTRACT (doy             FROM tsz1) v5,
           EXTRACT (epoch           FROM tsz2) v6,
           EXTRACT (hour            FROM tsz1) v7,
           EXTRACT (isodow          FROM tsz2) v8,
           EXTRACT (isoyear         FROM tsz1) v9,
           EXTRACT (microseconds    FROM tsz2) v10,
           EXTRACT (millennium      FROM tsz1) v11,
           EXTRACT (milliseconds    FROM tsz2) v12,
           EXTRACT (minute          FROM tsz1) v13,
           EXTRACT (month           FROM tsz2) v14,
           EXTRACT (quarter         FROM tsz1) v15,
           EXTRACT (second          FROM tsz2) v16,
           EXTRACT (timezone        FROM tsz1) v17,
           EXTRACT (timezone_hour   FROM tsz2) v18,
           EXTRACT (timezone_minute FROM tsz1) v19,
           EXTRACT (week            FROM tsz2) v20,
           EXTRACT (year            FROM tsz1) v21
  INTO test44p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test44g EXCEPT SELECT * FROM test44p) ORDER BY id;
(SELECT * FROM test44p EXCEPT SELECT * FROM test44g) ORDER BY id;

-- extract() on 'interval' type
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, EXTRACT (microseconds FROM iv1) v1,
           EXTRACT (milliseconds FROM iv2) v2,
           EXTRACT (second       FROM iv1) v3,
           EXTRACT (minute       FROM iv2) v4,
           EXTRACT (hour         FROM iv1) v5,
           EXTRACT (day          FROM iv2) v6,
           EXTRACT (month        FROM iv1) v7,
           EXTRACT (quarter      FROM iv2) v8,
           EXTRACT (year         FROM iv1) v9,
           EXTRACT (decade       FROM iv2) v10,
           EXTRACT (century      FROM iv1) v11,
           EXTRACT (millennium   FROM iv2) v12
  INTO test44g
  FROM rt_datetime
 WHERE id > 0;
SELECT id, EXTRACT (microseconds FROM iv1) v1,
           EXTRACT (milliseconds FROM iv2) v2,
           EXTRACT (second       FROM iv1) v3,
           EXTRACT (minute       FROM iv2) v4,
           EXTRACT (hour         FROM iv1) v5,
           EXTRACT (day          FROM iv2) v6,
           EXTRACT (month        FROM iv1) v7,
           EXTRACT (quarter      FROM iv2) v8,
           EXTRACT (year         FROM iv1) v9,
           EXTRACT (decade       FROM iv2) v10,
           EXTRACT (century      FROM iv1) v11,
           EXTRACT (millennium   FROM iv2) v12
  INTO test45g
  FROM rt_datetime
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, EXTRACT (microseconds FROM iv1) v1,
           EXTRACT (milliseconds FROM iv2) v2,
           EXTRACT (second       FROM iv1) v3,
           EXTRACT (minute       FROM iv2) v4,
           EXTRACT (hour         FROM iv1) v5,
           EXTRACT (day          FROM iv2) v6,
           EXTRACT (month        FROM iv1) v7,
           EXTRACT (quarter      FROM iv2) v8,
           EXTRACT (year         FROM iv1) v9,
           EXTRACT (decade       FROM iv2) v10,
           EXTRACT (century      FROM iv1) v11,
           EXTRACT (millennium   FROM iv2) v12
  INTO test45p
  FROM rt_datetime
 WHERE id > 0;

(SELECT * FROM test45g EXCEPT SELECT * FROM test45p) ORDER BY id;
(SELECT * FROM test45p EXCEPT SELECT * FROM test45g) ORDER BY id;

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA regtest_dtype_time_temp CASCADE;
