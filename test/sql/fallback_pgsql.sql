---
--- Test for CPU fallback and GPU kernel suspend / resume on PostgreSQL table
---
SET pg_strom.regression_test_mode = on;

SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_arrow_index_temp CASCADE;
CREATE SCHEMA regtest_arrow_index_temp;
RESET client_min_messages;

SET search_path = regtest_arrow_index_temp,public;

-- disables SeqScan and kernel source
SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;

-- prepare table
-- test for CPU fallback / GPU kernel suspend/resume
CREATE TABLE fallback_data (
  id    int,
  aid   int,
  cat   text,
  x     float,
  y     float,
  memo  text
);
SELECT pgstrom.random_setseed(20190714);
INSERT INTO fallback_data (
  SELECT x, pgstrom.random_int(0.5, 1, 4000),
            CASE floor(random()*26)
            WHEN 0 THEN 'aaa'
            WHEN  1 THEN 'bbb'
            WHEN  2 THEN 'ccc'
            WHEN  3 THEN 'ddd'
            WHEN  4 THEN 'eee'
            WHEN  5 THEN 'fff'
            WHEN  6 THEN 'ggg'
            WHEN  7 THEN 'hhh'
            WHEN  8 THEN 'iii'
            WHEN  9 THEN 'jjj'
            WHEN 10 THEN 'kkk'
            WHEN 11 THEN 'lll'
            WHEN 12 THEN 'mmm'
            WHEN 13 THEN 'nnn'
            WHEN 14 THEN 'ooo'
            WHEN 15 THEN 'ppp'
            WHEN 16 THEN 'qqq'
            WHEN 17 THEN 'rrr'
            WHEN 18 THEN 'sss'
            WHEN 19 THEN 'ttt'
            WHEN 20 THEN 'uuu'
            WHEN 21 THEN 'vvv'
            WHEN 22 THEN 'www'
            WHEN 23 THEN 'xxx'
            WHEN 24 THEN 'yyy'
            ELSE 'zzz'
            END,
            pgstrom.random_float(2,-1000.0,1000.0),
            pgstrom.random_float(2,-1000.0,1000.0),
            pgstrom.random_text_len(2, 200)
    FROM generate_series(1,400001) x);
UPDATE fallback_data
   SET memo = md5(memo) || md5(memo)
 WHERE id = 400001;
UPDATE fallback_data
   SET memo = memo || '-' || memo || '-' || memo || '-' || memo
 WHERE id = 400001;
UPDATE fallback_data
   SET memo = memo || '-' || memo || '-' || memo || '-' || memo
 WHERE id = 400001;
UPDATE fallback_data
   SET memo = memo || '-' || memo || '-' || memo || '-' || memo
 WHERE id = 400001;

CREATE TABLE fallback_small (
  aid   int,
  z     float,
  md5   varchar(32)
);
INSERT INTO fallback_small (
  SELECT x, pgstrom.random_float(2,-1000.0,1000.0),
            md5(x::text)
    FROM generate_series(1,4000) x);

CREATE TABLE fallback_enlarge (
  aid   int,
  z     float,
  md5   char(200)
);
INSERT INTO fallback_enlarge (
  SELECT x / 5, pgstrom.random_float(2,-1000.0,1000.0),
            md5(x::text)
    FROM generate_series(1,20000) x);



-- GpuScan  with CPU fallback
SET pg_strom.enabled = on;
SET pg_strom.cpu_fallback = off;
VACUUM ANALYZE;
EXPLAIN (verbose, costs off)
SELECT id, x+y v1, substring(memo, 1, 20) v2
  INTO test01g
  FROM fallback_data
 WHERE memo LIKE '%abc%';
SELECT id, x+y v1, substring(memo, 1, 20) v2
  INTO test01g
  FROM fallback_data
 WHERE memo LIKE '%abc%';	-- error
SET pg_strom.cpu_fallback = on;
SELECT id, x+y v1, substring(memo, 1, 20) v2
  INTO test01g
  FROM fallback_data
 WHERE memo LIKE '%abc%';
SET pg_strom.enabled = off;
SELECT id, x+y v1, substring(memo, 1, 20) v2
  INTO test01p
  FROM fallback_data
 WHERE memo LIKE '%abc%';
(SELECT * FROM test01g EXCEPT SELECT * FROM test01p) ORDER BY id;
(SELECT * FROM test01p EXCEPT SELECT * FROM test01g) ORDER BY id;
RESET pg_strom.cpu_fallback;

-- GpuScan with GPU kernel suspend/resume
SET pg_strom.enabled = on;
VACUUM ANALYZE;
EXPLAIN (verbose, costs off)
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test02g
  FROM fallback_data
 WHERE id > 0;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test02g
  FROM fallback_data
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test02p
  FROM fallback_data
 WHERE id > 0;
(SELECT * FROM test02g EXCEPT SELECT * FROM test02p) ORDER BY id;
(SELECT * FROM test02p EXCEPT SELECT * FROM test02g) ORDER BY id;

-- GpuScan with GPU kernel suspend/resume and CPU fallback
SET pg_strom.enabled = on;
VACUUM ANALYZE;
EXPLAIN (verbose, costs off)
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03g
  FROM fallback_data
 WHERE memo LIKE '%abc%' OR id > 0;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03g
  FROM fallback_data
 WHERE memo LIKE '%abc%' OR id > 0;
SET pg_strom.enabled = off;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03p
  FROM fallback_data
 WHERE memo LIKE '%abc%' OR id > 0;
(SELECT * FROM test03g EXCEPT SELECT * FROM test03p) ORDER BY id;
(SELECT * FROM test03p EXCEPT SELECT * FROM test03g) ORDER BY id;
RESET pg_strom.cpu_fallback;

-- GpuJoin with CPU fallback
SET pg_strom.enabled = on;
VACUUM ANALYZE;
EXPLAIN (verbose, costs off)
SELECT id, x+y+z v, memo
  INTO test10g
  FROM fallback_data d NATURAL JOIN fallback_small s
 WHERE memo LIKE '%abc%';
SELECT id, x+y+z v, memo
  INTO test10g
  FROM fallback_data d NATURAL JOIN fallback_small s
 WHERE memo LIKE '%abc%';
SET pg_strom.enabled = off;
SELECT id, x+y+z v, memo
  INTO test10p
  FROM fallback_data d NATURAL JOIN fallback_small s
 WHERE memo LIKE '%abc%';
(SELECT * FROM test10g EXCEPT SELECT * FROM test10p) ORDER BY id;
(SELECT * FROM test10p EXCEPT SELECT * FROM test10g) ORDER BY id;
RESET pg_strom.cpu_fallback;

-- GpuJoin with GPU kernel suspend / resume
SET pg_strom.enabled = on;
VACUUM ANALYZE;
EXPLAIN (verbose, costs off)
SELECT * INTO test11g
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 1000;
SELECT * INTO test11g
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 1000;
SET pg_strom.enabled = off;
SELECT * INTO test11p
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 1000;
(SELECT * FROM test11g EXCEPT SELECT * FROM test11p) ORDER BY id;
(SELECT * FROM test11p EXCEPT SELECT * FROM test11g) ORDER BY id;


-- GpuJoin with GPU kernel suspend / resume, and CPU fallback
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT *
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 2500 AND memo LIKE '%ab%';
SELECT * INTO test12g
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 2500 AND memo LIKE '%ab%';
SET pg_strom.enabled = off;
SELECT * INTO test12p
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 2500 AND memo LIKE '%ab%';
(SELECT * FROM test12g EXCEPT SELECT * FROM test12p) ORDER BY id LIMIT 10;
(SELECT * FROM test12p EXCEPT SELECT * FROM test12g) ORDER BY id LIMIT 10;
RESET pg_strom.cpu_fallback;
