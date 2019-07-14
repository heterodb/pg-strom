---
--- Test for CPU fallback and GPU kernel suspend / resume on PostgreSQL table
---
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_fallback_pgsql_temp CASCADE;
CREATE SCHEMA regtest_fallback_pgsql_temp;
RESET client_min_messages;

SET search_path = regtest_fallback_pgsql_temp,public;
CREATE TABLE regtest_data (
  id    int,
  aid   int,
  cat   text,
  x     float,
  y     float,
  memo  text
);
SELECT setseed(0.20190714);
INSERT INTO regtest_data (
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
UPDATE regtest_data
   SET memo = md5(memo) || md5(memo)
 WHERE id = 400001;
UPDATE regtest_data
   SET memo = memo || '-' || memo || '-' || memo || '-' || memo
 WHERE id = 400001;
UPDATE regtest_data
   SET memo = memo || '-' || memo || '-' || memo || '-' || memo
 WHERE id = 400001;
UPDATE regtest_data
   SET memo = memo || '-' || memo || '-' || memo || '-' || memo
 WHERE id = 400001;
VACUUM ANALYZE regtest_data;

-- disables SeqScan and kernel source
SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET pg_strom.debug_kernel_source = off;

-- GPU scan with CPU fallback
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, x+y v1, substring(memo, 1, 20) v2
  INTO test01g
  FROM regtest_data
 WHERE memo LIKE '%abc%';
SELECT id, x+y v1, substring(memo, 1, 20) v2
  INTO test01g
  FROM regtest_data
 WHERE memo LIKE '%abc%';	-- error
SET pg_strom.cpu_fallback = on;
SELECT id, x+y v1, substring(memo, 1, 20) v2
  INTO test01g
  FROM regtest_data
 WHERE memo LIKE '%abc%';
SET pg_strom.enabled = off;
SELECT id, x+y v1, substring(memo, 1, 20) v2
  INTO test01p
  FROM regtest_data
 WHERE memo LIKE '%abc%';
(SELECT * FROM test01g EXCEPT SELECT * FROM test01p) ORDER BY id;
(SELECT * FROM test01p EXCEPT SELECT * FROM test01g) ORDER BY id;
RESET pg_strom.cpu_fallback;

-- GpuScan with GPU kernel suspend/resume
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test02g
  FROM regtest_data
 WHERE id > 0;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test02g
  FROM regtest_data
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test02p
  FROM regtest_data
 WHERE id > 0;
(SELECT * FROM test02g EXCEPT SELECT * FROM test02p) ORDER BY id;
(SELECT * FROM test02p EXCEPT SELECT * FROM test02g) ORDER BY id;

-- GpuScan with GPU kernel suspend/resume and CPU fallback
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03g
  FROM regtest_data
 WHERE memo LIKE '%abc%' OR id > 0;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03g
  FROM regtest_data
 WHERE memo LIKE '%abc%' OR id > 0;		-- error
SET pg_strom.cpu_fallback = on;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03g
  FROM regtest_data
 WHERE memo LIKE '%abc%' OR id > 0;
SET pg_strom.enabled = off;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03p
  FROM regtest_data
 WHERE memo LIKE '%abc%' OR id > 0;
(SELECT * FROM test03g EXCEPT SELECT * FROM test03p) ORDER BY id;
(SELECT * FROM test03p EXCEPT SELECT * FROM test03g) ORDER BY id;
RESET pg_strom.cpu_fallback;
