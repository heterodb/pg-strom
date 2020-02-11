---
--- Test for CPU fallback and GPU kernel suspend / resume on PostgreSQL table
---
SET pg_strom.regression_test_mode = on;

-- this test uses pre-built test table
SET search_path = pg_temp,pgstrom_regress,public;

-- disables SeqScan and kernel source
SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET pg_strom.debug_kernel_source = off;

-- GpuScan  with CPU fallback
SET pg_strom.enabled = on;
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
EXPLAIN (verbose, costs off)
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03g
  FROM fallback_data
 WHERE memo LIKE '%abc%' OR id > 0;
SELECT id, x+y a, x-y b, x+1 c, y+1 d, x+2 e, y+2 f, x+3 g, y+4 h, memo
  INTO test03g
  FROM fallback_data
 WHERE memo LIKE '%abc%' OR id > 0;		-- Error
SET pg_strom.cpu_fallback = on;
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
EXPLAIN (verbose, costs off)
SELECT id, x+y+z v, memo
  INTO test10g
  FROM fallback_data d NATURAL JOIN fallback_small s
 WHERE memo LIKE '%abc%';
SELECT id, x+y+z v, memo
  INTO test10g
  FROM fallback_data d NATURAL JOIN fallback_small s
 WHERE memo LIKE '%abc%';		-- Error
SET pg_strom.cpu_fallback = on;
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
SELECT *
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 2500 AND memo LIKE '%ab%';	-- Error
SET pg_strom.cpu_fallback = on;
SELECT * INTO test12g
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 2500 AND memo LIKE '%ab%';
SET pg_strom.enabled = off;
SELECT * INTO test12p
  FROM fallback_data d NATURAL JOIN fallback_enlarge l
 WHERE l.aid < 2500 AND memo LIKE '%ab%';
(SELECT * FROM test12g EXCEPT SELECT * FROM test12p) ORDER BY id;
(SELECT * FROM test12p EXCEPT SELECT * FROM test12g) ORDER BY id;
RESET pg_strom.cpu_fallback;
