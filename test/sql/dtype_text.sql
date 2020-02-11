---
--- Micro test cases for text / varchar operators / functions
---
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
DROP SCHEMA IF EXISTS regtest_dtype_text_temp CASCADE;
CREATE SCHEMA regtest_dtype_text_temp;
RESET client_min_messages;

---
--- check database encoding (must be UTF-8)
---
SELECT getdatabaseencoding();

--- test database creation
SET search_path = regtest_dtype_text_temp,public;
CREATE TABLE rt_text (
  id    int,
  bc1   char(80)    COLLATE "C",
  bc2   char(80)    COLLATE "C",
  vc1   varchar(80) COLLATE "C",
  vc2   varchar(80) COLLATE "C",
  tc1   text        COLLATE "C",
  tc2   text        COLLATE "C",
  tj1   text        COLLATE "ja_JP",
  tj2   text        COLLATE "ja_JP"
);
SELECT pgstrom.random_setseed(20190616);
INSERT INTO rt_text (
  SELECT x, pgstrom.random_text_len(1,  80),
            pgstrom.random_text_len(1,  80),
            pgstrom.random_text_len(1,  80),
            pgstrom.random_text_len(1,  80),
            pgstrom.random_text_len(1, 160),
            pgstrom.random_text_len(1, 160),
            pgstrom.random_text_len(1, 160),
            pgstrom.random_text_len(1, 160)
    FROM generate_series(1,3000) x
);
UPDATE rt_text SET tc1 = vc1 || '-' || tc1 || '-' || vc2
 WHERE id % 100 = 57;
VACUUM ANALYZE;

-- force to use GpuScan, instead of SeqScan
SET enable_seqscan = off;
-- not to print kernel source code
SET pg_strom.debug_kernel_source = off;

-- type cast (relabel) operator
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, bc1::text, bc2::varchar,
           vc1::char, vc2::text,
           tc1::char, tc2::varchar
  INTO test01g
  FROM rt_text
 WHERE id > 0;
SELECT id, bc1::text, bc2::varchar,
           vc1::char, vc2::text,
           tc1::char, tc2::varchar
  INTO test01g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, bc1::text, bc2::varchar,
           vc1::char, vc2::text,
           tc1::char, tc2::varchar
  INTO test01p
  FROM rt_text
 WHERE id > 0;

(SELECT * FROM test01g EXCEPT SELECT * FROM test01p) ORDER BY id;
(SELECT * FROM test01p EXCEPT SELECT * FROM test01g) ORDER BY id;

-- comparison operators
-- collation != C cannot run comparison
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, bc1 = bc2 eq, bc1 <> bc2 ne,
           bc1 > bc2 lt, bc1 >= bc2 le,
           bc1 < bc2 gt, bc1 <= bc2 ge
  INTO test10g
  FROM rt_text
 WHERE id > 0;
SELECT id, bc1 = bc2 eq, bc1 <> bc2 ne,
           bc1 > bc2 lt, bc1 >= bc2 le,
           bc1 < bc2 gt, bc1 <= bc2 ge
  INTO test10g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, bc1 = bc2 eq, bc1 <> bc2 ne,
           bc1 > bc2 lt, bc1 >= bc2 le,
           bc1 < bc2 gt, bc1 <= bc2 ge
  INTO test10p
  FROM rt_text
 WHERE id > 0;
(SELECT * FROM test10g EXCEPT SELECT * FROM test10p) ORDER BY id;
(SELECT * FROM test10p EXCEPT SELECT * FROM test10g) ORDER BY id;


SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, tc1 = tc2 eq, tc1 <> tc2 ne,
           tc1 > tc2 lt, tc1 >= tc2 le,
           tc1 < tc2 gt, tc1 <= tc2 ge
  INTO test11g
  FROM rt_text
 WHERE id > 0;
SELECT id, tc1 = tc2 eq, tc1 <> tc2 ne,
           tc1 > tc2 lt, tc1 >= tc2 le,
           tc1 < tc2 gt, tc1 <= tc2 ge
  INTO test11g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, tc1 = tc2 eq, tc1 <> tc2 ne,
           tc1 > tc2 lt, tc1 >= tc2 le,
           tc1 < tc2 gt, tc1 <= tc2 ge
  INTO test11p
  FROM rt_text
 WHERE id > 0;
(SELECT * FROM test11g EXCEPT SELECT * FROM test11p) ORDER BY id;
(SELECT * FROM test11p EXCEPT SELECT * FROM test11g) ORDER BY id;


SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, tj1 = tj2 eq, tj1 <> tj2 ne,
           tj1 > tj2 lt, tj1 >= tj2 le,
           tj1 < tj2 gt, tj1 <= tj2 ge
  INTO test12g
  FROM rt_text
 WHERE id > 0;
SELECT id, tj1 = tj2 eq, tj1 <> tj2 ne,
           tj1 > tj2 lt, tj1 >= tj2 le,
           tj1 < tj2 gt, tj1 <= tj2 ge
  INTO test12g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, tj1 = tj2 eq, tj1 <> tj2 ne,
           tj1 > tj2 lt, tj1 >= tj2 le,
           tj1 < tj2 gt, tj1 <= tj2 ge
  INTO test12p
  FROM rt_text
 WHERE id > 0;
(SELECT * FROM test12g EXCEPT SELECT * FROM test12p) ORDER BY id;
(SELECT * FROM test12p EXCEPT SELECT * FROM test12g) ORDER BY id;

-- LIKE & ILIKE operators
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, bc1 LIKE '%ab%cd%' v1,
           bc2 LIKE '%bc_de%' v2,
           tj1 LIKE '%cd%ef%' v3,
           tj2 LIKE '%de_fg%' v4,
           tc1 LIKE '%ef_gh%' v5,
           tc2 LIKE '%fg_hi%' v6
  INTO test20g
  FROM rt_text
 WHERE id > 0;
SELECT id, bc1 LIKE '%ab%cd%' v1,
           bc2 LIKE '%bc_de%' v2,
           tj1 LIKE '%cd%ef%' v3,
           tj2 LIKE '%de_fg%' v4,
           tc1 LIKE '%ef_gh%' v5,
           tc2 LIKE '%fg_hi%' v6
  INTO test20g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, bc1 LIKE '%ab%cd%' v1,
           bc2 LIKE '%bc_de%' v2,
           tj1 LIKE '%cd%ef%' v3,
           tj2 LIKE '%de_fg%' v4,
           tc1 LIKE '%ef_gh%' v5,
           tc2 LIKE '%fg_hi%' v6
  INTO test20p
  FROM rt_text
 WHERE id > 0;
(SELECT * FROM test20g EXCEPT SELECT * FROM test20p) ORDER BY id;
(SELECT * FROM test20p EXCEPT SELECT * FROM test20g) ORDER BY id;

-- ILIKE is valid only non-multibyte encoding
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, bc1 ILIKE '%ab%cd%' v1,
           bc2 ILIKE '%bc_de%' v2,
           tj1 ILIKE '%cd%ef%' v3,
           tj2 ILIKE '%de_fg%' v4,
           tc1 ILIKE '%ef_gh%' v5,
           tc2 ILIKE '%fg_hi%' v6
  INTO test21g
  FROM rt_text
 WHERE id > 0;
SELECT id, bc1 ILIKE '%ab%cd%' v1,
           bc2 ILIKE '%bc_de%' v2,
           tj1 ILIKE '%cd%ef%' v3,
           tj2 ILIKE '%de_fg%' v4,
           tc1 ILIKE '%ef_gh%' v5,
           tc2 ILIKE '%fg_hi%' v6
  INTO test21g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, bc1 ILIKE '%ab%cd%' v1,
           bc2 ILIKE '%bc_de%' v2,
           tj1 ILIKE '%cd%ef%' v3,
           tj2 ILIKE '%de_fg%' v4,
           tc1 ILIKE '%ef_gh%' v5,
           tc2 ILIKE '%fg_hi%' v6
  INTO test21p
  FROM rt_text
 WHERE id > 0;
(SELECT * FROM test21g EXCEPT SELECT * FROM test21p) ORDER BY id;
(SELECT * FROM test21p EXCEPT SELECT * FROM test21g) ORDER BY id;

-- '||' operator (textcat) can work on fixed max-length text
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, length(tc1) v1,
       tc1 || '_foo'  v2,
       substring(tc1, 1, 10) || '_bar' v3,
       vc1 || '_baz' v4,
       vc1 || '--' || vc2 v5
  INTO test30g
  FROM rt_text
 WHERE id > 0;
SELECT id, length(tc1) v1,
       tc1 || '_foo'  v2,
       substring(tc1, 1, 10) || '_bar' v3,
       vc1 || '_baz' v4,
       vc1 || '--' || vc2 v5
  INTO test30g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, length(tc1) v1,
       tc1 || '_foo'  v2,
       substring(tc1, 1, 10) || '_bar' v3,
       vc1 || '_baz' v4,
       vc1 || '--' || vc2 v5
  INTO test30p
  FROM rt_text
 WHERE id > 0;
(SELECT * FROM test30g EXCEPT SELECT * FROM test30p) ORDER BY id;
(SELECT * FROM test30p EXCEPT SELECT * FROM test30g) ORDER BY id;

-- substring
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, length(tc2) v1, length(tj2) v2,
       substring(tj1, 8, id % 15 + 5) v3,
       substring(tj1, id % 10 + 1) v4,
       substring(vc1 || '-hoge-' || vc2, id % 15 + 5, 20) v5
  INTO test31g
  FROM rt_text
 WHERE id > 0;
SELECT id, length(tc2) v1, length(tj2) v2,
       substring(tj1, 8, id % 15 + 5) v3,
       substring(tj1, id % 10 + 1) v4,
       substring(vc1 || '-hoge-' || vc2, id % 15 + 5, 20) v5
  INTO test31g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, length(tc2) v1, length(tj2) v2,
       substring(tj1, 8, id % 15 + 5) v3,
       substring(tj1, id % 10 + 1) v4,
       substring(vc1 || '-hoge-' || vc2, id % 15 + 5, 20) v5
  INTO test31p
  FROM rt_text
 WHERE id > 0;
(SELECT * FROM test31g EXCEPT SELECT * FROM test31p) ORDER BY id;
(SELECT * FROM test31p EXCEPT SELECT * FROM test31g) ORDER BY id;

-- variadic concat() also works on fixed max-length text
SET pg_strom.enabled = on;
EXPLAIN (costs off, verbose)
SELECT id, length(tc1) v1,
       concat(tc1, '_foo') v2,
	   concat(substring(tc1, 1, 10), '_bar') v3,
	   concat(substring(tc1, 1, 8), '--hoge--', substring(tc1, 8, 16)) v4,
	   concat(substring(tc1, 1, 8), substring(tc2, 1, 8), vc1, vc2) v5
  INTO test32g
  FROM rt_text
 WHERE id > 0;
SELECT id, length(tc1) v1,
       concat(tc1, '_foo') v2,
	   concat(substring(tc1, 1, 10), '_bar') v3,
	   concat(substring(tc1, 1, 8), '--hoge--', substring(tc1, 8, 16)) v4,
	   concat(substring(tc1, 1, 8), substring(tc2, 1, 8), vc1, vc2) v5
  INTO test32g
  FROM rt_text
 WHERE id > 0;
SET pg_strom.enabled = off;
SELECT id, length(tc1) v1,
       concat(tc1, '_foo') v2,
	   concat(substring(tc1, 1, 10), '_bar') v3,
	   concat(substring(tc1, 1, 8), '--hoge--', substring(tc1, 8, 16)) v4,
	   concat(substring(tc1, 1, 8), substring(tc2, 1, 8), vc1, vc2) v5
  INTO test32p
  FROM rt_text
 WHERE id > 0;
(SELECT * FROM test32g EXCEPT SELECT * FROM test32p) ORDER BY id;
(SELECT * FROM test32p EXCEPT SELECT * FROM test32g) ORDER BY id;

-- cleanup temporary resource
SET client_min_messages = error;
DROP SCHEMA regtest_dtype_text_temp CASCADE;
