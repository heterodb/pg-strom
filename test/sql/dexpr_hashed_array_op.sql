---
--- Test for HashedArrayOp optimization
---
SET search_path = public;
SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET pg_strom.explain_developer_mode = on;

VACUUM ANALYZE lineorder;

-- IN-list with 3 constants keeps ScalarArrayOpAny
EXPLAIN (verbose, costs off)
SELECT lo_orderkey
  FROM lineorder
 WHERE lo_shipmode IN ('RAIL','SHIP','MAIL');

-- IN-list with 4 constants should switch to HashedArrayOp
EXPLAIN (verbose, costs off)
SELECT lo_orderkey
  FROM lineorder
 WHERE lo_shipmode IN ('RAIL','SHIP','MAIL','TRUCK');

SET pg_strom.enabled = on;
SELECT lo_orderkey
  INTO pg_temp.test01g
  FROM lineorder
 WHERE lo_shipmode IN ('RAIL','SHIP','MAIL','TRUCK')
   AND lo_orderkey % 97 = 0;
SET pg_strom.enabled = off;
SELECT lo_orderkey
  INTO pg_temp.test01c
  FROM lineorder
 WHERE lo_shipmode IN ('RAIL','SHIP','MAIL','TRUCK')
   AND lo_orderkey % 97 = 0;
(SELECT * FROM pg_temp.test01g EXCEPT SELECT * FROM pg_temp.test01c);
(SELECT * FROM pg_temp.test01c EXCEPT SELECT * FROM pg_temp.test01g);

-- CASE expr WHEN with 4 branches should switch to HashedArrayOp
SET pg_strom.enabled = on;
EXPLAIN (verbose, costs off)
SELECT lo_linenumber,
       CASE lo_linenumber % 5
       WHEN 0 THEN 'A'
       WHEN 1 THEN 'B'
       WHEN 2 THEN 'C'
       WHEN 3 THEN 'D'
       ELSE      'X'
       END AS tag
  FROM lineorder
 WHERE lo_shipmode IN ('RAIL','SHIP','MAIL','TRUCK');

SELECT lo_linenumber,
       CASE lo_linenumber % 5
       WHEN 0 THEN 'A'
       WHEN 1 THEN 'B'
       WHEN 2 THEN 'C'
       WHEN 3 THEN 'D'
       ELSE      'X'
       END AS tag
  INTO pg_temp.test02g
  FROM lineorder
 WHERE lo_shipmode IN ('RAIL','SHIP','MAIL','TRUCK')
   AND lo_orderkey % 97 = 0;
SET pg_strom.enabled = off;
SELECT lo_linenumber,
       CASE lo_linenumber % 5
       WHEN 0 THEN 'A'
       WHEN 1 THEN 'B'
       WHEN 2 THEN 'C'
       WHEN 3 THEN 'D'
       ELSE      'X'
       END AS tag
  INTO pg_temp.test02c
  FROM lineorder
 WHERE lo_shipmode IN ('RAIL','SHIP','MAIL','TRUCK')
   AND lo_orderkey % 97 = 0;
(SELECT * FROM pg_temp.test02g EXCEPT SELECT * FROM pg_temp.test02c);
(SELECT * FROM pg_temp.test02c EXCEPT SELECT * FROM pg_temp.test02g);