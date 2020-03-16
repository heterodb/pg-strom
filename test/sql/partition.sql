---
--- Test cases for asymmetric partition-wise GpuJoin
---
SET search_path = pgstrom_regress,public;
SET pg_strom.regression_test_mode = on;

-- INNER JOIN
RESET pg_strom.enabled;
EXPLAIN (costs off)
SELECT count(*), label, sum(a.x), sum(b.y), sum(c.z)
  FROM ptable p, atable a, btable b, ctable c
 WHERE p.aid = a.aid AND p.bid = b.bid AND p.cid = c.cid
 GROUP BY label
 ORDER BY label;

SELECT count(*), label, sum(a.x), sum(b.y), sum(c.z)
  FROM ptable p, atable a, btable b, ctable c
 WHERE p.aid = a.aid AND p.bid = b.bid AND p.cid = c.cid
 GROUP BY label
 ORDER BY label;

SET pg_strom.enabled = off;
SELECT count(*), label, sum(a.x), sum(b.y), sum(c.z)
  FROM ptable p, atable a, btable b, ctable c
 WHERE p.aid = a.aid AND p.bid = b.bid AND p.cid = c.cid
 GROUP BY label
 ORDER BY label;

-- LEFT OUTER JOIN
RESET pg_strom.enabled;
EXPLAIN (costs off)
SELECT count(*), label, sum(a.x), sum(b.y), sum(c.z)
  FROM ptable p
  LEFT OUTER JOIN atable a ON p.aid = a.aid
  LEFT OUTER JOIN btable b ON p.bid = b.bid
  LEFT OUTER JOIN ctable c ON p.cid = c.cid
 GROUP BY label
 ORDER BY label;

SELECT count(*), label, sum(a.x), sum(b.y), sum(c.z)
  FROM ptable p
  LEFT OUTER JOIN atable a ON p.aid = a.aid
  LEFT OUTER JOIN btable b ON p.bid = b.bid
  LEFT OUTER JOIN ctable c ON p.cid = c.cid
 GROUP BY label
 ORDER BY label;

SET pg_strom.enabled = off;
SELECT count(*), label, sum(a.x), sum(b.y), sum(c.z)
  FROM ptable p
  LEFT OUTER JOIN atable a ON p.aid = a.aid
  LEFT OUTER JOIN btable b ON p.bid = b.bid
  LEFT OUTER JOIN ctable c ON p.cid = c.cid
 GROUP BY label
 ORDER BY label;

--
-- RIGHT/FULL OUTER JOIN is not supported right now
--
