--
-- A simple workload introduced in the wikipage
--
-- It intends to join a large fact table with multiple master tables,
-- like construction of 
--
EXPLAIN ANALYZE SELECT cat, count(*), avg(ax) FROM t0 NATURAL JOIN t1 GROUP BY cat;

EXPLAIN ANALYZE SELECT cat, count(*), avg(ax) FROM t0 NATURAL JOIN t1 NATURAL JOIN t2 GROUP BY cat;

EXPLAIN ANALYZE SELECT cat, count(*), avg(ax) FROM t0 NATURAL JOIN t1 NATURAL JOIN t2 NATURAL JOIN t3 GROUP BY cat;

EXPLAIN ANALYZE SELECT cat, count(*), avg(ax) FROM t0 NATURAL JOIN t1 NATURAL JOIN t2 NATURAL JOIN t3 NATURAL JOIN t4 GROUP BY cat;

EXPLAIN ANALYZE SELECT cat, count(*), avg(ax) FROM t0 NATURAL JOIN t1 NATURAL JOIN t2 NATURAL JOIN t3 NATURAL JOIN t4 NATURAL JOIN t5 GROUP BY cat;

EXPLAIN ANALYZE SELECT cat, count(*), avg(ax) FROM t0 NATURAL JOIN t1 NATURAL JOIN t2 NATURAL JOIN t3 NATURAL JOIN t4 NATURAL JOIN t5 NATURAL JOIN t6 GROUP BY cat;

EXPLAIN ANALYZE SELECT cat, count(*), avg(ax) FROM t0 NATURAL JOIN t1 NATURAL JOIN t2 NATURAL JOIN t3 NATURAL JOIN t4 NATURAL JOIN t5 NATURAL JOIN t6 NATURAL JOIN t7 GROUP BY cat;

EXPLAIN ANALYZE SELECT cat, count(*), avg(ax) FROM t0 NATURAL JOIN t1 NATURAL JOIN t2 NATURAL JOIN t3 NATURAL JOIN t4 NATURAL JOIN t5 NATURAL JOIN t6 NATURAL JOIN t7 NATURAL JOIN t8 GROUP BY cat;

--
-- Add a little complicated qualifiers
--
EXPLAIN ANALYZE SELECT cat, count(*) FROM t0 NATURAL JOIN t1 NATURAL JOIN t2
        WHERE sqrt((ax-20)^2 + (bx-30)^2) < 10 GROUP BY cat;

--
-- Misc aggregations
--
EXPLAIN ANALYZE SELECT cat,
  SUM(ax * (100.0 - bx) / 100.0),
  SUM(ax * (100.0 - cx) / 100.0),
  COUNT(*) FILTER (WHERE ax between 20.0 and 40.0),
  COUNT(*) FILTER (WHERE ax between 40.0 and 60.0)
FROM t0 NATURAL JOIN t1 NATURAL JOIN t2 NATURAL JOIN t3
WHERE ax between 20.0 and 60.0
GROUP BY cat
ORDER BY cat;

---
--- Index access is better, so PG-Strom should not used
---
EXPLAIN SELECT * FROM t0 NATURAL JOIN t1 WHERE id between 5001 and 6000 AND atext like '%abc%';
