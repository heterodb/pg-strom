---
--- Test for gpusort
---
\t on
SET pg_strom.regression_test_mode = on;
SET client_min_messages = error;
RESET client_min_messages;

SET search_path = regtest_dtype_range_temp,public;

-- Generate answer data
SET pg_strom.enabled = off;
DROP TABLE IF EXISTS gpusort_answer1;
SELECT * INTO gpusort_answer1 FROM (
    SELECT c_region, c_nation, c_city, SUM(lo_revenue) lo_rev,
            RANK() OVER(PARTITION BY c_region, c_nation
                        ORDER BY SUM(lo_revenue)) rank
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE rank < 4;

DROP TABLE IF EXISTS gpusort_answer2;
SELECT * INTO gpusort_answer2 FROM (
    SELECT c_region, c_nation, c_city, AVG(lo_revenue) lo_rev,
            DENSE_RANK() OVER(PARTITION BY c_region, c_nation
                        ORDER BY AVG(lo_revenue) desc) dense_rank
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE dense_rank <= 4;

DROP TABLE IF EXISTS gpusort_answer3;
SELECT * INTO gpusort_answer3 FROM (
    SELECT c_region, c_nation, c_city, MAX(lo_revenue) lo_rev,
            ROW_NUMBER() OVER(PARTITION BY c_region, c_nation
                        ORDER BY MAX(lo_revenue)) row_number
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE row_number <= 4;


SET pg_strom.enabled = on;
SET pg_strom.cpu_fallback = off;
DROP TABLE IF EXISTS gpusort_result1;
EXPLAIN(costs off, verbose)
SELECT * INTO gpusort_result1 FROM (
    SELECT c_region, c_nation, c_city, SUM(lo_revenue) lo_rev,
            RANK() OVER(PARTITION BY c_region, c_nation
                        ORDER BY SUM(lo_revenue)) rank
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE rank < 4;
SELECT * INTO gpusort_result1 FROM (
    SELECT c_region, c_nation, c_city, SUM(lo_revenue) lo_rev,
            RANK() OVER(PARTITION BY c_region, c_nation
                        ORDER BY SUM(lo_revenue)) rank
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE rank < 4;

(SELECT * FROM gpusort_result1 EXCEPT SELECT * FROM gpusort_answer1);
(SELECT * FROM gpusort_answer1 EXCEPT SELECT * FROM gpusort_result1);


DROP TABLE IF EXISTS gpusort_result2;
EXPLAIN(costs off, verbose)
SELECT * INTO gpusort_result2 FROM (
    SELECT c_region, c_nation, c_city, AVG(lo_revenue) lo_rev,
            DENSE_RANK() OVER(PARTITION BY c_region, c_nation
                        ORDER BY AVG(lo_revenue) desc) dense_rank
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE dense_rank <= 4;
SELECT * INTO gpusort_result2 FROM (
    SELECT c_region, c_nation, c_city, AVG(lo_revenue) lo_rev,
            DENSE_RANK() OVER(PARTITION BY c_region, c_nation
                        ORDER BY AVG(lo_revenue) desc) dense_rank
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE dense_rank <= 4;

(SELECT * FROM gpusort_result2 EXCEPT SELECT * FROM gpusort_answer2);
(SELECT * FROM gpusort_answer2 EXCEPT SELECT * FROM gpusort_result2);


DROP TABLE IF EXISTS gpusort_result3;
EXPLAIN(costs off, verbose)
SELECT * INTO gpusort_result3 FROM (
    SELECT c_region, c_nation, c_city, MAX(lo_revenue) lo_rev,
            ROW_NUMBER() OVER(PARTITION BY c_region, c_nation
                        ORDER BY MAX(lo_revenue)) row_number
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE row_number <= 4;
SELECT * INTO gpusort_result3 FROM (
    SELECT c_region, c_nation, c_city, MAX(lo_revenue) lo_rev,
            ROW_NUMBER() OVER(PARTITION BY c_region, c_nation
                        ORDER BY MAX(lo_revenue)) row_number
        FROM lineorder, customer
    WHERE lo_custkey = c_custkey
        AND lo_shipmode IN ('RAIL','SHIP')
        GROUP BY c_region, c_nation, c_city
) subqry
WHERE row_number <= 4;

(SELECT * FROM gpusort_result3 EXCEPT SELECT * FROM gpusort_answer3);
(SELECT * FROM gpusort_answer3 EXCEPT SELECT * FROM gpusort_result3);

DROP TABLE gpusort_answer1;
DROP TABLE gpusort_answer2;
DROP TABLE gpusort_answer3;