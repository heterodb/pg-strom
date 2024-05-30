---
--- Test for SELECT DISTINCT
---
SET search_path = public;
SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;

---
--- Scan + Distinct
---
explain (verbose, costs off)
select distinct s_city, s_nation
  from supplier
 where s_region = 'ASIA'
 order by s_city, s_nation;

select distinct s_city, s_nation
  from supplier
 where s_region = 'ASIA'
 order by s_city, s_nation;


--
-- Join + Distinct
--
explain (verbose, costs off)
select distinct c_nation, lo_orderpriority
  from lineorder, customer
 where lo_custkey = c_custkey
   and c_region = 'AMERICA'
   and lo_shipmode = 'AIR'
 order by c_nation, lo_orderpriority;

select distinct c_nation, lo_orderpriority
  from lineorder, customer
 where lo_custkey = c_custkey
   and c_region = 'AMERICA'
   and lo_shipmode = 'AIR'
 order by c_nation, lo_orderpriority;

--
-- partial distinct
--
explain (verbose, costs off)
select distinct on (p_type) p_type, p_color, p_size
  into pg_temp.test01g
  from part
 where p_color in ('lemon', 'green', 'plum');

select distinct on (p_type) p_type, p_color, p_size
  into pg_temp.test01g
  from part
 where p_color in ('lemon', 'green', 'plum');

SET pg_strom.enabled = off;
select distinct on (p_type) p_type, p_color, p_size
  into pg_temp.test01c
  from part
 where p_color in ('lemon', 'green', 'plum');

SELECT p_type FROM pg_temp.test01g EXCEPT SELECT p_type FROM pg_temp.test01c;
SELECT p_type FROM pg_temp.test01g EXCEPT SELECT p_type FROM pg_temp.test01g;

