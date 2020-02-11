SET pg_strom.regression_test_mode = on;
SET search_path = pgstrom_regress,public;
SET pg_strom.debug_kernel_source = off;
--Q4_2
SET pg_strom.enabled = on;
select d_year, s_nation, p_category,
sum(lo_revenue - lo_supplycost) as profit
from date1, customer, supplier, part, lineorder
  where lo_custkey = c_custkey
  and lo_suppkey = s_suppkey
  and lo_partkey = p_partkey
  and lo_orderdate = d_datekey
  and c_region = 'AMERICA'
  and s_region = 'AMERICA'
  and (d_year = 1997 or d_year = 1998)
  and (p_mfgr = 'MFGR#1'
   or p_mfgr = 'MFGR#2')
group by d_year, s_nation, p_category
order by d_year, s_nation, p_category;

SET pg_strom.enabled = off;
select d_year, s_nation, p_category,
sum(lo_revenue - lo_supplycost) as profit
from date1, customer, supplier, part, lineorder
  where lo_custkey = c_custkey
  and lo_suppkey = s_suppkey
  and lo_partkey = p_partkey
  and lo_orderdate = d_datekey
  and c_region = 'AMERICA'
  and s_region = 'AMERICA'
  and (d_year = 1997 or d_year = 1998)
  and (p_mfgr = 'MFGR#1'
   or p_mfgr = 'MFGR#2')
group by d_year, s_nation, p_category
order by d_year, s_nation, p_category;
