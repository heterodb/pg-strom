SET pg_strom.regression_test_mode = on;
SET search_path = pgstrom_regress,public;
SET pg_strom.debug_kernel_source = off;
--Q4_3
SET pg_strom.enabled = on;
select d_year, s_city, p_brand1,
sum(lo_revenue - lo_supplycost) as profit_Q4_3
from date1, customer, supplier, part, lineorder
  where lo_custkey = c_custkey
  and lo_suppkey = s_suppkey
  and lo_partkey = p_partkey
  and lo_orderdate = d_datekey
  and c_region = 'AMERICA'
  and s_nation = 'UNITED STATES'
  and (d_year = 1997 or d_year = 1998)
  and p_category = 'MFGR#14'
group by d_year, s_city, p_brand1
order by d_year, s_city, p_brand1;

SET pg_strom.enabled = off;
select d_year, s_city, p_brand1,
sum(lo_revenue - lo_supplycost) as profit_Q4_3
from date1, customer, supplier, part, lineorder
  where lo_custkey = c_custkey
  and lo_suppkey = s_suppkey
  and lo_partkey = p_partkey
  and lo_orderdate = d_datekey
  and c_region = 'AMERICA'
  and s_nation = 'UNITED STATES'
  and (d_year = 1997 or d_year = 1998)
  and p_category = 'MFGR#14'
group by d_year, s_city, p_brand1
order by d_year, s_city, p_brand1;
