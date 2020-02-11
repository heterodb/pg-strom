SET pg_strom.regression_test_mode = on;
SET search_path = pgstrom_regress,public;
SET pg_strom.debug_kernel_source = off;
-- Q1_2
SET pg_strom.enabled = off;
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date1
where lo_orderdate = d_datekey
  and d_yearmonthnum = 199401
  and lo_discount between 4 and 6
  and lo_quantity between 26 and 35;

SET pg_strom.enabled = on;
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date1
where lo_orderdate = d_datekey
  and d_yearmonthnum = 199401
  and lo_discount between 4 and 6
  and lo_quantity between 26 and 35;
