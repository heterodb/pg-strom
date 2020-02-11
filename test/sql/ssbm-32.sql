SET pg_strom.regression_test_mode = on;
SET search_path = pgstrom_regress,public;
SET pg_strom.debug_kernel_source = off;
--Q3_2
SET pg_strom.enabled = on;
select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date1
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_nation = 'UNITED STATES'
and s_nation = 'UNITED STATES'
and d_year >= 1992 and d_year <= 1997
  group by c_city, s_city, d_year
  order by d_year asc, revenue desc;

SET pg_strom.enabled = off;
select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date1
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_nation = 'UNITED STATES'
and s_nation = 'UNITED STATES'
and d_year >= 1992 and d_year <= 1997
  group by c_city, s_city, d_year
  order by d_year asc, revenue desc;
