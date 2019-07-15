SET search_path = pgstrom_regress,public;
SET pg_strom.debug_kernel_source = off;
--Q4_1
EXPLAIN(costs off, verbose)
select d_year, c_nation,  sum(lo_revenue - lo_supplycost) as profit
from date1, customer, supplier, part, lineorder
    where lo_custkey = c_custkey
       and lo_suppkey = s_suppkey
       and lo_partkey = p_partkey
       and lo_orderdate = d_datekey
       and c_region = 'AMERICA'
       and s_region = 'AMERICA'
       and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
    group by d_year, c_nation
    order by d_year, c_nation;

select d_year, c_nation,  sum(lo_revenue - lo_supplycost) as profit
from date1, customer, supplier, part, lineorder
    where lo_custkey = c_custkey
       and lo_suppkey = s_suppkey
       and lo_partkey = p_partkey
       and lo_orderdate = d_datekey
       and c_region = 'AMERICA'
       and s_region = 'AMERICA'
       and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
    group by d_year, c_nation
    order by d_year, c_nation;
