SET search_path = pgstrom_regress,public;
SET pg_strom.debug_kernel_source = off;
--Q3_1
EXPLAIN(costs off, verbose)
select c_nation, s_nation, d_year, sum(lo_revenue)
as revenue from customer, lineorder, supplier, date1
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_region = 'ASIA'  and s_region = 'ASIA'
and d_year >= 1992 and d_year <= 1997
  group by c_nation, s_nation, d_year
  order by d_year asc, revenue desc;

select c_nation, s_nation, d_year, sum(lo_revenue)
as revenue from customer, lineorder, supplier, date1
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_region = 'ASIA'  and s_region = 'ASIA'
and d_year >= 1992 and d_year <= 1997
  group by c_nation, s_nation, d_year
  order by d_year asc, revenue desc;
