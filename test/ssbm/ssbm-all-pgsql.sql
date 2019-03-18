\timing on

SET max_parallel_workers_per_gather = 24;
SET pg_strom.enabled = off;
SET parallel_setup_cost = 100000;

--Q1_1

explain
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,date1
where lo_orderdate = d_datekey
and d_year = 1993
and lo_discount between 1 and 3
and lo_quantity < 25;

select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,date1
where lo_orderdate = d_datekey
and d_year = 1993
and lo_discount between 1 and 3
and lo_quantity < 25;

--Q1_2

explain
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date1
where lo_orderdate = d_datekey
  and d_yearmonthnum = 199401
  and lo_discount between 4 and 6
  and lo_quantity between 26 and 35;

select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date1
where lo_orderdate = d_datekey
  and d_yearmonthnum = 199401
  and lo_discount between 4 and 6
  and lo_quantity between 26 and 35;

--Q1_3

explain
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date1
where lo_orderdate = d_datekey
  and d_weeknuminyear = 6
  and d_year = 1994
  and lo_discount between 5 and 7
  and lo_quantity between 26 and 35;

select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date1
where lo_orderdate = d_datekey
  and d_weeknuminyear = 6
  and d_year = 1994
  and lo_discount between 5 and 7
  and lo_quantity between 26 and 35;

--Q2_1

explain
select sum(lo_revenue), d_year, p_brand1
from lineorder, date1, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 'MFGR#12'
and s_region = 'AMERICA'
  group by d_year, p_brand1
  order by d_year, p_brand1;

select sum(lo_revenue), d_year, p_brand1
from lineorder, date1, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 'MFGR#12'
and s_region = 'AMERICA'
  group by d_year, p_brand1
  order by d_year, p_brand1;

--Q2_2

explain
select sum(lo_revenue), d_year, p_brand1
  from lineorder, date1, part, supplier
  where lo_orderdate = d_datekey
    and lo_partkey = p_partkey
    and lo_suppkey = s_suppkey
    and p_brand1 between
           'MFGR#2221' and 'MFGR#2228'
    and s_region = 'ASIA'
  group by d_year, p_brand1
  order by d_year, p_brand1;

select sum(lo_revenue), d_year, p_brand1
  from lineorder, date1, part, supplier
  where lo_orderdate = d_datekey
    and lo_partkey = p_partkey
    and lo_suppkey = s_suppkey
    and p_brand1 between
           'MFGR#2221' and 'MFGR#2228'
    and s_region = 'ASIA'
  group by d_year, p_brand1
  order by d_year, p_brand1;

--Q2_3

explain
select sum(lo_revenue), d_year, p_brand1
  from lineorder, date1, part, supplier
  where lo_orderdate = d_datekey
    and lo_partkey = p_partkey
    and lo_suppkey = s_suppkey
     and p_brand1 = 'MFGR#2221'
     and s_region = 'EUROPE'
  group by d_year, p_brand1
  order by d_year, p_brand1;

select sum(lo_revenue), d_year, p_brand1
  from lineorder, date1, part, supplier
  where lo_orderdate = d_datekey
    and lo_partkey = p_partkey
    and lo_suppkey = s_suppkey
     and p_brand1 = 'MFGR#2221'
     and s_region = 'EUROPE'
  group by d_year, p_brand1
  order by d_year, p_brand1;

--Q3_1

explain
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

--Q3_2

explain
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

--Q3_3

explain
select c_city,s_city,d_year,sum(lo_revenue) as revenue
from customer,lineorder,supplier,date1
where lo_custkey = c_custkey
  and lo_suppkey = s_suppkey
  and lo_orderdate = d_datekey
  and (c_city='UNITED KI1' or c_city='UNITED KI5')
  and (s_city='UNITED KI1' or s_city='UNITED KI5')
  and d_year >= 1992 and d_year <= 1997
  group by c_city, s_city, d_year
  order by d_year asc,revenue desc;

select c_city,s_city,d_year,sum(lo_revenue) as revenue
from customer,lineorder,supplier,date1
where lo_custkey = c_custkey
  and lo_suppkey = s_suppkey
  and lo_orderdate = d_datekey
  and (c_city='UNITED KI1' or c_city='UNITED KI5')
  and (s_city='UNITED KI1' or s_city='UNITED KI5')
  and d_year >= 1992 and d_year <= 1997
  group by c_city, s_city, d_year
  order by d_year asc,revenue desc;

--Q3_4

explain
select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date1
   where lo_custkey = c_custkey
     and lo_suppkey = s_suppkey
     and lo_orderdate = d_datekey
      and (c_city='UNITED KI1' or c_city='UNITED KI5')
    and (s_city='UNITED KI1' or s_city='UNITED KI5')
    and d_yearmonth = 'Dec1997'
    group by c_city, s_city, d_year
  order by d_year asc, revenue desc;

select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date1
   where lo_custkey = c_custkey
     and lo_suppkey = s_suppkey
     and lo_orderdate = d_datekey
      and (c_city='UNITED KI1' or c_city='UNITED KI5')
    and (s_city='UNITED KI1' or s_city='UNITED KI5')
    and d_yearmonth = 'Dec1997'
    group by c_city, s_city, d_year
  order by d_year asc, revenue desc;

--Q4_1

explain
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
    order by d_year, c_nation ;

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
    order by d_year, c_nation ;

--Q4_2

explain
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

--Q4_3

explain
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

\q

