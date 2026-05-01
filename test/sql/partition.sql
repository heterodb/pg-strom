---
--- Test cases for partition-wise GpuJoin/GpuPreAgg
---
SET search_path = public;
SET pg_strom.regression_test_mode = on;

---
--- Simple GPU-Scan (blocked by issue #1026)
---


---
--- Simpe GPU-PreAgg without JOIN
---
explain (costs off)
select sum(lo_extendedprice*lo_discount) as revenue
from plineorder
where lo_orderdate between 19930101 and 19931231
and lo_discount between 1 and 3
and lo_quantity < 25;

select sum(lo_extendedprice*lo_discount) as revenue
from plineorder
where lo_orderdate between 19930101 and 19931231
and lo_discount between 1 and 3
and lo_quantity < 25;

explain (costs off)
select lo_orderpriority, sum(lo_extendedprice*lo_discount) as revenue
  from plineorder
 where lo_orderdate between 19930101 and 19931231
   and lo_discount between 1 and 3
   and lo_quantity < 25
 group by lo_orderpriority
 order by lo_orderpriority;

select lo_orderpriority, sum(lo_extendedprice*lo_discount) as revenue
  from plineorder
 where lo_orderdate between 19930101 and 19931231
   and lo_discount between 1 and 3
   and lo_quantity < 25
 group by lo_orderpriority
 order by lo_orderpriority;

---
--- GPU-PreAgg with JOIN
---
explain (costs off)
select c_city,s_city,(lo_orderdate/10000)::int d_year,sum(lo_revenue) as revenue
  from customer,plineorder,supplier
 where lo_custkey = c_custkey
   and lo_suppkey = s_suppkey
   and (c_city='UNITED KI1' or c_city='UNITED KI5')
   and (s_city='UNITED KI1' or s_city='UNITED KI5')
   and lo_orderdate between 19930701 and 19960630
 group by 1,2,3
 order by 3 asc, 2 desc;

select c_city,s_city,(lo_orderdate/10000)::int d_year,sum(lo_revenue) as revenue
  from customer,plineorder,supplier
 where lo_custkey = c_custkey
   and lo_suppkey = s_suppkey
   and (c_city='UNITED KI1' or c_city='UNITED KI5')
   and (s_city='UNITED KI1' or s_city='UNITED KI5')
   and lo_orderdate between 19930701 and 19960630
 group by 1,2,3
 order by 3 asc, 2 desc;

---
--- only heap and arrow by partition pruning
---
explain (costs off)
select c_city,s_city,(lo_orderdate/10000)::int d_year,lo_shipmode, sum(lo_revenue) as revenue
  from customer,plineorder,supplier
 where lo_custkey = c_custkey
   and lo_suppkey = s_suppkey
   and (c_city='UNITED KI1' or c_city='UNITED KI5')
   and (s_city='UNITED KI1' or s_city='UNITED KI5')
   and lo_orderdate between 19920701 and 19970630
   and lo_shipmode in ('SHIP','RAIL')
 group by 1,2,3,4
 order by 3 asc, 2 desc;

select c_city,s_city,(lo_orderdate/10000)::int d_year,lo_shipmode, sum(lo_revenue) as revenue
  from customer,plineorder,supplier
 where lo_custkey = c_custkey
   and lo_suppkey = s_suppkey
   and (c_city='UNITED KI1' or c_city='UNITED KI5')
   and (s_city='UNITED KI1' or s_city='UNITED KI5')
   and lo_orderdate between 19920701 and 19970630
   and lo_shipmode in ('SHIP','RAIL')
 group by 1,2,3,4
 order by 3 asc, 2 desc;

---
--- control by parameters
---
SET pg_strom.enable_partitionwise_gpupreagg = off;

explain (costs off)
select c_city,s_city,(lo_orderdate/10000)::int,sum(lo_revenue) as revenue
  from customer,plineorder,supplier
 where lo_custkey = c_custkey
   and lo_suppkey = s_suppkey
   and (c_city='UNITED KI1' or c_city='UNITED KI5')
   and (s_city='UNITED KI1' or s_city='UNITED KI5')
   and lo_orderdate between 19930701 and 19960630
 group by 1,2,3
 order by 3 asc, 4 desc;

select c_city,s_city,(lo_orderdate/10000)::int,sum(lo_revenue) as revenue
  from customer,plineorder,supplier
 where lo_custkey = c_custkey
   and lo_suppkey = s_suppkey
   and (c_city='UNITED KI1' or c_city='UNITED KI5')
   and (s_city='UNITED KI1' or s_city='UNITED KI5')
   and lo_orderdate between 19930701 and 19960630
 group by 1,2,3
 order by 3 asc, 4 desc;

RESET pg_strom.enable_partitionwise_gpupreagg;
SET pg_strom.enable_partitionwise_gpujoin = off;

explain (costs off)
select c_city,s_city,(lo_orderdate/10000)::int,sum(lo_revenue) as revenue
  from customer,plineorder,supplier
 where lo_custkey = c_custkey
   and lo_suppkey = s_suppkey
   and (c_city='UNITED KI1' or c_city='UNITED KI5')
   and (s_city='UNITED KI1' or s_city='UNITED KI5')
   and lo_orderdate between 19930701 and 19960630
 group by 1,2,3
 order by 3 asc, 4 desc;

select c_city,s_city,(lo_orderdate/10000)::int,sum(lo_revenue) as revenue
  from customer,plineorder,supplier
 where lo_custkey = c_custkey
   and lo_suppkey = s_suppkey
   and (c_city='UNITED KI1' or c_city='UNITED KI5')
   and (s_city='UNITED KI1' or s_city='UNITED KI5')
   and lo_orderdate between 19930701 and 19960630
 group by 1,2,3
 order by 3 asc, 4 desc;
