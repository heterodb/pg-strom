-- $ID$
-- TPC-H/TPC-R Local Supplier Volume Query (Q5)
-- Functional Query Definition
-- Approved February 1998
select n_name,
       sum(l_extendedprice * (1 - l_discount)) as revenue
  from customer,
       orders,
       lineitem,
       supplier,
       nation,
       region
 where c_custkey = o_custkey
   and l_orderkey = o_orderkey
   and l_suppkey = s_suppkey
   and c_nationkey = s_nationkey
   and s_nationkey = n_nationkey
   and n_regionkey = r_regionkey
   and r_name = 'AMERICA'
   and o_orderdate >= '1996-01-01'::date
   and o_orderdate <  '1996-01-01'::date + '1 year'::interval
 group by n_name
 order by revenue desc;
