-- $ID$
-- TPC-H/TPC-R Order Priority Checking Query (Q4)
-- Functional Query Definition
-- Approved February 1998
select o_orderpriority,
       count(*) as order_count
  from orders
 where o_orderdate >= '1997-05-01'::date
   and o_orderdate < '1997-05-01'::date + '3 months'::interval
   and exists (select *
                 from lineitem
                where l_orderkey = o_orderkey
                  and l_commitdate < l_receiptdate)
 group by o_orderpriority
 order by o_orderpriority;
