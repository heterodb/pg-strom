-- $ID$
-- TPC-H/TPC-R Top Supplier Query (Q15)
-- Functional Query Definition
-- Approved February 1998
WITH revenue_0 AS (
  select l_suppkey                               supplier_no,
         sum(l_extendedprice * (1 - l_discount)) total_revenue
    from lineitem
   where l_shipdate >= '1995-02-01'::date
     and l_shipdate <  '1995-02-01'::date + '3 months'::interval
   group by l_suppkey
)
select s_suppkey,
       s_name,
       s_address,
       s_phone,
       total_revenue
  from supplier,
       revenue_0
 where s_suppkey = supplier_no
   and total_revenue = (select max(total_revenue)
                          from revenue_0)
 order by s_suppkey;
