-- $ID$
-- TPC-H/TPC-R Forecasting Revenue Change Query (Q6)
-- Functional Query Definition
-- Approved February 1998
select sum(l_extendedprice * l_discount) as revenue
  from lineitem
 where l_shipdate >= '1996-01-01'::date
   and l_shipdate < '1996-01-01'::date + '1 year'::interval
   and l_discount between 0.09 - 0.01 and 0.09 + 0.01
   and l_quantity < 24;
