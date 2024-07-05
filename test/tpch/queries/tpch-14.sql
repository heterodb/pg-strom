-- $ID$
-- TPC-H/TPC-R Promotion Effect Query (Q14)
-- Functional Query Definition
-- Approved February 1998
select 100.00 * sum(case when p_type like 'PROMO%'
                         then l_extendedprice * (1 - l_discount)
                         else 0
                    end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
  from lineitem,
       part
 where l_partkey = p_partkey
   and l_shipdate >= '1993-11-01'::date
   and l_shipdate < '1993-11-01'::date + '1 month'::interval;
