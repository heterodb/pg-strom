SET search_path=tpch;
DROP TABLE IF EXISTS result_store;
SET pg_strom.gpudirect_enabled = ON; SET pg_strom.pinned_inner_buffer_threshold = 0;
EXPLAIN ANALYZE
select supp_nation,
       cust_nation,
       l_year,
       sum(volume) as revenue
  into result_store 
  from (select n1.n_name as supp_nation,
               n2.n_name as cust_nation,
               extract(year from l_shipdate) as l_year,
               l_extendedprice * (1 - l_discount) as volume
          from supplier,
               lineitem,
               orders,
               customer,
               nation n1,
               nation n2
         where s_suppkey = l_suppkey
           and o_orderkey = l_orderkey
           and c_custkey = o_custkey
           and s_nationkey = n1.n_nationkey
           and c_nationkey = n2.n_nationkey
           and ((n1.n_name = 'UNITED STATES' and n2.n_name = 'INDIA') or
		(n1.n_name = 'INDIA' and n2.n_name = 'UNITED STATES'))
           and o_orderdate between date '1995-01-01' and date '1996-12-31'
       ) as shipping
 group by supp_nation,
          cust_nation,
          l_year
 order by supp_nation,
          cust_nation,
          l_year;
select * from result_store;
