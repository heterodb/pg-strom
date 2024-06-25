-- $ID$
-- TPC-H/TPC-R Potential Part Promotion Query (Q20)
-- Function Query Definition
-- Approved February 1998
select s_name,
       s_address
  from supplier,
       nation
 where s_suppkey in (select ps_suppkey
                       from partsupp
                      where ps_partkey in (select p_partkey
                                             from part
                                            where p_name like 'mint%'
                                          )
                        and ps_availqty > (select 0.5 * sum(l_quantity)
                                             from lineitem
                                            where l_partkey = ps_partkey
                                              and l_suppkey = ps_suppkey
                                              and l_shipdate >= '1993-01-01'::date
                                              and l_shipdate <  '1993-01-01'::date + '1 year'::interval
                                          )
                    )
   and s_nationkey = n_nationkey
   and n_name = 'UNITED STATES'
 order by s_name;
