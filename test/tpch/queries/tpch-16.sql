-- $ID$
-- TPC-H/TPC-R Parts/Supplier Relationship Query (Q16)
-- Functional Query Definition
-- Approved February 1998
select p_brand,
       p_type,
       p_size,
       count(distinct ps_suppkey) as supplier_cnt
  from partsupp,
       part
 where p_partkey = ps_partkey
   and p_brand <> 'Brand#15'
   and p_type not like 'STANDARD BRUSHED%'
   and p_size in (5, 29, 4, 49, 30, 42, 33, 39)
   and ps_suppkey not in (select s_suppkey
                            from supplier
                           where s_comment like '%Customer%Complaints%')
 group by p_brand,
          p_type,
          p_size
 order by supplier_cnt desc,
          p_brand,
          p_type,
          p_size;
