-- $ID$
-- TPC-H/TPC-R Global Sales Opportunity Query (Q22)
-- Functional Query Definition
-- Approved February 1998
select cntrycode,
       count(*) as numcust,
       sum(c_acctbal) as totacctbal
  from (select substring(c_phone from 1 for 2) as cntrycode,
               c_acctbal
          from customer
         where substring(c_phone from 1 for 2) in ('34', '24', '30', '26', '29', '17', '33')
           and c_acctbal > (select avg(c_acctbal)
                              from customer
                             where c_acctbal > 0.00
                               and substring(c_phone from 1 for 2) in ('34', '24', '30', '26', '29', '17', '33')
                           )
           and not exists (select *
                             from orders
                            where o_custkey = c_custkey
                          )
       ) as custsale
 group by cntrycode
 order by cntrycode;
