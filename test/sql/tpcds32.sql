
select  sum(cs_ext_discount_amt)  as "excess discount amount" 
into pg_temp.tpcds_q32
from 
   catalog_sales 
   ,item 
   ,date_dim
where
i_manufact_id = 269
and i_item_sk = cs_item_sk 
and d_date between '1998-03-18' and 
        (cast('1998-03-18' as date) + '90 days'::interval)
and d_date_sk = cs_sold_date_sk 
and cs_ext_discount_amt  
     > ( 
         select 
            1.3 * avg(cs_ext_discount_amt) 
         from 
            catalog_sales 
           ,date_dim
         where 
              cs_item_sk = i_item_sk 
          and d_date between '1998-03-18' and
                             (cast('1998-03-18' as date) + '90 days'::interval)
          and d_date_sk = cs_sold_date_sk 
      ) 
limit 100;



--- validation check
(SELECT * FROM pg_temp.tpcds_q32.sql
 EXCEPT
 SELECT * FROM public.tpcds_q32.sql);
(SELECT * FROM public.tpcds_q32.sql
 EXCEPT
 SELECT * FROM pg_temp.tpcds_q32.sql);
