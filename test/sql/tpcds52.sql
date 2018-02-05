
select  dt.d_year
 	,item.i_brand_id brand_id
 	,item.i_brand brand
 	,sum(ss_ext_sales_price) ext_price
into pg_temp.tpcds_q52
 from date_dim dt
     ,store_sales
     ,item
 where dt.d_date_sk = store_sales.ss_sold_date_sk
    and store_sales.ss_item_sk = item.i_item_sk
    and item.i_manager_id = 1
    and dt.d_moy=12
    and dt.d_year=1998
 group by dt.d_year
 	,item.i_brand
 	,item.i_brand_id
 order by dt.d_year
 	,ext_price desc
 	,brand_id
limit 100 ;




--- validation check
(SELECT * FROM pg_temp.tpcds_q52.sql
 EXCEPT
 SELECT * FROM public.tpcds_q52.sql);
(SELECT * FROM public.tpcds_q52.sql
 EXCEPT
 SELECT * FROM pg_temp.tpcds_q52.sql);
