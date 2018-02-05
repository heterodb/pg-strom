
select  i_brand_id brand_id, i_brand brand,
 	sum(ss_ext_sales_price) ext_price
into pg_temp.tpcds_q55
 from date_dim, store_sales, item
 where d_date_sk = ss_sold_date_sk
 	and ss_item_sk = i_item_sk
 	and i_manager_id=36
 	and d_moy=12
 	and d_year=2001
 group by i_brand, i_brand_id
 order by ext_price desc, i_brand_id
limit 100 ;




--- validation check
(SELECT * FROM pg_temp.tpcds_q55.sql
 EXCEPT
 SELECT * FROM public.tpcds_q55.sql);
(SELECT * FROM public.tpcds_q55.sql
 EXCEPT
 SELECT * FROM pg_temp.tpcds_q55.sql);
