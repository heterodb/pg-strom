
with results as
 (select i_item_id,
        ca_country,
        ca_state, 
        ca_county,
        avg( cast(cs_quantity as numeric(12,2))) agg1,
        avg( cast(cs_list_price as numeric(12,2))) agg2,
        avg( cast(cs_coupon_amt as numeric(12,2))) agg3,
        avg( cast(cs_sales_price as numeric(12,2))) agg4,
        avg( cast(cs_net_profit as numeric(12,2))) agg5,
        avg( cast(c_birth_year as numeric(12,2))) agg6,
        avg( cast(cd1.cd_dep_count as numeric(12,2))) agg7 
 from catalog_sales, customer_demographics cd1, customer_demographics cd2, customer, customer_address, date_dim, item
 where cs_sold_date_sk = d_date_sk and
       cs_item_sk = i_item_sk and
       cs_bill_cdemo_sk = cd1.cd_demo_sk and
       cs_bill_customer_sk = c_customer_sk and
       cd1.cd_gender = 'M' and 
       cd1.cd_education_status = 'College' and
       c_current_cdemo_sk = cd2.cd_demo_sk and
       c_current_addr_sk = ca_address_sk and
       c_birth_month in (9,5,12,4,1,10) and
       d_year = 2001 and
       ca_state in ('ND','WI','AL','NC','OK','MS','TN')
 group by i_item_id, ca_country, ca_state, ca_county)
  select  i_item_id, ca_country, ca_state, ca_county, agg1, agg2, agg3, agg4, agg5, agg6, agg7
into pg_temp.tpcds_q18
 from (
 select i_item_id, ca_country, ca_state, ca_county, agg1, agg2, agg3, agg4, agg5, agg6, agg7 from results
 union
 select i_item_id, ca_country, ca_state, NULL as county, avg(agg1) agg1, avg(agg2) agg2, avg(agg3) agg3,
   avg(agg4) agg4, avg(agg5) agg5, avg(agg6) agg6, avg(agg7) agg7 from results
 group by i_item_id, ca_country, ca_state
 union
 select i_item_id, ca_country, NULL as ca_state, NULL as county, avg(agg1) agg1, avg(agg2) agg2, avg(agg3) agg3,
   avg(agg4) agg4, avg(agg5) agg5, avg(agg6) agg6, avg(agg7) agg7 from results
 group by i_item_id, ca_country
 union
 select i_item_id, NULL as ca_country, NULL as ca_state, NULL as county, avg(agg1) agg1, avg(agg2) agg2, avg(agg3) agg3,
   avg(agg4) agg4, avg(agg5) agg5, avg(agg6) agg6, avg(agg7) agg7 from results
 group by i_item_id
 union
 select NULL AS i_item_id, NULL as ca_country, NULL as ca_state, NULL as county, avg(agg1) agg1, avg(agg2) agg2, avg(agg3) agg3,
   avg(agg4) agg4, avg(agg5) agg5, avg(agg6) agg6, avg(agg7) agg7 from results
 ) foo
 order by ca_country, ca_state, ca_county, i_item_id
 limit 100;




--- validation check
(SELECT * FROM pg_temp.tpcds_q18.sql
 EXCEPT
 SELECT * FROM public.tpcds_q18.sql);
(SELECT * FROM public.tpcds_q18.sql
 EXCEPT
 SELECT * FROM pg_temp.tpcds_q18.sql);
