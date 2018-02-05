
select  a.ca_state state, count(*) cnt
 into pg_temp.tpcds_q06x
 from customer_address a
     ,customer c
     ,store_sales s
     ,date_dim d
     ,item i
     ,(select i_category, avg(i_current_price)::numeric(7,2) i_current_price
       from item
       group by i_category) j
 where       a.ca_address_sk = c.c_current_addr_sk
 	and c.c_customer_sk = s.ss_customer_sk
 	and s.ss_sold_date_sk = d.d_date_sk
 	and s.ss_item_sk = i.i_item_sk
 	and d.d_month_seq = 
 	     (select distinct (d_month_seq)
 	      from date_dim
               where d_year = 2000
 	        and d_moy = 2 )
        and i.i_category = j.i_category
        and i.i_current_price > 1.2 * j.i_current_price
 group by a.ca_state
 having count(*) >= 10
 order by cnt 
 limit 100;

--- validation check
(SELECT * FROM pg_temp.tpcds_q06x.sql
 EXCEPT
 SELECT * FROM public.tpcds_q06x.sql);
(SELECT * FROM public.tpcds_q06x.sql
 EXCEPT
 SELECT * FROM pg_temp.tpcds_q06x.sql);
