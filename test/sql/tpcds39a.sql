with inv as
(select w_warehouse_name,w_warehouse_sk,i_item_sk,d_moy
       ,stdev,mean, case mean when 0 then null else stdev/mean end cov
 from(select w_warehouse_name,w_warehouse_sk,i_item_sk,d_moy
            ,stddev_samp(inv_quantity_on_hand) stdev,avg(inv_quantity_on_hand) mean
      from inventory
          ,item
          ,warehouse
          ,date_dim
      where inv_item_sk = i_item_sk
        and inv_warehouse_sk = w_warehouse_sk
        and inv_date_sk = d_date_sk
        and d_year =2000
      group by w_warehouse_name,w_warehouse_sk,i_item_sk,d_moy) foo
 where case mean when 0 then 0 else stdev/mean end > 1)
select inv1.w_warehouse_sk   w_warehouse_sk1
      ,inv1.i_item_sk        i_item_sk1
      ,inv1.d_moy            d_moy1
      ,inv1.mean             mean1
      ,inv1.cov              cov1
      ,inv2.w_warehouse_sk   w_warehouse_sk2
      ,inv2.i_item_sk        i_item_sk2
      ,inv2.d_moy            d_moy2
      ,inv2.mean             mean2
      ,inv2.cov              cov2
into pg_temp.tpcds_q39a
from inv inv1,inv inv2
where inv1.i_item_sk = inv2.i_item_sk
  and inv1.w_warehouse_sk =  inv2.w_warehouse_sk
  and inv1.d_moy=1
  and inv2.d_moy=1+1
order by inv1.w_warehouse_sk,inv1.i_item_sk,inv1.d_moy,inv1.mean,inv1.cov
        ,inv2.d_moy,inv2.mean, inv2.cov
;


--- validation check
(SELECT * FROM pg_temp.tpcds_q39a.sql
 EXCEPT
 SELECT * FROM public.tpcds_q39a.sql);
(SELECT * FROM public.tpcds_q39a.sql
 EXCEPT
 SELECT * FROM pg_temp.tpcds_q39a.sql);
