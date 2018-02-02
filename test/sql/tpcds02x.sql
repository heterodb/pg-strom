
with wscs as
 (select sold_date_sk
        ,sales_price
  from  (select ws_sold_date_sk sold_date_sk
              ,ws_ext_sales_price sales_price
        from web_sales 
        union all
        select cs_sold_date_sk sold_date_sk
              ,cs_ext_sales_price sales_price
        from catalog_sales) u),
 wswscs as 
 (select d_week_seq,
        sum(sales_price::real) FILTER (WHERE d_day_name='Sunday') sun_sales,
        sum(sales_price::real) FILTER (WHERE d_day_name='Monday') mon_sales,
        sum(sales_price::real) FILTER (WHERE d_day_name='Tuesday') tue_sales,
        sum(sales_price::real) FILTER (WHERE d_day_name='Wednesday') wed_sales,
        sum(sales_price::real) FILTER (WHERE d_day_name='Thursday') thu_sales,
        sum(sales_price::real) FILTER (WHERE d_day_name='Friday') fri_sales,
        sum(sales_price::real) FILTER (WHERE d_day_name='Saturday') sat_sales
 from wscs
     ,date_dim
 where d_date_sk = sold_date_sk
 group by d_week_seq)
 select d_week_seq1
       ,sun_sales1/sun_sales2::numeric(12,2)  a
       ,mon_sales1/mon_sales2::numeric(12,2)  b
       ,tue_sales1/tue_sales2::numeric(12,2)  c
       ,wed_sales1/wed_sales2::numeric(12,2)  d
       ,thu_sales1/thu_sales2::numeric(12,2)  e
       ,fri_sales1/fri_sales2::numeric(12,2)  f
       ,sat_sales1/sat_sales2::numeric(12,2)  g
 into tpcds_q02
 from
 (select wswscs.d_week_seq d_week_seq1
        ,sun_sales sun_sales1
        ,mon_sales mon_sales1
        ,tue_sales tue_sales1
        ,wed_sales wed_sales1
        ,thu_sales thu_sales1
        ,fri_sales fri_sales1
        ,sat_sales sat_sales1
  from wswscs,date_dim 
  where date_dim.d_week_seq = wswscs.d_week_seq and
        d_year = 2001) y,
 (select wswscs.d_week_seq d_week_seq2
        ,sun_sales sun_sales2
        ,mon_sales mon_sales2
        ,tue_sales tue_sales2
        ,wed_sales wed_sales2
        ,thu_sales thu_sales2
        ,fri_sales fri_sales2
        ,sat_sales sat_sales2
  from wswscs
      ,date_dim 
  where date_dim.d_week_seq = wswscs.d_week_seq and
        d_year = 2001+1) z
 where d_week_seq1=d_week_seq2-53
 order by d_week_seq1;


