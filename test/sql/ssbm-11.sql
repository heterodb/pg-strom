-- Q1_1
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,date1
where lo_orderdate = d_datekey
and d_year = 1993
and lo_discount between 1 and 3
and lo_quantity < 25;
