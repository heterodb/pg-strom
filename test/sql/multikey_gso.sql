--#
--#       Gpu Sort Multi-Key TestCases. 
--#

set pg_strom.debug_force_gpusort to on;
set pg_strom.gpu_setup_cost=0;
set random_page_cost=1000000;   --# force off index_scan.   
set pg_strom.enable_gpusort to on;                                                                                                                                       
set client_min_messages to warning;

select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;

select * from (select row_number() over (order by float_x desc,nume_x desc,smlint_x desc,integer_x desc,bigint_x desc,real_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by float_x  asc,nume_x  asc,smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by float_x  asc,nume_x desc,smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by float_x desc,nume_x  asc,smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;

select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;

select * from (select row_number() over (order by float_x desc,nume_x desc,smlint_x desc,integer_x desc,bigint_x desc,real_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by float_x  asc,nume_x  asc,smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by float_x  asc,nume_x desc,smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by float_x desc,nume_x  asc,smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;

select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
