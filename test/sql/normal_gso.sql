--#
--#       Gpu Sort Simple TestCases. 
--#

set pg_strom.debug_force_gpusort to on;
set pg_strom.gpu_setup_cost=0;
set random_page_cost=1000000;   --# force off index_scan.   
set pg_strom.enable_gpusort to on;                                                                                                                                       
set client_min_messages to warning;

select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

