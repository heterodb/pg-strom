--#
--#       Gpu Sort in Merge Join TestCases. 
--#

set pg_strom.debug_force_gpusort to on;
set pg_strom.gpu_setup_cost=0;
set random_page_cost=1000000;   --# force off index_scan.   
set pg_strom.enable_gpusort to on;                                                                                                                                       
set client_min_messages to warning;
set pg_strom.enable_gpuhashjoin to off;
set enable_hashjoin to off;

--smallint
prepare t1 as select * from (select row_number() over (order by t1.smlint_x desc) as rowid,t1.smlint_x,t2.smlint_x from strom_test t1, strom_test t2 where t1.smlint_x=t2.smlint_x and t1.id%100=0) as t where t.rowid%1000=0;
execute t1;
deallocate t1;

prepare t1 as select * from (select row_number() over (order by t1.smlint_x asc) as rowid,t1.smlint_x,t2.smlint_x from strom_test t1, strom_test t2 where t1.smlint_x=t2.smlint_x and t1.id%100=0) as t where t.rowid%1000=0;
execute t1;
deallocate t1;

--integer
prepare t1 as select * from (select row_number() over (order by t1.integer_x desc) as rowid,t1.integer_x,t2.integer_x from strom_test t1, strom_test t2 where t1.integer_x=t2.integer_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

prepare t1 as select * from (select row_number() over (order by t1.integer_x asc) as rowid,t1.integer_x,t2.integer_x from strom_test t1, strom_test t2 where t1.integer_x=t2.integer_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

--bigint
prepare t1 as select * from (select row_number() over (order by t1.bigint_x desc) as rowid,t1.bigint_x,t2.bigint_x from strom_test t1, strom_test t2 where t1.bigint_x=t2.bigint_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

prepare t1 as select * from (select row_number() over (order by t1.bigint_x asc) as rowid,t1.bigint_x,t2.bigint_x from strom_test t1, strom_test t2 where t1.bigint_x=t2.bigint_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

--real
prepare t1 as select * from (select row_number() over (order by t1.real_x desc) as rowid,t1.real_x,t2.real_x from strom_test t1, strom_test t2 where t1.real_x=t2.real_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

prepare t1 as select * from (select row_number() over (order by t1.real_x asc) as rowid,t1.real_x,t2.real_x from strom_test t1, strom_test t2 where t1.real_x=t2.real_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

--float
prepare t1 as select * from (select row_number() over (order by t1.float_x desc) as rowid,t1.float_x,t2.float_x from strom_test t1, strom_test t2 where t1.float_x=t2.float_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

prepare t1 as select * from (select row_number() over (order by t1.float_x asc) as rowid,t1.float_x,t2.float_x from strom_test t1, strom_test t2 where t1.float_x=t2.float_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

--numeric
prepare t1 as select * from (select row_number() over (order by t1.nume_x desc) as rowid,t1.nume_x,t2.nume_x from strom_test t1, strom_test t2 where t1.nume_x=t2.nume_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

prepare t1 as select * from (select row_number() over (order by t1.nume_x asc) as rowid,t1.nume_x,t2.nume_x from strom_test t1, strom_test t2 where t1.nume_x=t2.nume_x and t1.id%100=0) as t where t.rowid%10=0;
execute t1;
deallocate t1;

