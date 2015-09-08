--#
--#       Gpu Sort by Grouping TestCases. 
--#

set pg_strom.debug_force_gpusort to on;
set pg_strom.gpu_setup_cost=0;
set random_page_cost=1000000;   --# force off index_scan.   
set pg_strom.enable_gpusort to on;                                                                                                                                       
set client_min_messages to warning;
set enable_hashagg to off;      --# force off HashAggregate

--smallint
select row_number() over () as rowid, smlint_x from strom_test where id between     1 and 10000 group by smlint_x;
select row_number() over () as rowid, smlint_x from strom_test where id between 10001 and 20000 group by smlint_x;
select row_number() over () as rowid, smlint_x from strom_test where id between 20001 and 30000 group by smlint_x;
select row_number() over () as rowid, smlint_x from strom_test where id between 30001 and 40000 group by smlint_x;

--integer
select * from ( select row_number() over () as rowid, integer_x from strom_test where id between     1 and 10000 group by integer_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 10001 and 20000 group by integer_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 20001 and 30000 group by integer_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 30001 and 40000 group by integer_x ) as t;

--bigint
select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between     1 and 10000 group by bigint_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 10001 and 20000 group by bigint_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 20001 and 30000 group by bigint_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 30001 and 40000 group by bigint_x ) as t;

--real
select * from ( select row_number() over () as rowid, real_x from strom_test where id between     1 and 10000 group by real_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, real_x from strom_test where id between 10001 and 20000 group by real_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, real_x from strom_test where id between 20001 and 30000 group by real_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, real_x from strom_test where id between 30001 and 40000 group by real_x ) as t;

--float
select * from ( select row_number() over () as rowid, float_x from strom_test where id between     1 and 10000 group by float_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, float_x from strom_test where id between 10001 and 20000 group by float_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, float_x from strom_test where id between 20001 and 30000 group by float_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, float_x from strom_test where id between 30001 and 40000 group by float_x ) as t;

--numeric
select * from ( select row_number() over () as rowid, nume_x from strom_test where id between     1 and 10000 group by nume_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 10001 and 20000 group by nume_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 20001 and 30000 group by nume_x ) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 30001 and 40000 group by nume_x ) as t;
