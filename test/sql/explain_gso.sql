--#
--#       Gpu Scan Explain TestCases. 
--#  [TODO] Do not test commented-out queries until GPUSort supports TOAST data process. 
--#         If will support it, please remake expected outs and test_init.sql
--#

set pg_strom.debug_force_gpusort to on;
set enable_hashagg to off;      --# force off HashAggregate
set random_page_cost=1000000;   --# force off index_scan.
set pg_strom.enable_gpuhashjoin to off;
set pg_strom.enable_gpusort to off;
set client_min_messages to warning;

set pg_strom.enabled=off;

-- normal case
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

--grouping
explain (verbose, costs off, timing off) select row_number() over () as rowid, smlint_x from strom_test where id between     1 and 10000 group by smlint_x;
explain (verbose, costs off, timing off) select row_number() over () as rowid, smlint_x from strom_test where id between 10001 and 20000 group by smlint_x;
explain (verbose, costs off, timing off) select row_number() over () as rowid, smlint_x from strom_test where id between 20001 and 30000 group by smlint_x;
explain (verbose, costs off, timing off) select row_number() over () as rowid, smlint_x from strom_test where id between 30001 and 40000 group by smlint_x;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, integer_x from strom_test where id between     1 and 10000 group by integer_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 10001 and 20000 group by integer_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 20001 and 30000 group by integer_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 30001 and 40000 group by integer_x ) as t;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between     1 and 10000 group by bigint_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 10001 and 20000 group by bigint_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 20001 and 30000 group by bigint_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 30001 and 40000 group by bigint_x ) as t;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, real_x from strom_test where id between     1 and 10000 group by real_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, real_x from strom_test where id between 10001 and 20000 group by real_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, real_x from strom_test where id between 20001 and 30000 group by real_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, real_x from strom_test where id between 30001 and 40000 group by real_x ) as t;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, float_x from strom_test where id between     1 and 10000 group by float_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, float_x from strom_test where id between 10001 and 20000 group by float_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, float_x from strom_test where id between 20001 and 30000 group by float_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, float_x from strom_test where id between 30001 and 40000 group by float_x ) as t;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, nume_x from strom_test where id between     1 and 10000 group by nume_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 10001 and 20000 group by nume_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 20001 and 30000 group by nume_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 30001 and 40000 group by nume_x ) as t;

--merge
set enable_hashjoin to off;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.smlint_x desc) as rowid,t1.smlint_x,t2.smlint_x from strom_test t1, strom_test t2 where t1.smlint_x=t2.smlint_x and t1.id%100=0) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.smlint_x asc) as rowid,t1.smlint_x,t2.smlint_x from strom_test t1, strom_test t2 where t1.smlint_x=t2.smlint_x and t1.id%100=0) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.integer_x desc) as rowid,t1.integer_x,t2.integer_x from strom_test t1, strom_test t2 where t1.integer_x=t2.integer_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.integer_x asc) as rowid,t1.integer_x,t2.integer_x from strom_test t1, strom_test t2 where t1.integer_x=t2.integer_x and t1.id%100=0) as t where t.rowid%10=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.bigint_x desc) as rowid,t1.bigint_x,t2.bigint_x from strom_test t1, strom_test t2 where t1.bigint_x=t2.bigint_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.bigint_x asc) as rowid,t1.bigint_x,t2.bigint_x from strom_test t1, strom_test t2 where t1.bigint_x=t2.bigint_x and t1.id%100=0) as t where t.rowid%10=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.real_x desc) as rowid,t1.real_x,t2.real_x from strom_test t1, strom_test t2 where t1.real_x=t2.real_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.real_x asc) as rowid,t1.real_x,t2.real_x from strom_test t1, strom_test t2 where t1.real_x=t2.real_x and t1.id%100=0) as t where t.rowid%10=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.float_x desc) as rowid,t1.float_x,t2.float_x from strom_test t1, strom_test t2 where t1.float_x=t2.float_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.float_x asc) as rowid,t1.float_x,t2.float_x from strom_test t1, strom_test t2 where t1.float_x=t2.float_x and t1.id%100=0) as t where t.rowid%10=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.nume_x desc) as rowid,t1.nume_x,t2.nume_x from strom_test t1, strom_test t2 where t1.nume_x=t2.nume_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.nume_x asc) as rowid,t1.nume_x,t2.nume_x from strom_test t1, strom_test t2 where t1.nume_x=t2.nume_x and t1.id%100=0) as t where t.rowid%10=0;

--multikey
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x desc,nume_x desc,smlint_x desc,integer_x desc,bigint_x desc,real_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x  asc,nume_x  asc,smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x  asc,nume_x desc,smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x desc,nume_x  asc,smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x desc,nume_x desc,smlint_x desc,integer_x desc,bigint_x desc,real_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x  asc,nume_x  asc,smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x  asc,nume_x desc,smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x desc,nume_x  asc,smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;

--gpusort on zero table
explain (verbose, costs off, timing off) select * from strom_zero_test order by smlint_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by integer_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by bigint_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by real_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by float_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by nume_x;

--gpusort by text-key
explain (verbose, costs off, timing off) select * from (select row_number() over (order by char_x desc) as rowid,char_x from strom_string_test) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by char_x asc) as rowid,char_x from strom_string_test) as t where t.rowid%100=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by nchar_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by nchar_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by vchar_x desc) as rowid,vchar_x from strom_string_test) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by vchar_x asc) as rowid,vchar_x from strom_string_test) as t where t.rowid%100=0;

-- explain (verbose, costs off, timing off) select * from (select row_number() over (order by nvchar_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
-- explain (verbose, costs off, timing off) select * from (select row_number() over (order by nvchar_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;

-- explain (verbose, costs off, timing off) select * from (select row_number() over (order by text_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
-- explain (verbose, costs off, timing off) select * from (select row_number() over (order by text_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;

set pg_strom.enabled=on;
set pg_strom.gpu_setup_cost=0;
set enable_hashagg to off;      --# force off HashAggregate
set random_page_cost=1000000;   --# force off index_scan.   
set pg_strom.enable_gpusort to on;
set pg_strom.enable_gpuhashjoin to off;                                                                                                                                       
set client_min_messages to warning;

-- normal case
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,smlint_x from (select smlint_x, row_number() over (order by smlint_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,integer_x from (select integer_x, row_number() over (order by integer_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,bigint_x from (select bigint_x, row_number() over (order by bigint_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,real_x from (select real_x, row_number() over (order by real_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,float_x from (select float_x, row_number() over (order by float_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x desc) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select rowid,nume_x from (select nume_x, row_number() over (order by nume_x asc ) as rowid from strom_test where id between 30001 and 40000) as t where t.rowid%1000=0;

--grouping
explain (verbose, costs off, timing off) select row_number() over () as rowid, smlint_x from strom_test where id between     1 and 10000 group by smlint_x;
explain (verbose, costs off, timing off) select row_number() over () as rowid, smlint_x from strom_test where id between 10001 and 20000 group by smlint_x;
explain (verbose, costs off, timing off) select row_number() over () as rowid, smlint_x from strom_test where id between 20001 and 30000 group by smlint_x;
explain (verbose, costs off, timing off) select row_number() over () as rowid, smlint_x from strom_test where id between 30001 and 40000 group by smlint_x;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, integer_x from strom_test where id between     1 and 10000 group by integer_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 10001 and 20000 group by integer_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 20001 and 30000 group by integer_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, integer_x from strom_test where id between 30001 and 40000 group by integer_x ) as t;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between     1 and 10000 group by bigint_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 10001 and 20000 group by bigint_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 20001 and 30000 group by bigint_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, bigint_x from strom_test where id between 30001 and 40000 group by bigint_x ) as t;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, real_x from strom_test where id between     1 and 10000 group by real_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, real_x from strom_test where id between 10001 and 20000 group by real_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, real_x from strom_test where id between 20001 and 30000 group by real_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, real_x from strom_test where id between 30001 and 40000 group by real_x ) as t;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, float_x from strom_test where id between     1 and 10000 group by float_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, float_x from strom_test where id between 10001 and 20000 group by float_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, float_x from strom_test where id between 20001 and 30000 group by float_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, float_x from strom_test where id between 30001 and 40000 group by float_x ) as t;

explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, nume_x from strom_test where id between     1 and 10000 group by nume_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 10001 and 20000 group by nume_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 20001 and 30000 group by nume_x ) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from ( select row_number() over () as rowid, nume_x from strom_test where id between 30001 and 40000 group by nume_x ) as t;

--merge
set enable_hashjoin to off;
set pg_strom.enable_gpuhashjoin to off;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.smlint_x desc) as rowid,t1.smlint_x,t2.smlint_x from strom_test t1, strom_test t2 where t1.smlint_x=t2.smlint_x and t1.id%100=0) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.smlint_x asc) as rowid,t1.smlint_x,t2.smlint_x from strom_test t1, strom_test t2 where t1.smlint_x=t2.smlint_x and t1.id%100=0) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.integer_x desc) as rowid,t1.integer_x,t2.integer_x from strom_test t1, strom_test t2 where t1.integer_x=t2.integer_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.integer_x asc) as rowid,t1.integer_x,t2.integer_x from strom_test t1, strom_test t2 where t1.integer_x=t2.integer_x and t1.id%100=0) as t where t.rowid%10=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.bigint_x desc) as rowid,t1.bigint_x,t2.bigint_x from strom_test t1, strom_test t2 where t1.bigint_x=t2.bigint_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.bigint_x asc) as rowid,t1.bigint_x,t2.bigint_x from strom_test t1, strom_test t2 where t1.bigint_x=t2.bigint_x and t1.id%100=0) as t where t.rowid%10=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.real_x desc) as rowid,t1.real_x,t2.real_x from strom_test t1, strom_test t2 where t1.real_x=t2.real_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.real_x asc) as rowid,t1.real_x,t2.real_x from strom_test t1, strom_test t2 where t1.real_x=t2.real_x and t1.id%100=0) as t where t.rowid%10=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.float_x desc) as rowid,t1.float_x,t2.float_x from strom_test t1, strom_test t2 where t1.float_x=t2.float_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.float_x asc) as rowid,t1.float_x,t2.float_x from strom_test t1, strom_test t2 where t1.float_x=t2.float_x and t1.id%100=0) as t where t.rowid%10=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.nume_x desc) as rowid,t1.nume_x,t2.nume_x from strom_test t1, strom_test t2 where t1.nume_x=t2.nume_x and t1.id%100=0) as t where t.rowid%10=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by t1.nume_x asc) as rowid,t1.nume_x,t2.nume_x from strom_test t1, strom_test t2 where t1.nume_x=t2.nume_x and t1.id%100=0) as t where t.rowid%10=0;

--multikey
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x desc,nume_x desc,smlint_x desc,integer_x desc,bigint_x desc,real_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x  asc,nume_x  asc,smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x  asc,nume_x desc,smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x desc,nume_x  asc,smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc) as rowid,* from strom_test where id between     1 and 10000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x desc,nume_x desc,smlint_x desc,integer_x desc,bigint_x desc,real_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x  asc,nume_x  asc,smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x  asc,nume_x desc,smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by float_x desc,nume_x  asc,smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc) as rowid,* from strom_test where id between 10001 and 20000) as t where t.rowid%1000=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x desc,bigint_x desc,real_x desc,float_x desc,nume_x desc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x  asc,bigint_x  asc,real_x  asc,float_x  asc,nume_x  asc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x  asc,integer_x desc,bigint_x  asc,real_x desc,float_x  asc,nume_x desc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by smlint_x desc,integer_x  asc,bigint_x desc,real_x  asc,float_x desc,nume_x  asc) as rowid,* from strom_test where id between 20001 and 30000) as t where t.rowid%1000=0;

--gpusort on zero table
explain (verbose, costs off, timing off) select * from strom_zero_test order by smlint_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by integer_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by bigint_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by real_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by float_x;
explain (verbose, costs off, timing off) select * from strom_zero_test order by nume_x;

--gpusort by text-key
explain (verbose, costs off, timing off) select * from (select row_number() over (order by char_x desc) as rowid,char_x from strom_string_test) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by char_x asc) as rowid,char_x from strom_string_test) as t where t.rowid%100=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by nchar_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by nchar_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;

explain (verbose, costs off, timing off) select * from (select row_number() over (order by vchar_x desc) as rowid,vchar_x from strom_string_test) as t where t.rowid%100=0;
explain (verbose, costs off, timing off) select * from (select row_number() over (order by vchar_x asc) as rowid,vchar_x from strom_string_test) as t where t.rowid%100=0;

-- explain (verbose, costs off, timing off) select * from (select row_number() over (order by nvchar_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
-- explain (verbose, costs off, timing off) select * from (select row_number() over (order by nvchar_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;

-- explain (verbose, costs off, timing off) select * from (select row_number() over (order by text_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
-- explain (verbose, costs off, timing off) select * from (select row_number() over (order by text_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
