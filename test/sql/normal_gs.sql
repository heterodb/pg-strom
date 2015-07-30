--#
--#       Gpu Scan Simple TestCases. 
--#

set enable_seqscan to off;
set enable_bitmapscan to off;
set enable_indexscan to off;
set random_page_cost=1000000;   --# force off index_scan.
set pg_strom.enable_gpuhashjoin to off;
set pg_strom.enable_gpupreagg to off;
set pg_strom.enable_gpusort to off;
set client_min_messages to warning;

-- normal
select  smlint_x    from strom_test order by id limit 100;
select  integer_x    from strom_test order by id limit 100;
select  bigint_x    from strom_test order by id limit 100;
select  real_x    from strom_test order by id limit 100;
select  float_x    from strom_test order by id limit 100;
select  nume_x    from strom_test order by id limit 100;
select  smlsrl_x    from strom_test order by id limit 100;
select  serial_x    from strom_test order by id limit 100;
select  bigsrl_x    from strom_test order by id limit 100;

-- where
select  smlint_x    from strom_test where abs(smlint_x) between 1 and 1000 order by id limit 100;
select  integer_x    from strom_test where abs(integer_x) between 100000 and 1000000 order by id limit 100;
select  bigint_x    from strom_test where abs(bigint_x) between 1000000000000 and 10000000000000 order by id limit 100;
select  real_x    from strom_test where abs(real_x) between 0.001 and 0.01 order by id limit 100;
select  float_x    from strom_test where abs(float_x) between 0.001 and 0.01 order by id limit 100;
select  nume_x    from strom_test where abs(nume_x) between 0.001 and 0.01 order by id limit 100;
select  smlsrl_x    from strom_test where abs(smlsrl_x) between 1 and 1000 order by id limit 100;
select  serial_x    from strom_test where abs(serial_x) between 100000 and 1000000 order by id limit 100;
select  bigsrl_x    from strom_test where abs(bigsrl_x) between 1000000000000 and 10000000000000 order by id limit 100;

-- NULL
select  id,smlint_x    from strom_test where abs(smlint_x) IS NULL order by id limit 100;
select  id,integer_x    from strom_test where abs(integer_x) IS NULL order by id limit 100;
select  id,bigint_x    from strom_test where abs(bigint_x) IS NULL order by id limit 100;
select  id,real_x    from strom_test where abs(real_x) IS NULL order by id limit 100;
select  id,float_x    from strom_test where abs(float_x) IS NULL order by id limit 100;
select  id,nume_x    from strom_test where abs(nume_x) IS NULL order by id limit 100;
select  id,smlsrl_x    from strom_test where abs(smlsrl_x) IS NULL order by id limit 100;
select  id,serial_x    from strom_test where abs(serial_x) IS NULL order by id limit 100;
select  id,bigsrl_x    from strom_test where abs(bigsrl_x) IS NULL order by id limit 100;


-- NOT NULL
select  id,smlint_x    from strom_test where abs(smlint_x) IS NOT NULL order by id limit 100;
select  id,integer_x    from strom_test where abs(integer_x) IS NOT NULL order by id limit 100;
select  id,bigint_x    from strom_test where abs(bigint_x) IS NOT NULL order by id limit 100;
select  id,real_x    from strom_test where abs(real_x) IS NOT NULL order by id limit 100;
select  id,float_x    from strom_test where abs(float_x) IS NOT NULL order by id limit 100;
select  id,nume_x    from strom_test where abs(nume_x) IS NOT NULL order by id limit 100;
select  id,smlsrl_x    from strom_test where abs(smlsrl_x) IS NOT NULL order by id limit 100;
select  id,serial_x    from strom_test where abs(serial_x) IS NOT NULL order by id limit 100;
select  id,bigsrl_x    from strom_test where abs(bigsrl_x) IS NOT NULL order by id limit 100;
