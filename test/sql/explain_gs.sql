--#
--#       Gpu Scan Explain TestCases. 
--#

set enable_seqscan to off;
set enable_bitmapscan to off;
set enable_indexscan to off;
set random_page_cost=1000000;   --# force off index_scan.
set pg_strom.enable_gpuhashjoin to off;
set pg_strom.enable_gpupreagg to off;
set pg_strom.enable_gpusort to off;
set client_min_messages to warning;

set pg_strom.enabled=off;
-- normal
explain (verbose, costs off, timing off) select  smlint_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  integer_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  bigint_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  real_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  float_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  nume_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  smlsrl_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  serial_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  bigsrl_x    from strom_test order by id limit 100;

-- where
explain (verbose, costs off, timing off) select  smlint_x    from strom_test where abs(smlint_x) between 1 and 1000 order by id limit 100;
explain (verbose, costs off, timing off) select  integer_x    from strom_test where abs(integer_x) between 100000 and 1000000 order by id limit 100;
explain (verbose, costs off, timing off) select  bigint_x    from strom_test where abs(bigint_x) between 1000000000000 and 10000000000000 order by id limit 100;
explain (verbose, costs off, timing off) select  real_x    from strom_test where abs(real_x) between 0.001 and 0.01 order by id limit 100;
explain (verbose, costs off, timing off) select  float_x    from strom_test where abs(float_x) between 0.001 and 0.01 order by id limit 100;
explain (verbose, costs off, timing off) select  nume_x    from strom_test where abs(nume_x) between 0.001 and 0.01 order by id limit 100;
explain (verbose, costs off, timing off) select  smlsrl_x    from strom_test where abs(smlsrl_x) between 1 and 1000 order by id limit 100;
explain (verbose, costs off, timing off) select  serial_x    from strom_test where abs(serial_x) between 100000 and 1000000 order by id limit 100;
explain (verbose, costs off, timing off) select  bigsrl_x    from strom_test where abs(bigsrl_x) between 1000000000000 and 10000000000000 order by id limit 100;

-- NULL
explain (verbose, costs off, timing off) select  id,smlint_x    from strom_test where abs(smlint_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,integer_x    from strom_test where abs(integer_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,bigint_x    from strom_test where abs(bigint_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,real_x    from strom_test where abs(real_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,float_x    from strom_test where abs(float_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,nume_x    from strom_test where abs(nume_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,smlsrl_x    from strom_test where abs(smlsrl_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,serial_x    from strom_test where abs(serial_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,bigsrl_x    from strom_test where abs(bigsrl_x) IS NULL order by id limit 100;


-- NOT NULL
explain (verbose, costs off, timing off) select  id,smlint_x    from strom_test where abs(smlint_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,integer_x    from strom_test where abs(integer_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,bigint_x    from strom_test where abs(bigint_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,real_x    from strom_test where abs(real_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,float_x    from strom_test where abs(float_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,nume_x    from strom_test where abs(nume_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,smlsrl_x    from strom_test where abs(smlsrl_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,serial_x    from strom_test where abs(serial_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,bigsrl_x    from strom_test where abs(bigsrl_x) IS NOT NULL order by id limit 100;

set pg_strom.enabled=on;
set pg_strom.enable_gpusort to off;
-- normal
explain (verbose, costs off, timing off) select  smlint_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  integer_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  bigint_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  real_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  float_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  nume_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  smlsrl_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  serial_x    from strom_test order by id limit 100;
explain (verbose, costs off, timing off) select  bigsrl_x    from strom_test order by id limit 100;

-- where
explain (verbose, costs off, timing off) select  smlint_x    from strom_test where abs(smlint_x) between 1 and 1000 order by id limit 100;
explain (verbose, costs off, timing off) select  integer_x    from strom_test where abs(integer_x) between 100000 and 1000000 order by id limit 100;
explain (verbose, costs off, timing off) select  bigint_x    from strom_test where abs(bigint_x) between 1000000000000 and 10000000000000 order by id limit 100;
explain (verbose, costs off, timing off) select  real_x    from strom_test where abs(real_x) between 0.001 and 0.01 order by id limit 100;
explain (verbose, costs off, timing off) select  float_x    from strom_test where abs(float_x) between 0.001 and 0.01 order by id limit 100;
explain (verbose, costs off, timing off) select  nume_x    from strom_test where abs(nume_x) between 0.001 and 0.01 order by id limit 100;
explain (verbose, costs off, timing off) select  smlsrl_x    from strom_test where abs(smlsrl_x) between 1 and 1000 order by id limit 100;
explain (verbose, costs off, timing off) select  serial_x    from strom_test where abs(serial_x) between 100000 and 1000000 order by id limit 100;
explain (verbose, costs off, timing off) select  bigsrl_x    from strom_test where abs(bigsrl_x) between 1000000000000 and 10000000000000 order by id limit 100;

-- NULL
explain (verbose, costs off, timing off) select  id,smlint_x    from strom_test where abs(smlint_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,integer_x    from strom_test where abs(integer_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,bigint_x    from strom_test where abs(bigint_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,real_x    from strom_test where abs(real_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,float_x    from strom_test where abs(float_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,nume_x    from strom_test where abs(nume_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,smlsrl_x    from strom_test where abs(smlsrl_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,serial_x    from strom_test where abs(serial_x) IS NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,bigsrl_x    from strom_test where abs(bigsrl_x) IS NULL order by id limit 100;


-- NOT NULL
explain (verbose, costs off, timing off) select  id,smlint_x    from strom_test where abs(smlint_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,integer_x    from strom_test where abs(integer_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,bigint_x    from strom_test where abs(bigint_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,real_x    from strom_test where abs(real_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,float_x    from strom_test where abs(float_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,nume_x    from strom_test where abs(nume_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,smlsrl_x    from strom_test where abs(smlsrl_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,serial_x    from strom_test where abs(serial_x) IS NOT NULL order by id limit 100;
explain (verbose, costs off, timing off) select  id,bigsrl_x    from strom_test where abs(bigsrl_x) IS NOT NULL order by id limit 100;


-- division by zero with GpuScan
set pg_strom.enabled=on;
explain (verbose on, costs off) select * from strom_test where smlint_x/(id%1000) = 1;
explain (verbose on, costs off) select * from strom_test where integer_x/(id%1000) = 1;
explain (verbose on, costs off) select * from strom_test where bigint_x/(id%1000) = 1;
explain (verbose on, costs off) select * from strom_test where real_x/(id%1000) = 1;
explain (verbose on, costs off) select * from strom_test where float_x/(id%1000) = 1;

set pg_strom.enabled=off;
explain (verbose on, costs off) select * from strom_test where smlint_x/(id%1000) = 1;
explain (verbose on, costs off) select * from strom_test where integer_x/(id%1000) = 1;
explain (verbose on, costs off) select * from strom_test where bigint_x/(id%1000) = 1;
explain (verbose on, costs off) select * from strom_test where real_x/(id%1000) = 1;
explain (verbose on, costs off) select * from strom_test where float_x/(id%1000) = 1;
