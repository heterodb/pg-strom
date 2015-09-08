--#
--#       Gpu Scan OverFlow TestCases. 
--#

set enable_seqscan to off;
set enable_bitmapscan to off;
set enable_indexscan to off;
set random_page_cost=1000000;   --# force off index_scan.
set pg_strom.enable_gpuhashjoin to off;
set pg_strom.enable_gpupreagg to off;
set pg_strom.enable_gpusort to off;
set client_min_messages to warning;

-- overflow
select  (smlint_x+1)::smallint    from strom_overflow_test order by id limit 100;
select  (integer_x+1)::integer    from strom_overflow_test order by id limit 100;
select  (bigint_x+1)::bigint    from strom_overflow_test order by id limit 100;
select  (real_x^2)::real    from strom_overflow_test order by id limit 100;
select  (float_x^2)::float    from strom_overflow_test order by id limit 100;
select  (nume_x^10000)::numeric    from strom_overflow_test order by id limit 100;
select  (smlsrl_x^2)::smallint    from strom_overflow_test order by id limit 100;
select  (serial_x^2)::integer    from strom_overflow_test order by id limit 100;
select  (bigsrl_x^2)::bigint    from strom_overflow_test order by id limit 100;

