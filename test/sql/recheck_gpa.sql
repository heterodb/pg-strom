--#
--#       Gpu PreAggregate TestCases with RECHECK
--#

set pg_strom.gpu_setup_cost=0;
set pg_strom.debug_force_gpupreagg to on;
set pg_strom.enable_gpusort to off;
set client_min_messages to notice;
set extra_float_digits to -3;

-- NO RECHECK
select sum(0);
select sum(1E+48);
select sum(1E-32);
-- RECHECKED BY CPU.
select sum(1E-33);
select sum(1E+49);
select sum(1E+1000);
select sum(1E-1000);

-- division by zero with GpuPreAggregate
prepare p1 as select sum(smlint_x/(id%1000)) from strom_test;
execute p1;
deallocate p1;

prepare p1 as select sum(integer_x/(id%1000)) from strom_test;
execute p1;
deallocate p1;

prepare p1 as select sum(bigint_x/(id%1000)) from strom_test;
execute p1;
deallocate p1;

prepare p1 as select sum(real_x/(id%1000)) from strom_test;
execute p1;
deallocate p1;

prepare p1 as select sum(float_x/(id%1000)) from strom_test;
execute p1;
deallocate p1;
