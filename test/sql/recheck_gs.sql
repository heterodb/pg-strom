--#
--#       Gpu Scan TestCases with CPU Recheck
--#

set pg_strom.gpu_setup_cost=0;
set pg_strom.enable_gpusort to off;
set client_min_messages to notice;

-- NO RECHECK
select * from strom_test where smlint_x<1e+48 and id%1000=0 order by id;
select * from strom_test where smlint_x>-1e+48 and id%1000=0 order by id;
select * from strom_test where smlint_x>1e-32 and id%1000=0 order by id;
select * from strom_test where smlint_x<-1e-32 and id%1000=0 order by id;

select * from strom_test where integer_x<1e+48 and id%1000=0 order by id;
select * from strom_test where integer_x>-1e+48 and id%1000=0 order by id;
select * from strom_test where integer_x>1e-32 and id%1000=0 order by id;
select * from strom_test where integer_x<-1e-32 and id%1000=0 order by id;

select * from strom_test where bigint_x<1e+48 and id%1000=0 order by id;
select * from strom_test where bigint_x>-1e+48 and id%1000=0 order by id;
select * from strom_test where bigint_x>1e-32 and id%1000=0 order by id;
select * from strom_test where bigint_x<-1e-32 and id%1000=0 order by id;

select * from strom_test where real_x::numeric<1e+10 and id%1000=0 order by id;
select * from strom_test where real_x::numeric>-1e+10 and id%1000=0 order by id;
select * from strom_test where real_x::numeric>1e-10 and id%1000=0 order by id;
select * from strom_test where real_x::numeric<-1e-10 and id%1000=0 order by id;

select * from strom_test where float_x::numeric<1e+10 and id%1000=0 order by id;
select * from strom_test where float_x::numeric>-1e+10 and id%1000=0 order by id;
select * from strom_test where float_x::numeric>1e-10 and id%1000=0 order by id;
select * from strom_test where float_x::numeric<-1e-10 and id%1000=0 order by id;

select * from strom_test where nume_x<1e+10 and id%1000=0 order by id;
select * from strom_test where nume_x>-1e+10 and id%1000=0 order by id;
select * from strom_test where nume_x>1e-10 and id%1000=0 order by id;
select * from strom_test where nume_x<-1e-10 and id%1000=0 order by id;

-- RECHECKED BY CPU.
select * from strom_test where smlint_x<1e+49 and id%1000=0 order by id;
select * from strom_test where smlint_x>-1e+49 and id%1000=0 order by id;
select * from strom_test where smlint_x>1e-33 and id%1000=0 order by id;
select * from strom_test where smlint_x<-1e-33 and id%1000=0 order by id;

select * from strom_test where integer_x<1e+49 and id%1000=0 order by id;
select * from strom_test where integer_x>-1e+49 and id%1000=0 order by id;
select * from strom_test where integer_x>1e-33 and id%1000=0 order by id;
select * from strom_test where integer_x<-1e-33 and id%1000=0 order by id;

select * from strom_test where bigint_x<1e+49 and id%1000=0 order by id;
select * from strom_test where bigint_x>-1e+49 and id%1000=0 order by id;
select * from strom_test where bigint_x>1e-33 and id%1000=0 order by id;
select * from strom_test where bigint_x<-1e-33 and id%1000=0 order by id;

select * from strom_test where real_x::numeric<1e+1000 and id%1000=0 order by id;
select * from strom_test where real_x::numeric>-1e+1000 and id%1000=0 order by id;
select * from strom_test where real_x::numeric>1e-1000 and id%1000=0 order by id;
select * from strom_test where real_x::numeric<-1e-1000 and id%1000=0 order by id;

select * from strom_test where float_x::numeric<1e+1000 and id%1000=0 order by id;
select * from strom_test where float_x::numeric>-1e+1000 and id%1000=0 order by id;
select * from strom_test where float_x::numeric>1e-1000 and id%1000=0 order by id;
select * from strom_test where float_x::numeric<-1e-1000 and id%1000=0 order by id;

select * from strom_test where nume_x<1e+1000 and id%1000=0 order by id;
select * from strom_test where nume_x>-1e+1000 and id%1000=0 order by id;
select * from strom_test where nume_x>1e-1000 and id%1000=0 order by id;
select * from strom_test where nume_x<-1e-1000 and id%1000=0 order by id;


-- division by zero with GpuScan
prepare p1 as select * from strom_test where smlint_x/(id%1000) = 1;
execute p1;
deallocate p1;

prepare p1 as select * from strom_test where integer_x/(id%1000) = 1;
execute p1;
deallocate p1;

prepare p1 as select * from strom_test where bigint_x/(id%1000) = 1;
execute p1;
deallocate p1;

prepare p1 as select * from strom_test where real_x/(id%1000) = 1;
execute p1;
deallocate p1;

prepare p1 as select * from strom_test where float_x/(id%1000) = 1;
execute p1;
deallocate p1;
