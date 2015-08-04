--#
--#       Gpu Sort on Zero-record Table TestCases. 
--#

set pg_strom.debug_force_gpusort to on;
set pg_strom.gpu_setup_cost=0;
set random_page_cost=1000000;   --# force off index_scan.   
set pg_strom.enable_gpusort to on;                                                                                                                                       
set client_min_messages to warning;

select * from strom_zero_test order by smlint_x;
select * from strom_zero_test order by integer_x;
select * from strom_zero_test order by bigint_x;
select * from strom_zero_test order by real_x;
select * from strom_zero_test order by float_x;
select * from strom_zero_test order by nume_x;
