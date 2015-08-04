--#
--#       Gpu Text Sort TestCases. 
--#  [TODO] Do not test commented-out queries until GPUSort supports TOAST data process. 
--#         If will support it, please remake expected outs and test_init.sql
--#

set pg_strom.debug_force_gpusort to on;
set pg_strom.gpu_setup_cost=0;
set random_page_cost=1000000;   --# force off index_scan.   
set pg_strom.enable_gpusort to on;                                                                                                                                       
set client_min_messages to warning;


select * from (select row_number() over (order by char_x desc) as rowid,char_x from strom_string_test) as t where t.rowid%100=0;
select * from (select row_number() over (order by char_x asc) as rowid,char_x from strom_string_test) as t where t.rowid%100=0;

select * from (select row_number() over (order by nchar_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
select * from (select row_number() over (order by nchar_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;

select * from (select row_number() over (order by vchar_x desc) as rowid,vchar_x from strom_string_test) as t where t.rowid%100=0;
select * from (select row_number() over (order by vchar_x asc) as rowid,vchar_x from strom_string_test) as t where t.rowid%100=0;

-- select * from (select row_number() over (order by nvchar_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
-- select * from (select row_number() over (order by nvchar_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;

-- select * from (select row_number() over (order by text_x desc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
-- select * from (select row_number() over (order by text_x asc) as rowid,id from strom_string_test) as t where t.rowid%100=0;
