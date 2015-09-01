--#
--#       Gpu Sort TestCases with Date/Time types.
--#   [TODO] Time/Interval type has not implemented yet.
--#          If their types will be able to sort by Gpu,
--#          then we need refactor this test cases.
--#

set pg_strom.gpu_setup_cost=0;
set enable_hashagg to off;      --# force off HashAggregate
set random_page_cost=1000000;   --# force off index_scan.   
set pg_strom.enable_gpusort to on;
set pg_strom.enable_gpuhashjoin to off;                                                                                                                                      
set client_min_messages to warning;
set datestyle to 'Postgres, MDY';

--gpusort order by date/time types.

-- timezone JST-9
set timezone to 'JST-9';
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timestamp_x from strom_time_test group by timestamp_x ) as t where t.rowid%50=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timestamptz_x from strom_time_test group by timestamptz_x ) as t where t.rowid%50=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, date_x from strom_time_test group by date_x ) as t where t.rowid%50=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, time_x from strom_time_test group by time_x ) as t where t.rowid%50=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timetz_x from strom_time_test group by timetz_x ) as t where t.rowid%50=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, interval_x from strom_time_test group by interval_x ) as t where t.rowid%50=0;

-- timezone America/New_York
set timezone to 'America/New_York';
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timestamp_x from strom_time_test group by timestamp_x ) as t where t.rowid%50=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timestamptz_x from strom_time_test group by timestamptz_x ) as t where t.rowid%50=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, date_x from strom_time_test group by date_x ) as t where t.rowid%50=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, time_x from strom_time_test group by time_x ) as t where t.rowid%50=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timetz_x from strom_time_test group by timetz_x ) as t where t.rowid%50=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, interval_x from strom_time_test group by interval_x ) as t where t.rowid%50=0;

-- timezone Europe/Moscow
set timezone to 'Europe/Moscow';
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamp_x from (select timestamp_x, row_number() over (order by timestamp_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timestamp_x from strom_time_test group by timestamp_x ) as t where t.rowid%50=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timestamptz_x from (select timestamptz_x, row_number() over (order by timestamptz_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timestamptz_x from strom_time_test group by timestamptz_x ) as t where t.rowid%50=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,date_x from (select date_x, row_number() over (order by date_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, date_x from strom_time_test group by date_x ) as t where t.rowid%50=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,time_x from (select time_x, row_number() over (order by time_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, time_x from strom_time_test group by time_x ) as t where t.rowid%50=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,timetz_x from (select timetz_x, row_number() over (order by timetz_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, timetz_x from strom_time_test group by timetz_x ) as t where t.rowid%50=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x asc ) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x asc NULLS FIRST) as rowid from strom_time_test) as t where t.rowid%100=0;
select rowid,interval_x from (select interval_x, row_number() over (order by interval_x desc NULLS LAST) as rowid from strom_time_test) as t where t.rowid%100=0;
select * from ( select row_number() over () as rowid, interval_x from strom_time_test group by interval_x ) as t where t.rowid%50=0;
