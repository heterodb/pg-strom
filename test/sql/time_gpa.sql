--#
--#       Gpu PreAggregate TestCases with Date/Time types.
--# 
--#   [TODO] Time/Interval type has not implemented yet.
--#          If their types will be able to aggregate by Gpu,
--#          then we need refactor this test cases.
--#

-- global configuration
set pg_strom.gpu_setup_cost to 0;
set pg_strom.debug_force_gpupreagg to on;
set pg_strom.enable_gpusort to off;
set client_min_messages to warning;
set datestyle to 'Postgres, MDY';

-- timezone JST-9
set timezone to 'JST-9';
select key,max(timestamp_x) from strom_time_test group by key order by key;
select key,max(timestamptz_x) from strom_time_test group by key order by key;
select key,max(date_x) from strom_time_test group by key order by key;
select key,max(time_x) from strom_time_test group by key order by key;
select key,max(timetz_x) from strom_time_test group by key order by key;
select key,max(interval_x) from strom_time_test group by key order by key;
select key,min(timestamp_x) from strom_time_test group by key order by key;
select key,min(timestamptz_x) from strom_time_test group by key order by key;
select key,min(date_x) from strom_time_test group by key order by key;
select key,min(time_x) from strom_time_test group by key order by key;
select key,min(timetz_x) from strom_time_test group by key order by key;
select key,min(interval_x) from strom_time_test group by key order by key;
select key,count(timestamp_x) from strom_time_test group by key order by key;
select key,count(timestamptz_x) from strom_time_test group by key order by key;
select key,count(date_x) from strom_time_test group by key order by key;
select key,count(time_x) from strom_time_test group by key order by key;
select key,count(timetz_x) from strom_time_test group by key order by key;
select key,count(interval_x) from strom_time_test group by key order by key;
select key,avg(time_x) from strom_time_test group by key order by key;
select key,avg(interval_x) from strom_time_test group by key order by key;
select max(timestamp_x) from strom_time_test;
select max(timestamptz_x) from strom_time_test;
select max(date_x) from strom_time_test;
select max(time_x) from strom_time_test;
select max(timetz_x) from strom_time_test;
select max(interval_x) from strom_time_test;
select min(timestamp_x) from strom_time_test;
select min(timestamptz_x) from strom_time_test;
select min(date_x) from strom_time_test;
select min(time_x) from strom_time_test;
select min(timetz_x) from strom_time_test;
select min(interval_x) from strom_time_test;
select count(timestamp_x) from strom_time_test;
select count(timestamptz_x) from strom_time_test;
select count(date_x) from strom_time_test;
select count(time_x) from strom_time_test;
select count(timetz_x) from strom_time_test;
select count(interval_x) from strom_time_test;
select avg(time_x) from strom_time_test;
select avg(interval_x) from strom_time_test;

-- timezone America/New_York
set timezone to 'America/New_York';
select key,max(timestamp_x) from strom_time_test group by key order by key;
select key,max(timestamptz_x) from strom_time_test group by key order by key;
select key,max(date_x) from strom_time_test group by key order by key;
select key,max(time_x) from strom_time_test group by key order by key;
select key,max(timetz_x) from strom_time_test group by key order by key;
select key,max(interval_x) from strom_time_test group by key order by key;
select key,min(timestamp_x) from strom_time_test group by key order by key;
select key,min(timestamptz_x) from strom_time_test group by key order by key;
select key,min(date_x) from strom_time_test group by key order by key;
select key,min(time_x) from strom_time_test group by key order by key;
select key,min(timetz_x) from strom_time_test group by key order by key;
select key,min(interval_x) from strom_time_test group by key order by key;
select key,count(timestamp_x) from strom_time_test group by key order by key;
select key,count(timestamptz_x) from strom_time_test group by key order by key;
select key,count(date_x) from strom_time_test group by key order by key;
select key,count(time_x) from strom_time_test group by key order by key;
select key,count(timetz_x) from strom_time_test group by key order by key;
select key,count(interval_x) from strom_time_test group by key order by key;
select key,avg(time_x) from strom_time_test group by key order by key;
select key,avg(interval_x) from strom_time_test group by key order by key;
select max(timestamp_x) from strom_time_test;
select max(timestamptz_x) from strom_time_test;
select max(date_x) from strom_time_test;
select max(time_x) from strom_time_test;
select max(timetz_x) from strom_time_test;
select max(interval_x) from strom_time_test;
select min(timestamp_x) from strom_time_test;
select min(timestamptz_x) from strom_time_test;
select min(date_x) from strom_time_test;
select min(time_x) from strom_time_test;
select min(timetz_x) from strom_time_test;
select min(interval_x) from strom_time_test;
select count(timestamp_x) from strom_time_test;
select count(timestamptz_x) from strom_time_test;
select count(date_x) from strom_time_test;
select count(time_x) from strom_time_test;
select count(timetz_x) from strom_time_test;
select count(interval_x) from strom_time_test;
select avg(time_x) from strom_time_test;
select avg(interval_x) from strom_time_test;

-- timezone Europe/Moscow
set timezone to 'Europe/Moscow';
select key,max(timestamp_x) from strom_time_test group by key order by key;
select key,max(timestamptz_x) from strom_time_test group by key order by key;
select key,max(date_x) from strom_time_test group by key order by key;
select key,max(time_x) from strom_time_test group by key order by key;
select key,max(timetz_x) from strom_time_test group by key order by key;
select key,max(interval_x) from strom_time_test group by key order by key;
select key,min(timestamp_x) from strom_time_test group by key order by key;
select key,min(timestamptz_x) from strom_time_test group by key order by key;
select key,min(date_x) from strom_time_test group by key order by key;
select key,min(time_x) from strom_time_test group by key order by key;
select key,min(timetz_x) from strom_time_test group by key order by key;
select key,min(interval_x) from strom_time_test group by key order by key;
select key,count(timestamp_x) from strom_time_test group by key order by key;
select key,count(timestamptz_x) from strom_time_test group by key order by key;
select key,count(date_x) from strom_time_test group by key order by key;
select key,count(time_x) from strom_time_test group by key order by key;
select key,count(timetz_x) from strom_time_test group by key order by key;
select key,count(interval_x) from strom_time_test group by key order by key;
select key,avg(time_x) from strom_time_test group by key order by key;
select key,avg(interval_x) from strom_time_test group by key order by key;
select max(timestamp_x) from strom_time_test;
select max(timestamptz_x) from strom_time_test;
select max(date_x) from strom_time_test;
select max(time_x) from strom_time_test;
select max(timetz_x) from strom_time_test;
select max(interval_x) from strom_time_test;
select min(timestamp_x) from strom_time_test;
select min(timestamptz_x) from strom_time_test;
select min(date_x) from strom_time_test;
select min(time_x) from strom_time_test;
select min(timetz_x) from strom_time_test;
select min(interval_x) from strom_time_test;
select count(timestamp_x) from strom_time_test;
select count(timestamptz_x) from strom_time_test;
select count(date_x) from strom_time_test;
select count(time_x) from strom_time_test;
select count(timetz_x) from strom_time_test;
select count(interval_x) from strom_time_test;
select avg(time_x) from strom_time_test;
select avg(interval_x) from strom_time_test;
