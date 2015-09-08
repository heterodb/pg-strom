--#
--#       Gpu PreAggregate Complex TestCases. 
--#

set pg_strom.debug_force_gpupreagg to on;
set pg_strom.enable_gpusort to off;
set client_min_messages to warning;
set extra_float_digits to -3;

-- Device Filter
select key, avg(float_x)  from strom_test  where float_x >= 0.5 and float_x <= 0.6 group by key order by key;

-- select distinct
select key,count(distinct smlint_x)  from strom_test group by key order by key;

-- (group by x+y)
select key+1000,count(float_x)  from strom_test group by key+1000 order by key+1000;

-- (group by x::varchar)
select key::varchar(20),count(float_x)  from strom_test group by key::varchar(20) order by key;

-- (group by x::numeric)
select key::numeric,count(float_x)  from strom_test group by key::numeric order by key;

-- aggfunc( x + y + z )
select key, avg(float_x + float_y + float_z)  from strom_mix  group by key order by key;
