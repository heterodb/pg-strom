---#
---#    GpuSort Closed Issue (Remapping of variable) Test Cases
---#    #137, ...
---#

set pg_strom.debug_force_gpusort to on;
set pg_strom.enable_gpusort to on;
set enable_sort to off;
set log_min_messages = debug1;
set client_min_messages to warning;

-- #137
drop table if exists i137;
create table i137(id integer,t1 text collate "C",t2 text collate "C");
insert into i137 select generate_series(1,100) , 'i137','i137';
select * from i137  where id%2=0 order by id,t1,t2;
