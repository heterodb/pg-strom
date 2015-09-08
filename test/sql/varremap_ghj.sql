---#
---#	GpuHashJoin Closed Issue (Remapping of variable) Test Cases
---#    #109, ...
---#

set pg_strom.gpu_setup_cost=0;
set pg_strom.enable_gpupreagg to off;
set pg_strom.enable_gpusort to off;
set random_page_cost=1000000;   --# force off index_scan.                                                                                                                                          
set client_min_messages to warning;

-- #109
drop table if exists i109;
create table i109 as select generate_series(1,1000)::integer  id, 'c1'::text as c1, 'c2'::text as c2;
-- explain select b.c1,a.c2 from i109 as a inner join i109 as b on a.id=b.id limit 1;
select b.c1,a.c2 from i109 as a inner join i109 as b on a.id=b.id limit 1;


