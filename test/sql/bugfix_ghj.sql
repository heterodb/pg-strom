---#
---#	GpuHashJoin Closed Issue Test Cases
---#

-- off to any join plan
set enable_hashjoin to off;
set enable_mergejoin to off;
set enable_nestloop to off;
-- off to any scan plan
set enable_indexscan to off;
set enable_bitmapscan to off;

set client_min_messages to warning;

-- #109
drop table ttest;
create table ttest(id integer,c1 real,c2 text);
alter table ttest alter c2 set storage external;
insert into ttest select generate_series(1,1),1000,repeat('a',3000)||generate_series(1,10);

select a.id,b.id,a.c1,b.c1,substring(a.c2,1,1),substring(b.c2,1,10) from ttest a inner join ttest b on a.id=b.id;

