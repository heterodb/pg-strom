set client_min_messages to notice;
set extra_float_digits to -3;
set pg_strom.debug_force_gpupreagg to on;
set pg_strom.enabled=on;

-- NO RECHECK
select sum(0);
select sum(1E+48);
select sum(1E-32);
-- RECHECKED BY CPU.
select sum(1E-33);
select sum(1E+49);
select sum(1E+1000);
select sum(1E-1000);
