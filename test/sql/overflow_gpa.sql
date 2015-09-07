--#
--#       Gpu PreAggregate OverFlowed TestCases.
--#
--#   "About Calculation errors in floating-point number"
--#
--#   stddev(float4/float8) and corr(float4/float8) except 
--#   from this test case because their functions may return
--#   wrong results due to floating point error.
--# 
--#   Please refer to the following for more information.
--#   https://github.com/pg-strom/devel/wiki/Known-Issues
--#

set pg_strom.debug_force_gpupreagg to on;
set pg_strom.enable_gpusort to off;
set client_min_messages to warning;

-- smallint
select key, avg(smlint_x)::smallint            from strom_overflow_test  group by key order by key;
select key, count(smlint_x)::smallint          from strom_overflow_test  group by key order by key;
select key, max(smlint_x)::smallint            from strom_overflow_test  group by key order by key;
select key, min(smlint_x)::smallint            from strom_overflow_test  group by key order by key;
select key, sum(smlint_x)::smallint            from strom_overflow_test  group by key order by key;
select key, stddev(smlint_x)::smallint          from strom_overflow_test  group by key order by key;
select key, stddev_pop(smlint_x)::smallint      from strom_overflow_test  group by key order by key;
select key, stddev_samp(smlint_x)::smallint     from strom_overflow_test  group by key order by key;
select key, variance(smlint_x)::smallint        from strom_overflow_test  group by key order by key;
select key, var_pop(smlint_x)::smallint         from strom_overflow_test  group by key order by key;
select key, var_samp(smlint_x)::smallint        from strom_overflow_test  group by key order by key;
select key,corr(smlint_x,smlint_x)::smallint  from strom_overflow_test  group by key order by key;
select key,covar_pop(smlint_x,smlint_x)::smallint  from strom_overflow_test  group by key order by key;
select key,covar_samp(smlint_x,smlint_x)::smallint  from strom_overflow_test  group by key order by key;

--integer
select key, avg(integer_x)::integer            from strom_overflow_test  group by key order by key;
select key, count(integer_x)::integer            from strom_overflow_test  group by key order by key;
select key, max(integer_x)::integer              from strom_overflow_test  group by key order by key;
select key, min(integer_x)::integer              from strom_overflow_test  group by key order by key;
select key, sum(integer_x)::integer              from strom_overflow_test  group by key order by key;
select key, stddev(integer_x)::integer           from strom_overflow_test  group by key order by key;
select key, stddev_pop(integer_x)::integer       from strom_overflow_test  group by key order by key;
select key, stddev_samp(integer_x)::integer      from strom_overflow_test  group by key order by key;
select key, variance(integer_x)::integer         from strom_overflow_test  group by key order by key;
select key, var_pop(integer_x)::integer          from strom_overflow_test  group by key order by key;
select key, var_samp(integer_x)::integer         from strom_overflow_test  group by key order by key;
select key,corr(integer_x,integer_x)::integer   from strom_overflow_test  group by key order by key;
select key,covar_pop(integer_x,integer_x)::integer   from strom_overflow_test  group by key order by key;
select key,covar_samp(integer_x,integer_x)::integer   from strom_overflow_test  group by key order by key;

--bigint
select key, avg(bigint_x)::bigint            from strom_overflow_test  group by key order by key;
select key, count(bigint_x)::bigint           from strom_overflow_test  group by key order by key;
select key, max(bigint_x)::bigint             from strom_overflow_test  group by key order by key;
select key, min(bigint_x)::bigint             from strom_overflow_test  group by key order by key;
select key, sum(bigint_x)::bigint             from strom_overflow_test  group by key order by key;
select key, stddev(bigint_x)::bigint          from strom_overflow_test  group by key order by key;
select key, stddev_pop(bigint_x)::bigint      from strom_overflow_test  group by key order by key;
select key, stddev_samp(bigint_x)::bigint     from strom_overflow_test  group by key order by key;
select key, variance(bigint_x)::bigint        from strom_overflow_test  group by key order by key;
select key, var_pop(bigint_x)::bigint         from strom_overflow_test  group by key order by key;
select key, var_samp(bigint_x)::bigint        from strom_overflow_test  group by key order by key;
select key,corr(bigint_x,bigint_x)::bigint  from strom_overflow_test  group by key order by key;
select key,covar_pop(bigint_x,bigint_x)::bigint  from strom_overflow_test  group by key order by key;
select key,covar_samp(bigint_x,bigint_x)::bigint from strom_overflow_test  group by key order by key;

--real
select key, avg(real_x)::real            from strom_overflow_test  group by key order by key;
select key, count(real_x)::real           from strom_overflow_test  group by key order by key;
select key, max(real_x)::real             from strom_overflow_test  group by key order by key;
select key, min(real_x)::real             from strom_overflow_test  group by key order by key;
select key, sum(real_x)::real             from strom_overflow_test  group by key order by key;
-- select key, stddev(real_x)::real          from strom_overflow_test  group by key order by key;
-- select key, stddev_pop(real_x)::real      from strom_overflow_test  group by key order by key;
-- select key, stddev_samp(real_x)::real     from strom_overflow_test  group by key order by key;
select key, variance(real_x)::real        from strom_overflow_test  group by key order by key;
select key, var_pop(real_x)::real         from strom_overflow_test  group by key order by key;
select key, var_samp(real_x)::real        from strom_overflow_test  group by key order by key;
-- select key,corr(real_x,real_x)::real  from strom_overflow_test  group by key order by key;
select key,covar_pop(real_x,real_x)::real  from strom_overflow_test  group by key order by key;
select key,covar_samp(real_x,real_x)::real  from strom_overflow_test  group by key order by key;

--float
select key, avg(float_x)::float            from strom_overflow_test  group by key order by key;
select key, count(float_x)::float           from strom_overflow_test  group by key order by key;
select key, max(float_x)::float             from strom_overflow_test  group by key order by key;
select key, min(float_x)::float             from strom_overflow_test  group by key order by key;
select key, sum(float_x)::float             from strom_overflow_test  group by key order by key;
-- select key, stddev(float_x)::float          from strom_overflow_test  group by key order by key;
-- select key, stddev_pop(float_x)::float      from strom_overflow_test  group by key order by key;
-- select key, stddev_samp(float_x)::float     from strom_overflow_test  group by key order by key;
select key, variance(float_x)::float        from strom_overflow_test  group by key order by key;
select key, var_pop(float_x)::float         from strom_overflow_test  group by key order by key;
select key, var_samp(float_x)::float        from strom_overflow_test  group by key order by key;
-- select key,corr(float_x,float_x)::float  from strom_overflow_test  group by key order by key;
select key,covar_pop(float_x,float_x)::float  from strom_overflow_test  group by key order by key;
select key,covar_samp(float_x,float_x)::float  from strom_overflow_test  group by key order by key;

--numeric
select key, avg(nume_x)::numeric            from strom_overflow_test  group by key order by key;
select key, count(nume_x)::numeric           from strom_overflow_test  group by key order by key;
select key, max(nume_x)::numeric             from strom_overflow_test  group by key order by key;
select key, min(nume_x)::numeric             from strom_overflow_test  group by key order by key;
select key, sum(nume_x)::numeric             from strom_overflow_test  group by key order by key;
select key, stddev(nume_x)::numeric          from strom_overflow_test  group by key order by key;
select key, stddev_pop(nume_x)::numeric      from strom_overflow_test  group by key order by key;
select key, stddev_samp(nume_x)::numeric     from strom_overflow_test  group by key order by key;
select key, variance(nume_x)::numeric        from strom_overflow_test  group by key order by key;
select key, var_pop(nume_x)::numeric         from strom_overflow_test  group by key order by key;
select key, var_samp(nume_x)::numeric        from strom_overflow_test  group by key order by key;
select key,round(corr(nume_x,nume_x)::numeric,14)  from strom_overflow_test  group by key order by key;
select key,trunc(covar_pop(nume_x,nume_x)::numeric*1e-42,13)  from strom_overflow_test  group by key order by key;
select key,trunc(covar_samp(nume_x,nume_x)::numeric*1e-42,13)  from strom_overflow_test  group by key order by key;

--smallserial
select key, avg(smlsrl_x)::smallint            from strom_overflow_test  group by key order by key;
select key, count(smlsrl_x)::smallint              from strom_overflow_test  group by key order by key;
select key, max(smlsrl_x)::smallint                from strom_overflow_test  group by key order by key;
select key, min(smlsrl_x)::smallint                from strom_overflow_test  group by key order by key;
select key, sum(smlsrl_x)::smallint                from strom_overflow_test  group by key order by key;
select key, stddev(smlsrl_x)::smallint             from strom_overflow_test  group by key order by key;
select key, stddev_pop(smlsrl_x)::smallint         from strom_overflow_test  group by key order by key;
select key, stddev_samp(smlsrl_x)::smallint        from strom_overflow_test  group by key order by key;
select key, variance(smlsrl_x)::smallint           from strom_overflow_test  group by key order by key;
select key, var_pop(smlsrl_x)::smallint            from strom_overflow_test  group by key order by key;
select key, var_samp(smlsrl_x)::smallint           from strom_overflow_test  group by key order by key;
select key,corr(smlsrl_x,smlsrl_x)::smallint     from strom_overflow_test  group by key order by key;
select key,covar_pop(smlsrl_x,smlsrl_x)::smallint     from strom_overflow_test  group by key order by key;
select key,covar_samp(smlsrl_x,smlsrl_x)::smallint     from strom_overflow_test  group by key order by key;

--serial
select key, avg(serial_x)::integer            from strom_overflow_test  group by key order by key;
select key, count(serial_x)::integer           from strom_overflow_test  group by key order by key;
select key, max(serial_x)::integer                from strom_overflow_test  group by key order by key;
select key, min(serial_x)::integer                from strom_overflow_test  group by key order by key;
select key, sum(serial_x)::integer                from strom_overflow_test  group by key order by key;
select key, stddev(serial_x)::integer             from strom_overflow_test  group by key order by key;
select key, stddev_pop(serial_x)::integer         from strom_overflow_test  group by key order by key;
select key, stddev_samp(serial_x)::integer        from strom_overflow_test  group by key order by key;
select key, variance(serial_x)::integer           from strom_overflow_test  group by key order by key;
select key, var_pop(serial_x)::integer            from strom_overflow_test  group by key order by key;
select key, var_samp(serial_x)::integer           from strom_overflow_test  group by key order by key;
select key,corr(serial_x,serial_x)::integer     from strom_overflow_test  group by key order by key;
select key,covar_pop(serial_x,serial_x)::integer    from strom_overflow_test  group by key order by key;
select key,covar_samp(serial_x,serial_x)::integer     from strom_overflow_test  group by key order by key;

--bigserial
select key, avg(bigsrl_x)::bigint            from strom_overflow_test  group by key order by key;
select key, count(bigsrl_x)::bigint          from strom_overflow_test  group by key order by key;
select key, max(bigsrl_x)::bigint            from strom_overflow_test  group by key order by key;
select key, min(bigsrl_x)::bigint            from strom_overflow_test  group by key order by key;
select key, sum(bigsrl_x)::bigint            from strom_overflow_test  group by key order by key;
select key, stddev(bigsrl_x)::bigint         from strom_overflow_test  group by key order by key;
select key, stddev_pop(bigsrl_x)::bigint     from strom_overflow_test  group by key order by key;
select key, stddev_samp(bigsrl_x)::bigint    from strom_overflow_test  group by key order by key;
select key, variance(bigsrl_x)::bigint       from strom_overflow_test  group by key order by key;
select key, var_pop(bigsrl_x)::bigint        from strom_overflow_test  group by key order by key;
select key, var_samp(bigsrl_x)::bigint       from strom_overflow_test  group by key order by key;
select key,corr(bigsrl_x,bigsrl_x)::bigint from strom_overflow_test  group by key order by key;
select key,covar_pop(bigsrl_x,bigsrl_x)::bigint from strom_overflow_test  group by key order by key;
select key,covar_samp(bigsrl_x,bigsrl_x)::bigint from strom_overflow_test  group by key order by key;
