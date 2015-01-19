--#
--#       Gpu PreAggregate TestCases with GROUP BY.
--#

set pg_strom.debug_force_gpupreagg to on;
set client_min_messages to warning;
set extra_float_digits to -3;

-- smallint
select key, avg(smlint_x)            from strom_test  group by key order by key;
select key, count(smlint_x)          from strom_test  group by key order by key;
select key, max(smlint_x)            from strom_test  group by key order by key;
select key, min(smlint_x)            from strom_test  group by key order by key;
select key, sum(smlint_x)            from strom_test  group by key order by key;
select key, stddev(smlint_x)         from strom_test  group by key order by key;
select key, stddev_pop(smlint_x)     from strom_test  group by key order by key;
select key, stddev_samp(smlint_x)    from strom_test  group by key order by key;
select key, variance(smlint_x)       from strom_test  group by key order by key;
select key, var_pop(smlint_x)        from strom_test  group by key order by key;
select key, var_samp(smlint_x)       from strom_test  group by key order by key;
select key,corr(smlint_x,smlint_z) from strom_mix  group by key order by key;
select key,covar_pop(smlint_x,smlint_z) from strom_mix  group by key order by key;
select key,covar_samp(smlint_x,smlint_z) from strom_mix  group by key order by key;
select key,corr(smlint_y,smlint_z) from strom_mix  group by key order by key;
select key,covar_pop(smlint_y,smlint_z) from strom_mix  group by key order by key;
select key,covar_samp(smlint_y,smlint_z) from strom_mix  group by key order by key;

--integer
select key, avg(integer_x)            from strom_test  group by key order by key;
select key, count(integer_x)          from strom_test  group by key order by key;
select key, max(integer_x)            from strom_test  group by key order by key;
select key, min(integer_x)            from strom_test  group by key order by key;
select key, sum(integer_x)            from strom_test  group by key order by key;
select key, stddev(integer_x)         from strom_test  group by key order by key;
select key, stddev_pop(integer_x)     from strom_test  group by key order by key;
select key, stddev_samp(integer_x)    from strom_test  group by key order by key;
select key, variance(integer_x)       from strom_test  group by key order by key;
select key, var_pop(integer_x)        from strom_test  group by key order by key;
select key, var_samp(integer_x)       from strom_test  group by key order by key;
select key,corr(integer_x,integer_z) from strom_mix  group by key order by key;
select key,covar_pop(integer_x,integer_z) from strom_mix  group by key order by key;
select key,covar_samp(integer_x,integer_z) from strom_mix  group by key order by key;
select key,corr(integer_y,integer_z) from strom_mix  group by key order by key;
select key,covar_pop(integer_y,integer_z) from strom_mix  group by key order by key;
select key,covar_samp(integer_y,integer_z) from strom_mix  group by key order by key;

--bigint
select key, avg(bigint_x)            from strom_test  group by key order by key;
select key, count(bigint_x)          from strom_test  group by key order by key;
select key, max(bigint_x)            from strom_test  group by key order by key;
select key, min(bigint_x)            from strom_test  group by key order by key;
select key, sum(bigint_x)            from strom_test  group by key order by key;
select key, stddev(bigint_x)         from strom_test  group by key order by key;
select key, stddev_pop(bigint_x)     from strom_test  group by key order by key;
select key, stddev_samp(bigint_x)    from strom_test  group by key order by key;
select key, variance(bigint_x)       from strom_test  group by key order by key;
select key, var_pop(bigint_x)        from strom_test  group by key order by key;
select key, var_samp(bigint_x)       from strom_test  group by key order by key;
select key,corr(bigint_x,bigint_z) from strom_mix  group by key order by key;
select key,covar_pop(bigint_x,bigint_z) from strom_mix  group by key order by key;
select key,covar_samp(bigint_x,bigint_z) from strom_mix  group by key order by key;
select key,corr(bigint_y,bigint_z) from strom_mix  group by key order by key;
select key,covar_pop(bigint_y,bigint_z) from strom_mix  group by key order by key;
select key,covar_samp(bigint_y,bigint_z) from strom_mix  group by key order by key;

--real
select key, avg(real_x)            from strom_test  group by key order by key;
select key, count(real_x)          from strom_test  group by key order by key;
select key, max(real_x)            from strom_test  group by key order by key;
select key, min(real_x)            from strom_test  group by key order by key;
select key, sum(real_x)            from strom_test  group by key order by key;
select key, stddev(real_x)         from strom_test  group by key order by key;
select key, stddev_pop(real_x)     from strom_test  group by key order by key;
select key, stddev_samp(real_x)    from strom_test  group by key order by key;
select key, variance(real_x)       from strom_test  group by key order by key;
select key, var_pop(real_x)        from strom_test  group by key order by key;
select key, var_samp(real_x)       from strom_test  group by key order by key;
select key,corr(real_x,real_z) from strom_mix  group by key order by key;
select key,covar_pop(real_x,real_z) from strom_mix  group by key order by key;
select key,covar_samp(real_x,real_z) from strom_mix  group by key order by key;
select key,corr(real_y,real_z) from strom_mix  group by key order by key;
select key,covar_pop(real_y,real_z) from strom_mix  group by key order by key;
select key,covar_samp(real_y,real_z) from strom_mix  group by key order by key;

--float
select key, avg(float_x)            from strom_test  group by key order by key;
select key, count(float_x)          from strom_test  group by key order by key;
select key, max(float_x)            from strom_test  group by key order by key;
select key, min(float_x)            from strom_test  group by key order by key;
select key, sum(float_x)            from strom_test  group by key order by key;
select key, stddev(float_x)         from strom_test  group by key order by key;
select key, stddev_pop(float_x)     from strom_test  group by key order by key;
select key, stddev_samp(float_x)    from strom_test  group by key order by key;
select key, variance(float_x)       from strom_test  group by key order by key;
select key, var_pop(float_x)        from strom_test  group by key order by key;
select key, var_samp(float_x)       from strom_test  group by key order by key;
select key,corr(float_x,float_z) from strom_mix  group by key order by key;
select key,covar_pop(float_x,float_z) from strom_mix  group by key order by key;
select key,covar_samp(float_x,float_z) from strom_mix  group by key order by key;
select key,corr(float_y,float_z) from strom_mix  group by key order by key;
select key,covar_pop(float_y,float_z) from strom_mix  group by key order by key;
select key,covar_samp(float_y,float_z) from strom_mix  group by key order by key;

--numeric
select key, avg(nume_x)            from strom_test  group by key order by key;
select key, count(nume_x)          from strom_test  group by key order by key;
select key, max(nume_x)            from strom_test  group by key order by key;
select key, min(nume_x)            from strom_test  group by key order by key;
select key, sum(nume_x)            from strom_test  group by key order by key;
select key, stddev(nume_x)         from strom_test  group by key order by key;
select key, stddev_pop(nume_x)     from strom_test  group by key order by key;
select key, stddev_samp(nume_x)    from strom_test  group by key order by key;
select key, variance(nume_x)       from strom_test  group by key order by key;
select key, var_pop(nume_x)        from strom_test  group by key order by key;
select key, var_samp(nume_x)       from strom_test  group by key order by key;
select key,corr(nume_x,nume_z) from strom_mix  group by key order by key;
select key,covar_pop(nume_x,nume_z) from strom_mix  group by key order by key;
select key,covar_samp(nume_x,nume_z) from strom_mix  group by key order by key;
select key,corr(nume_y,nume_z) from strom_mix  group by key order by key;
select key,covar_pop(nume_y,nume_z) from strom_mix  group by key order by key;
select key,covar_samp(nume_y,nume_z) from strom_mix  group by key order by key;

--smallserial
select key, avg(smlsrl_x)            from strom_test  group by key order by key;
select key, count(smlsrl_x)          from strom_test  group by key order by key;
select key, max(smlsrl_x)            from strom_test  group by key order by key;
select key, min(smlsrl_x)            from strom_test  group by key order by key;
select key, sum(smlsrl_x)            from strom_test  group by key order by key;
select key, stddev(smlsrl_x)         from strom_test  group by key order by key;
select key, stddev_pop(smlsrl_x)     from strom_test  group by key order by key;
select key, stddev_samp(smlsrl_x)    from strom_test  group by key order by key;
select key, variance(smlsrl_x)       from strom_test  group by key order by key;
select key, var_pop(smlsrl_x)        from strom_test  group by key order by key;
select key, var_samp(smlsrl_x)       from strom_test  group by key order by key;
select key,corr(smlsrl_x,smlsrl_z) from strom_mix  group by key order by key;
select key,covar_pop(smlsrl_x,smlsrl_z) from strom_mix  group by key order by key;
select key,covar_samp(smlsrl_x,smlsrl_z) from strom_mix  group by key order by key;
select key,corr(smlsrl_y,smlsrl_z) from strom_mix  group by key order by key;
select key,covar_pop(smlsrl_y,smlsrl_z) from strom_mix  group by key order by key;
select key,covar_samp(smlsrl_y,smlsrl_z) from strom_mix  group by key order by key;

--serial
select key, avg(serial_x)            from strom_test  group by key order by key;
select key, count(serial_x)          from strom_test  group by key order by key;
select key, max(serial_x)            from strom_test  group by key order by key;
select key, min(serial_x)            from strom_test  group by key order by key;
select key, sum(serial_x)            from strom_test  group by key order by key;
select key, stddev(serial_x)         from strom_test  group by key order by key;
select key, stddev_pop(serial_x)     from strom_test  group by key order by key;
select key, stddev_samp(serial_x)    from strom_test  group by key order by key;
select key, variance(serial_x)       from strom_test  group by key order by key;
select key, var_pop(serial_x)        from strom_test  group by key order by key;
select key, var_samp(serial_x)       from strom_test  group by key order by key;
select key,corr(serial_x,serial_z) from strom_mix  group by key order by key;
select key,covar_pop(serial_x,serial_z) from strom_mix  group by key order by key;
select key,covar_samp(serial_x,serial_z) from strom_mix  group by key order by key;
select key,corr(serial_y,serial_z) from strom_mix  group by key order by key;
select key,covar_pop(serial_y,serial_z) from strom_mix  group by key order by key;
select key,covar_samp(serial_y,serial_z) from strom_mix  group by key order by key;

--bigserial
select key, avg(bigsrl_x)            from strom_test  group by key order by key;
select key, count(bigsrl_x)          from strom_test  group by key order by key;
select key, max(bigsrl_x)            from strom_test  group by key order by key;
select key, min(bigsrl_x)            from strom_test  group by key order by key;
select key, sum(bigsrl_x)            from strom_test  group by key order by key;
select key, stddev(bigsrl_x)         from strom_test  group by key order by key;
select key, stddev_pop(bigsrl_x)     from strom_test  group by key order by key;
select key, stddev_samp(bigsrl_x)    from strom_test  group by key order by key;
select key, variance(bigsrl_x)       from strom_test  group by key order by key;
select key, var_pop(bigsrl_x)        from strom_test  group by key order by key;
select key, var_samp(bigsrl_x)       from strom_test  group by key order by key;
select key,corr(bigsrl_x,bigsrl_z) from strom_mix  group by key order by key;
select key,covar_pop(bigsrl_x,bigsrl_z) from strom_mix  group by key order by key;
select key,covar_samp(bigsrl_x,bigsrl_z) from strom_mix  group by key order by key;
select key,corr(bigsrl_y,bigsrl_z) from strom_mix  group by key order by key;
select key,covar_pop(bigsrl_y,bigsrl_z) from strom_mix  group by key order by key;
select key,covar_samp(bigsrl_y,bigsrl_z) from strom_mix  group by key order by key;
