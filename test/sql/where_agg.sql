--#
--#       Gpu PreAggregate TestCases with WHERE.
--#

set pg_strom.debug_force_gpupreagg to on;
set client_min_messages to warning;
set extra_float_digits to -3;

-- smallint
select  avg(smlint_x)            from strom_test where key=1 group by key order by key;
select  count(smlint_x)          from strom_test where key=1 group by key order by key;
select  max(smlint_x)            from strom_test where key=1 group by key order by key;
select  min(smlint_x)            from strom_test where key=1 group by key order by key;
select  sum(smlint_x)            from strom_test where key=1 group by key order by key;
select  stddev(smlint_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(smlint_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(smlint_x)    from strom_test where key=1 group by key order by key;
select  variance(smlint_x)       from strom_test where key=1 group by key order by key;
select  var_pop(smlint_x)        from strom_test where key=1 group by key order by key;
select  var_samp(smlint_x)       from strom_test where key=1 group by key order by key;
select corr(smlint_x,smlint_x) from strom_test where key=1 group by key order by key;
select covar_pop(smlint_x,smlint_x) from strom_test where key=1 group by key order by key;
select covar_samp(smlint_x,smlint_x) from strom_test where key=1 group by key order by key;

--integer
select  avg(integer_x)            from strom_test where key=1 group by key order by key;
select  count(integer_x)          from strom_test where key=1 group by key order by key;
select  max(integer_x)            from strom_test where key=1 group by key order by key;
select  min(integer_x)            from strom_test where key=1 group by key order by key;
select  sum(integer_x)            from strom_test where key=1 group by key order by key;
select  stddev(integer_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(integer_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(integer_x)    from strom_test where key=1 group by key order by key;
select  variance(integer_x)       from strom_test where key=1 group by key order by key;
select  var_pop(integer_x)        from strom_test where key=1 group by key order by key;
select  var_samp(integer_x)       from strom_test where key=1 group by key order by key;
select corr(integer_x,integer_x) from strom_test where key=1 group by key order by key;
select covar_pop(integer_x,integer_x) from strom_test where key=1 group by key order by key;
select covar_samp(integer_x,integer_x) from strom_test where key=1 group by key order by key;

--bigint
select  avg(bigint_x)            from strom_test where key=1 group by key order by key;
select  count(bigint_x)          from strom_test where key=1 group by key order by key;
select  max(bigint_x)            from strom_test where key=1 group by key order by key;
select  min(bigint_x)            from strom_test where key=1 group by key order by key;
select  sum(bigint_x)            from strom_test where key=1 group by key order by key;
select  stddev(bigint_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(bigint_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(bigint_x)    from strom_test where key=1 group by key order by key;
select  variance(bigint_x)       from strom_test where key=1 group by key order by key;
select  var_pop(bigint_x)        from strom_test where key=1 group by key order by key;
select  var_samp(bigint_x)       from strom_test where key=1 group by key order by key;
select corr(bigint_x,bigint_x) from strom_test where key=1 group by key order by key;
select covar_pop(bigint_x,bigint_x) from strom_test where key=1 group by key order by key;
select covar_samp(bigint_x,bigint_x) from strom_test where key=1 group by key order by key;

--real
select  avg(real_x)            from strom_test where key=1 group by key order by key;
select  count(real_x)          from strom_test where key=1 group by key order by key;
select  max(real_x)            from strom_test where key=1 group by key order by key;
select  min(real_x)            from strom_test where key=1 group by key order by key;
select  sum(real_x)            from strom_test where key=1 group by key order by key;
select  stddev(real_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(real_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(real_x)    from strom_test where key=1 group by key order by key;
select  variance(real_x)       from strom_test where key=1 group by key order by key;
select  var_pop(real_x)        from strom_test where key=1 group by key order by key;
select  var_samp(real_x)       from strom_test where key=1 group by key order by key;
select corr(real_x,real_x) from strom_test where key=1 group by key order by key;
select covar_pop(real_x,real_x) from strom_test where key=1 group by key order by key;
select covar_samp(real_x,real_x) from strom_test where key=1 group by key order by key;

--float
select  avg(float_x)            from strom_test where key=1 group by key order by key;
select  count(float_x)          from strom_test where key=1 group by key order by key;
select  max(float_x)            from strom_test where key=1 group by key order by key;
select  min(float_x)            from strom_test where key=1 group by key order by key;
select  sum(float_x)            from strom_test where key=1 group by key order by key;
select  stddev(float_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(float_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(float_x)    from strom_test where key=1 group by key order by key;
select  variance(float_x)       from strom_test where key=1 group by key order by key;
select  var_pop(float_x)        from strom_test where key=1 group by key order by key;
select  var_samp(float_x)       from strom_test where key=1 group by key order by key;
select corr(float_x,float_x) from strom_test where key=1 group by key order by key;
select covar_pop(float_x,float_x) from strom_test where key=1 group by key order by key;
select covar_samp(float_x,float_x) from strom_test where key=1 group by key order by key;

--numeric
select  avg(nume_x)            from strom_test where key=1 group by key order by key;
select  count(nume_x)          from strom_test where key=1 group by key order by key;
select  max(nume_x)            from strom_test where key=1 group by key order by key;
select  min(nume_x)            from strom_test where key=1 group by key order by key;
select  sum(nume_x)            from strom_test where key=1 group by key order by key;
select  stddev(nume_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(nume_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(nume_x)    from strom_test where key=1 group by key order by key;
select  variance(nume_x)       from strom_test where key=1 group by key order by key;
select  var_pop(nume_x)        from strom_test where key=1 group by key order by key;
select  var_samp(nume_x)       from strom_test where key=1 group by key order by key;
select corr(nume_x,nume_x) from strom_test where key=1 group by key order by key;
select covar_pop(nume_x,nume_x) from strom_test where key=1 group by key order by key;
select covar_samp(nume_x,nume_x) from strom_test where key=1 group by key order by key;

--smallserial
select  avg(smlsrl_x)            from strom_test where key=1 group by key order by key;
select  count(smlsrl_x)          from strom_test where key=1 group by key order by key;
select  max(smlsrl_x)            from strom_test where key=1 group by key order by key;
select  min(smlsrl_x)            from strom_test where key=1 group by key order by key;
select  sum(smlsrl_x)            from strom_test where key=1 group by key order by key;
select  stddev(smlsrl_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(smlsrl_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(smlsrl_x)    from strom_test where key=1 group by key order by key;
select  variance(smlsrl_x)       from strom_test where key=1 group by key order by key;
select  var_pop(smlsrl_x)        from strom_test where key=1 group by key order by key;
select  var_samp(smlsrl_x)       from strom_test where key=1 group by key order by key;
select corr(smlsrl_x,smlsrl_x) from strom_test where key=1 group by key order by key;
select covar_pop(smlsrl_x,smlsrl_x) from strom_test where key=1 group by key order by key;
select covar_samp(smlsrl_x,smlsrl_x) from strom_test where key=1 group by key order by key;

--serial
select  avg(serial_x)            from strom_test where key=1 group by key order by key;
select  count(serial_x)          from strom_test where key=1 group by key order by key;
select  max(serial_x)            from strom_test where key=1 group by key order by key;
select  min(serial_x)            from strom_test where key=1 group by key order by key;
select  sum(serial_x)            from strom_test where key=1 group by key order by key;
select  stddev(serial_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(serial_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(serial_x)    from strom_test where key=1 group by key order by key;
select  variance(serial_x)       from strom_test where key=1 group by key order by key;
select  var_pop(serial_x)        from strom_test where key=1 group by key order by key;
select  var_samp(serial_x)       from strom_test where key=1 group by key order by key;
select corr(serial_x,serial_x) from strom_test where key=1 group by key order by key;
select covar_pop(serial_x,serial_x) from strom_test where key=1 group by key order by key;
select covar_samp(serial_x,serial_x) from strom_test where key=1 group by key order by key;

--bigserial
select  avg(bigsrl_x)            from strom_test where key=1 group by key order by key;
select  count(bigsrl_x)          from strom_test where key=1 group by key order by key;
select  max(bigsrl_x)            from strom_test where key=1 group by key order by key;
select  min(bigsrl_x)            from strom_test where key=1 group by key order by key;
select  sum(bigsrl_x)            from strom_test where key=1 group by key order by key;
select  stddev(bigsrl_x)         from strom_test where key=1 group by key order by key;
select  stddev_pop(bigsrl_x)     from strom_test where key=1 group by key order by key;
select  stddev_samp(bigsrl_x)    from strom_test where key=1 group by key order by key;
select  variance(bigsrl_x)       from strom_test where key=1 group by key order by key;
select  var_pop(bigsrl_x)        from strom_test where key=1 group by key order by key;
select  var_samp(bigsrl_x)       from strom_test where key=1 group by key order by key;
select corr(bigsrl_x,bigsrl_x) from strom_test where key=1 group by key order by key;
select covar_pop(bigsrl_x,bigsrl_x) from strom_test where key=1 group by key order by key;
select covar_samp(bigsrl_x,bigsrl_x) from strom_test where key=1 group by key order by key;
