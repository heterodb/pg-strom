--#
--#       Gpu PreAggregate TestCases on Zero record Table.
--#

set pg_strom.debug_force_gpupreagg to on;
set pg_strom.enable_gpusort to off;
set client_min_messages to warning;

-- smallint
select  avg(smlint_x)            from strom_zero_test ;
select  count(smlint_x)          from strom_zero_test ;
select  max(smlint_x)            from strom_zero_test ;
select  min(smlint_x)            from strom_zero_test ;
select  sum(smlint_x)            from strom_zero_test ;
select  stddev(smlint_x)         from strom_zero_test ;
select  stddev_pop(smlint_x)     from strom_zero_test ;
select  stddev_samp(smlint_x)    from strom_zero_test ;
select  variance(smlint_x)       from strom_zero_test ;
select  var_pop(smlint_x)        from strom_zero_test ;
select  var_samp(smlint_x)       from strom_zero_test ;
select corr(smlint_x,smlint_x) from strom_zero_test ;
select covar_pop(smlint_x,smlint_x) from strom_zero_test ;
select covar_samp(smlint_x,smlint_x) from strom_zero_test ;

--integer
select  avg(integer_x)            from strom_zero_test ;
select  count(integer_x)          from strom_zero_test ;
select  max(integer_x)            from strom_zero_test ;
select  min(integer_x)            from strom_zero_test ;
select  sum(integer_x)            from strom_zero_test ;
select  stddev(integer_x)         from strom_zero_test ;
select  stddev_pop(integer_x)     from strom_zero_test ;
select  stddev_samp(integer_x)    from strom_zero_test ;
select  variance(integer_x)       from strom_zero_test ;
select  var_pop(integer_x)        from strom_zero_test ;
select  var_samp(integer_x)       from strom_zero_test ;
select corr(integer_x,integer_x) from strom_zero_test ;
select covar_pop(integer_x,integer_x) from strom_zero_test ;
select covar_samp(integer_x,integer_x) from strom_zero_test ;

--bigint
select  avg(bigint_x)            from strom_zero_test ;
select  count(bigint_x)          from strom_zero_test ;
select  max(bigint_x)            from strom_zero_test ;
select  min(bigint_x)            from strom_zero_test ;
select  sum(bigint_x)            from strom_zero_test ;
select  stddev(bigint_x)         from strom_zero_test ;
select  stddev_pop(bigint_x)     from strom_zero_test ;
select  stddev_samp(bigint_x)    from strom_zero_test ;
select  variance(bigint_x)       from strom_zero_test ;
select  var_pop(bigint_x)        from strom_zero_test ;
select  var_samp(bigint_x)       from strom_zero_test ;
select corr(bigint_x,bigint_x) from strom_zero_test ;
select covar_pop(bigint_x,bigint_x) from strom_zero_test ;
select covar_samp(bigint_x,bigint_x) from strom_zero_test ;

--real
select  avg(real_x)            from strom_zero_test ;
select  count(real_x)          from strom_zero_test ;
select  max(real_x)            from strom_zero_test ;
select  min(real_x)            from strom_zero_test ;
select  sum(real_x)            from strom_zero_test ;
select  stddev(real_x)         from strom_zero_test ;
select  stddev_pop(real_x)     from strom_zero_test ;
select  stddev_samp(real_x)    from strom_zero_test ;
select  variance(real_x)       from strom_zero_test ;
select  var_pop(real_x)        from strom_zero_test ;
select  var_samp(real_x)       from strom_zero_test ;
select corr(real_x,real_x) from strom_zero_test ;
select covar_pop(real_x,real_x) from strom_zero_test ;
select covar_samp(real_x,real_x) from strom_zero_test ;

--float
select  avg(float_x)            from strom_zero_test ;
select  count(float_x)          from strom_zero_test ;
select  max(float_x)            from strom_zero_test ;
select  min(float_x)            from strom_zero_test ;
select  sum(float_x)            from strom_zero_test ;
select  stddev(float_x)         from strom_zero_test ;
select  stddev_pop(float_x)     from strom_zero_test ;
select  stddev_samp(float_x)    from strom_zero_test ;
select  variance(float_x)       from strom_zero_test ;
select  var_pop(float_x)        from strom_zero_test ;
select  var_samp(float_x)       from strom_zero_test ;
select corr(float_x,float_x) from strom_zero_test ;
select covar_pop(float_x,float_x) from strom_zero_test ;
select covar_samp(float_x,float_x) from strom_zero_test ;

--numeric
select  avg(nume_x)            from strom_zero_test ;
select  count(nume_x)          from strom_zero_test ;
select  max(nume_x)            from strom_zero_test ;
select  min(nume_x)            from strom_zero_test ;
select  sum(nume_x)            from strom_zero_test ;
select  stddev(nume_x)         from strom_zero_test ;
select  stddev_pop(nume_x)     from strom_zero_test ;
select  stddev_samp(nume_x)    from strom_zero_test ;
select  variance(nume_x)       from strom_zero_test ;
select  var_pop(nume_x)        from strom_zero_test ;
select  var_samp(nume_x)       from strom_zero_test ;
select corr(nume_x,nume_x) from strom_zero_test ;
select covar_pop(nume_x,nume_x) from strom_zero_test ;
select covar_samp(nume_x,nume_x) from strom_zero_test ;

--smallserial
select  avg(smlsrl_x)            from strom_zero_test ;
select  count(smlsrl_x)          from strom_zero_test ;
select  max(smlsrl_x)            from strom_zero_test ;
select  min(smlsrl_x)            from strom_zero_test ;
select  sum(smlsrl_x)            from strom_zero_test ;
select  stddev(smlsrl_x)         from strom_zero_test ;
select  stddev_pop(smlsrl_x)     from strom_zero_test ;
select  stddev_samp(smlsrl_x)    from strom_zero_test ;
select  variance(smlsrl_x)       from strom_zero_test ;
select  var_pop(smlsrl_x)        from strom_zero_test ;
select  var_samp(smlsrl_x)       from strom_zero_test ;
select corr(smlsrl_x,smlsrl_x) from strom_zero_test ;
select covar_pop(smlsrl_x,smlsrl_x) from strom_zero_test ;
select covar_samp(smlsrl_x,smlsrl_x) from strom_zero_test ;

--serial
select  avg(serial_x)            from strom_zero_test ;
select  count(serial_x)          from strom_zero_test ;
select  max(serial_x)            from strom_zero_test ;
select  min(serial_x)            from strom_zero_test ;
select  sum(serial_x)            from strom_zero_test ;
select  stddev(serial_x)         from strom_zero_test ;
select  stddev_pop(serial_x)     from strom_zero_test ;
select  stddev_samp(serial_x)    from strom_zero_test ;
select  variance(serial_x)       from strom_zero_test ;
select  var_pop(serial_x)        from strom_zero_test ;
select  var_samp(serial_x)       from strom_zero_test ;
select corr(serial_x,serial_x) from strom_zero_test ;
select covar_pop(serial_x,serial_x) from strom_zero_test ;
select covar_samp(serial_x,serial_x) from strom_zero_test ;

--bigserial
select  avg(bigsrl_x)            from strom_zero_test ;
select  count(bigsrl_x)          from strom_zero_test ;
select  max(bigsrl_x)            from strom_zero_test ;
select  min(bigsrl_x)            from strom_zero_test ;
select  sum(bigsrl_x)            from strom_zero_test ;
select  stddev(bigsrl_x)         from strom_zero_test ;
select  stddev_pop(bigsrl_x)     from strom_zero_test ;
select  stddev_samp(bigsrl_x)    from strom_zero_test ;
select  variance(bigsrl_x)       from strom_zero_test ;
select  var_pop(bigsrl_x)        from strom_zero_test ;
select  var_samp(bigsrl_x)       from strom_zero_test ;
select corr(bigsrl_x,bigsrl_x) from strom_zero_test ;
select covar_pop(bigsrl_x,bigsrl_x) from strom_zero_test ;
select covar_samp(bigsrl_x,bigsrl_x) from strom_zero_test ;
