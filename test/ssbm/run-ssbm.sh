#!/bin/sh

YMD=`date +%Y%m%d`
DIR=~/ssbm-logs
CWD=`dirname $0`
DBNAME="ssbm"

if [ -n "$1" ]; then
  DBNAME="$1"
fi

mkdir -p ${DIR} || exit 1

sudo sysctl -w vm.drop_caches=1
psql ${DBNAME} -a -f ${CWD}/ssbm-all-pgsql.sql > ${DIR}/log_pgsql_${DBNAME}_${YMD}a.log

sudo sysctl -w vm.drop_caches=1
psql ${DBNAME} -a -f ${CWD}/ssbm-all-pgsql.sql > ${DIR}/log_pgsql_${DBNAME}_${YMD}b.log

sudo sysctl -w vm.drop_caches=1
psql ${DBNAME} -a -f ${CWD}/ssbm-all-pgsql.sql > ${DIR}/log_pgsql_${DBNAME}_${YMD}c.log

sudo sysctl -w vm.drop_caches=1
psql ${DBNAME} -a -f ${CWD}/ssbm-all-strom.sql > ${DIR}/log_strom_${DBNAME}_${YMD}a.log

sudo sysctl -w vm.drop_caches=1
psql ${DBNAME} -a -f ${CWD}/ssbm-all-strom.sql > ${DIR}/log_strom_${DBNAME}_${YMD}b.log

sudo sysctl -w vm.drop_caches=1
psql ${DBNAME} -a -f ${CWD}/ssbm-all-strom.sql > ${DIR}/log_strom_${DBNAME}_${YMD}c.log
