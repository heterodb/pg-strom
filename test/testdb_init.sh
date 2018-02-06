#!/bin/sh

DIR="`dirname $0`"
REGRESS_DBNAME="$1"
PSQL="$2"

if [ -z "$REGRESS_DBNAME" ]; then
  echo "regression test database name was not specified"
  exit 1
elif [ "$REGRESS_DBNAME" != "contrib_regression_pg_strom" ]; then
  echo "regression test database is not normal: $REGRESS_DBNAME"
fi

if [ -z "$PSQL" ]; then
  PSQL="psql"
fi
PG_RESTORE="`dirname \`which $PSQL\``/pg_restore"

#
# check whether $REGRESS_DBNAME exists
#
SQL="SELECT count(*) > 0 FROM pg_database WHERE datname = '$REGRESS_DBNAME'"
RV=`$PSQL -Atq postgres -c "$SQL"`
if [ $? -ne 0 ]; then
  echo "failed to connect PostgreSQL server"
  exit 1
fi
if [ "$RV" != "t" ]; then
  SQL="CREATE DATABASE \"$REGRESS_DBNAME\""
  $PSQL -c "$SQL"
  if [ $? -ne 0 ]; then
    echo "failed on [$SQL]"
    exit 1
  fi
fi

#
# check required modules
#
SQL="SELECT count(*) FROM pg_extension WHERE extname = 'pg_strom'"
RV=`$PSQL -Atq -c "$SQL" "$REGRESS_DBNAME"`
if [ $? -ne 0 -o "$RV" != "1" ]; then
  $PSQL -c "CREATE EXTENSION pg_strom" "$REGRESS_DBNAME"
  if [ $? -ne 0 ]; then
    echo "failed on CREATE EXTENSION pg_strom"
    exit 1
  fi
fi

SQL="SELECT count(*) FROM pg_extension WHERE extname = 'pgcrypto'"
RV=`$PSQL -Atq -c "$SQL" "$REGRESS_DBNAME"`
if [ $? -ne 0 -o "$RV" != "1" ]; then
  $PSQL -c "CREATE EXTENSION pgcrypto" "$REGRESS_DBNAME"
  if [ $? -ne 0 ]; then
    echo "failed on CREATE EXTENSION pgcrypto"
    exot 1
  fi
fi

#
# construct test database if not exists
#
SQL="SELECT dv_version FROM dbgen_version"
RV=`$PSQL -Atq -c "$SQL" "$REGRESS_DBNAME"`
if [ $? -ne 0 -o "$RV" != "2.7.0" ]; then
  if [ ! -e "$DIR/tpcds_sf5.dump" ]; then
    # TODO: add curl -O ... from swdc
    echo "No tpcds_sf5.dump file"
    exit 1
  fi
  DUMPFILE="$DIR/tpcds_sf5.dump"
  $PG_RESTORE -d "$REGRESS_DBNAME" -v -e --single-transaction "$DUMPFILE"
  if [ $? -ne 0 ]; then
    echo "failed on $PG_RESTORE with $DUMPFILE"
    exit 1
  fi
fi

SQL="SELECT public.pgstrom_regression_test_revision()"
RV=`$PSQL -Atq -c "$SQL" "$REGRESS_DBNAME"`
if [ $? -ne 0 -o "$RV" != "20180124" ]; then
  $PSQL $REGRESS_DBNAME -f ${DIR}/testdb_init.sql
  if [ $? -ne 0 ]; then
    echo "failed on testdb_init.sql"
    exit 1
  fi
  RV=`$PSQL -Atq -c "$SQL" "$REGRESS_DBNAME"`
  if [ $? -ne 0 -o "$RV" != "20180124" ]; then
    echo "testdb_init.sql could not setup database correctly"
    exit 1
  fi
fi

#
# Run test for lo_export_gpu / lo_import_gpu
#
echo "====== independent test cases ======"
${DIR}/testapp_largeobject -d ${REGRESS_DBNAME}

exit 0
