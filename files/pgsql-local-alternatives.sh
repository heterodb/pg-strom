#!/bin/sh

PG_MAJOR_VERSIONS="15 16 17 18"
PG_COMMANDS="postgres pg_ctl psql initdb pg_dump pg_dumpall pg_restore pg_config pg_isready createdb createuser dropdb dropuser clusterdb reindexdb vacuumdb pgbench"

# cleanup first
alternatives --display pgsql-local | grep priority | while read PG_CONFIG A B PRIO;
do
	alternatives --remove pgsql-local ${PG_CONFIG}
done

# install the link
for v in ${PG_MAJOR_VERSIONS}
do
	OPT="--install /usr/local/bin/pg_config pgsql-local /usr/local/pgsql-${v}/bin/pg_config ${v}0"
	OPT="${OPT} --slave /opt/pgdata pgsql-local-pgdata /opt/pgdata${v}"
    for cmd in ${PG_COMMANDS}
	do
      OPT="${OPT} --slave /usr/local/bin/${cmd} pgsql-local-${cmd} /usr/local/pgsql-${v}/bin/${cmd}"
	done
	alternatives ${OPT}
done
