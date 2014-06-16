# Makefile of pg_strom
EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

MODULE_big = pg_strom
OBJS  = main.o shmem.o codegen.o mqueue.o restrack.o debug.o grafter.o \
	tcache.o datastore.o gpuscan.o gpusort.o gpuhashjoin.o \
	opencl_entry.o opencl_serv.o opencl_devinfo.o opencl_devprog.o \
	opencl_common.o opencl_gpuscan.o opencl_gpusort.o opencl_hashjoin.o \
	opencl_textlib.o opencl_timelib.o


PG_CONFIG = pg_config
PGSTROM_DEBUG := $(shell $(PG_CONFIG) --configure | grep -q "'--enable-debug'" && echo "-Werror -Wall -O0 -DPGSTROM_DEBUG=1")
PG_CPPFLAGS := $(PGSTROM_DEBUG)
EXTRA_CLEAN := opencl_common.c opencl_gpuscan.c \
		opencl_gpusort.c opencl_hashjoin.c \
		opencl_textlib.c opencl_timelib.c

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

opencl_common.c: opencl_common.h
	(echo "const char *pgstrom_opencl_common_code ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@

opencl_gpuscan.c: opencl_gpuscan.h
	(echo "const char *pgstrom_opencl_gpuscan_code ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@

opencl_gpusort.c: opencl_gpusort.h
	(echo "const char *pgstrom_opencl_gpusort_code ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@

opencl_hashjoin.c: opencl_hashjoin.h
	(echo "const char *pgstrom_opencl_hashjoin_code ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@

opencl_textlib.c: opencl_textlib.h
	(echo "const char *pgstrom_opencl_textlib_code ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@

opencl_timelib.c: opencl_timelib.h
	(echo "const char *pgstrom_opencl_timelib_code ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@
