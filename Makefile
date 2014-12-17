# Makefile of pg_strom
EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

MODULE_big = pg_strom
OBJS  = main.o shmem.o codegen.o mqueue.o restrack.o grafter.o \
	datastore.o gpuscan.o gpuhashjoin.o gpupreagg.o \
	opencl_entry.o opencl_serv.o opencl_devinfo.o opencl_devprog.o \
	opencl_common.o opencl_gpuscan.o opencl_gpupreagg.o opencl_hashjoin.o \
	opencl_textlib.o opencl_timelib.o opencl_numeric.o

PG_CONFIG = pg_config
PGSTROM_DEBUG := $(shell $(PG_CONFIG) --configure | grep -q "'--enable-debug'" && echo "-Wall -DPGSTROM_DEBUG=1 -O0")
PG_CPPFLAGS := $(PGSTROM_DEBUG)
EXTRA_CLEAN := opencl_common.c opencl_gpuscan.c \
		opencl_gpupreagg.c opencl_hashjoin.c \
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

opencl_hashjoin.c: opencl_hashjoin.h
	(echo "const char *pgstrom_opencl_hashjoin_code ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@

opencl_gpupreagg.c: opencl_gpupreagg.h
	(echo "const char *pgstrom_opencl_gpupreagg_code ="; \
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

opencl_numeric.c: opencl_numeric.h
	(echo "const char *pgstrom_opencl_numeric_code ="; \
	 sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	     -e 's/^/  "/g' -e 's/$$/\\n"/g'< $^; \
	 echo ";") > $@
