# Makefile of pg_strom
MODULE_big = pg_strom
OBJS = pg_strom.o utilcmds.o blkload.o plan.o scan.o devinfo.o

PG_CPPFLAGS = -I/usr/local/cuda/include
SHLIB_LINK := -lOpenCL

EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
