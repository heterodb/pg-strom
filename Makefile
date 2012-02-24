# Makefile of pg_strom
MODULE_big = pg_strom
OBJS = main.o shmseg.o opencl_serv.o opencl_catalog.o

DATA = opencl_kernel.cl

#OBJS = pg_strom.o utilcmds.o blkload.o plan.o scan.o nvcc.o devinfo.o

CUDA_DIR := /usr/local/cuda

#PG_CPPFLAGS = -I$(CUDA_DIR)/include -DNVCC_CMD_DEFAULT=\"$(CUDA_DIR)/bin/nvcc\"
#SHLIB_LINK := -lcuda -Wl,-rpath,'$(CUDA_DIR)/lib64' -Wl,-rpath,'$(CUDA_DIR)/lib'

PG_CPPFLAGS = -I$(CUDA_DIR)/include
SHLIB_LINK := -lOpenCL

EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
