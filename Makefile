# Makefile of pg_strom
MODULE_big = pg_strom
OBJS = main.o shmseg.o plan.o exec.o opencl_serv.o opencl_catalog.o

DATA_built = opencl_kernel

OPENCL_DIR := /usr/local/cuda
OPENCL_INCLUDE := $(OPENCL_DIR)/include

#PG_CPPFLAGS = -I$(CUDA_DIR)/include -DNVCC_CMD_DEFAULT=\"$(CUDA_DIR)/bin/nvcc\"
#SHLIB_LINK := -lcuda -Wl,-rpath,'$(CUDA_DIR)/lib64' -Wl,-rpath,'$(CUDA_DIR)/lib'

PG_CPPFLAGS = -I$(OPENCL_INCLUDE)
SHLIB_LINK := -lOpenCL

EXTENSION = pg_strom
DATA = pg_strom--1.0.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

opencl_kernel: opencl_kernel.cl opencl_catalog.h
	$(CC) -E -xc $< -o $@
