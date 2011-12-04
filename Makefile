# pg_rapid Makefile

MODULE = pg_strom
OBJS = pg_strom.o utilcmds.o blkload.o plan.o scan.o cuda_api.o

CUDAROOT := /usr/local/cuda
NVCC := $(shell which nvcc 2>/dev/null || echo $(CUDAROOT)/bin/nvcc)
PG_CONFIG = pg_config

PG_SHAREDIR	:= $(shell $(PG_CONFIG) --sharedir)
PG_LIBDIR	:= $(shell $(PG_CONFIG) --pkglibdir)

NVCFLAGS  := -O2 --shared -I. -D_GNU_SOURCE
NVCFLAGS  += -I$(shell $(PG_CONFIG) --includedir-server)
NVCFLAGS  += -g -Xcompiler=-Wall -Xcompiler=-g -Xcompiler=-fpic
NVLDFLAGS := -Xlinker=--as-needed
NVLDFLAGS += -Xlinker=-rpath,$(shell $(PG_CONFIG) --libdir)
NVLDFLAGS += -Xlinker=-rpath,$(CUDAROOT)/lib64
NVLDFLAGS += -Xlinker=-rpath,$(CUDAROOT)/lib

all: $(MODULE).so

install: $(MODULE).so
	mkdir -p $(PG_SHAREDIR)/extension
	mkdir -p $(PG_LIBDIR)
	install -m 644 pg_strom--1.0.sql $(PG_SHAREDIR)/extension
	install -m 644 pg_strom.control $(PG_SHAREDIR)/extension
	install -m 755 $^ $(PG_LIBDIR)

clean:
	rm -f $(MODULE).so $(OBJS)

$(MODULE).so: $(OBJS)
	$(NVCC) $(NVCFLAGS) -o $@ $^ $(NVLDFLAGS)
.c.o:
	$(NVCC) $(NVCFLAGS) -c $< $(NVLDFLAGS)
.cu.o:
	$(NVCC) --x=c $(NVCFLAGS) -c $< $(NVLDFLAGS)
.SUFFIXES: .c .cu .o
