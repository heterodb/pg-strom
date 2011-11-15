# pg_rapid Makefile

MODULE_big = pg_rapid
OBJS = pg_rapid.o utilcmds.o blkload.o

NVCC := $(shell which nvcc 2>/dev/null)


PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

XCOMP	:= $(shell echo $(CFLAGS) $(CFLAGS_SL) -fPIC | sed 's/[ ]\+/,/g')
XLINK	:= $(shell echo $(LDFLAGS_SL) $(LDFLAGS) | sed 's/[ ]\+/,/g')
NVCCOPT	:= --x=c --shared -I. -I$(INCLUDEDIR-SERVER) -Xcompiler=$(XCOMP)

.cu.o:
	$(NVCC) -c $< $(NVCCOPT)
.SUFFIXES: .cu .o
