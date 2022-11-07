#
# Makefile for dpu_strom_serv
#
PG_CONFIG ?= pg_config
CC ?= gcc

DPUSERV_OBJS = dpuserv.o dpu_common.o dpu_basetype.o \
               dpu_numeric.o dpu_timelib.o dpu_textlib.o dpu_misclib.o
DPUSERB_HEADS = dpuserv.h arrow_defs.h xpu_common.h xpu_basetype.h \
                xpu_basetype.h xpu_numeric.h xpu_textlib.h \
                xpu_timelib.h xpu_misclib.h

CFLAGS  := -Wall -g -O2 -D_GNU_SOURCE \
           -Wno-sign-compare \
           -I $(shell $(PG_CONFIG) --includedir)
LDFLAGS := $(shell $(PG_CONFIG) --ldflags)

dpuserv: $(DPUSERV_OBJS)
	$(CC) -o $@ $(DPUSERV_OBJS) $(LDFLAGS) -lpthread -lm

.c.o: $(DPUSERB_HEADS)
	$(CC) $(CFLAGS) -c -o $@ $<
.cc.o: $(DPUSERB_HEADS)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f dpuserv $(DPUSERV_OBJS)
