#
# Makefile for dpu_strom_serv
#
CC ?= gcc

DPUSERV_OBJS = dpuserv.o xpu_common.o xpu_basetype.o \
               xpu_numeric.o xpu_timelib.o xpu_textlib.o xpu_misclib.o \
               xpu_jsonlib.o xpu_postgis.o
DPUSERB_HEADS = dpuserv.h arrow_defs.h xpu_common.h xpu_basetype.h \
                xpu_basetype.h xpu_numeric.h xpu_textlib.h \
                xpu_timelib.h xpu_misclib.h

CFLAGS  := -Wall -g -O3 -D_GNU_SOURCE \
           -Wno-sign-compare \
           -DCUDA_MAXTHREADS_PER_BLOCK=1
LDFLAGS := -lpthread -lm -lstdc++
ifeq ($(PGSTROM_DEBUG),1)
CFLAGS += -O0
endif

dpuserv: $(DPUSERV_OBJS)
	$(CC) -o $@ $(DPUSERV_OBJS) $(LDFLAGS)

.PHONY: headers

headers: $(DPUSERB_HEADS)

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<
.cc.o:
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f dpuserv $(DPUSERV_OBJS)
