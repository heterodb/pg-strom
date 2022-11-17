#
# Makefile for dpu_strom_serv
#
CC ?= gcc

DPUSERV_OBJS = dpuserv.o xpu_common.o xpu_basetype.o \
               xpu_numeric.o xpu_timelib.o xpu_textlib.o xpu_misclib.o
DPUSERB_HEADS = dpuserv.h arrow_defs.h xpu_common.h xpu_basetype.h \
                xpu_basetype.h xpu_numeric.h xpu_textlib.h \
                xpu_timelib.h xpu_misclib.h

CFLAGS  := -Wall -g -O2 -D_GNU_SOURCE \
           -Wno-sign-compare
LDFLAGS := -lpthread -lm -lstdc++

dpuserv: $(DPUSERV_OBJS)
	$(CC) -o $@ $(DPUSERV_OBJS) $(LDFLAGS)

.c.o: $(DPUSERB_HEADS)
	$(CC) $(CFLAGS) -c -o $@ $<
.cc.o: $(DPUSERB_HEADS)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f dpuserv $(DPUSERV_OBJS)
