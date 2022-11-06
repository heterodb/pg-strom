#
# Makefile for dpu_strom_serv
#
PG_CONFIG ?= pg_config

DPUSERV_OBJS = dpuserv.o
DPUSERB_HEADS = dpuserv.h arrow_defs.h xpu_common.h xpu_basetype.h \
                xpu_basetype.h xpu_numeric.h xpu_textlib.h \
                xpu_timelib.h xpu_misclib.h

CFLAGS  := $(shell $(PG_CONFIG) --cflags) \
           $(shell $(PG_CONFIG) --cppflags) \
           -I $(shell $(PG_CONFIG) --includedir)
LDFLAGS := $(shell $(PG_CONFIG) --ldflags)

dpuserv: $(DPUSERV_OBJS)
	$(CC) -o $@ $(DPUSERV_OBJS) $(LDFLAGS) -lpthread

.c.o: $(DPUSERB_HEADS)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f dpuserv $(DPUSERV_OBJS)
