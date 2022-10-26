#
# Makefile for pcap2arrow
#
PG_CONFIG        ?= pg_config
MYSQL_CONFIG     ?= mysql_config

HAS_PG_CONFIG = $(shell which $(PG_CONFIG)>/dev/null 2>&1 && echo yes)
HAS_MYSQL_CONFIG = $(shell which $(MYSQL_CONFIG)>/dev/null 2>&1 && echo yes)

ALL_PROGS = pcap2arrow arrow2csv
ifeq ($(HAS_PG_CONFIG),yes)
ALL_PROGS += pg2arrow
endif
ifeq ($(HAS_MYSQL_CONFIG),yes)
ALL_PROGS += mysql2arrow
endif
PG2ARROW_OBJS    = __pgsql2arrow.o pgsql_client.o \
                   arrow_nodes.o arrow_write.o arrow_pgsql.o
MYSQL2ARROW_OBJS = __mysql2arrow.o mysql_client.o \
                   arrow_nodes.o arrow_write.o
PCAP2ARROW_OBJS  = pcap2arrow.o arrow_nodes.o arrow_write.o
ARROW2CSV_OBJS   = arrow2csv.o arrow_nodes.o
CLEAN_OBJS = $(PG2ARROW_OBJS) $(MYSQL2ARROW_OBJS) \
             $(PCAP2ARROW_OBJS) $(ARROW2CSV_OBJS) \
             pcap2arrow arrow2csv pg2arrow mysql2arrow

CFLAGS = -O2 -fPIC -g -Wall -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64
ifeq ($(HAS_PG_CONFIG),yes)
CFLAGS += -I $(shell $(PG_CONFIG) --includedir)
endif
ifeq ($(HAS_MYSQL_CONFIG),yes)
CFLAGS += $(shell $(MYSQL_CONFIG) --include)
endif

PREFIX		?= /usr/local

all: $(ALL_PROGS)

#
# Only if pg_config is available
#
ifeq ($(HAS_PG_CONFIG),yes)
pg2arrow: $(PG2ARROW_OBJS)
	$(CC) -o $@ $(PG2ARROW_OBJS) -lpq \
	$(shell $(PG_CONFIG) --ldflags) \
	-L $(shell $(PG_CONFIG) --libdir)

install-pg2arrow: pg2arrow
	mkdir -p $(DESTDIR)$(PREFIX)/bin && \
	install -m 0755 pg2arrow $(DESTDIR)$(PREFIX)/bin

__pgsql2arrow.o: sql2arrow.c
	$(CC) $(CFLAGS) -D__PG2ARROW__ -c -o $@ $<
endif

#
# Only if mysql_config is available
#
ifeq ($(HAS_MYSQL_CONFIG),yes)
mysql2arrow: $(MYSQL2ARROW_OBJS)
	$(CC) -o $@ $(MYSQL2ARROW_OBJS) \
	$(shell $(MYSQL_CONFIG) --libs) \
	-Wl,-rpath,$(shell $(MYSQL_CONFIG) --variable=pkglibdir)

install-mysql2arrow: mysql2arrow
	mkdir -p $(DESTDIR)$(PREFIX)/bin && \
	install -m 0755 mysql2arrow $(DESTDIR)$(PREFIX)/bin

__mysql2arrow.o: sql2arrow.c
	$(CC) $(CFLAGS) -D__MYSQL2ARROW__ -c -o $@ $<
endif

#
# Pcap2Arrow
#
pcap2arrow: $(PCAP2ARROW_OBJS)
	$(CC) -o $@ $(PCAP2ARROW_OBJS) -lpthread -lpfring -lpcap

install-pcap2arrow: pcap2arrow
	mkdir -p $(DESTDIR)$(PREFIX)/bin && \
	install -m 0755 pcap2arrow $(DESTDIR)$(PREFIX)/bin

#
# Arrow2CSV
#
install-arrow2csv: arrow2csv
	mkdir -p $(DESTDIR)$(PREFIX)/bin && \
	install -m 0755 arrow2csv $(DESTDIR)$(PREFIX)/bin

arrow2csv: $(ARROW2CSV_OBJS)
	$(CC) -o $@ $(ARROW2CSV_OBJS)

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

install:  $(addprefix install-,$(ALL_PROGS))

clean:
	rm -f $(CLEAN_OBJS)
