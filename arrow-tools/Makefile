#
# Makefile for pcap2arrow
#
PG_CONFIG        ?= pg_config
MYSQL_CONFIG     ?= mysql_config
PROG             = pg2arrow mysql2arrow pcap2arrow arrow2csv
PG2ARROW_OBJS    = __pgsql2arrow.o pgsql_client.o \
                   arrow_nodes.o arrow_write.o arrow_pgsql.o
MYSQL2ARROW_OBJS = __mysql2arrow.o mysql_client.o \
                   arrow_nodes.o arrow_write.o
PCAP2ARROW_OBJS  = pcap2arrow.o arrow_nodes.o arrow_write.o
ARROW2CSV_OBJS   = arrow2csv.o arrow_nodes.o
ALL_OBJS = $(PG2ARROW_OBJS) $(MYSQL2ARROW_OBJS) \
           $(PCAP2ARROW_OBJS) $(ARROW2CSV_OBJS)

CFLAGS = -O2 -fPIC -g -Wall -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64
CFLAGS += -I $(shell $(PG_CONFIG) --includedir)
CFLAGS += $(shell $(MYSQL_CONFIG) --cflags)
#CFLAGS += -O0

PREFIX		?= /usr/local

all: $(PROG)

pg2arrow: $(PG2ARROW_OBJS)
	$(CC) -o $@ $(PG2ARROW_OBJS) -lpq \
	$(shell $(PG_CONFIG) --ldflags) \
	-L $(shell $(PG_CONFIG) --libdir)

mysql2arrow: $(MYSQL2ARROW_OBJS)
	$(CC) -o $@ $(MYSQL2ARROW_OBJS) \
	$(shell $(MYSQL_CONFIG) --libs) \
	-Wl,-rpath,$(shell $(MYSQL_CONFIG) --variable=pkglibdir)

pcap2arrow: $(PCAP2ARROW_OBJS)
	$(CC) -o $@ $(PCAP2ARROW_OBJS) -lpthread -lpfring -lpcap

arrow2csv: $(ARROW2CSV_OBJS)
	$(CC) -o $@ $(ARROW2CSV_OBJS)

__pgsql2arrow.o: sql2arrow.c
	$(CC) $(CFLAGS) -D__PG2ARROW__ -c -o $@ $<

__mysql2arrow.o: sql2arrow.c
	$(CC) $(CFLAGS) -D__MYSQL2ARROW__ -c -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

install-pg2arrow: pg2arrow
	mkdir -p $(DESTDIR)$(PREFIX)/bin && \
	install -m 0755 pg2arrow $(DESTDIR)$(PREFIX)/bin

install-mysql2arrow: mysql2arrow
	mkdir -p $(DESTDIR)$(PREFIX)/bin && \
	install -m 0755 mysql2arrow $(DESTDIR)$(PREFIX)/bin

install-pcap2arrow: pcap2arrow
	mkdir -p $(DESTDIR)$(PREFIX)/bin && \
	install -m 0755 pcap2arrow $(DESTDIR)$(PREFIX)/bin

install-arrow2csv: arrow2csv
	mkdir -p $(DESTDIR)$(PREFIX)/bin && \
	install -m 0755 arrow2csv $(DESTDIR)$(PREFIX)/bin

install: install-pg2arrow install-mysql2arrow \
         install-pcap2arrow install-arrow2csv

clean:
	rm -f $(PROG) $(ALL_OBJS)
