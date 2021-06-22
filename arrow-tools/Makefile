#
# Makefile for pcap2arrow
#
PG_CONFIG	?= pg_config
MYSQL_CONFIG	?= mysql_config
PROG		= pg2arrow mysql2arrow pcap2arrow
PG2ARROW_OBJS	= sql2arrow.o pgsql_client.o arrow_nodes.o arrow_write.o arrow_pgsql.o
MYSQL2ARROW_OBJS= sql2arrow.o mysql_client.o arrow_nodes.o arrow_write.o
PCAP2ARROW_OBJS	= pcap2arrow.o arrow_nodes.o arrow_write.o

CFLAGS = -O2 -fPIC -g -Wall -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64
CFLAGS += -I $(shell $(PG_CONFIG) --includedir)
CFLAGS += $(shell $(MYSQL_CONFIG) --cflags)
#CFLAGS += -O0

PREFIX := /usr/local

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

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

install: $(PROG)
	install -m 0755 $(PROG) $(PREFIX)/bin

clean:
	rm -f $(PROG) $(PG2ARROW_OBJS) $(PCAP2ARROW_OBJS)
