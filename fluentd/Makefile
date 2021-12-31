CC ?= gcc
MODULE = arrow_file.so
OBJS = arrow_nodes.o arrow_write.o arrow_ruby.o
HEAD = arrow_defs.h arrow_ipc.h float2.h

ifneq ($(TD_AGENT),1)
INC_PATHS = . /usr/include
LIB_PATHS = /usr/lib64
DEST_PATH = /usr/lib64/ruby/
else
INC_PATHS = . /opt/td-agent/include/ruby-2.7.0 /usr/include
LIB_PATHS = /opt/td-agent/lib /usr/lib64
DEST_PATH = /opt/td-agent/lib/ruby/2.7.0/x86_64-linux/
endif

CFLAGS = -g -O2 -fPIC -pipe -Wall \
         $(addprefix -I ,$(INC_PATHS)) \
         -Wp,-D_FORTIFY_SOURCE=2 \
         -Wp,-D_GLIBCXX_ASSERTIONS \
         -D_GNU_SOURCE \
         -fexceptions \
         -fstack-protector-strong \
         -grecord-gcc-switches \
         -mtune=generic \
         -fasynchronous-unwind-tables \
         -fstack-clash-protection \
         -fcf-protection

LDFLAGS = -shared \
          $(addprefix -L ,$(LIB_PATHS)) \
          -Wl,-z,relro -Wl,-z,now \
          -fstack-protector-strong \
          -rdynamic -Wl,-export-dynamic

all:	$(MODULE)

$(MODULE): $(OBJS) $(HEAD)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) -lruby

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

install: $(MODULE)
	install -m 755 $(MODULE) $(DEST_PATH)


clean:
	rm -f $(MODULE) $(OBJS)
