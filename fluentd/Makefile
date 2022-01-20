CC ?= gcc
MODULE = arrow_file.so
OBJS = arrow_nodes.o arrow_write.o arrow_ruby.o
HEAD = arrow_defs.h arrow_ipc.h float2.h
SCRIPT = arrow_file_out.rb

INC_PATHS = .
ifneq ($(TD_AGENT),1)
__DEST_PATH = /usr/lib64/ruby
__RUBY_PATH = /usr/share/ruby
else
INC_PATHS += $(shell dirname `ls /opt/td-agent/include/*/ruby.h`)
LIB_PATHS += /opt/td-agent/lib
__DEST_PATH = $(shell dirname `find /opt/td-agent | grep '/zlib.so$$'`)
__RUBY_PATH = $(shell dirname `find /opt/td-agent | grep '/matrix.rb$$'`)
endif
INC_PATHS += /usr/include
LIB_PATHS += /usr/lib64
ifneq ($(DESTDIR),)
DEST_PATH = $(DESTDIR)$(__DEST_PATH)
RUBY_PATH = $(DESTDIR)$(__RUBY_PATH)
else
DEST_PATH = $(__DEST_PATH)
RUBY_PATH = $(__RUBY_PATH)
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

install: $(MODULE) $(SCRIPT)
	mkdir -p $(DEST_PATH) $(RUBY_PATH)
	install -m 755 $(MODULE) $(DEST_PATH)
	install -m 644 $(SCRIPT) $(RUBY_PATH)

uninstall:
	rm -f $(DEST_PATH)/$(MODULE)
	rm -f $(RUBY_PATH)/$(SCRIPT)

clean:
	rm -f $(MODULE) $(OBJS)
