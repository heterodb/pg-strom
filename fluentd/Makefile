RUBY_GEM=gem
ifeq ($(TD_AGENT), 1)
RUBY_GEM=/opt/td-agent/bin/fluent-gem
endif
GEM_SPEC=fluent-plugin-arrow-file.gemspec
GEM_VERSION=$(shell cat $(GEM_SPEC) | awk '/spec.version/{print $$3}' | sed 's/\"//g')
GEM_PACKAGE=fluent-plugin-arrow-file-$(GEM_VERSION).gem

__RUBY_SCRIPTS=out_arrow_file.rb
RUBY_SCRIPTS=$(addprefix lib/fluent/plugin/,$(__RUBY_SCRIPTS))
__EXT_SOURCE=arrow_write.c arrow_nodes.c \
             arrow_defs.h arrow_ipc.h float2.h
EXT_SOURCE=$(addprefix ../src/,$(__EXT_SOURCE)) arrow_ruby.c
EXT_DIR=ext/arrow_file_write
EXT_CONF=$(EXT_DIR)/extconf.rb

$(GEM_PACKAGE): $(RUBY_SCRIPTS) $(EXT_SOURCE) $(EXT_CONF)
	cp -f $(EXT_SOURCE) $(EXT_DIR)
	$(RUBY_GEM) build

install: $(GEM_PACKAGE)
	$(RUBY_GEM) install $(GEM_PACKAGE)

uninstall:
	$(RUBY_GEM) uninstall fluent-plugin-arrow-file

clean:
	rm -rf $(EXT_MODULE) $(GEM_PACKAGE) tmp \
	       $(addprefix ext/arrow_file_write,$(notdir $(EXT_SOURCE)))
