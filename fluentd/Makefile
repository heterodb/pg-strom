RUBY_GEM=gem
RUBY_RAKE=rake
ifeq ($(TD_AGENT), 1)
RUBY_GEM=/opt/td-agent/bin/fluent-gem
RUBY_RAKE=/opt/td-agent/bin/rake
endif
GEM_SPEC=fluent-plugin-arrow-file.gemspec
GEM_VERSION=$(shell cat $(GEM_SPEC) | awk '/spec.version/{print $$3}' | sed 's/\"//g')
GEM_PACKAGE=fluent-plugin-arrow-file-$(GEM_VERSION).gem
GEM_PREFIX=lib/fluent/plugin
GEM_SCRIPTS=$(GEM_PREFIX)/out_arrow_file.rb

__EXT_SOURCE=arrow_ruby.c arrow_write.c arrow_nodes.c \
             arrow_defs.h arrow_ipc.h float2.h extconf.rb
EXT_SOURCE=$(addprefix ext/arrow_file_write/,$(__EXT_SOURCE))
EXT_MODULE=$(GEM_PREFIX)/arrow_file_write.so

$(GEM_PACKAGE): $(EXT_MODULE) $(GEM_SCRIPTS)
	$(RUBY_GEM) build

$(EXT_MODULE): $(EXT_SOURCE)
	rm -rf tmp
	$(RUBY_RAKE) compile

install: $(GEM_PACKAGE)
	$(RUBY_GEM) install $(GEM_PACKAGE)

uninstall:
	$(RUBY_GEM) uninstall fluent-plugin-arrow-file

clean:
	rm -rf $(EXT_MODULE) $(GEM_PACKAGE) tmp
