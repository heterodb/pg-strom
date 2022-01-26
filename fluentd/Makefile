GITHASH ?= HEAD

RUBY_GEM=gem
RUBY_RAKE=rake
ifeq ($(TD_AGENT), 1)
RUBY_GEM=/opt/td-agent/bin/fluent-gem
RUBY_RAKE=/opt/td-agent/bin/rake
endif
GEM_SPEC=fluent-plugin-arrow-file.gemspec
GEM_VERSION=$(shell cat $(GEM_SPEC) | awk '/spec.version/{print $$3}' | sed 's/\"//g')
GEM_RELEASE ?= 1
GEM_NAME=fluent-plugin-arrow-file
GEM_PACKAGE=$(GEM_NAME)-$(GEM_VERSION).gem
TAR_NAME=$(GEM_NAME)-$(GEM_VERSION).tar
TGZ_NAME=$(TAR_NAME).gz

__RUBY_SCRIPTS=out_arrow_file.rb
RUBY_SCRIPTS=$(addprefix lib/fluent/plugin/,$(__RUBY_SCRIPTS))
__EXT_SOURCE=arrow_write.c arrow_nodes.c \
             arrow_defs.h arrow_ipc.h float2.h
EXT_SOURCE=$(addprefix ../src/,$(__EXT_SOURCE)) arrow_ruby.c
EXT_DIR=ext/arrow_file_write
EXT_CONF=$(EXT_DIR)/extconf.rb

BIN_MODULE=lib/fluent/plugin/arrow_file_write.so

__SOURCEDIR = $(shell rpmbuild -E %{_sourcedir})
__SPECDIR = $(shell rpmbuild -E %{_specdir})
__SPECFILE = fluent-plugin-arrow-file.spec

$(BIN_MODULE): $(EXT_SOURCE)
	cp -f $(EXT_SOURCE) $(EXT_DIR)
	$(RUBY_RAKE) compile

$(GEM_PACKAGE): $(RUBY_SCRIPTS) $(EXT_SOURCE) $(EXT_CONF)
	cp -f $(EXT_SOURCE) $(EXT_DIR)
	$(RUBY_GEM) build

gem: $(GEM_PACKAGE)

tarball:
	(cd .. && git archive --format=tar              \
                      --prefix=$(basename $(TAR_NAME))/ \
                      -o fluentd/$(TAR_NAME) $(GITHASH) \
                      fluentd src) &&                   \
         gzip -f $(TAR_NAME) && test -e $(TGZ_NAME) || exit 1

rpm: tarball
	cp -f $(TGZ_NAME) $(__SOURCEDIR)
	sed -e "s/@@PLUGIN_VERSION@@/$(GEM_VERSION)/g" \
            -e "s/@@PLUGIN_RELEASE@@/$(GEM_RELEASE)/g" \
            -e "s/@@PLUGIN_TARBALL@@/$(TGZ_NAME)/g"    \
            < $(__SPECFILE).in > $(__SPECDIR)/$(__SPECFILE)
	rpmbuild -ba $(__SPECDIR)/$(__SPECFILE)


install: $(GEM_PACKAGE)
	$(RUBY_GEM) install $(GEM_PACKAGE)

uninstall:
	$(RUBY_GEM) uninstall fluent-plugin-arrow-file

clean:
	rm -rf $(EXT_MODULE) $(GEM_PACKAGE) tmp \
	       $(addprefix ext/arrow_file_write/,$(notdir $(EXT_SOURCE))) \
           $(BIN_MODULE) $(TAR_NAME)*
