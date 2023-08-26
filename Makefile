# ----------------------------------------
#
# Makefile for packaging
#
# ----------------------------------------
include VERSION

__PGSTROM_TGZ := pg_strom-$(PGSTROM_VERSION)
PGSTROM_TGZ   := $(__PGSTROM_TGZ).tar.gz

__SOURCEDIR = $(shell rpmbuild -E %{_sourcedir})
__SPECDIR = $(shell rpmbuild -E %{_specdir})


all: rpm

tarball:
	git archive --format=tar.gz \
	            --prefix=$(__PGSTROM_TGZ)/ \
	            -o $(PGSTROM_TGZ) HEAD src

rpm: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/systemd-pg_strom.conf > \
		$(__SOURCEDIR)/systemd-pg_strom.conf || exit 1
	git show --format=raw $(GITHASH):files/pg_strom.spec.in | \
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"			\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"			\
		    -e "s/@@STROM_TARBALL@@/$(PGSTROM_TGZ)/g"		\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g"			\
		    -e "s/@@PGSQL_VERSION@@/$(PG_MAJORVERSION)/g" >	\
		$(__SPECDIR)/pg_strom-PG$(PG_MAJORVERSION).spec
	rpmbuild -ba $(__SPECDIR)/pg_strom-PG$(PG_MAJORVERSION).spec \
		--undefine=_debugsource_packages
