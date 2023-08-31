# ----------------------------------------
#
# Makefile for packaging
#
# ----------------------------------------
include Makefile.common

__PGSTROM_TGZ := pg_strom-$(PGSTROM_VERSION)
PGSTROM_TGZ   := $(__PGSTROM_TGZ).tar.gz

__SOURCEDIR = $(shell rpmbuild -E %{_sourcedir})
__SPECDIR = $(shell rpmbuild -E %{_specdir})

default: rpm-pg_strom

all: rpm-pg_strom rpm-mysql2arrow rpm-pcap2arrow

tarball:
	git archive --format=tar.gz \
	            --prefix=$(__PGSTROM_TGZ)/ \
	            -o $(PGSTROM_TGZ) $(GITHASH) \
	            LICENSE Makefile Makefile.common \
	            src arrow-tools test/ssbm

rpm-pg_strom: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/systemd-pg_strom.conf > \
		$(__SOURCEDIR)/systemd-pg_strom.conf || exit 1
	git show --format=raw $(GITHASH):files/pg_strom.spec.in |	\
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"		\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"		\
		    -e "s/@@STROM_TARBALL@@/$(__PGSTROM_TGZ)/g"		\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g"		\
		    -e "s/@@PGSQL_VERSION@@/$(PG_MAJORVERSION)/g" >	\
		$(__SPECDIR)/pg_strom-PG$(PG_MAJORVERSION).spec
	rpmbuild -ba $(__SPECDIR)/pg_strom-PG$(PG_MAJORVERSION).spec \
		--undefine=_debugsource_packages

rpm-mysql2arrow: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/mysql2arrow.spec.in |	\
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"		\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"		\
		    -e "s/@@STROM_TARBALL@@/$(__PGSTROM_TGZ)/g"		\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g" >		\
		$(__SPECDIR)/mysql2arrow.spec
	git show --format=raw $(GITHASH):files/pg_strom.spec.in |	\
		awk 'BEGIN {flag=0;} /^%changelog$$/{flag=1; next;} { if (flag>0) print; }' >> \
		$(__SPECDIR)/mysql2arrow.spec
	rpmbuild -ba $(__SPECDIR)/mysql2arrow.spec			\
		--undefine=_debugsource_packages

rpm-pcap2arrow: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/pcap2arrow.spec.in |	\
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"		\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"		\
		    -e "s/@@STROM_TARBALL@@/$(__PGSTROM_TGZ)/g"		\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g" >		\
		$(__SPECDIR)/pcap2arrow.spec
	git show --format=raw $(GITHASH):files/pg_strom.spec.in |	\
		awk 'BEGIN {flag=0;} /^%changelog$$/{flag=1; next;} { if (flag>0) print; }' >> \
		$(__SPECDIR)/mysql2arrow.spec
	rpmbuild -ba $(__SPECDIR)/pcap2arrow.spec			\
		--undefine=_debugsource_packages
