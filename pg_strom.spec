Name: pg_strom
Version: %{strom_version}
Release: 1%{?dist}
Summary: GPU utilized SQL query accelerator for PostgreSQL
Group: Applications/Databases
License: GPL 2.0
URL: https://github.com/pg-strom/devel
Source0: pg_strom-%{strom_version}.tar.gz
BuildRequires: postgresql%{pgsql_pkgver}         >= %{pgsql_minver}
BuildRequires: postgresql%{pgsql_pkgver}-devel   >= %{pgsql_minver}
%if %{?pgsql_maxver:1}%{!?pgsql_maxver:0}
BuildRequires: postgresql%{pgsql_pkgver}         < %{pgsql_maxver}
BuildRequires: postgresql%{pgsql_pkgver}-devel   < %{pgsql_maxver}
%endif
BuildRequires: cuda-misc-headers-%{cuda_pkgver}  >= %{cuda_minver}
BuildRequires: cuda-nvrtc-dev-%{cuda_pkgver}     >= %{cuda_minver}
%if %{?cuda_maxver:1}%{!?cuda_maxver:0}
BuildRequires: cuda-misc-headers-%{cuda_pkgver}  < %{cuda_maxver}
BuildRequires: cuda-nvrtc-dev-%{cuda_pkgver}     < %{cuda_maxver}
%endif
Requires: postgresql%{pgsql_pkgver}-server       >= %{pgsql_minver}
Requires: postgresql%{pgsql_pkgver}-libs         >= %{pgsql_minver}
%if %{?pgsql_maxver:1}%{!?pgsql_maxver:0}
Requires: postgresql%{pgsql_pkgver}-server       <  %{pgsql_maxver}
Requires: postgresql%{pgsql_pkgver}-libs         <  %{pgsql_maxver}
%endif
Requires: cuda-nvrtc-%{cuda_pkgver}              >= %{cuda_minver}
Requires: cuda-cudart-dev-%{cuda_pkgver}         >= %{cuda_minver}
%if %{?cuda_maxver:1}%{!?cuda_maxver:0}
Requires: cuda-nvrtc-%{cuda_pkgver}              <  %{cuda_maxver}
Requires: cuda-cudart-dev-%{cuda_pkgver}         <  %{cuda_maxver}
%endif

BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root

%define __pg_config %(rpm -ql postgresql%{pgsql_pkgver} | grep /bin/pg_config$)
%define __pkglibdir %(%{__pg_config} --pkglibdir)
%define __pkgbindir %(%{__pg_config} --bindir)
%define __pkgsharedir %(%{__pg_config} --sharedir)
%define __cuda_lib64 %(rpm -ql cuda-nvrtc-dev-%{cuda_pkgver} | grep /lib64$)

%description
PG-Strom is an extension for PostgreSQL, to accelerate analytic queries
towards large data set using the capability of GPU devices.

%prep
%setup -q

%build
rm -rf %{buildroot}
%{__make} PG_CONFIG=%{__pg_config}
echo %{__cuda_lib64} > pgstrom-cuda-lib64.conf

%install
#rm -rf %{buildroot}
%{__make} PG_CONFIG=%{__pg_config} DESTDIR=%{buildroot} install

# config to use CUDA/NVRTC
%{__mkdir} -p %{buildroot}/etc/ld.so.conf.d
%{__install} -Dp pgstrom-cuda-lib64.conf %{buildroot}/etc/ld.so.conf.d/pgstrom-cuda-lib64.conf

%clean
rm -rf %{buildroot}

%post
ldconfig

%postun
ldconfig

%files
%defattr(-,root,root,-)
%doc LICENSE README.md
%{__pkglibdir}/pg_strom.so
%{__pkgbindir}/gpuinfo
%{__pkgsharedir}/extension/*
%config(noreplace) /etc/ld.so.conf.d/pgstrom-cuda-lib64.conf

%changelog
* Sat May 14 2016 KaiGai Kohei <kaigai@kaigai.gr.jp>
- initial RPM specfile
