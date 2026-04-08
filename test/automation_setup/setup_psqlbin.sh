#!/bin/bash


cd /actions-runner/psqlbin/
rm -rf *.tar.bz2 postgresql-*
wget https://ftp.postgresql.org/pub/source/v18.3/postgresql-18.3.tar.bz2
wget https://ftp.postgresql.org/pub/source/v17.9/postgresql-17.9.tar.bz2
wget https://ftp.postgresql.org/pub/source/v16.13/postgresql-16.13.tar.bz2

tar -xjf postgresql-18.3.tar.bz2
tar -xjf postgresql-17.9.tar.bz2
tar -xjf postgresql-16.13.tar.bz2

function build_psql() {
    local version=$1
    local targetpath=/actions-runner/psqlbin/pgbin-$version
    cd /actions-runner/psqlbin/postgresql-$version
    mkdir -p ${targetpath}
    ./configure --prefix=${targetpath}
    make -j$(nproc)
    make install
}

build_psql 18.3
build_psql 17.9
build_psql 16.13