#!/bin/bash


function usage() {
    echo "USAGE: bash load.sh <temporary directory path> <size(GB)>"
    echo " generate data files and load to PostgreSQL"
}

function get_disk_availability(){
    df --output=avail $1 2>/dev/null | grep "^[0-9]*$"
}

function check_disk_availability(){
    available_size=$(get_disk_availability ${1})
    required_size=$(($2 * 1024 * 1024))
    [[ $available_size -lt $required_size ]] && return 1 || return 0
}

function main(){
    # parameter validation check
    ## check the number of parameters
    if [[ $# -lt 2 ]]; then
        echo $#
        usage
        exit 1
    fi
    # check the first parameter is a directory
    if [[ ! -d $1 ]]; then
        echo "ERROR: $1 is not a directory or does not exist."
        usage
        exit 1
    fi
    # check the second parameter is a number
    if [[ ! $2 =~ ^[0-9]+$ ]]; then
        echo "ERROR: $2 is not a number."
        usage
        exit 1
    fi

    # check the disk availability
    check_disk_availability $@
    if [[ $? -ne 0 ]]; then
        echo "ERROR: disk space is not enough."
        exit 1
    fi

    # set the temporary directory path
    temporary_dir_path="${1}/$(date +%Y%m%d%H%M%S)"
    size=${2}
    # remove the first two parameters
    shift
    shift

    # PostgreSQL is running?
    psql $@ -c "SELECT 1" 2>&1 > /dev/null 
    if [[ $? -ne 0 ]]; then
        echo "ERROR: PostgreSQL is not running."
        exit 1
    fi

    # check the dbgen is available
    tpch_dir_path="$(cd $(dirname $0); pwd)/../tpch"
    dbgen_path="${tpch_dir_path}/dbgen-tpch"
    current_path=$(pwd)
    if [[ ! -e ${dbgen_path} ]]; then
        cd ${tpch_dir_path}
        make -j$(nproc)
        cd ${current_path}
    fi

    # create a temporary directory
    mkdir -p ${temporary_dir_path}
    cd ${temporary_dir_path}
    ${dbgen_path} -s ${size}
    cd ${current_path}

    # create a table
    {
    cat << EOL
DROP SCHEMA IF EXISTS tpch CASCADE;
CREATE SCHEMA tpch;
SET search_path = tpch;
EOL
    cat ${tpch_dir_path}/tpch-ddl.sql
    } | psql $@

    # load data
    grep "CREATE TABLE" ${tpch_dir_path}/tpch-ddl.sql | awk '{print $3}' | while read table_name; do
        echo "\COPY tpch.${table_name} FROM '${temporary_dir_path}/${table_name}.tbl' DELIMITER '|' CSV;"
    done | psql $@

    # remove the temporary directory
    rm -rf ${temporary_dir_path}
}

main $@