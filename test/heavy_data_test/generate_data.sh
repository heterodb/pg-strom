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
    if [[ $# -ne 2 ]]; then
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
    temporary_dir_path="${1}/$(date +%Y%m%d%H%M%S)"
    mkdir -p ${temporary_dir_path}
    cd ${temporary_dir_path}
    ${dbgen_path} -s ${2}
}

main $@