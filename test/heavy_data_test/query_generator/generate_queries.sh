#!/bin/bash


function convert_query(){
    echo "SET pg_strom.gpudirect_enabled = off;"
    echo "EXPLAIN ANALYZE"
    grep -v "^--" $1 | sed '0,/from/ s/from/into temporary result_store from/'
    echo ";"
    echo "select * from result_store;"
}

function run_all(){
    script_dir_path="$(cd $(dirname $0); pwd)"
    patterns_path="${script_dir_path}/query_patterns"
    conditions_dir_path="${script_dir_path}/conditions"
    query_path="${script_dir_path}/.."

    find ${patterns_path} -type f -name "*.sql" | sort -n | while read file; do
        find ${conditions_dir_path} -type f -name "*.sh" | sort -n | while read condition; do
            condition_label=$(basename ${condition} .sh)
            file_label=$(basename ${file} .sql)
            source ${condition}
            {
                echo "SET search_path=tpch;"
                echo "DROP TABLE IF EXISTS result_store;"
                # set parameters
                echo ${PARAMETERS_CONDITION}
                # change table name
                sed "s/{{{ORDERS_TABLE_NAME}}}/${ORDERS_TABLE_NAME}/" $file
                echo "select * from result_store;"
            } > ${query_path}/${condition_label}_${file_label}.sql
        done
    done
}

run_all $@