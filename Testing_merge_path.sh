#!/bin/bash
Build_DIR=./build
Backend_ALG=mergepath
for arg in $@;
do
    if [[ $arg == "--prefix="* ]];
    then
        Build_DIR=$(echo $arg | cut -d'=' -f2)
    fi
    if [[ $arg == "--mode="* ]];
    then
        Backend_ALG=$(echo $arg | cut -d'=' -f2)
    fi
done

Dataset_DIR=./dataset/europar
Test_DIR=./
dataset_files=$(find $Dataset_DIR -type f -name "*.mtx")
Test_Failed_List=()
Test_Passed_List=()
for file in ${dataset_files[@]}; do
    for max_ncols in 32 64 #256 1024 16384 32768 524288 1048576
    do
        Command_Str="$Build_DIR/bin/gspmm --mode=$Backend_ALG --max_ncols $max_ncols $file"
        echo "$Command_Str"
        $Build_DIR/bin/gspmm --mode=$Backend_ALG --max_ncols $max_ncols $file
        if [ $? -eq 0 ]; then
            echo -e "\e[32m$Command_Str ---- Passed"
        else
            echo -e "\e[31m$Command_Str ---- Failed"
        fi
        echo -e "\e[0m\n"
    done
done