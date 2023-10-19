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
        echo "$Build_DIR/bin/gspmm --mode=$Backend_ALG --max_ncols $max_ncols $file"
        $Build_DIR/bin/gspmm --mode=$Backend_ALG --max_ncols $max_ncols $file
        if [ $? -eq 0 ]; then
            Test_Passed_List+=("$Build_DIR/bin/gspmm --mode=$Backend_ALG --max_ncols $max_ncols $file")
        else
            Test_Failed_List+=("$Build_DIR/bin/gspmm --mode=$Backend_ALG --max_ncols $max_ncols $file")
        fi
    done
done

clear
if [ ${#Test_Passed_List[@]} -gt 0 ];then
    echo "Test Passed List:"
fi
for case in ${Test_Passed_List[@]}
do
    echo "   $case"
done

if [ ${#Test_Failed_List[@]} -gt 0 ];then
    echo "Test Failed List:"
fi
for case in ${Test_Failed_List[@]}
do
    echo "   $case"
done