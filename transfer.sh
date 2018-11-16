#!/bin/bash

export PYTHONPATH=:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim

#ckpts=($(ls Challenge/test/))

readarray ckpts < test_diff.txt
len=${#ckpts[@]}
j=0
for i in "${ckpts[@]}"
do
   drive upload --parent 1fLzGHvaa_EIaZUklNueN0gcknw6-UPqj -f Challenge/test/${i}
   if ! ((j % 200)); then 
	echo "\n\n\n\n$j images uploaded\n\n\n\n of $len";
   fi
   j=$((j+1))
done

echo "Upload finished!"

