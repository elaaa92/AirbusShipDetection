#!/bin/bash

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

