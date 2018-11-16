#!/bin/bash

./export.sh

MODEL_DIRECTORY=`pwd`'/models/model'

echo "Evaluating"

cd $MODEL_DIRECTORY
ckpts=( $(find ./*fine* | grep -o '[0-9]*'))

cd /usr/local/lib/python2.7/dist-packages/tensorflow/models/research/object_detection/legacy/

for i in "${ckpts[@]}"
do
   CHECKPOINT_DIR=$MODEL_DIRECTORY/fine_tuned_model-$i
   EVAL_DIR=$MODEL_DIRECTORY/eval
   PIPELINE_CONFIG_PATH=$MODEL_DIRECTORY/fine_tuned_model-$i/pipeline.config

   python eval.py --logtostderr --checkpoint_dir=${CHECKPOINT_DIR} --eval_dir=${EVAL_DIR} --pipeline_config_path=${PIPELINE_CONFIG_PATH}
done
