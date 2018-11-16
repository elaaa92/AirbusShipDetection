#!/bin/bash

./export.sh

PROJECT_DIRECTORY=`pwd`

PIPELINE_CONFIG_PATH=$PROJECT_DIRECTORY/models/model/pipeline.config
MODEL_DIR=$PROJECT_DIRECTORY/models/model
NUM_EVAL_STEPS=2000

echo 'Fix configuration file'

python config_adjust.py

echo train

for i in `seq 28380 1000 50000`;
do
    NUM_TRAIN_STEPS=${i}
    cd /usr/local/lib/python2.7/dist-packages/tensorflow/models/research
    echo "python object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --num_eval_steps=${NUM_EVAL_STEPS} --alsologtostderr"
    python object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --num_eval_steps=${NUM_EVAL_STEPS} --alsologtostderr
    cd $PROJECT_DIRECTORY
    ./save.sh
done
echo 'Train completed'
