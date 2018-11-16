#!/bin/bash

export PYTHONPATH=:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim

#export CUDA_VISIBLE_DEVICES=""

PROJECT_DIRECTORY=./

echo "Training"
PIPELINE_CONFIG_PATH=$PROJECT_DIRECTORY/models/model/pipeline.config
MODEL_DIR=$PROJECT_DIRECTORY/models/model
NUM_TRAIN_STEPS=80000
NUM_EVAL_STEPS=2000

cd /usr/local/lib/python2.7/dist-packages/tensorflow/models/research
    python object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --num_eval_steps=${NUM_EVAL_STEPS} --alsologtostderr

    cd $PROJECT_DIRECTORY
    ./save.sh
done

: '
    cd

    ask=`zenity --info --title "Training completed" --text "Test or shutdown?" --ok-label="cancel" --extra-button="shutdown" --extra-button="test" --timeout=60`

    if [ "$ask" == "test" ]; then
        cd $PROJECT_DIRECTORY
        python test.py
    fi

    zenity --info --title "Job completed" --text "Shutting down" --timeout=60
    umount /media/elisa/D43E3FC63E3FA082
    poweroff
'

