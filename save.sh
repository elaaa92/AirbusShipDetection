#!/bin/bash

export PYTHONPATH=:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim

PROJECT_DIRECTORY=`pwd`
echo "Saving"
cd $PROJECT_DIRECTORY
ckpts=( $(find models/model/*ckpt*.meta | grep -o '[0-9]*'))

cd /usr/local/lib/python2.7/dist-packages/tensorflow/models/research/object_detection
for i in "${ckpts[@]}"
do
   PIPELINE_CONFIG_PATH=$PROJECT_DIRECTORY/models/model/pipeline.config
   FINE_TUNED_MODEL=$PROJECT_DIRECTORY/models/model/fine_tuned_model-$i
   TRAINED_CHECKPOINT_PREFIX=$PROJECT_DIRECTORY/models/model/model.ckpt-$i
   mkdir $FINE_TUNED_MODEL
   python export_inference_graph.py --input_type image_tensor --pipeline_config_path ${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix ${TRAINED_CHECKPOINT_PREFIX} --output_directory ${FINE_TUNED_MODEL}
   cp $PROJECT_DIRECTORY/models/model/pipeline.config $PROJECT_DIRECTORY/models/model/fine_tuned_model-$i/pipeline.config
done

