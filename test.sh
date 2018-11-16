#!/bin/bash

export PYTHONPATH=:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research:/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim

PROJECT_DIRECTORY=`pwd`

echo "Saving"
./save.sh

echo "Testing"
cd $PROJECT_DIRECTORY
python test.py

: '
	cd
	ask=`zenity --info --title "Testing completed" --text "Train or shutdown?" --ok-label="cancel" --extra-button="shutdown" --extra-button="train" --timeout=60`

	if [ "$ask" == "train" ]; then
	    cd $PROJECT_DIRECTORY
	    ./train.sh
	fi

	zenity --info --title "Job completed" --text "Shutting down" --timeout=60
	umount /media/elisa/D43E3FC63E3FA082
	poweroff
'
