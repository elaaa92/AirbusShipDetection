# AirbusShipDetection

Installazione delle api per tensorflow  
https://github.com/tensorflow/tensorflow  
  
Creazione del dataset in formato riconosciuto TFRecord  
python generate_tf.py  
  
Modelli pre-trained:  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
  
Lancio del training  
Copiare il modello desiderato nella cartella models/model  
./train_step.sh  
  
Monitoraggio modello corrente (dalla cartella radice del progetto)  
tensorboard --logdir ./models/model/eval_eval/  
  
Comparazione modelli  
tensorboard --logdir=faster_rcnn:./models/base-adapted/faster_rcnn_resnet50_coco_2018_01_28_adapted/eval,ssd_mobilenet:./models/base-adapted/ssd_mobilenet_v1_adapted_v2/eval_eval,ssd_inception:./models/base-adapted/ssd_inception_v2_coco_2018_01_28_adapted/eval_eval  

Valutazione modelli  
./eval.sh  
  
Salvataggio modello (viene già chiamata alla fine di ogni ciclo di training)  
./config_adjust  
./save.sh  
