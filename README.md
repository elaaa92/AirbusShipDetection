# AirbusShipDetection

Le immagini devono essere all'interno delle cartelle Challenge/train e Challenge/test

Installazione delle api per tensorflow  
https://github.com/tensorflow/tensorflow  
  
Creazione del dataset in formato riconosciuto TFRecord  
python generate_tf.py  
  
Modelli pre-trained:  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
  
Lancio del training  
Copiare il modello desiderato (interno delle cartelle fine_tuned_model) nella cartella models/model  
./train_step.sh  
  
Monitoraggio modello corrente (dalla cartella radice del progetto)  
tensorboard --logdir ./models/model/eval_eval/  
  
Comparazione modelli  
tensorboard --logdir=faster_rcnn:path_to_model_eval_dir,ssd_mobilenet:path_to_model_eval_dir,ssd_inception:path_to_model_eval_dir  

Valutazione modelli  
./eval.sh  
  
Salvataggio dei checkpoint models/model (viene già chiamata alla fine di ogni ciclo di training)  
./config_adjust  
./save.sh  
  
Test dei checkpoint in models/model  
./test.sh  
