# AirbusShipDetection

Le immagini devono essere all'interno delle cartelle Challenge/train e Challenge/test

Installazione delle api per tensorflow  
https://github.com/tensorflow/tensorflow 

Download di models all'interno della cartella del progetto
git clone https://github.com/tensorflow/models  
  
Creazione del dataset in formato riconosciuto TFRecord  
./generate_tf.sh  
  
Modelli pre-trained:  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
  
Lancio del training  
Copiare il modello desiderato (file ckpt, graph ecc) nella cartella models/model  
./train_step.sh  
  
Monitoraggio modello corrente  
tensorboard --logdir ./models/model/eval_eval/  
  
Comparazione modelli  
tensorboard --logdir=modelname1:path_to_model1_eval_dir,modelname2:path_to_model2_eval_dir,modelname3:path_to_model3_eval_dir  

Valutazione modelli  
./eval.sh  
  
Salvataggio dei checkpoint models/model (viene già chiamata alla fine di ogni ciclo di training)  
./config_adjust  
./save.sh  
  
Generazione dei file con segmentazione delle navi del test set usando checkpoint e modelli salvati in models/model  
./test.sh  
I file vengono salvati in Challenge/submission.  
