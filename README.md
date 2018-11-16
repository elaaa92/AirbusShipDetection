# AirbusShipDetection

L'applicazione si riferisce al task definito dalla challenge Airbus Ship Detection  
https://www.kaggle.com/c/airbus-ship-detection  

Il dataset può essere scaricato all'indirizzo  
https://www.kaggle.com/c/airbus-ship-detection/data  

Le immagini devono essere all'interno delle cartelle Challenge/train e Challenge/test

Installazione delle api per tensorflow  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md  
NB prima di installare le cocoapi (come indicato nella guida) scaricare models  
cd <path_to_tensorflow>  
git clone https://github.com/tensorflow/models  
  
Creazione del dataset in formato riconosciuto TFRecord  
./generate_tf.sh  
  
Modelli pre-trained:  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
  
Lancio del training  
Copiare il modello desiderato (tutti i file ckpt e pipeline.config) nella cartella models/model  
./train_step.sh  
NB nel pipeline.config è necessario impostare l'opzione batch_size con un valore opportuno a seconda delle capacità della propria macchina; se non particolarmente performante è possibile mettere batch_size=1.  
NNB Assicurarsi che le configurazioni dei modelli siano quelle dell'ultima versione di tensorflow, perché in alternativa potrebbero non funzionare correttamente  
  
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
