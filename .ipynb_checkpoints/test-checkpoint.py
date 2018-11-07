from ships_classifier import ShipsClassifier
from matplotlib import pyplot
from os import listdir
from os.path import isfile, join, isdir
from sys import path
import numpy as np
import re

model_path = "./models/model"
input_path = "./Challenge/test/"
output_path="./Challenge/"
#Get available models
models = [f for f in listdir(model_path) if isdir(join(model_path, f)) and "fine" in f]
#Get images from directory
image_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
image_files.sort()
nfiles=len(image_files)

for model in models:
   model_id=re.findall('\d+',model)[0]
   print('Testing model '+model_id)
   sc = ShipsClassifier(model)
   found=0
   read=0
   with open(output_path+'submission-'+str(model_id)+'.csv',"w") as f:
      for filename in image_files:
         img = pyplot.imread(input_path+filename)
         (boxes, scores, classes, num) = sc.get_classification(img)
         #There is a row for each class, but in this work there is only one class
         read = read+1
         boxes = list(filter(lambda x: x.any(), boxes[0]))
         if len(boxes) > 0:
            f.write(filename+","+str(boxes)+"\n")
            found=found+len(boxes)
         else:
            f.write(filename+",-\n")
         if read%1000==0:
            print(str(read)+" images read of "+str(nfiles)+", "+str(found) + " ships found")
