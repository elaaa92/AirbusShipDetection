import tensorflow as tf
from os import listdir,getcwd
from os.path import isfile, join, dirname
from sys import path

from PIL import Image
import re
import csv
dataset_util_path = dirname(tf.__file__) + '/models/research/object_detection/utils'
if not dataset_util_path in path: path.append(dataset_util_path)
import dataset_util
import random

size = 768

def bb_from_rlmask(rlmask):
  #list of run (start lenght format)
  rl_list = re.findall("[^ ]+ [^ ]+",rlmask)
  xmin = 768
  ymin = 768
  xmax = -1
  ymax = -1
  #for each row construct a row bounding box
  for i in range(len(rl_list)):
    start = int(rl_list[i].split(" ")[0])
    lenght = int(rl_list[i].split(" ")[1])
    #Rows of run extrema
    y1 = start / size
    y2 = (start + lenght) / size
    #Record one row at time
    while y1 <= y2:
       #Single row bounding box
       x1 = (start % size)
       x2 = (start + lenght) % size
       xmin = x1 if x1 < xmin else xmin		#Left x coordinate
       xmax = x2 if x2 > xmax else xmax 	#Right x coordinate
       ymin = y1 if y1 < ymin else ymin 	#Top y coordinate
       ymax = y1 if y1 > ymax else ymax		#Bottom y coordinate
       #print((xmin,xmax,ymin,ymax))
       #Pick next row
       lenght = (x1 + lenght) % size
       y1 = y1 + 1
       start = y1 * size
  return (xmin,xmax,ymin,ymax)


def create_tf(input_path,filename,bb_list):
  height = size # Image height
  width = size # Image width
  encoded_image_data = tf.gfile.GFile(input_path+filename).read() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'
  xmins = [i[0]/float(size) for i in bb_list] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [i[1]/float(size) for i in bb_list] # List of normalized right x coordinates in bounding box (1 per box)
  ymins = [i[2]/float(size) for i in bb_list] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [i[3]/float(size) for i in bb_list] # List of normalized bottom y coordinates in bounding box (1 per box)
  classes_text = ['ship'] * len(bb_list) # List of string class name of bounding box (1 per box)
  classes = [1] * len(bb_list) # List of integer class id of bounding box (1 per box)
  tf_image = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_image


def main(_):
  bb_list = []
  input_path = "./Challenge/train/"
  output_path = "./data/"
 
  #Adjust config
  conf=''
  with open('./models/model/pipeline.config',"r+") as f:
    conf = f.read()
    label_path = re.findall('PATH_TO_BE_CONFIGURED/\w*\.pbtxt',conf)
    conf=conf.replace(label_path[0],getcwd()+'/data/label_map.pbtxt')
    record_paths = re.findall('PATH_TO_BE_CONFIGURED/\w*\.record',conf)
    conf=conf.replace(record_paths[0],getcwd()+'/data/train.record')
    conf=conf.replace(record_paths[1],getcwd()+'/data/train.record')
    saved_model_path = re.findall('PATH_TO_BE_CONFIGURED/\w*\.ckpt',conf)
    conf=conf.replace(saved_model_path[0],getcwd()+'/models/model/model.ckpt')
    num_classes = re.findall('num_classes:\s\d*',conf)
    conf=conf.replace(num_classes[0],'num_classes: 1')
    train_config = re.findall('}\ntrain_input_reader',conf)
    conf=conf.replace(train_config[0],'  batch_queue_capacity: 2\n  prefetch_queue_capacity: 2\n}\ntrain_input_reader')
    f.seek(0)
    f.write(conf)

  #Write label infos
  with open(output_path+'label_map.pbtxt',"w") as f:
    f.write("item {\nid: 1\nname: 'Ship'\n}")

  rlm = []
  with open ('./Challenge/train_ship_segmentations.csv','rb') as csvfile:
    rlm = list(csv.reader(csvfile))
  #Remove row with infos
  rlm = rlm[1:]
  #Use images with ships only
  selected = list(filter(lambda x: x[1] != '', rlm))
  #Names of images with ships
  names = set(map(lambda x: x[0],selected))
  #Number of images with ships
  nselected = len(names)

  #Get images from directory
  image_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
  image_files.sort()

  #Select subset for evaluation
  neval = int(nselected/10)
  eval_set=random.sample(range(1, nselected), neval)

  i = 0
  j = 0
  writer_train = tf.python_io.TFRecordWriter(output_path+'train.record')
  writer_eval = tf.python_io.TFRecordWriter(output_path+'eval.record')
  for filename in image_files:
    bb_list = []
    while i<len(rlm) and rlm[i][0] == filename:
      if rlm[i][1] != '':
        bb_list.append(bb_from_rlmask(rlm[i][1]))
      i = i+1
    tf_image = create_tf(input_path,filename,bb_list)
    if len(bb_list) > 0:
      if j in eval_set:
        writer_eval.write(tf_image.SerializeToString())
      else:
        writer_train.write(tf_image.SerializeToString())
      if j%1000 == 0:
        print(str(j) + " of " + str(nselected))
      j = j+1
  writer_eval.close()
  writer_train.close()

  #Check number of images with ships
  if (nselected == j):
    print(str(nselected)+" images correctly converted:")
    print(str(neval)+" for eval and "+str(nselected-neval)+" for training model.")

#  Debug, show bounding boxes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
input_path = "./Challenge/train/"
filename = "0a1a7f395.jpg"
runlenghtmask = "210236 2 211004 5 211771 9 212539 12 213307 15 214075 19 214842 23 215610 26 216378 29 217145 33 217913 36 218681 39 219448 43 220216 46 220984 49 221752 52 222519 57 223287 60 224057 61 224828 61 225599 61 226370 60 227141 57 227913 52 228684 49 229455 46 230226 41 230997 38 231768 2 231772 31 232543 27 233314 24 234086 20 234857 17 235628 13 236398 11 237166 12 237937 9 238708 5 239479 2"
image = Image.open(input_path+filename)
fig,ax = plt.subplots(1)
bb_list = bb_from_rlmask(runlenghtmask)
for i in range(len(bb_list)):
     rect = patches.Rectangle((bb_list[i][3],bb_list[i][0]),bb_list[i][2]-bb_list[i][3]+1,bb_list[i][1]-bb_list[i][0]+1,linewidth=1,edgecolor='r',facecolor='none')
     ax.add_patch(rect)
     ax.imshow(image)
plt.show()

if __name__ == '__main__':
  tf.app.run()
