import tensorflow as tf
from os import listdir, getcwd
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
    x=[]
    y=[]
    #for each row construct a row bounding box
    for i in range(len(rl_list)):
	#Runs are 1-indexed
        start = float(rl_list[i].split(" ")[0])-1
        lenght = float(rl_list[i].split(" ")[1])
        #Rows of run extrema
        x.append(start / size)
        x.append((start+lenght) / size)
        y.append(start % size)
        y.append((start + lenght) % size)
    xmin=int(round(min(x)))
    xmax=int(round(max(x)))
    ymin=int(round(min(y)))
    ymax=int(round(max(y)))
    return (xmin,xmax,ymin,ymax) #left, right, top, bottom

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
 
    #Write label infos
    with open(output_path+'label_map.pbtxt',"w") as f:
        f.write("item {\nid: 1\nname: 'ship'\n}")

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
    neval = int(3*nselected/10)
    eval_set=random.sample(range(1, nselected), neval)

    i = 0
    j = 0
    k = 0
    writer_train = tf.python_io.TFRecordWriter(output_path+'train.record')
    writer_eval = tf.python_io.TFRecordWriter(output_path+'eval.record')
    for filename in image_files:
        bb_list = []
	#Skip absent images in rlm
	if rlm[i][0] != filename:
	    print('skipped ' + filename)
            i = i+1
	else:
            while i<len(rlm) and rlm[i][0] == filename:
                if rlm[i][1] != '':
                    bb_list.append(bb_from_rlmask(rlm[i][1]))
                i = i+1
            if len(bb_list) > 0:
	        tf_image = create_tf(input_path,filename,bb_list)
                if j in eval_set:
                    writer_eval.write(tf_image.SerializeToString())
		    k = k+1
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
        print(str(k)+" for eval and "+str(j-k)+" for training model.")

"""
#  Debug, show bounding boxes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
input_path = "./Challenge/train/"
filename = "00113a75c.jpg"
rlm=["401790 1 402557 3 403325 5 404092 7 404859 9 405627 11 406394 13 407162 15 407929 17 408696 20 409464 21 410231 23 411000 24 411770 23 412539 23 413308 24 414078 23 414847 24 415617 23 416386 23 417155 24 417925 23 418694 23 419464 23 420233 23 421002 21 421772 18 422541 17 423310 15 424080 12 424849 10 425619 7 426388 6 427157 4 427927 1",
    "110888 2 111654 4 112420 7 113186 9 113953 11 114722 10 115490 11 116259 10 117027 11 117796 10 118564 11 119333 10 120101 10 120870 10 121638 10 122407 10 123175 10 123944 10 124712 10 125481 10 126249 10 127018 10 127786 10 128555 9 129323 10 130092 9 130860 10 131629 9 132397 10 133166 9 133934 10 134703 9 135471 10 136240 9 137008 7 137777 4 138545 2",
    "394109 1 394876 4 395644 5 396411 8 397178 10 397946 12 398713 14 399482 14 400252 14 401021 14 401791 14 402561 13 403330 11 404099 10 404869 7 405638 5 406407 4 407177 1",
    "123159 1 123925 4 124691 6 125458 8 126227 7 126995 8 127764 7 128532 8 129301 7 130069 8 130838 7 131607 7 132375 7 133144 6 133912 7 134681 6 135449 7 136218 6 136987 6 137755 6 138524 6 139292 6 140061 6 140829 6 141598 6 142367 5 143135 6 143904 5 144672 6 145441 5 146209 4 146978 1",
    "111647 1 112413 3 113179 6 113945 8 114711 10 115477 13 116244 14 117012 15 117781 14 118549 15 119318 14 120086 15 120855 14 121623 15 122392 14 123160 14 123929 14 124697 14 125466 14 126234 14 127003 14 127771 14 128540 14 129309 13 130077 14 130846 13 131614 13 132383 13 133151 13 133920 13 134689 12 135457 13 136226 12 136994 13 137763 12 138531 13 139300 12 139313 1 140069 13 140837 14 141606 13 142374 14 143143 13 143911 14 144680 13 145447 15 146215 13 146983 11 147752 8 148520 6 149289 3 150057 1",
    "536114 1 536880 4 537646 6 538413 8 539179 10 539946 12 540715 12 541483 12 542252 12 543020 12 543789 12 544558 12 545326 12 546095 12 546863 12 547632 12 548401 12 549169 12 549938 12 550706 13 551475 12 552244 12 553012 12 553781 12 554550 12 555318 12 556087 12 556855 12 557624 12 558393 12 559161 12 559930 12 560698 12 561467 12 562236 10 563004 8 563773 5 564541 4 565310 1",
    "172862 2 173628 4 174394 7 175160 9 175926 12 176692 14 177459 16 178227 16 178996 16 179764 16 180533 16 181301 16 182070 16 182838 16 183607 16 184375 16 185144 16 185912 16 186681 16 187449 16 188218 16 188986 16 189755 16 190523 16 191292 16 192060 16 192829 16 193597 16 194366 16 195134 16 195903 15 196671 16 197440 15 198208 16 198977 15 199745 16 200513 16 201282 16 202050 16 202819 16 203587 16 204356 16 205124 16 205893 16 206661 16 207430 16 208198 16 208967 16 209735 16 210504 16 211272 16 212041 16 212809 16 213578 16 214346 16 215115 16 215883 16 216652 16 217420 16 218189 13 218957 11 219726 8 220494 6 221263 3 222031 1"]
image = Image.open(input_path+filename)
fig,ax = plt.subplots(1)
bb_list = []
i=0
while i<len(rlm):
    if rlm[i] != '':
        bb_list.append(bb_from_rlmask(rlm[i]))
    i = i+1
for bb in bb_list:
    print(map(lambda e: e/float(size),bb))
    ax.add_patch(patches.Rectangle((bb[0],bb[3]),bb[1]-bb[0],bb[2]-bb[3],linewidth=1,edgecolor='r',facecolor='none'))
ax.imshow(image)
plt.show()
"""

if __name__ == '__main__':
    tf.app.run()


