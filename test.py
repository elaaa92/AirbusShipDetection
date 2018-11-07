from ships_classifier import ShipsClassifier
from matplotlib import pyplot
from os import listdir, getcwd
from os.path import isfile, join, isdir
from sys import path
import numpy as np
import re
import tensorflow as tf
import datetime

size = 768

def get_overlap_mask(bs):
    boxes = bs
    #Order by x then by y coordinate of upper left corner
    boxes = sorted(bs,key=lambda box: (box[0],box[2]))
    nboxes = len(boxes)
    omask = np.zeros((size, size))

    for i in range(0,nboxes):
        for j in range(i+1,nboxes):
            #If there is overlap
            xmin=max(boxes[i][0],boxes[j][0])
            xmax=min(boxes[i][2],boxes[j][2])
            ymin=max(boxes[i][1],boxes[j][1])
            ymax=min(boxes[i][3],boxes[j][3])
            if xmin <= xmax and ymin <= ymax:
                omask[ymin:ymax+1,xmin:xmax+1] = 1
    return omask

def rlmask_from_boxes(boxes,scores,thresh):
    rlm=[]
    nboxes=0
    #Fetch unnormalized coordinates
    #(xmin,ymin,xmax,ymax)
    boxes=[(int(bb[0]*size),int(bb[1]*size),int(bb[2]*size),int(bb[3]*size)) for bb in boxes]
    mass = max([max(bb) for bb in boxes])
    if (mass > 768 *768):
	print('Overflow')
    scores=[max(ss) for ss in scores]
    #Sort by scores
    sbs = zip(scores,boxes)
    sbs.sort(key=lambda sb: -sb[0])
    #Filter out boxes with score under the threshold
    sbs = list(filter(lambda sb: sb[0] > thresh,sbs))
    if (len(sbs) > 0):
        boxes = list(map(lambda sb: sb[1],sbs))
        nboxes = len(boxes)
        omask = get_overlap_mask(boxes)
        b = 0
        #Filter out boxes with complete overlap
        while b < nboxes:
            box = boxes[b]
            smask = np.zeros((size, size))
            smask[box[1]:box[3]+1,box[0]:box[2]+1] = 1
            smask = np.subtract(smask,omask)
            smask[smask!=1] = 0
            #Ordered by rows
            rows,cols = np.nonzero(smask) 
            if len(rows)==0:
                #Complete overlap: delete box and update overlap mask
                #print("Deleted" + str(box))
	        for bb in range(b,nboxes-1):
                    boxes[bb] = boxes[bb+1]
		    scores[bb] = scores[bb+1]
                nboxes = nboxes - 1
                omask = get_overlap_mask(boxes[0:nboxes])
                b = b-1
            b = b+1
        #Get runs removing overlapping areas
        b = 0
        while b < nboxes:
	    box = boxes[b]
            smask = np.zeros((size, size))
            smask[box[1]:box[3]+1,box[0]:box[2]+1] = 1
            smask = np.subtract(smask,omask)
            smask[smask!=1] = 0
            #Ordered by rows
            rows,cols = np.nonzero(smask) 
            #if len(rows)!=0:
            cr = zip(cols,rows)
            #Ordered by columns
            cr.sort(key=lambda x: (x[0],x[1]))
            cols=list(map(lambda x: x[0],cr))
            rows=list(map(lambda x: x[1],cr))
            ship = ""
            start = size*cols[0]+rows[0]+1
            i = 0
            while i<len(rows):
                lenght=0
                flag = True
                while flag:
                    lenght = lenght+1
                    i = i+1
                    #Same column, sequential rows or sequential columns and peripheal rows => same run
                    flag = i<len(rows) and ((cols[i] == cols[i-1] and rows[i] == rows[i-1]+1) or
                                       (cols[i] == cols[i-1]+1 and rows[i-1]==size-1 and rows[i]==0))
                #Close run
                ship=ship+" "+str(start)+" "+str(lenght)  
	        #New run
                if i<len(rows):
                    start = size*cols[i]+rows[i]+1
            ship=ship[1:]
            rlm.append(ship)
            #Next box
            b = b+1	
    return (rlm,nboxes)

def main(_):
    model_path = "./models/model"
    input_path = "./Challenge/test/"
    output_path="./Challenge/submission/"
    #Get available models
    models = [f for f in listdir(model_path) if isdir(join(model_path, f)) and "fine" in f]
    models.reverse()
    #Get images from directory
    image_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    image_files.sort()
    nfiles=len(image_files)
    threshs = [0.95]

    for model in models:
        model_id=re.findall('\d+',model)[0]
        print('Testing model '+model_id)
        print('./models/model/'+model+'/frozen_inference_graph.pb')
        sc = ShipsClassifier(model)
        with open('./models/model/'+model+'/pipeline.config',"r+") as f:
            conf = f.read()
            saved_model_path = re.findall('fine_tune_checkpoint: .+',conf)
            conf=conf.replace(saved_model_path[0],'fine_tune_checkpoint: \"'+getcwd()+'/models/model/model.ckpt\"')
            f.seek(0)
            f.write(conf)
        found=[0 for i in range(len(threshs))]
        read=0
        files = [open(output_path+'submission-'+str(model_id)+'-'+str(thresh)+'.csv',"w") for thresh in threshs]
	for f in files:
	    f.write('ImageId,EncodedPixels\n')
        strs = ['' for thresh in threshs]
        for filename in image_files:
            img = pyplot.imread(input_path+filename)
            (boxes, scores, classes, num) = sc.get_classification(img)
            #There is a row for each class, but in this work there is only one class
            read = read+1
            boxes = list(filter(lambda x: x.any(), boxes[0]))
            if len(boxes) > 0:
		#Write run lenght code
	        for i in range(len(threshs)):
	            rlm,nboxes=rlmask_from_boxes(boxes,scores,threshs[i])
		    if len(rlm) == 0:
			strs[i] = strs[i] + filename+", \n"
		    for run in rlm: 
			strs[i] = strs[i] + filename+","+run+"\n"
		    	#Write boxes
		    	#for bb in boxes: 
                    	#    f.write(filename+","+str(bb)+"\n")
                    found[i]=found[i]+nboxes
            else:
		for i in range(len(threshs)):
		    strs[i] = strs[i] + filename+", \n"
	    if read%100==0 or read == nfiles:
		for i in range(len(threshs)):
		    files[i].write(strs[i])
		    strs[i] = ''
            if read%1000==0:
		for i in range(len(threshs)):
                    print(str(datetime.datetime.now()) + ' - ' + str(read)+" images read of "+str(nfiles)+", "+str(found[i]) + " ships found with threshold " + str(threshs[i]))
	for f in files:
	    f.close()

if __name__ == '__main__':
    tf.app.run()
