# Import the converted model's classi
from shapenet_tensor import RCNN_Fine_Tune as MyNet

import tensorflow as tf
import numpy as np
from PIL import Image


g_caffe_prob_keys = ['fc-azimuth','fc-elevation','fc-tilt']

batch_size = 1

images= tf.placeholder(tf.float32, [batch_size,227,227,3]) 
net = MyNet({'data':images})

azimuth = net.layers['fc-azimuth']
elevation = net.layers['fc-elevation']
tilt = net.layers['fc-tilt']


with tf.Session() as sesh:

    #img = Image.open('test.jpg')
    img = Image.open('test9.jpg')
    img = np.array(img)
    
    # Load the data
    net.load('shapenet_tensor.npy', sesh)
    
    feed = {images:[img]}
    az, ev, ti = sesh.run([azimuth,elevation,tilt], feed_dict=feed)
    prob_lists = []
    prob_lists.append(az)
    prob_lists.append(ev)
    prob_lists.append(ti)
    #print az[0][0][360:2*306]
    #print len(az), len(az[0]), len(az[0][0]), az[0][0][0] 
   
    result_keys = g_caffe_prob_keys
    # EXTRACT PRED FROM PROBS
    preds = []
    for k in range(len(result_keys)):
        preds.append([])
    class_idx = 0 
    # pred is the class with highest prob within
    # class_idx*360~class_idx*360+360-1
    for k in range(len(result_keys)):
        probs = prob_lists[k][0]
        probs = probs[class_idx*360:(class_idx+1)*360]
        #print probs
        #print probs.argmax()
        pred = probs.argmax() + class_idx*360
        preds[k].append(pred)
    print preds
