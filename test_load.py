# Import the converted model's classi
from shapenet_tensor import RCNN_Fine_Tune as MyNet

import tensorflow as tf
import numpy as np
from PIL import Image




batch_size = 1

images= tf.placeholder(tf.float32, [batch_size,227,227,3]) 
net = MyNet({'data':images})

azimuth = net.layers['fc-azimuth']

with tf.Session() as sesh:

    img = Image.open('test.jpg')
    img = np.array(img)
    #print img
    
    # Load the data
    net.load('shapenet_tensor.npy', sesh)
    # Forward pass
    #output = sesh.run(net.get_output(), ...)
    
    feed = {images:[img]}
    az = sesh.run([azimuth], feed_dict=feed)
    print az[0]
    print len(az), len(az[0]), len(az[0][0]), az[0][0][0] 

