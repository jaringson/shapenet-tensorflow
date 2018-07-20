#!/usr/bin/env python
import numpy as np 
import tensorflow as tf
import res_batch_utils
import cut_backgrounds
from scipy.misc import imsave
from scipy.misc import imsave
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import average, add, Input, Lambda, Dense, GlobalAveragePooling2D, concatenate
from keras import backend as K
import os
import time



SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 16
EXBIG_SIZE = 24

plt.rc('font', size=EXBIG_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=EXBIG_SIZE)  # fontsize of the figure titl

sess = tf.Session()
K.set_session(sess)

def get_stats(sess, batch, writer, fig,i, testing=False):
    prefix = 'Training'
    if testing:
        prefix = 'Testing'
    summary_str,ac,ec,tc,loss_r,a_c,e_c,t_c = sess.run([
                merged_summary_op,
                a_acc,e_acc,t_acc,loss,
		a_conv,e_conv,t_conv],
                feed_dict={
                x: batch[0],
                a_: batch[1],
                e_: batch[2],
                t_: batch[3],
		dist_a: batch[4],
		dist_e: batch[5],
		dist_t: batch[6],
                sigma_: sigma_val,
                keep_prob: kp_in})
    y_max = np.max(a_c[0,:])+0.01*np.max(a_c[0,:])
    plt.clf()
    plt.bar(range(-180,180), a_c[0,:], 1, color='black')
    plt.bar(batch[1][0][0]*180/np.pi, y_max, 0.2, edgecolor='red')
    plt.title('Yaw')
    plt.xticks(np.arange(-180,181,60))
    plt.xlim([-181,181])
    plt.ylim([0,y_max])
    plt.xlabel('Yaw Angle (deg)')
    plt.pause(0.00001)
    fig.savefig('tf_logs/'+BASE_DIR+'/'+str(i)+prefix+'_azimuth.eps',format='eps')

    p_max = np.max(e_c[0,:])+0.01*np.max(e_c[0,:])
    plt.clf()
    plt.bar(range(-90,90),e_c[0,:],1,color='black')
    plt.bar(batch[2][0][0]*180/np.pi, p_max, 0.2, edgecolor='red')
    plt.title('Pitch')
    plt.xticks(np.arange(-90,91,60))
    plt.xlim([-91,91])
    plt.ylim([0,p_max])
    plt.xlabel('Pitch Angle (deg)')
    plt.pause(0.00001)
    fig.savefig('tf_logs/'+BASE_DIR+'/'+str(i)+prefix+'_elevation.eps',format='eps')
	
    r_max = np.max(t_c[0,:])+0.01*np.max(t_c[0,:])
    plt.clf()
    plt.bar(range(-180,180),t_c[0,:],1,color='black')
    plt.bar(batch[3][0][0]*180/np.pi, r_max, 0.2, edgecolor='red')
    plt.title('Roll')
    plt.xticks(np.arange(-180,181,60))
    plt.xlim([-181,181]) 
    plt.ylim([0,r_max])
    plt.xlabel('Roll Angle (deg)')
    plt.pause(0.00001)
    fig.savefig('tf_logs/'+BASE_DIR+'/'+str(i)+prefix+'_tilt.eps',format='eps')


    im = np.array(batch[0][0])
    im = im.reshape([50,50,3])
    imsave('./tf_logs/' +BASE_DIR+'/'+str(i)+prefix+'_image.png',im)    
 
    print(prefix+": %d, %g, %g, %g, %g "%(i, ac, ec, tc, loss_r))
    writer.add_summary(summary_str,i)

# placeholders
x = tf.placeholder(tf.float32, shape=[None,50*50*3])
a_ = tf.placeholder(tf.float32, shape=[None, 1])
e_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])
sigma_ = tf.placeholder(tf.float32)
dist_a = tf.placeholder(tf.int32, shape=[None, 360])
dist_e = tf.placeholder(tf.int32, shape=[None, 180])
dist_t = tf.placeholder(tf.int32, shape=[None,360])

keep_prob = tf.placeholder(tf.float32)

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)(x)

#print(base_model.summary())

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(base_model)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
a_conv = Dense(360, activation='softmax',name='a_conv')(x)
e_conv = Dense(180, activation='softmax',name='e_conv')(x)
t_conv = Dense(360, activation='softmax',name='t_conv')(x)

loss_a = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_a, tf.float32)/sigma_) * tf.log(tf.clip_by_value(a_conv,1e-10,1.0)), axis=1))
loss_e = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_e, tf.float32)/sigma_) * tf.log(tf.clip_by_value(e_conv,1e-10,1.0)), axis=1))
loss_t = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_t, tf.float32)/sigma_) * tf.log(tf.clip_by_value(t_conv,1e-10,1.0)), axis=1))

loss = loss_a+lost_e+loss_t


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


with tf.name_scope('Optimizer'):
    train_step = tf.train.AdamOptimizer().minimize(loss)

with tf.name_scope('Accuracy'):
    a_acc = tf.reduce_mean(tf.abs(a_-tf.reduce_max(a_conv,axis=1)))
    e_acc = tf.reduce_mean(tf.abs(e_-tf.reduce_max(e_conv,axis=1))) 
    t_acc = tf.reduce_mean(tf.abs(t_-tf.reduce_max(t_conv,axis=1)))

acc_summary = tf.summary.scalar( 'azimuth accuracy', a_acc )
acc_summary = tf.summary.scalar( 'elevation accuracy', e_acc )
acc_summary = tf.summary.scalar( 'tilt accuracy', t_acc )
loss_summary = tf.summary.scalar( 'loss', loss )

merged_summary_op = tf.summary.merge_all()

BASE_DIR = 'a_keras'


train_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/train",graph=sess.graph)
test_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/test")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
# saver.restore(sess, 'tf_logs/j_back/shapenet.ckpt')

max_steps = 100000

fig = plt.figure(0)
print("step, azimuth, elevation, tilt, loss")

for i in range(max_steps):
    sigma_val = 1.0# 1.0/(1+i*0.001) 
    kp_in = 0.50
    batch = res_batch_utils.next_batch(50)
    '''print sess.run([
                a_conv, loss_a ,loss
                ],
                feed_dict={
                x: batch[0],
                a_: batch[1],
                e_: batch[2],
                t_: batch[3],
		dist_a: batch[4],
		dist_e: batch[5],
		dist_t: batch[6],
                sigma_: sigma_val,
                keep_prob: kp_in})
    '''
    if i%100 == 0:
        get_stats(sess, batch, train_writer, fig)
        saver.save(sess, "tf_logs/"+BASE_DIR+"/shapenet.ckpt")
    
    if i%500 == 0:
        test_batch = res_batch_utils.next_batch(50, testing=True)
        get_stats(sess, test_batch, test_writer, fig, testing=True)
        cut_backgrounds.cut(10)   
 
    train_step.run(feed_dict={
                x: batch[0],
                a_: batch[1],
                e_: batch[2],
                t_: batch[3],
                dist_a: batch[4],
                dist_e: batch[5],
                dist_t: batch[6],
                sigma_: sigma_val,
                keep_prob: kp_in})

saver.save(sess, "tf_logs/"+BASE_DIR+"/shapenet.ckpt")
train_writer.close()
test_writer.close()




