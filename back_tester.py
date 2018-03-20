#!/usr/bin/env python
import numpy as np 
import tensorflow as tf
import back_batch_utils
import cut_backgrounds
from scipy.misc import imsave
import matplotlib.pyplot as plt
import os, glob
from PIL import Image


tf.reset_default_graph()
sess = tf.InteractiveSession()

BATCH_SIZE = 100
LAMBDA = 10 # Gradient penalty lambda hyperparameter

g_scope = 0
#d_scope = 0
noise = 0

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def conv_t( x, filter_size=3, stride=1, num_filters=64, is_output=False,out_size=None, name="conv_t"):
    with tf.variable_scope(name) as scope:
    
   
        x_shape = x.get_shape().as_list()
        W = tf.get_variable("W_conv_t", [filter_size, filter_size, num_filters,x_shape[3]], initializer = tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("B_conv_t", [num_filters], initializer = tf.contrib.layers.variance_scaling_initializer())
        
        if out_size ==None:
            outsize = x_shape
            outsize[0] = BATCH_SIZE
            outsize[1] *= 2
            outsize[2] *= 2
            outsize[3] = W.get_shape().as_list()[2]

        h = []
        

        if not is_output:
            h = tf.nn.relu(tf.nn.conv2d_transpose(x, W, output_shape=outsize, strides=[1, stride, stride, 1], padding="SAME") + b)        
        
        else:
            h = tf.nn.conv2d_transpose(x, W, output_shape=outsize, strides=[1,stride,stride,1], padding='SAME') + b
    

        return h #result

def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, name="conv"):
    with tf.variable_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        W = tf.get_variable("W_conv", [filter_size, filter_size, x_shape[3], num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("B_conv", [num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
        h = []
        result = []
        

        if not is_output:
            h = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b)
            #result = h
        else:
            h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b


        return h #result

def fc( x, out_size=50, is_output=False, name="fc" ):
    with tf.variable_scope(name) as scope:
        shape = x.get_shape().as_list()
        W = tf.get_variable("W_fc", [shape[1], out_size], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("B_fc", [out_size], initializer=tf.contrib.layers.variance_scaling_initializer())
        
        h = []
        if not is_output:
            h = tf.nn.relu(tf.matmul(x,W)+b)
        else:
            h = tf.matmul(x,W)+b
        return h



# placeholders
x = tf.placeholder(tf.float32, shape=[None,150*150*3])
a_ = tf.placeholder(tf.float32, shape=[None, 1])
e_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])
sigma_ = tf.placeholder(tf.float32)
dist_a = tf.placeholder(tf.int32, shape=[None, 360])
dist_e = tf.placeholder(tf.int32, shape=[None, 180])
dist_t = tf.placeholder(tf.int32, shape=[None,360])


keep_prob = tf.placeholder(tf.float32)

x_img = tf.reshape(x, [-1,150,150,3])

c1 = conv(x_img,num_filters=12,stride=2,name='C1')
print c1.get_shape()
c2 = conv(c1,num_filters=48,stride=2,name='C2')
print c2.get_shape()
c3 = conv(c2,num_filters=192,stride=2,name='C3')
print c3.get_shape()
c4 = conv(c3,num_filters=768,stride=2,name='C4')
print c4.get_shape()

last = c4

shape = last.get_shape().as_list()
f_flat = tf.reshape(last,[-1,shape[1]*shape[2]*shape[3]])
f1 = fc(f_flat,out_size=1000,name='F1')
print f1.get_shape()
f2 = fc(f1,out_size=500,name='F2')
f2_drop = tf.nn.dropout(f2, keep_prob)

a_conv = tf.nn.softmax(fc(f2_drop,out_size=360,is_output=True,name='az'))
e_conv = tf.nn.softmax(fc(f2_drop,out_size=180,is_output=True,name='el'))
t_conv = tf.nn.softmax(fc(f2_drop,out_size=360,is_output=True,name='ti'))

with tf.name_scope('Cost'):
    loss_a = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_a, tf.float32)/sigma_) * tf.log(tf.clip_by_value(a_conv,1e-10,1.0)), axis=1))
    loss_e = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_e, tf.float32)/sigma_) * tf.log(tf.clip_by_value(e_conv,1e-10,1.0)), axis=1))
    loss_t = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_t, tf.float32)/sigma_) * tf.log(tf.clip_by_value(t_conv,1e-10,1.0)), axis=1)) 
    loss = loss_a+loss_e+loss_t 
with tf.name_scope('Optimizer'):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.name_scope('Accuracy'):
    a_acc = tf.reduce_mean(tf.abs(a_-a_conv))
    e_acc = tf.reduce_mean(tf.abs(e_-e_conv)) 
    t_acc = tf.reduce_mean(tf.abs(t_-t_conv))

acc_summary = tf.summary.scalar( 'azimuth accuracy', a_acc )
acc_summary = tf.summary.scalar( 'elevation accuracy', e_acc )
acc_summary = tf.summary.scalar( 'tilt accuracy', t_acc )
loss_summary = tf.summary.scalar( 'loss', loss )

merged_summary_op = tf.summary.merge_all()



#train_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/train",graph=sess.graph)
#test_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/test")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, 'tf_logs/q/shapenet.ckpt')


test_dir = './motion_capture/'
out_dir = './testing_output_q/'

all_pics = glob.glob(test_dir+'*.png')

backgrounds = './rock_canyon_cut'
all_backgrounds = glob.glob(backgrounds + '/*')

f = open(test_dir+'/views.txt')
all_views = f.readlines()


fig = plt.figure(0)
print("step, azimuth, elevation, tilt, loss")


#print all_pics

for i in range(len(all_pics)):
    sigma_val = 1.0 #1.0/(1+i*0.001) 
    kp_in = 0.50

    img = Image.open(test_dir + str(i)+'.png')
    angles = all_views[i].split(' ')




    yaw_d = int(angles[2])
    pitch_d = int(angles[1])
    roll_d = int(angles[0])
    if yaw_d > 180:
	yaw_d = yaw_d - 360
    
    yaw = np.radians(yaw_d)
    pitch = np.radians(pitch_d)
    roll = np.radians(roll_d)

    yaw_dist = []
    pitch_dist = []
    roll_dist = []

    if yaw_d < 0:
        yaw_dist = np.concatenate(( np.arange(180+yaw_d,0,-1), np.arange(0,180), np.arange(180, 180+yaw_d, -1) ))
    else:
        yaw_dist = np.concatenate(( np.arange(180-yaw_d,180), np.arange(180,0,-1), np.arange(0,180-yaw_d) ))

    if pitch_d  < 0:
        pitch_dist = np.concatenate(( np.arange(90+pitch_d,0,-1), np.arange(0,90), np.arange(90, 90+pitch_d, -1) ))
    else:
        pitch_dist = np.concatenate(( np.arange(90-pitch_d,90), np.arange(90,0,-1), np.arange(0,90-pitch_d) ))

    if roll_d < 0:
        roll_dist = np.concatenate(( np.arange(180+roll_d,0,-1), np.arange(0,180), np.arange(180, 180+roll_d, -1) ))     
    else:
        roll_dist = np.concatenate(( np.arange(180-roll_d,180), np.arange(180,0,-1), np.arange(0,180-roll_d) ))

    rand_back = all_backgrounds[np.random.randint(len(all_backgrounds))]
    background = Image.open(rand_back)
    background.paste(img, (background.size[0]/2-img.size[0]/2 + np.random.randint(-20,20),
                                       background.size[1]/2-img.size[1]/2 + np.random.randint(-20,20)),img)

    background.save(out_dir+str(i)+'.png')
    img = np.array(background).flatten() / 255.0
    
    summary_str,ac,ec,tc,loss_r,a_c,e_c,t_c = sess.run([
                merged_summary_op,
                a_acc,e_acc,t_acc,loss,
		a_conv,e_conv,t_conv],
                feed_dict={
                x: [img.tolist()],
                a_: [[yaw]],
                e_: [[pitch]],
                t_: [[roll]],
		dist_a: [yaw_dist],
		dist_e: [pitch_dist],
		dist_t: [roll_dist],
                sigma_: sigma_val,
                keep_prob: kp_in})
    plt.clf()
    plt.bar(range(-180,180),a_c[0,:],1)
    plt.title('Yaw: '+str(yaw_d))
    plt.pause(0.00001)
    fig.savefig(out_dir+str(i)+'_azimuth.png')

    plt.clf()
    plt.bar(range(-90,90),e_c[0,:],1)
    plt.title('Pitch: '+str(pitch_d))
    plt.pause(0.00001)
    fig.savefig(out_dir+str(i)+'_elevation.png')
	
    plt.clf()
    plt.bar(range(-180,180),t_c[0,:],1)
    plt.title('Roll: '+str(roll_d))
    plt.pause(0.00001)
    fig.savefig(out_dir+str(i)+'_tilt.png')
 
    print(": %d, %g, %g, %g, %g "%(i, ac, ec, tc, loss_r))
    #writer.add_summary(summary_str,i)

#saver.save(sess, "tf_logs/"+BASE_DIR+"/shapenet.ckpt")
#train_writer.close()
#test_writer.close()
