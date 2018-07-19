#!/usr/bin/env python
import numpy as np 
import tensorflow as tf
import back_batch_utils
import cut_backgrounds
from scipy.misc import imsave
import matplotlib.pyplot as plt
import os, glob
from PIL import Image


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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
x = tf.placeholder(tf.float32, shape=[None,50*50*3])
a_ = tf.placeholder(tf.float32, shape=[None, 1])
e_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])
sigma_ = tf.placeholder(tf.float32)
dist_a = tf.placeholder(tf.int32, shape=[None, 360])
dist_e = tf.placeholder(tf.int32, shape=[None, 180])
dist_t = tf.placeholder(tf.int32, shape=[None,360])


keep_prob = tf.placeholder(tf.float32)

x_img = tf.reshape(x, [-1,50,50,3])

c1 = conv(x_img,num_filters=12,stride=2,name='C1')
print c1.get_shape()
c2 = conv(c1,num_filters=48,stride=2,name='C2')
print c2.get_shape()
c3 = conv(c2,num_filters=192,stride=2,name='C3')
print c3.get_shape()
c4 = conv(c3,num_filters=768,stride=2,name='C4')
print c4.get_shape()
c5 = conv(c4,num_filters=768,stride=1,name='C5')
print c5.get_shape()

last = c5

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
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.name_scope('Accuracy'):
    a_acc = tf.reduce_mean(tf.abs(a_-tf.reduce_max(a_conv,axis=1)))
    e_acc = tf.reduce_mean(tf.abs(e_-tf.reduce_max(e_conv,axis=1))) 
    t_acc = tf.reduce_mean(tf.abs(t_-tf.reduce_max(t_conv,axis=1)))

acc_summary = tf.summary.scalar( 'azimuth accuracy', a_acc )
acc_summary = tf.summary.scalar( 'elevation accuracy', e_acc )
acc_summary = tf.summary.scalar( 'tilt accuracy', t_acc )
loss_summary = tf.summary.scalar( 'loss', loss )

merged_summary_op = tf.summary.merge_all()


sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, 'tf_logs/j_back/shapenet.ckpt')


test_dir = './mc_timber/'
out_dir = './testing_output_j-back_t/'

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
    kp_in = 1.0

    img = Image.open(test_dir + str(i)+'.png')
    #img = img.resize((50,50))
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

    background = background.resize((50,50))
    background.save(out_dir+str(i)+'.png')
    img = np.array(background).flatten() / 255.0
    
    a_c,e_c,t_c = sess.run([
		a_conv,e_conv,t_conv],
                feed_dict={
                x: [img.tolist()],
                keep_prob: kp_in})
    
    y_max = np.max(a_c[0,:])+0.01*np.max(a_c[0,:])
    plt.clf()
    plt.bar(range(-180,180), a_c[0,:], 1, color='black')
    plt.bar(yaw_d, y_max, 0.2, edgecolor='red')
    plt.title('Yaw')
    plt.xticks(np.arange(-180,181,60))
    plt.xlim([-181,181])
    plt.ylim([0,y_max])
    plt.xlabel('Yaw Angle (deg)')
    plt.pause(0.00001)
    fig.savefig(out_dir+str(i)+'_azimuth.eps',format='eps')

    p_max = np.max(e_c[0,:])+0.01*np.max(e_c[0,:])
    plt.clf()
    plt.bar(range(-90,90),e_c[0,:],1,color='black')
    plt.bar(pitch_d, p_max, 0.2, edgecolor='red')
    plt.title('Pitch')
    plt.xticks(np.arange(-90,91,60))
    plt.xlim([-91,91])
    plt.ylim([0,p_max])
    plt.xlabel('Pitch Angle (deg)')
    plt.pause(0.00001)
    fig.savefig(out_dir+str(i)+'_elevation.eps',format='eps')

    r_max = np.max(t_c[0,:])+0.01*np.max(t_c[0,:])
    plt.clf()
    plt.bar(range(-180,180),t_c[0,:],1,color='black')
    plt.bar(roll_d, r_max, 0.2, edgecolor='red')
    plt.title('Roll')
    plt.xticks(np.arange(-180,181,60))
    plt.xlim([-181,181])
    plt.ylim([0,r_max])
    plt.xlabel('Roll Angle (deg)')
    plt.pause(0.00001)
    fig.savefig(out_dir+str(i)+'_tilt.eps',format='eps')
 
    print(": %d "%(i))
    #writer.add_summary(summary_str,i)

#saver.save(sess, "tf_logs/"+BASE_DIR+"/shapenet.ckpt")
#train_writer.close()
#test_writer.close()

