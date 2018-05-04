#!/usr/bin/env python
import tensorflow as tf
from scipy.misc import imsave
import res_batch_utils
import numpy as np
import vgg16

import cut_backgrounds
from scipy.misc import imsave
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

def get_stats(sess, batch, writer, fig, testing=False):
    prefix = 'Training'
    if testing:
        prefix = 'Testing'
    summary_str,loss_r,a_c,e_c,t_c = sess.run([
                merged_summary_op,
                loss,
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
    plt.clf()
    plt.bar(range(-180,180),a_c[0,:],1)
    plt.title(prefix+' Yaw: '+str(batch[1][0][0]*180/np.pi))
    plt.pause(0.00001)
    fig.savefig('tf_logs/'+BASE_DIR+'/'+prefix+'_azimuth.png')

    plt.clf()
    plt.bar(range(-90,90),e_c[0,:],1)
    plt.title(prefix+' Pitch: '+str(batch[2][0][0]*180/np.pi))
    plt.pause(0.00001)
    fig.savefig('tf_logs/'+BASE_DIR+'/'+prefix+'_elevation.png')
	
    plt.clf()
    plt.bar(range(-180,180),t_c[0,:],1)
    plt.title(prefix+' Roll: '+str(batch[3][0][0]*180/np.pi))
    plt.pause(0.00001)
    fig.savefig('tf_logs/'+BASE_DIR+'/'+prefix+'_tilt.png')


    from scipy.misc import imsave
    im = np.array(batch[0][0])
    im = im.reshape([224,224,3])
    imsave('./tf_logs/' +BASE_DIR+'/'+prefix+'_image.png',im)    
 
    print(prefix+": %d,  %g "%(i, loss_r))
    writer.add_summary(summary_str,i)

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
x = tf.placeholder(tf.float32, shape=[None, 224*224*3], name="images")
a_ = tf.placeholder(tf.float32, shape=[None, 1])
e_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])
sigma_ = tf.placeholder(tf.float32)
dist_a = tf.placeholder(tf.int32, shape=[None, 360])
dist_e = tf.placeholder(tf.int32, shape=[None, 180])
dist_t = tf.placeholder(tf.int32, shape=[None,360])

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1,224,224,3])
vgg = vgg16.vgg16( x_image, 'vgg16_weights.npz', sess )

#layers = [ 'conv5_1','conv5_2' ]
#ops = [ getattr( vgg, x ) for x in layers ]

print(vgg.conv5_3)
print(vgg.conv5_3.get_shape())
#vgg_acts = sess.run( ops, feed_dict={vgg.imgs: x_image} )


#print(ops[1].get_shape())
last = vgg.conv5_3
shape = last.get_shape().as_list()
f_flat = tf.reshape(last,[-1,shape[1]*shape[2]*shape[3]])
f1 = fc(f_flat,out_size=1000,name='F1')
print f1.get_shape()
f2 = fc(f1,out_size=500,name='F2')
f2_drop = f2 #tf.nn.dropout(f2, keep_prob)

a_conv = tf.nn.softmax(fc(f2_drop,out_size=360,is_output=True,name='az'))
e_conv = tf.nn.softmax(fc(f2_drop,out_size=180,is_output=True,name='el'))
t_conv = tf.nn.softmax(fc(f2_drop,out_size=360,is_output=True,name='ti'))


sess.run( tf.global_variables_initializer())
#vgg.load_weights( 'vgg16_weights.npz', sess )


with tf.name_scope('Cost'):
    loss_a = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_a, tf.float32)/sigma_) * tf.log(tf.clip_by_value(a_conv,1e-10,1.0)), axis=1))
    loss_e = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_e, tf.float32)/sigma_) * tf.log(tf.clip_by_value(e_conv,1e-10,1.0)), axis=1))
    loss_t = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_t, tf.float32)/sigma_) * tf.log(tf.clip_by_value(t_conv,1e-10,1.0)), axis=1)) 
    loss = loss_a+loss_e+loss_t 
train_vars = [v for v in tf.global_variables() if v.name == 'F2/W_fc:0' or v.name=='F1/B_fc:0' or v.name=='F2/W_fc:0' or v.name=='F2/B_fc:0' or v.name=='az/W_fc:0' or v.name=='az/B_fc:0' or v.name=='el/W_fc:0' or v.name=='el/B_fc:0' or v.name=='ti/W_fc:0' or v.name=='ti/B_fc:0']
print(train_vars)
with tf.name_scope('Optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss,var_list=train_vars)

loss_summary = tf.summary.scalar( 'loss', loss )

merged_summary_op = tf.summary.merge_all()

BASE_DIR = 'a_vgg'


train_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/train",graph=sess.graph)
test_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/test")

sess.run(tf.global_variables_initializer())

#all_vars = [v for v in tf.global_variables()]
#print(all_vars)


saver = tf.train.Saver()
#saver.restore(sess, 'tf_logs/q/shapenet.ckpt')

max_steps = 100000

fig = plt.figure(0)
print("step, azimuth, elevation, tilt, loss")

for i in range(max_steps):
    sigma_val = 1.0 #1.0/(1+i*0.001) 
    kp_in = 0.50
    batch = res_batch_utils.next_batch(50)
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
