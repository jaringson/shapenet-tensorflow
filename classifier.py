#!/usr/bin/env python
import numpy as np 
import tensorflow as tf
import batch_utils
from scipy.misc import imsave

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
x = tf.placeholder(tf.float32, shape=[None,100*100])
a_ = tf.placeholder(tf.float32, shape=[None, 1])
e_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])
sigma_ = tf.placeholder(tf.float32)

keep_prob = tf.placeholder(tf.float32)

x_img = tf.reshape(x, [-1,100,100,1])

c1 = conv(x_img,num_filters=200,stride=2,name='C1')
print c1.get_shape()
c2 = conv(c1,num_filters=800,stride=2,name='C2')
print c2.get_shape()
c3 = conv(c2,num_filters=800,stride=2,name='C3')
print c3.get_shape()

last = c3

shape = last.get_shape().as_list()
f_flat = tf.reshape(last,[-1,shape[1]*shape[2]*shape[3]])
f1 = fc(f_flat,out_size=1000,name='F1')
print f1.get_shape()
f2 = fc(f1,out_size=100,name='F2')

a_conv =  tf.sigmoid(fc(f2,out_size=1,is_output=True,name='az')) * 360
e_conv =     tf.tanh(fc(f2,out_size=1,is_output=True,name='el')) * 90.0
t_conv =  tf.sigmoid(fc(f2,out_size=1,is_output=True,name='tr')) * 360

with tf.name_scope('Cost'):
    a = a_ * np.pi/180.0 
    a_conv *= np.pi/180.0 
    e = e_ * np.pi/180.0 
    e_conv *= np.pi/180.0 
    t = t_ * np.pi/180.0 
    t_conv *= np.pi/180.0 
    
    temp = tf.sin(e)*tf.sin(e_conv) + tf.cos(e) * tf.cos(e_conv) * tf.cos(tf.abs(a-a_conv))  
    delta_angle = tf.acos(tf.clip_by_value(tf.sin(e)*tf.sin(e_conv) + tf.cos(e) * tf.cos(e_conv) * tf.cos(tf.abs(a-a_conv)), -0.9999, 0.9999))
    print delta_angle.get_shape()
    inner = delta_angle + tf.abs(t - t_conv)
    print inner.get_shape()
    d_loss = tf.reduce_mean(inner)
    print d_loss.get_shape()

with tf.name_scope('Optimizer'):
    train_step = tf.train.AdamOptimizer(1e-2).minimize(d_loss)

with tf.name_scope('Accuracy'):
    a_acc = tf.reduce_mean(tf.abs(a-a_conv))
    e_acc = tf.reduce_mean(tf.abs(e-e_conv)) 
    t_acc = tf.reduce_mean(tf.abs(t-t_conv))

acc_summary = tf.summary.scalar( 'azimuth accuracy', a_acc )
acc_summary = tf.summary.scalar( 'elevation accuracy', e_acc )
acc_summary = tf.summary.scalar( 'tilt accuracy', t_acc )
loss_summary = tf.summary.scalar( 'loss', d_loss )

merged_summary_op = tf.summary.merge_all()

BASE_DIR = 'a'

train_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR,graph=sess.graph)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

max_steps = 5000

print("step, azimuth, elevation, tilt, loss")
for i in range(max_steps):
    batch = batch_utils.next_batch(50)
    '''print sess.run([
                temp ,d_loss
                ],
                feed_dict={
                x: batch[0],
                a_: batch[1],
                e_: batch[2],
                t_: batch[3],
                sigma_: 1/max_steps,
                keep_prob: 0.5})
    '''
    if i%10 == 0:
        summary_str,ac,ec,tc,loss_r = sess.run([
                merged_summary_op,
                a_acc,e_acc,t_acc,d_loss],
                feed_dict={
                x: batch[0],
                a_: batch[1],
                e_: batch[2],
                t_: batch[3],
                sigma_: 1/max_steps,
                keep_prob: 0.5})

        print("Train: %d, %g, %g, %g, %g "%(i, ac, ec, tc, loss_r))
        train_writer.add_summary(summary_str,i)
        saver.save(sess, "tf_logs/"+BASE_DIR+"/shapenet.ckpt")

    train_step.run(feed_dict={
                x: batch[0],
                a_: batch[1],
                e_: batch[2],
                t_: batch[3],
                sigma_: 1/max_steps,
                keep_prob: 0.5})

saver.save(sess, "tf_logs/"+BASE_DIR+"/shapenet.ckpt")
train_writer.close()


