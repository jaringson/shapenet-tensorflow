#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import batch_utils

sess = tf.InteractiveSession()



# placeholders
x = tf.placeholder(tf.float32, shape=[None,100*100])
a_ = tf.placeholder(tf.float32, shape=[None, 1])
e_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])
sigma_ = tf.placeholder(tf.float32)

keep_prob = tf.placeholder(tf.float32)

# initialization functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# layer functions
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,100,100,1])

with tf.variable_scope('classifyer_network') as class_scope:
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([100, 100, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([25*25*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope("azimuth"):
        W_fc2 = tf.get_variable( "W", [1024, 360], tf.float32,
                                  tf.random_normal_initializer( stddev=np.sqrt(2 / np.prod(h_fc1_drop.get_shape().as_list()[1:])) ) )
        b_fc2 = bias_variable([360])
        a_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.variable_scope("elevation"):
        W_fc2 = tf.get_variable( "W", [1024, 360], tf.float32,
                                  tf.random_normal_initializer( stddev=np.sqrt(2 / np.prod(h_fc1_drop.get_shape().as_list()[1:])) ) )
        b_fc2 = bias_variable([360])
        e_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.variable_scope("tilt"):
        W_fc2 = tf.get_variable( "W", [1024, 360], tf.float32,
                                  tf.random_normal_initializer( stddev=np.sqrt(2 / np.prod(h_fc1_drop.get_shape().as_list()[1:])) ) )
        b_fc2 = bias_variable([360])
        t_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


with tf.name_scope('Cost'):
    # a_ = in
    # a_conv = guess
    delta_angle = tf.acos(tf.sin(e_)*tf.sin(e_conv)+tf.cos(e_)*tf.cos(e_conv)*tf.cos(tf.abs(a_-a_conv)))
    d_loss = delta_angle + tf.sqrt(tf.square(t_)-tf.square(t_conv))
    #cross_entropies = tf.reduce_mean(-tf.reduce_sum(tf.exp(-d/sigma_)  * tf.log(tf.clip_by_value(a_conv,1e-10,1.0)), axis=[1])) + \
    #                    tf.reduce_mean(-tf.reduce_sum(tf.exp(-d/sigma_)  * tf.log(tf.clip_by_value(e_conv,1e-10,1.0)), axis=[1])) + \
    #                    tf.reduce_mean(-tf.reduce_sum(tf.exp(-d/sigma_)  * tf.log(tf.clip_by_value(t_conv,1e-10,1.0)), axis=[1]))


with tf.name_scope('Optimizer'):
    train_step = tf.train.AdamOptimizer(1e-6).minimize(d_loss)


with tf.name_scope('Accuracy'):
    a_correct_prediction = tf.equal(tf.argmax(a_conv,1), tf.argmax(a_,1))
    a_acc = tf.reduce_mean(tf.cast(a_correct_prediction, tf.float32))
    e_correct_prediction = tf.equal(tf.argmax(e_conv,1), tf.argmax(e_,1))
    e_acc = tf.reduce_mean(tf.cast(e_correct_prediction, tf.float32))
    t_correct_prediction = tf.equal(tf.argmax(t_conv,1), tf.argmax(t_,1))
    t_acc = tf.reduce_mean(tf.cast(t_correct_prediction, tf.float32))

acc_summary = tf.summary.scalar( 'azimuth accuracy', a_acc )
acc_summary = tf.summary.scalar( 'elevation accuracy', e_acc )
acc_summary = tf.summary.scalar( 'tilt accuracy', t_acc )
cost_summary = tf.summary.scalar( 'cost', cross_entropies )

merged_summary_op = tf.summary.merge_all()

BASE_DIR = 'a'

train_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR,graph=sess.graph)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
#saver.restore(sess, "tf_logs/x/classification_mode.ckpt")
#print("Classification mode model restored")

class_steps = 400
max_steps = 1 # 1600
m = 1./(class_steps - max_steps)
b = 1.0*max_steps/(max_steps - class_steps)
print("step, azimuth, elevation, tilt")
for i in range(max_steps):
    # batch = batch_utils.next_batch(150, min(1, i*slope + class_steps), i > class_steps)
    #batch = background.next_batch(50, min(1, i*m+b))
    batch = batch_utils.next_batch(50)

    if i%10 == 0:
        summary_str,ac,ec,tc = sess.run([
                merged_summary_op,
                a_acc,e_acc,t_acc],
                feed_dict={
                x: batch[0],
                a_: batch[1],
                e_: batch[2],
                t_: batch[3],
                sigma_: 1/max_steps,
                keep_prob: 0.5})

        print("Train: %d, %g, %g, %g "%(i, ac, ec, tc))
        train_writer.add_summary(summary_str,i)
	saver.save(sess, "tf_logs/"+BASE_DIR+"/classification_mode.ckpt")

    train_step.run(feed_dict={
                x: batch[0],
                a_: batch[1],
                e_: batch[2],
                t_: batch[3],
                sigma_: 1/max_steps,
                keep_prob: 0.5})

saver.save(sess, "tf_logs/"+BASE_DIR+"/classification_mode.ckpt")
train_writer.close()
