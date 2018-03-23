from __future__ import print_function
from __future__ import division

from pdb import set_trace as debugger
#from imageloader import ImageLoader
from tensorflow.python.training import moving_averages

import tensorflow as tf
import numpy as np

import os
import sys

from scipy.misc import imsave
import matplotlib.pyplot as plt
import cut_backgrounds

import res_batch_utils

batch_size = 50
xsize, ysize = 150, 150
resnet_units = 3

tf.reset_default_graph()
sess = tf.InteractiveSession()

def global_avg_pool(in_var, name='global_pool'):
    assert name is not None, 'Op name should be specified'
    # start global average pooling
    with tf.name_scope(name):
        input_shape = in_var.get_shape()
        assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

        inference = tf.reduce_mean(in_var, [1, 2])
        return inference


def max_pool(in_var, kernel_size=[1,2,2,1], strides=[1,1,1,1], 
            padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    assert strides is not None, 'Strides should be specified when performing max pooling'
    # start max pooling
    with tf.name_scope(name):
        input_shape = in_var.get_shape()
        assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

        inference = tf.nn.max_pool(in_var, kernel_size, strides, padding)
        return inference


def avg_pool_2d(in_var, kernel_size=[1,2,2,1], strides=None, 
                padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    assert strides is not None, 'Strides should be specified when performing average pooling'
    # start average pooling
    with tf.name_scope(name):
        input_shape = in_var.get_shape()
        assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

        inference = tf.nn.avg_pool(in_var, kernel_size, strides, padding)
        return inference

def conv_2d(in_var, out_channels, filters=[3,3], strides=[1,1,1,1], 
            padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    # start conv_2d
    with tf.name_scope(name):
        k_w, k_h = filters  # filter width/height
        W = tf.get_variable(name + "_W", [k_h, k_w, in_var.get_shape()[-1], out_channels],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name + "_b", [out_channels], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(in_var, W, strides=strides, padding=padding)
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv

def residual_block(in_var, nb_blocks, out_channels, batch_norm=True, strides=[1,1,1,1],
                    downsample=False, downsample_strides=[1,2,2,1], name=None):
    assert name is not None, 'Op name should be specified'
    # start residual block
    with tf.name_scope(name):
        resnet = in_var
        in_channels = in_var.get_shape()[-1].value

        # multiple layers for a single residual block
        for i in xrange(nb_blocks):
            identity = resnet
            ##################
            # apply convolution
            resnet = conv_2d(resnet, out_channels, 
                            strides=strides if not downsample else downsample_strides, 
                            name='{}_conv2d_{}'.format(name, i))
            # normalize batch before activations
            if batch_norm:
                resnet = batch_normalization(resnet, name='{}_batch_norm{}'.format(name, i))
            # apply activation function
            resnet = tf.nn.relu(resnet)
            # apply convolution again
            resnet = conv_2d(resnet, out_channels, strides=strides, name='{}_conv2d_{}{}'.format(name, i, 05))
            # normalize batch before activations or previous convolution
            if batch_norm:
                resnet = batch_normalization(resnet, name='{}_batch_norm{}'.format(name, i), reuse=True)
            # apply activation function
            resnet = tf.nn.relu(resnet)
            ##################
            # downsample
            if downsample:
                identity = avg_pool_2d(identity, strides=downsample_strides, name=name+'_avg_pool_2d')
            # projection to new dimension by padding
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity, [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels
            # add residual
            resnet = resnet + identity

        return resnet

def batch_normalization(in_var, beta=0.0, gamma=1.0, epsilon=1e-5, 
                        decay=0.9, name=None, reuse=None):
    assert name is not None, 'Op name should be specified'
    # start batch normalization with moving averages
    input_shape = in_var.get_shape().as_list()
    input_ndim = len(input_shape)

    with tf.variable_scope(name, reuse=reuse):
        gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002)
        beta = tf.get_variable(name+'_beta', shape=[input_shape[-1]],
                               initializer=tf.constant_initializer(beta),
                               trainable=True)
        gamma = tf.get_variable(name+'_gamma', shape=[input_shape[-1]],
                                initializer=gamma_init, trainable=True)

        axis = list(range(input_ndim - 1))
        moving_mean = tf.get_variable(name+'_moving_mean',
                        input_shape[-1:], initializer=tf.zeros_initializer,
                        trainable=False)
        moving_variance = tf.get_variable(name+'_moving_variance',
                            input_shape[-1:], initializer=tf.ones_initializer,
                            trainable=False)

        # define a function to update mean and variance
        def update_mean_var():
            mean, variance = tf.nn.moments(in_var, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)

        # only update mean and variance with moving average while training
        mean, var = tf.cond(is_training, update_mean_var, lambda: (moving_mean, moving_variance))

        inference = tf.nn.batch_normalization(in_var, mean, var, beta, gamma, epsilon)
        inference.set_shape(input_shape)

    return inference


def residual_network(_x):
    with tf.name_scope('residual_network') as scope:
        #_x = tf.reshape(x, [batch_size, xsize, ysize, 1])
        net = conv_2d(_x, 8, filters=[7,7], strides=[1,2,2,1], name='conv_0')
        net = max_pool(net, name='max_pool_0')
        net = residual_block(net, resnet_units, 8, name='resblock_1')
        net = residual_block(net, 1, 16, downsample=True, name='resblock_1-5')
        net = residual_block(net, resnet_units, 16, name='resblock_2')
        net = residual_block(net, 1, 24, downsample=True, name='resblock_2-5')
        net = residual_block(net, resnet_units+1, 24, name='resblock_3')
        net = residual_block(net, 1, 32, downsample=True, name='resblock_3-5')
        net = residual_block(net, resnet_units, 32, name='resblock_4')
        net = batch_normalization(net, name='batch_norm')
        net = tf.nn.relu(net)
        net = global_avg_pool(net)
        return net

def get_stats(sess, batch, writer, fig, testing=False):
    prefix = 'Training'
    if testing:
        prefix = 'Testing'
        a_c,e_c,t_c = sess.run([
                a_conv,e_conv,t_conv],
                feed_dict={
                x: batch[0],
		is_training:True})
    else:
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
		is_training:True})
        print(prefix+": %d, %g, %g, %g, %g "%(i, ac, ec, tc, loss_r))
        writer.add_summary(summary_str,i)

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
    im = im.reshape([150,150,3])
    imsave('./tf_logs/' +BASE_DIR+'/'+prefix+'_image.png',im)


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

# Create model
x = tf.placeholder(tf.float32, shape=[None,150*150*3])
a_ = tf.placeholder(tf.float32, shape=[None, 1])
e_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])
sigma_ = tf.placeholder(tf.float32)
dist_a = tf.placeholder(tf.int32, shape=[None, 360])
dist_e = tf.placeholder(tf.int32, shape=[None, 180])
dist_t = tf.placeholder(tf.int32, shape=[None,360])


x_r = tf.reshape(x, [batch_size, xsize, ysize, 3])
is_training = tf.placeholder(tf.bool)

with tf.variable_scope("residual") as scope:
    last = residual_network(x_r)

shape = last.get_shape().as_list()
print(shape)
#f_flat = tf.reshape(last,[-1,shape[1]*shape[2]*shape[3]])
#f1 = fc(f_flat,out_size=1000,name='F1')
#print(f1.get_shape())

a_conv = tf.nn.softmax(fc(last,out_size=360,is_output=True,name='az'))
e_conv = tf.nn.softmax(fc(last,out_size=180,is_output=True,name='el'))
t_conv = tf.nn.softmax(fc(last,out_size=360,is_output=True,name='ti'))

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

BASE_DIR = 'g'

train_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/train",graph=sess.graph)
test_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/test")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, 'tf_logs/e/shapenet.ckpt')

max_steps = 100000

fig = plt.figure(0)
print("step, azimuth, elevation, tilt, loss")

for i in range(max_steps):
    sigma_val = 1.0 #1.0/(1+i*0.001) 
    kp_in = 0.50
    batch = res_batch_utils.next_batch(50)
    '''print(sess.run([
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
                keep_prob: kp_in}))
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
                is_training:True })

saver.save(sess, "tf_logs/"+BASE_DIR+"/shapenet.ckpt")
train_writer.close()
test_writer.close()

