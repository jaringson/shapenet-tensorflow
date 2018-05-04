from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import cut_backgrounds
import res_batch_utils

from scipy.misc import imsave
import matplotlib.pyplot as plt



tf.reset_default_graph()
sess = tf.InteractiveSession()

def create_model_graph(model_info):
	""""Creates a graph from saved GraphDef file and returns a Graph object.
	Args:
	model_info: Dictionary containing information about the model architecture.
	Returns:
	Graph holding the trained Inception network, and various tensors we'll be
	manipulating.
	"""
	with tf.Graph().as_default() as graph:
		model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
	with gfile.FastGFile(model_path, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
			graph_def,
			name='',
			return_elements=[
				model_info['bottleneck_tensor_name'],
				model_info['resized_input_tensor_name'],
			]))
	return graph, bottleneck_tensor, resized_input_tensor

def create_model_graph(model_info):
	""""Creates a graph from saved GraphDef file and returns a Graph object.
	Args:
		model_info: Dictionary containing information about the model architecture.
	Returns:
		Graph holding the trained Inception network, and various tensors we'll be
		manipulating.
	"""
	with tf.Graph().as_default() as graph:
		model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
	with gfile.FastGFile(model_path, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
			graph_def,
			name='',
			return_elements=[
				model_info['bottleneck_tensor_name'],
				model_info['resized_input_tensor_name'],
			]))
	return graph, bottleneck_tensor, resized_input_tensor

def maybe_download_and_extract(data_url):
	"""Download and extract model tar file.
	If the pretrained model we're using doesn't already exist, this function
	downloads it from the TensorFlow.org website and unpacks it into a directory.
	Args:
	data_url: Web location of the tar file containing the pretrained model.
	"""
	dest_directory = FLAGS.model_dir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = data_url.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):

		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' %
	               (filename,
	                float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()

		filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
	                'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_model_info(architecture):
	"""Given the name of a model architecture, returns information about it.
	There are different base image recognition pretrained models that can be
	retrained using transfer learning, and this function translates from the name
	of a model to the attributes that are needed to download and train with it.
	Args:
	architecture: Name of a model architecture.
	Returns:
	Dictionary of information about the model, or None if the name isn't
	recognized
	Raises:
	ValueError: If architecture name is unknown.
	"""
	architecture = architecture.lower()
	if architecture == 'inception_v3':
		# pylint: disable=line-too-long
		data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
		# pylint: enable=line-too-long
		bottleneck_tensor_name = 'pool_3/_reshape:0'
		bottleneck_tensor_size = 2048
		input_width = 299
		input_height = 299
		input_depth = 3
		resized_input_tensor_name = 'Mul:0'
		model_file_name = 'classify_image_graph_def.pb'
		input_mean = 128
		input_std = 128
	else:
		tf.logging.error("Couldn't understand architecture name '%s'", architecture)
		raise ValueError('Unknown architecture', architecture)

	return {
	  'data_url': data_url,
	  'bottleneck_tensor_name': bottleneck_tensor_name,
	  'bottleneck_tensor_size': bottleneck_tensor_size,
	  'input_width': input_width,
	  'input_height': input_height,
	  'input_depth': input_depth,
	  'resized_input_tensor_name': resized_input_tensor_name,
	  'model_file_name': model_file_name,
	  'input_mean': input_mean,
	  'input_std': input_std,
	}

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

def main(_):
	# Needed to make sure the logging output is visible.
	# See https://github.com/tensorflow/tensorflow/issues/3047
	# tf.logging.set_verbosity(tf.logging.INFO)


	# Gather information about the model architecture we'll be using.
	model_info = create_model_info(FLAGS.architecture)
	if not model_info:
		tf.logging.error('Did not recognize architecture flag')
		return -1

	# Set up the pre-trained graph.
	maybe_download_and_extract(model_info['data_url'])
	graph, bottleneck_tensor, resized_input_tensor = (
	  create_model_graph(model_info))

	# placeholders
	a_ = tf.placeholder(tf.float32, shape=[None, 1])
	e_ = tf.placeholder(tf.float32, shape=[None, 1])
	t_ = tf.placeholder(tf.float32, shape=[None, 1])
	sigma_ = tf.placeholder(tf.float32)
	dist_a = tf.placeholder(tf.int32, shape=[None, 360])
	dist_e = tf.placeholder(tf.int32, shape=[None, 180])
	dist_t = tf.placeholder(tf.int32, shape=[None,360])
	keep_prob = tf.placeholder(tf.float32)

	bottleneck_input = tf.placeholder(tf.float32, shape=[None, 2048])

	#print(ops[1].get_shape())
	last = bottleneck_input
	# shape = last.get_shape().as_list()
	# f_flat = tf.reshape(last,[-1,shape[1]*shape[2]*shape[3]])
	f1 = fc(last,out_size=1000,name='F1')
	# print(f1.get_shape())
	f2 = fc(f1,out_size=500,name='F2')
	f2_drop = f2 #tf.nn.dropout(f2, keep_prob)

	a_conv = tf.nn.softmax(fc(f2_drop,out_size=360,is_output=True,name='az'))
	e_conv = tf.nn.softmax(fc(f2_drop,out_size=180,is_output=True,name='el'))
	t_conv = tf.nn.softmax(fc(f2_drop,out_size=360,is_output=True,name='ti'))


	sess.run( tf.global_variables_initializer())


	with tf.name_scope('Cost'):
	    loss_a = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_a, tf.float32)/sigma_) * tf.log(tf.clip_by_value(a_conv,1e-10,1.0)), axis=1))
	    loss_e = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_e, tf.float32)/sigma_) * tf.log(tf.clip_by_value(e_conv,1e-10,1.0)), axis=1))
	    loss_t = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_t, tf.float32)/sigma_) * tf.log(tf.clip_by_value(t_conv,1e-10,1.0)), axis=1)) 
	    loss = loss_a+loss_e+loss_t 
	train_vars = [v for v in tf.global_variables() if v.name == 'F2/W_fc:0' \
					or v.name=='F1/B_fc:0' or v.name=='F2/W_fc:0' \
					or v.name=='F2/B_fc:0' or v.name=='az/W_fc:0' \
					or v.name=='az/B_fc:0' or v.name=='el/W_fc:0' \
					or v.name=='el/B_fc:0' or v.name=='ti/W_fc:0' \
					or v.name=='ti/B_fc:0']
	# print(train_vars)
	with tf.name_scope('Optimizer'):
	    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss,var_list=train_vars)

	loss_summary = tf.summary.scalar( 'loss', loss )

	merged_summary_op = tf.summary.merge_all()

	BASE_DIR = 'a_inception'


	train_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/train",graph=sess.graph)
	test_writer = tf.summary.FileWriter("./tf_logs/"+BASE_DIR+"/test")

	sess.run(tf.global_variables_initializer())



	saver = tf.train.Saver()
	#saver.restore(sess, 'tf_logs/q/shapenet.ckpt')

	max_steps = 100000

	fig = plt.figure(0)
	print("step, azimuth, elevation, tilt, loss")

	for i in range(max_steps):
		sigma_val = 1.0 #1.0/(1+i*0.001) 
		kp_in = 0.50
		batch = res_batch_utils.next_batch(50)
		# if i%100 == 0:
		#     get_stats(sess, batch, train_writer, fig)
		#     saver.save(sess, "tf_logs/"+BASE_DIR+"/shapenet.ckpt")

		# if i%500 == 0:
		#     test_batch = res_batch_utils.next_batch(50, testing=True)
		#     get_stats(sess, test_batch, test_writer, fig, testing=True)
		#     cut_backgrounds.cut(10)   

		all_outputs = []
		for bat in batch[0]:
			bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: np.array(bat).reshape((1,299,299,3))})
			bottleneck_values = np.squeeze(bottleneck_values)
			all_outputs.append(bottleneck_values)

		train_step.run(feed_dict={
					bottleneck_input: all_outputs,
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--image_dir',
	  type=str,
	  default='',
	  help='Path to folders of labeled images.'
	)
	parser.add_argument(
	  '--output_graph',
	  type=str,
	  default='/tmp/output_graph.pb',
	  help='Where to save the trained graph.'
	)
	parser.add_argument(
	  '--intermediate_output_graphs_dir',
	  type=str,
	  default='/tmp/intermediate_graph/',
	  help='Where to save the intermediate graphs.'
	)
	parser.add_argument(
	  '--intermediate_store_frequency',
	  type=int,
	  default=0,
	  help="""\
	     How many steps to store intermediate graph. If "0" then will not
	     store.\
	  """
	)
	parser.add_argument(
	  '--output_labels',
	  type=str,
	  default='/tmp/output_labels.txt',
	  help='Where to save the trained graph\'s labels.'
	)
	parser.add_argument(
	  '--summaries_dir',
	  type=str,
	  default='/tmp/retrain_logs',
	  help='Where to save summary logs for TensorBoard.'
	)
	parser.add_argument(
	  '--how_many_training_steps',
	  type=int,
	  default=4000,
	  help='How many training steps to run before ending.'
	)
	parser.add_argument(
	  '--learning_rate',
	  type=float,
	  default=0.01,
	  help='How large a learning rate to use when training.'
	)
	parser.add_argument(
	  '--testing_percentage',
	  type=int,
	  default=10,
	  help='What percentage of images to use as a test set.'
	)
	parser.add_argument(
	  '--validation_percentage',
	  type=int,
	  default=10,
	  help='What percentage of images to use as a validation set.'
	)
	parser.add_argument(
	  '--eval_step_interval',
	  type=int,
	  default=10,
	  help='How often to evaluate the training results.'
	)
	parser.add_argument(
	  '--train_batch_size',
	  type=int,
	  default=100,
	  help='How many images to train on at a time.'
	)
	parser.add_argument(
	  '--test_batch_size',
	  type=int,
	  default=-1,
	  help="""\
	  How many images to test on. This test set is only used once, to evaluate
	  the final accuracy of the model after training completes.
	  A value of -1 causes the entire test set to be used, which leads to more
	  stable results across runs.\
	  """
	)
	parser.add_argument(
	  '--validation_batch_size',
	  type=int,
	  default=100,
	  help="""\
	  How many images to use in an evaluation batch. This validation set is
	  used much more often than the test set, and is an early indicator of how
	  accurate the model is during training.
	  A value of -1 causes the entire validation set to be used, which leads to
	  more stable results across training iterations, but may be slower on large
	  training sets.\
	  """
	)
	parser.add_argument(
	  '--print_misclassified_test_images',
	  default=False,
	  help="""\
	  Whether to print out a list of all misclassified test images.\
	  """,
	  action='store_true'
	)
	parser.add_argument(
	  '--model_dir',
	  type=str,
	  default='./tf_logs/imagenet',
	  help="""\
	  Path to classify_image_graph_def.pb,
	  imagenet_synset_to_human_label_map.txt, and
	  imagenet_2012_challenge_label_map_proto.pbtxt.\
	  """
	)
	parser.add_argument(
	  '--bottleneck_dir',
	  type=str,
	  default='/tmp/bottleneck',
	  help='Path to cache bottleneck layer values as files.'
	)
	parser.add_argument(
	  '--final_tensor_name',
	  type=str,
	  default='final_result',
	  help="""\
	  The name of the output classification layer in the retrained graph.\
	  """
	)
	parser.add_argument(
	  '--flip_left_right',
	  default=False,
	  help="""\
	  Whether to randomly flip half of the training images horizontally.\
	  """,
	  action='store_true'
	)
	parser.add_argument(
	  '--random_crop',
	  type=int,
	  default=0,
	  help="""\
	  A percentage determining how much of a margin to randomly crop off the
	  training images.\
	  """
	)
	parser.add_argument(
	  '--random_scale',
	  type=int,
	  default=0,
	  help="""\
	  A percentage determining how much to randomly scale up the size of the
	  training images by.\
	  """
	)
	parser.add_argument(
	  '--random_brightness',
	  type=int,
	  default=0,
	  help="""\
	  A percentage determining how much to randomly multiply the training image
	  input pixels up or down by.\
	  """
	)
	parser.add_argument(
	  '--architecture',
	  type=str,
	  default='inception_v3',
	  help="""\
	  Which model architecture to use. 'inception_v3' is the most accurate, but
	  also the slowest. For faster or smaller models, chose a MobileNet with the
	  form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
	  'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
	  pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
	  less accurate, but smaller and faster network that's 920 KB on disk and
	  takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
	  for more information on Mobilenet.\
	  """)
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
