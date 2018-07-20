#!/usr/bin/env python
import numpy as np 
import tensorflow as tf
import res_batch_utils
import cut_backgrounds
from scipy.misc import imsave
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import average, add, Input, Lambda, Dense, GlobalAveragePooling2D, concatenate
from keras import backend as K
import os
import time

sess = tf.Session()
K.set_session(sess)

def make_loss(part_conv, dist_part):
    #loss_part = tf.reduce_mean(-tf.reduce_sum(tf.exp(-tf.cast(dist_part, tf.float32)/1.0) * tf.log(tf.clip_by_value(part_conv,1e-10,1.0)), axis=1))
    loss_part = tf.exp(-tf.cast(dist_part, tf.float32)/1.0) * tf.log(tf.clip_by_value(part_conv,1e-10,1.0))
    return loss_part
    
def customLoss(yTrue,yPred):
    return yPred

dist_a = Input(shape=(360,1),name='dist_a')
dist_e = Input(shape=(180,1),name='dist_e')
dist_t = Input(shape=(360,1),name='dist_t')

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

#print(base_model.summary())

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
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


## this is the model we will train
#model = Model(inputs=[base_model.input,dist_a,dist_e,dist_t], outputs=loss)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

train_step = tf.train.AdamOptimizer(5e-6).minimize(loss)

sess.run(tf.global_variables_initializer())

for i in range(max_steps):
test_batch = res_batch_utils.next_batch(50, testing=True)


'''
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss=customLoss)

out_dir = 'a_keras'
steps_per_epoch = 150
nb_epoch = 2

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('tf_logs', out_dir, 'out.{epoch:03d}-{val_loss:.3f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: TensorBoard
tb = TensorBoard(log_dir=os.path.join('tf_logs', out_dir))

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=5)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('tf_logs', out_dir, 'training-' + \
    str(timestamp) + '.log'))

generator = res_batch_utils.next_batch(50)
val_generator = res_batch_utils.next_batch(50, testing=True)


# train the model on the new data for a few epochs
model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40)


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=customLoss)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40)


'''
