"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf

from cleverhans import initializers
from cleverhans.model import Model
from cleverhans.picklable_model import MLP, Conv2D, ReLU, Flatten, Linear
from cleverhans.picklable_model import Softmax


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
import numpy as np
#from PPDModel import get_ppd_model


class ModelBasicCNN(Model):
  def __init__(self, scope, nb_classes, nb_filters, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_filters = nb_filters

    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    self.fprop(tf.placeholder(tf.float32, [128, 28, 28, 1]))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    my_conv = functools.partial(
        tf.layers.conv2d, activation=tf.nn.relu,
        kernel_initializer=initializers.HeReLuNormalInitializer)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      y = my_conv(x, self.nb_filters, 8, strides=2, padding='same')
      y = my_conv(y, 2 * self.nb_filters, 6, strides=2, padding='valid')
      y = my_conv(y, 2 * self.nb_filters, 5, strides=1, padding='valid')
      logits = tf.layers.dense(
          tf.layers.flatten(y), self.nb_classes,
          kernel_initializer=initializers.HeReLuNormalInitializer)
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}



class PPDModel(Model):
  def __init__(self, scope, nb_classes, hparams, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_classes = nb_classes
    self.dataset = hparams['dataset']
    self.secret_seed = hparams['secret_seed']
    
    if self.dataset == 'mnist':
      self.fprop(tf.placeholder(tf.float32, [128, 28, 28, 1]))
    elif self.dataset == 'cifar' or self.dataset == 'cifar10':
      self.fprop(tf.placeholder(tf.float32, [128, 32, 32, 3]))
    # Put a reference to the params in self so that the params get pickled
    #self.params = self.get_params()

  def permute_pixels(self, images, seed):
    n_sample, img_r, img_c, n_channels = images.shape
    np.random.seed(seed)
    perm_idx = np.random.permutation(img_r*img_c)
    for idx in range(n_sample):
        for channel in range(n_channels):
            img = images[idx, :, :, channel].flatten()
            img = img[perm_idx]
            images[idx, :, :, channel] = img.reshape((img_r, img_c))
    return images

  def fprop(self, x, **kwargs):
    del kwargs
    
    x_train = self.permute_pixels(x, self.secret_seed)

    def pixel2phase(images):
      img_fft = tf.fft2d(tf.cast(images, tf.complex64))
      phase = tf.angle(img_fft)
      return phase
    
    #input_tensor = Input(name='input_images', shape=input_shape, dtype='float32')
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      inner = Lambda(pixel2phase, name='pixel2phase', output_shape=input_shape)(input_tensor)
      inner = tf.contrib.layers.flatten(inner)
      inner = tf.layers.dense(inner, units=800, kernel_initializer=tf.initializers.he_normal, activation=tf.nn.relu)
      inner = tf.layers.dense(inner, units=300, kernel_initializer=tf.initializers.he_normal, activation=tf.nn.relu)
      y_pred = tf.layers.dense(inner, units=10, kernel_initializer=tf.initializers.he_normal, activation=tf.nn.relu)
      
      #inner = Flatten(name='flatten')(inner)
      #inner = Dense(800,  name='dense800', kernel_initializer='he_normal', activation='relu')(inner)
      #inner = Dense(300,  name='dense300', kernel_initializer='he_normal', activation='relu')(inner)
      #y_pred = Dense(10,  name='dense10', kernel_initializer='he_normal', activation='softmax')(inner)

      return {self.O_LOGITS: y_pred,
              self.O_PROBS: tf.nn.softmax(logits=y_pred)}


  




def make_basic_picklable_cnn(nb_filters=64, nb_classes=10,
                             input_shape=(None, 28, 28, 1)):
  """The model for the picklable models tutorial.
  """
  layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
            ReLU(),
            Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
            ReLU(),
            Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
            ReLU(),
            Flatten(),
            Linear(nb_classes),
            Softmax()]
  model = MLP(layers, input_shape)
  return model
