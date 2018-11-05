#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
import numpy as np


# In[2]:


def get_ppd_model(name='mnist'):
    if name == 'mnist':
        input_shape = (28, 28, 1)
    else:
        input_shape = (32, 32, 3)  # cifar10

    def pixel2phase(images):
        img_fft = tf.fft2d(tf.cast(images, tf.complex64))
        phase = tf.angle(img_fft)
        return phase
    
    input_tensor = Input(name='input_images', shape=input_shape, dtype='float32')
    inner = Lambda(pixel2phase, name='pixel2phase', output_shape=input_shape)(input_tensor)
    inner = Flatten(name='flatten')(inner)
    inner = Dense(800,  name='dense800', kernel_initializer='he_normal', activation='relu')(inner)
    inner = Dense(300,  name='dense300', kernel_initializer='he_normal', activation='relu')(inner)
    y_pred = Dense(10,  name='dense10', kernel_initializer='he_normal', activation='softmax')(inner)

    return Model(inputs=[input_tensor], outputs=y_pred)


# In[3]:


# mnist_model = get_ppd_model('mnist')
# mnist_model.summary()
#
# cifar10_model = get_ppd_model('cifar10')
# cifar10_model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# I added permute_pixel block to go inside the keras model, but that yielded terrible accuracy results. (~18% on MNIST)
# The PPD paper mentions Pixel2Phase block being in tensorflow, but not the permutation block, so I took it out.
# I'll leave this code in case we end up needing it.

# def get_ppd_model_old(name='mnist', SECRET_SEED=0):
#     if name == 'mnist':
#         input_shape = (28, 28, 1)
#     else:
#         input_shape = (32, 32, 3)  # cifar10
#
#     def permute_pixel(args):
#         image_tensor = args  # image tensor is shape batch_size x image_rows x image_cols x n_channels
#         tf.set_random_seed(SECRET_SEED)
#         x = tf.reshape(image_tensor, [-1, input_shape[0]*input_shape[1], input_shape[2]])  # flatten per channel per image
#         x = tf.transpose(x, [1, 0, 2])  # move the pixels to first axis
#         x = tf.random_shuffle(x, seed=SECRET_SEED)  # shuffle pixels
#         x = tf.transpose(x, [1, 0, 2])
#         image_tensor = tf.reshape(x, [-1, input_shape[0], input_shape[1], input_shape[2]])  # reshape to original shape
#         return image_tensor
#
#
#     def pixel2phase(images):
#         img_fft = tf.fft2d(tf.cast(images, tf.complex64))
#         phase = tf.angle(img_fft)
#         return phase
#
#     input_tensor = Input(name='input_images', shape=input_shape, dtype='float32')
#     inner = Lambda(permute_pixel, name='permute_pixel', output_shape=input_shape)(input_tensor)
#     inner = Lambda(pixel2phase, name='pixel2phase', output_shape=input_shape)(inner)
#     inner = Flatten(name='flatten')(input_tensor)
#     inner = Dense(800,  name='dense800', kernel_initializer='he_normal', activation='relu')(inner)
#     inner = Dense(300,  name='dense300', kernel_initializer='he_normal', activation='relu')(inner)
#     y_pred = Dense(10,  name='dense10', kernel_initializer='he_normal', activation='softmax')(inner)
#
#     return Model(inputs=[input_tensor], outputs=y_pred)
#
# mnist_model = get_ppd_model('mnist', 23)
# mnist_model.summary()
#
# cifar10_model = get_ppd_model('cifar10', 87)
# cifar10_model.summary()

