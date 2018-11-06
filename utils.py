#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:37:02 2018

@author: harlinl
"""
import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Lambda

def get_dataset(name='mnist'):
    if name == 'mnist':
        dataset = keras.datasets.mnist
    else:
        dataset = keras.datasets.cifar10
    (x_train, y_train),(x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if name == 'mnist':
        x_train = np.expand_dims(x_train, axis=3)  # PPDModel expects 4 dimensional image data
        x_test = np.expand_dims(x_test, axis=3)
    print('x_train shape', x_train.shape)
    print('y_train shape',y_train.shape)
    print('x_test shape',x_test.shape)
    print('y_test shape',y_test.shape)
    return (x_train, y_train, x_test, y_test)


def permute_pixels(images, seed):
    n_sample, img_r, img_c, n_channels = images.shape
    np.random.seed(seed)
    perm_idx = np.random.permutation(img_r*img_c)
    permuted_images = np.zeros_like(images)
    for idx in range(n_sample):
        for channel in range(n_channels):
            img = images[idx, :, :, channel].flatten()
            img = img[perm_idx]
            permuted_images[idx, :, :, channel] = img.reshape((img_r, img_c))
    return permuted_images

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
