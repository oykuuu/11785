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
from keras.layers import Dense, Flatten, Input, Lambda, Conv2D, MaxPooling2D, Dropout



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


def get_ppd_model(name='mnist', ppd=True):
    if name == 'mnist':
        input_shape = (28, 28, 1)
    else:
        input_shape = (32, 32, 3)  # cifar10

    def pixel2phase(images):
        img_fft = tf.fft2d(tf.cast(images, tf.complex64))
        phase = tf.angle(img_fft)
        return phase
    
    input_tensor = Input(name='input_images', shape=input_shape, dtype='float32')
    
    if ppd:
        inner = Lambda(pixel2phase, name='pixel2phase', output_shape=input_shape)(input_tensor)
        inner = Flatten(name='flatten')(inner)
    else:
        inner = Flatten(name='flatten')(input_tensor)
    inner = Dense(800,  name='dense800', kernel_initializer='he_normal', activation='relu')(inner)
    inner = Dense(300,  name='dense300', kernel_initializer='he_normal', activation='relu')(inner)
    y_pred = Dense(10,  name='dense10', kernel_initializer='he_normal', activation='softmax')(inner)

    return Model(inputs=[input_tensor], outputs=y_pred)


# def get_ppd_model(name='mnist', ppd=True):
#     if name == 'mnist':
#         input_shape = (28, 28, 1)
#     else:
#         input_shape = (32, 32, 3)  # cifar10

#     def pixel2phase(images):
#         img_fft = tf.fft(tf.cast(images, tf.complex64))
#         phase = tf.angle(img_fft)
#         return phase
    
#     input_tensor = Input(name='input_images', shape=input_shape, dtype='float32')
    
    
#     inner = Flatten(name='flatten')(input_tensor)
#     inner = Dense(800,  name='dense800', kernel_initializer='he_normal', activation='relu')(inner)
    
#     if ppd:
#         inner = Lambda(pixel2phase, name='pixel2phase')(inner)
#         #inner = Flatten(name='flatten')(inner)
        
#     inner = Dense(300,  name='dense300', kernel_initializer='he_normal', activation='relu')(inner)
#     y_pred = Dense(10,  name='dense10', kernel_initializer='he_normal', activation='softmax')(inner)

#     return Model(inputs=[input_tensor], outputs=y_pred)


def get_cnn_model(name='mnist', ppd=True, layerno=0):
    if name == 'mnist':
        return mnist_cnn_model(ppd, layerno=layerno)
    else:
        return cifar10_cnn_model(ppd, layerno=layerno)

    
def mnist_cnn_model(ppd=True, layerno=0):
    input_shape = (28, 28, 1)
   
    def pixel2phase1d(images):
        img_fft = tf.fft(tf.cast(images, tf.complex64))
        phase = tf.angle(img_fft)
        return phase

    def pixel2phase(images):
        img_fft = tf.fft2d(tf.cast(images, tf.complex64))
        phase = tf.angle(img_fft)
        return phase
    
    input_tensor = Input(name='input_images', shape=input_shape, dtype='float32')
    if ppd and layerno==0:
        inner = Lambda(pixel2phase, name='pixel2phase', output_shape=input_shape)(input_tensor)
        inner = Conv2D(32, kernel_size=(3, 3), activation='relu')(inner)
    else:
        inner = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_tensor)
    
    if ppd and layerno==1:
        inner = Lambda(pixel2phase1d, name='pixel2phase1d')(inner)
    
    inner = Conv2D(64, (3, 3), activation='relu')(inner)
    if ppd and layerno==2:
        inner = Lambda(pixel2phase1d, name='pixel2phase1d')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Dropout(0.25)(inner)
    inner = Flatten(name='flatten')(inner)
    inner = Dense(128, activation='relu')(inner)
    if ppd and layerno==3:
        inner = Lambda(pixel2phase1d, name='pixel2phase1d')(inner)
    inner = Dropout(0.5)(inner)
    y_pred = Dense(10,  name='dense10', kernel_initializer='he_normal', activation='softmax')(inner)

    return Model(inputs=[input_tensor], outputs=y_pred)


def cifar10_cnn_model(ppd=True, layerno=0):
    input_shape = (32, 32, 3)  # cifar10

    def pixel2phase1d(images):
        img_fft = tf.fft(tf.cast(images, tf.complex64))
        phase = tf.angle(img_fft)
        return phase

    def pixel2phase(images):
        img_fft = tf.fft2d(tf.cast(images, tf.complex64))
        phase = tf.angle(img_fft)
        return phase
    
    input_tensor = Input(name='input_images', shape=input_shape, dtype='float32')
    if ppd and layerno==0:
        inner = Lambda(pixel2phase, name='pixel2phase', output_shape=input_shape)(input_tensor)
        inner = Conv2D(32, (3, 3), padding='same', activation='relu')(inner)
    else:
        inner = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    if ppd and layerno==1:
        inner = Lambda(pixel2phase1d, name='pixel2phase1d')(inner)
    inner = Conv2D(32, (3, 3), activation='relu')(inner)
    if ppd and layerno==2:
        inner = Lambda(pixel2phase1d, name='pixel2phase1d')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Dropout(0.25)(inner)
    inner = Conv2D(64, (3, 3), padding='same', activation='relu')(inner)
    if ppd and layerno==3:
        inner = Lambda(pixel2phase1d, name='pixel2phase1d')(inner)
    inner = Conv2D(64, (3, 3), padding='same', activation='relu')(inner)
    if ppd and layerno==4:
        inner = Lambda(pixel2phase1d, name='pixel2phase1d')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Dropout(0.25)(inner)
    inner = Flatten(name='flatten')(inner)
    inner = Dense(512, activation='relu')(inner)
    if ppd and layerno==5:
        inner = Lambda(pixel2phase1d, name='pixel2phase1d')(inner)
    inner = Dropout(0.5)(inner)
    y_pred = Dense(10,  name='dense10', kernel_initializer='he_normal', activation='softmax')(inner)

    return Model(inputs=[input_tensor], outputs=y_pred)
