#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
import matplotlib.pyplot as plt
from PPDModel import get_ppd_model
from keras import models


# In[2]:


def get_dataset(name='mnist'):
    if name == 'mnist':
        dataset = tf.keras.datasets.mnist
    else:
        dataset = tf.keras.datasets.cifar10
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
    for idx in range(n_sample):
        for channel in range(n_channels):
            img = images[idx, :, :, channel].flatten()
            img = img[perm_idx]
            images[idx, :, :, channel] = img.reshape((img_r, img_c))
    return images


# ## TRAINING ON MNIST DATASET

# In[3]:


SECRET_SEED = 23
(x_train, y_train, x_test, y_test) = get_dataset('mnist')
plt.figure()
plt.imshow(x_train[0,:,:,0])
plt.title('original image')

x_train = permute_pixels(x_train, SECRET_SEED)
x_test = permute_pixels(x_test, SECRET_SEED)
plt.figure()
plt.imshow(x_train[0,:,:,0])
plt.title('permuted image')
plt.show()


# In[4]:


model = get_ppd_model('mnist')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, verbose=1, batch_size=128, validation_data=(x_test, y_test))
model.save('models/mnist_trained_keras_model.hdf5')
print(model.evaluate(x_test, y_test))


# ## TRAINING ON CIFAR10 DATASET

# In[5]:


SECRET_SEED = 87
(x_train, y_train, x_test, y_test) = get_dataset('cifar10')
plt.figure()
plt.imshow(x_train[0])
plt.title('original image')

x_train = permute_pixels(x_train, SECRET_SEED)
x_test = permute_pixels(x_test, SECRET_SEED)
plt.figure()
plt.imshow(x_train[0])
plt.title('permuted image')
plt.show()


# In[6]:


model = get_ppd_model('cifar10')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, verbose=1, batch_size=128, validation_data=(x_test, y_test))
model.save('models/cifar10_trained_keras_model.hdf5')


# In[ ]:





# In[ ]:


# load saved model with these lines
#http://everettsprojects.com/2018/01/30/mnist-adversarial-examples.html
#import keras
#from cleverhans.utils_keras import KerasModelWrapper
#from keras.models import load_model
#keras.backend.set_learning_phase(False)
#keras_model = load_model('models/cifar10_trained_keras_model.hdf5')
#wrap = KerasModelWrapper(keras_model)


# In[ ]:




