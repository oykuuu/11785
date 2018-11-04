import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_dataset(name='mnist'):
    if name == 'cifar10':
        dataset = tf.keras.datasets.cifar10
    else:
        dataset = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print('x_train shape', x_train.shape)
    print('y_train shape',y_train.shape)
    print('x_test shape',x_test.shape)
    print('y_test shape',y_test.shape)
    return (x_train, y_train, x_test, y_test)


def permutateImages(images, seed):
    n_sample, img_r, img_c = images.shape
    np.random.seed(seed)
    perm_idx = np.random.permutation(img_r*img_c)
    for idx in range(n_sample):
        img = images[idx].flatten()
        img = img[perm_idx]
        images[idx] = img.reshape((img_r, img_c))
    return images


def pixel2phase(images):
    img_fft = np.fft.fft2(images)
    phase = np.angle(img_fft)
    return phase

seed = 0

(x_train, y_train, x_test, y_test) = get_dataset('mnist')
plt.imshow(x_train[0])
plt.title('original image')

x_train = permutateImages(x_train, seed)
x_test = permutateImages(x_test, seed)
plt.imshow(x_train[0])
plt.title('permutated image')

x_train = pixel2phase(x_train)
x_test = pixel2phase(x_test)
plt.imshow(x_train[0])
plt.title('phase of permutated image')
plt.show()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(800, activation=tf.nn.relu),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=64)
print(model.evaluate(x_test, y_test))