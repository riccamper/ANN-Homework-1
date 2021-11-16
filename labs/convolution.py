# Import needed libraries
import os
import time 
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np

# Test the Keras version
tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

# Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# Import the image as single channel
image = Image.open('labs\picture.jpg').convert('L')
print("Original image shape: ", image.size)
image = image.resize((512,512))
print("Resized image shape: ", image.size)
fig = plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.show()

# Convert the image into an array
image = np.array(image, dtype=np.float32)
image_h, image_w = image.shape[:2]

# Sobel filter (Edge detection)

# Function to plot image and filters
def plot_edges(orig_image, h_edge_image, v_edge_image, edge_image):
  print("Original image shape:", orig_image.shape)
  print("Horizontal edge image shape:", h_edge_image.shape)
  print("Vertical edge image shape:", v_edge_image.shape)
  print("Edge image shape:", edge_image.shape)
  fig, ax = plt.subplots(1, 4, figsize=(15, 45))
  ax[0].imshow(orig_image, cmap='gray')
  ax[1].imshow(h_edge_image, cmap='gray')
  ax[2].imshow(v_edge_image, cmap='gray')
  ax[3].imshow(edge_image, cmap='gray')
  plt.show()

# Define Sobel filters
# Horizontal filter
def kernel_h_init(shape, dtype=None, partition_info=None):
    kernel = tf.constant([[1,0,-1],
                          [2,0,-2],
                          [1,0,-1]], dtype=dtype)
    kernel = tf.reshape(kernel, shape)
    return kernel

# Vertical filter
def kernel_v_init(shape, dtype=None, partition_info=None):
    kernel = tf.constant([[1,2,1],
                          [0,0,0],
                          [-1,-2,-1]], dtype=dtype)
    kernel = tf.reshape(kernel, shape)
    return kernel

# Compute the edges by (manually) convolving the input with the filters
stride = 1
kernel_size = 3

h_kernel = kernel_h_init(shape=[3, 3], dtype=None)
v_kernel = kernel_v_init(shape=[3, 3], dtype=None)

h_edges = np.zeros([image_h, image_w])
v_edges = np.zeros([image_h, image_w])  
edges = np.zeros([image_h, image_w])

# Slide the filters over the image
for i in np.arange(0, image_h-kernel_size+1, stride):
    for j in np.arange(0, image_w-kernel_size+1, stride):
        # Apply the filter
        h_out = image[i:i+kernel_size,j:j+kernel_size] * h_kernel[:, :]
        h_out = tf.reduce_sum(h_out)
        v_out = image[i:i+kernel_size,j:j+kernel_size] * v_kernel[:, :]
        v_out = tf.reduce_sum(v_out)

        h_edges[i, j] = h_out
        v_edges[i, j] = v_out
        edges[i, j] = np.sqrt(h_out**2+v_out**2)

h_edges = h_edges[:image_h-kernel_size+1, :image_w-kernel_size+1]
v_edges = v_edges[:image_h-kernel_size+1, :image_w-kernel_size+1]
edges = edges[:image_h-kernel_size+1, :image_w-kernel_size+1]

plot_edges(image, h_edges, v_edges, edges)

# ----------------------------------------------------------

#2D Convolutial Layer - tfk.layers.Conv2D

# Create Conv2D layer
conv2d_h = tfkl.Conv2D(1, [kernel_size, kernel_size], strides=(stride, stride), 
                       kernel_initializer=kernel_h_init, input_shape=(image_h,image_w,1))

conv2d_v = tfkl.Conv2D(1, [kernel_size, kernel_size], strides=(stride, stride),
                       kernel_initializer=kernel_v_init, input_shape=(image_h,image_w,1))

h_edges_conv = conv2d_h(image[None, :, :, None]) # 'None' to add batch and channel dimensions
v_edges_conv = conv2d_v(image[None, :, :, None]) # 'None' to add batch and channel dimensions

h_edges_conv = h_edges_conv[0, :, :, 0]
v_edges_conv = v_edges_conv[0, :, :, 0]
edges_conv = np.sqrt(h_edges_conv**2+v_edges_conv**2)

# Check the result of the "manual" convolution and 
# the result of the Keras convolution are the same 
assert np.allclose(h_edges, h_edges_conv)
print("OK. Horizontal edges are the same!")
assert np.allclose(v_edges, v_edges_conv)
print("OK. Vertical edges are the same!")
assert np.allclose(edges, edges_conv)
print("OK. Edge magnutides are the same!")

plot_edges(image, h_edges_conv, v_edges_conv, edges_conv)

# ----------------------------------------------------------

# Padding

# What about input and Conv2D output shapes?
print("Original image shape:", image.shape)
print("Conv2D output shape:", h_edges_conv.shape)

# Create Conv2D layer with padding
conv2d_pad_h = tfkl.Conv2D(1, [kernel_size, kernel_size], strides=(stride, stride),
                           kernel_initializer=kernel_h_init, input_shape=(image_h,image_w,1), padding='same')

# Create Conv2D layer with padding
conv2d_pad_v = tfkl.Conv2D(1, [kernel_size, kernel_size], strides=(stride, stride),
                           kernel_initializer=kernel_v_init, input_shape=(image_h,image_w,1), padding='same')

h_edges_conv_pad = conv2d_pad_h(image[None, :, :, None])
v_edges_conv_pad = conv2d_pad_v(image[None, :, :, None])

h_edges_conv_pad = h_edges_conv_pad[0, :, :, 0]
v_edges_conv_pad = v_edges_conv_pad[0, :, :, 0]
edges_conv_pad = np.sqrt(h_edges_conv_pad**2+v_edges_conv_pad**2)

# What about the shapes now?
print("Original image shape:", image.shape)
print("Conv2D output shape:", h_edges_conv_pad.shape)

# If we remove the padding in the output, we obtain
# the same result of the convolution with no padding
assert np.allclose(h_edges_conv, h_edges_conv_pad[1:511, 1:511])
print("OK. Horizontal edges are the same!")
assert np.allclose(v_edges_conv, v_edges_conv_pad[1:511, 1:511])
print("OK. Vertical edges are the same!")
assert np.allclose(edges_conv, edges_conv_pad[1:511, 1:511])
print("OK. Edge magnutides are the same!")

# ----------------------------------------------------------

# Learning the Conv2D filter
# Horizontal edge model
h_edge_model = tfk.Sequential()
h_edge_model.add(tfkl.Conv2D(1, [kernel_size, kernel_size], strides=(stride, stride), 
                 kernel_initializer=tfk.initializers.GlorotUniform(seed=seed),
                 input_shape=(image_h,image_w,1), padding='valid'))

h_edge_model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam(learning_rate=1e-1))

# Vertical edge model
v_edge_model = tfk.Sequential()
v_edge_model.add(tfkl.Conv2D(1, [kernel_size, kernel_size], strides=(stride, stride), 
                 kernel_initializer=tfk.initializers.GlorotUniform(seed=seed),
                 input_shape=(image_h,image_w,1), padding='valid'))

v_edge_model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam(learning_rate=1e-1))

# Horizontal edge model training
h_edge_model.fit(
    x=image[None, ..., None], 
    y=h_edges[None, ..., None], 
    epochs=3000, batch_size=1,
    callbacks=[tfk.callbacks.EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)])

# Vertical edge model training
v_edge_model.fit(
    x=image[None, ..., None], 
    y=v_edges[None, ..., None], 
    epochs=3000, batch_size=1,
    callbacks=[tfk.callbacks.EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)])

# Compare the learned filter with the Sobel one
learned_h_kernel = h_edge_model.weights[0].numpy()
learned_v_kernel = v_edge_model.weights[0].numpy()

print("Learned horizontal edge filter")
print()
print(learned_h_kernel[..., 0, 0].round(1))
print()
print("Learned vertical edge filter")
print()
print(learned_v_kernel[..., 0, 0].round(1))

# Check if learned and original Sobel filters are the same
assert np.allclose(h_kernel, learned_h_kernel[..., 0, 0].round(1))
assert np.allclose(v_kernel, learned_v_kernel[..., 0, 0].round(1))
print("OK. Learned and original Sobel filters are the same!")

# ----------------------------------------------------------

# Convolutional Neural Network (CNN)

# Create example tensor 4x4
tensor = tf.reshape(tf.range(0, 4*4, dtype=tf.float32), [1, 4, 4, 1])
print("Tensor shape:", tensor.shape)
print("Tensor values:")
print(tensor[0, ..., 0])

# 2D Average Pooling
avg_pool2d = tfkl.AvgPool2D()
out = avg_pool2d(tensor)
print("Output shape:", out.shape)
print("Output values:")
print(out[0, ..., 0])

# 2D Max Pooling
max_pool2d = tfkl.MaxPool2D()
out = max_pool2d(tensor)
print("Output shape:", out.shape)
print("Output values:")
print(out[0, ..., 0])

# Global Average Pooling
global_avg_pool2d = tfkl.GlobalAvgPool2D()
out = global_avg_pool2d(tensor)
print("Output shape:", out.shape)
print("Output values:")
print(out[0, ..., 0])

