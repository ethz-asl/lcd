#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

python_root = '../'
sys.path.insert(0, python_root)

from model.datagenerator import ImageDataGenerator

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('/media/space_for_data/tmp_home_training/triplet_loss_batch_all_ckpt/bgr-d_model_epoch30.ckpt.meta')
saver.restore(sess, '/media/space_for_data/tmp_home_training/triplet_loss_batch_all_ckpt/bgr-d_model_epoch30.ckpt')

import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

# # Visualize kernels

conv1_kernels = sess.run('conv1/weights:0')
conv1_kernels_rgb = conv1_kernels[:,:,:3,:]
conv1_kernels_depth = conv1_kernels[:,:,-1,:]

from tools.visualization import vis_square
plt.rcParams['figure.figsize'] = (10, 10)

vis_square(conv1_kernels_rgb.transpose(3,0,1,2))
vis_square(conv1_kernels_depth.transpose(2,0,1))


# # Embeddings learned for test set
# According to ../split_dataset_with_labels_world.py
test = []
traj = 1
frames_total = 300
for frame_id in range(frames_total):
    if frame_id % 5 == 3:
        test.append(frame_id)
        continue
test_file = '/media/space_for_data/test (copy).txt'
test_set_size = 0
with open(test_file) as f:
    for i, l in enumerate(f):
        pass
    test_set_size = i + 1

print("Test set has {} lines ".format(test_set_size))

train_set_mean = np.array([22.4536157, 20.11461999, 5.61416132, 605.87199598])
test_generator = ImageDataGenerator(test_file, image_type='bgr-d',
                                    mean=train_set_mean)
x = graph.get_tensor_by_name('Placeholder:0')  # input images
labels = graph.get_tensor_by_name('Placeholder_1:0')  # labels of input images
keep_prob = graph.get_tensor_by_name('Placeholder_2:0')  # dropout probability
embeddings = graph.get_tensor_by_name('l2_normalize:0')

batch_size = int(x.shape[0])
test_embeddings_all = np.empty((0, int(embeddings.shape[1])), dtype=np.float32)
test_generator.reset_pointer()

for i in range(test_set_size / batch_size):
    batch_x, batch_labels = test_generator.next_batch(batch_size)
    output = sess.run(embeddings, feed_dict={x: batch_x,
                                                 labels: batch_labels,
                                                 keep_prob: 1.})
    test_embeddings_all = np.vstack([test_embeddings_all, output])
    # Display every 5 steps
    if i % 5 == 0:
        print("Embeddings got for {} lines ".format(test_embeddings_all.shape[0]))

lines_total = test_embeddings_all.shape[0]
print("Embeddings got for {} lines ".format(lines_total))


# # K-means clustering with obtained embeddings
from sklearn.cluster import KMeans

lines_features = test_embeddings_all

n_clusters = 12
kmeans = KMeans(n_clusters, init='k-means++').fit(lines_features)
cluster_labels = kmeans.labels_
count = cluster_labels.shape[0]

from tools.visualization import get_lines_world_coordinates_with_instances
data_lines_world = get_lines_world_coordinates_with_instances(trajectory=traj, frames=test)
print('shape of data_lines_world is here {0}'.format(data_lines_world.shape))
data_lines_world = data_lines_world[:count]
print('shape of data_lines_world is here {0}'.format(data_lines_world.shape))
instance_labels = data_lines_world[:, -1]

from tools.visualization import pcl_lines_for_plot

#pcl_lines_open3d = pcl_lines_for_plot(data_lines_world, lines_color=cluster_labels, visualizer='open3d')
pcl_lines_matplotlib = pcl_lines_for_plot(data_lines_world, lines_color=cluster_labels, visualizer='matplotlib')

#import open3d

#open3d.draw_geometries(pcl_lines_open3d[1:300])
from tools.visualization import plot_lines_with_matplotlib
plot_lines_with_matplotlib(pcl_lines_matplotlib)

#open3d.draw_geometries(pcl_lines_open3d[:])

# # See embeddings in the feature space, colored with instance label
from tensorflow.contrib.tensorboard.plugins import projector
LOG_DIR = os.path.abspath('../logs/tmp/embedding_logs')
# Create parent path if it doesn't exist
if not os.path.isdir(LOG_DIR): os.makedirs(LOG_DIR)
metadata = os.path.join(LOG_DIR, 'embedding_metadata.tsv')
test_embeddings = tf.Variable(test_embeddings_all, name='test_embeddings')

with open(metadata, 'w') as metadata_file:
    for label in instance_labels:
        metadata_file.write('%d\n' % label)

with tf.Session() as sess:
    saver = tf.train.Saver([test_embeddings])

    sess.run(test_embeddings.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'test_embeddings.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = test_embeddings.name
    print('test_embeddings: {0}'.format(test_embeddings))
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
