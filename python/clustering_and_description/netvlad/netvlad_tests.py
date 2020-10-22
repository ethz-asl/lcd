import cv2
import numpy as np
import tensorflow as tf
import sys
import os
import pickle

sys.path.append("../../../../netvlad_tf_open/python/")

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf.reset_default_graph()

image_batch = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3])

net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver()

sess = tf.InteractiveSession(config=config)
saver.restore(sess, nets.defaultCheckpoint())

RANDOM_LIGHTING = True

data_path = "/nvme/datasets/interiornet"
pickle_path = "interiornet_descriptors"

if RANDOM_LIGHTING:
    pickle_path = pickle_path + "_random"

files_list = [os.path.join(data_path, name) for name in os.listdir(data_path)
              if os.path.isdir(os.path.join(data_path, name))]

print("Computing NetVLAD descriptors.")

results = {}

for i, scene_dir in enumerate(files_list):
    if RANDOM_LIGHTING:
        rgb_dir = os.path.join(scene_dir, "random_lighting_cam0", "data")
    else:
        rgb_dir = os.path.join(scene_dir, "cam0", "data")

    print("{}% completed; computing scene {}".format(i * 1.0 / len(files_list) * 100., scene_dir))

    frame_list = [os.path.join(rgb_dir, name) for name in os.listdir(rgb_dir)
                  if os.path.isfile(os.path.join(rgb_dir, name))]

    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = sess.run(net_out, feed_dict={image_batch: np.expand_dims(frame, axis=0)})
        results[frame_path] = result

with open(pickle_path, 'wb') as f:
    pickle.dump(results, f)

