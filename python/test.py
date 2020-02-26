""" Obtain embeddings for a test dataset, given checkpoints and meta graph from
    a previously trained model. The embeddings can then be clustered and the
    result of the clustering is displayed, followed by ground-truth instances.
    Finally, a checkpoint with the retrieved embeddings can be saved, so that it
    can later be used for visualization with TensorBoard's Projector.
"""

import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.contrib import tensorboard
from timeit import default_timer as timer

from model.datagenerator import ImageDataGenerator
from tools.cluster_lines import cluster_lines_affinity_propagation, \
                                cluster_lines_kmeans, \
                                cluster_lines_aggr_clustering
from tools.lines_utils import get_label_with_line_center, get_geometric_info
from tools.visualization import pcl_lines_for_plot

python_root = '../'
sys.path.insert(0, python_root)

# Configuration.
# Cluster strategy. One of "kmeans", "aggr_clustering" and
# "affinity_propagation".
cluster_strategy = "aggr_clustering"

# Visualizer of the lines coloured with the instances from clustering. Possible
# values: 'open3d', 'matplotlib'.
visualizer = 'open3d'

# Whether or not the test dataset should be read as a pickle file.
# NOTE: non-pickle mode is currently deprecated and not fully supported.
read_as_pickle = True

# If True, the ground-truth instance labels are read from the input dataset (in
# pickle-file format, each line has a 7-dimensional entry called 'labels' in the
# format [start point (3x)] [end point (3x)] [instance label]. The instance
# label is read as the last of these values). The ground-truth instance labels
# are later used to display-ground truth labels and to assign each embedding
# in the checkpoint for TensorBoard's Projector a label (that can be used, e.g.,
# to colour the points by instance label).
# NOTE: in principle, for a generic test dataset it might be the case that
# ground-truth instance labels are not available (e.g., when testing on a real
# set without ground-truth labelling), and it therefore makes sense to set
# use_ground_truth_instance_labels to False. In this case, it is still expected
# that the pickle files contain an entry called 'labels', with the first six
# entries corresponding to [start point (3x)] [end point (3x)].
use_ground_truth_instance_labels = False

# Folder where the checkpoints and meta graph for the test are stored.
log_files_folder = './logs/240220_2216/'

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph(
    os.path.join(log_files_folder,
                 'triplet_loss_batch_all_wohlhart_lepetit_ckpt/bgr-d_model_epoch30.ckpt.meta'))
saver.restore(
    sess,
    os.path.join(log_files_folder,
                 'triplet_loss_batch_all_wohlhart_lepetit_ckpt/bgr-d_model_epoch30.ckpt'))

# Test dataset.
test_files = '/home/felix/line_ws/data/pickle_files/train_0/traj_1/pickled_val.pkl'
name_test_file = test_files.split('/')[-1].split('.')[0]

# Retrieve mean of the training set.
train_set_mean = sess.run('train_set_mean:0')

print("Train set mean is {}".format(train_set_mean))

test_generator = ImageDataGenerator(
    [test_files],
    image_type='bgr-d',
    mean=train_set_mean,
    read_as_pickle=read_as_pickle)

# Check if embeddings have already been generated (look for them at the path
# embeddings_path). If not, generate them.
embeddings_path = os.path.join(log_files_folder,
                               'embeddings_{}.npy'.format(name_test_file))
if os.path.isfile(embeddings_path):
    print("Using embeddings found at {}".format(embeddings_path))
    test_embeddings_all = np.load(embeddings_path)
else:
    graph = tf.get_default_graph()
    # Input image tensor.
    input_img = graph.get_tensor_by_name('input_img:0')
    # Dropout probability tensor.
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    # Embeddings tensor.
    try:
	# New models have the embedding tensor named 'embeddings'.
    	embeddings = graph.get_tensor_by_name('embeddings:0')
    except KeyError:
	# Old models have the embedding tensor named 'l2_normalize'.
    	embeddings = graph.get_tensor_by_name('l2_normalize:0')
    # Line type tensor.
    line_types = graph.get_tensor_by_name('line_types:0')
    # Geometric info tensor.
    try:
        geometric_info = graph.get_tensor_by_name('geometric_info:0')
    except KeyError:
        geometric_info_found = False
    else:
        geometric_info_found = True

    if (use_ground_truth_instance_labels):
        labels = graph.get_tensor_by_name('labels:0')

    batch_size = 128
    test_embeddings_all = np.empty(
        (0, int(embeddings.shape[1])), dtype=np.float32)
    test_generator.reset_pointer()

    if read_as_pickle:
        test_set_size = len(test_generator.pickled_labels)
    else:
        # TODO: fix so that test frames can be written directly
        # According to ../split_dataset_with_labels_world.py
        test = []
        traj = 1
        frames_total = 300
        for frame_id in range(frames_total):
            if frame_id % 5 == 3:
                test.append(frame_id)
                continue
        test_set_size = 0
        with open(test_files) as f:
            for i, l in enumerate(f):
                pass
            test_set_size = i + 1

    print("Test set has {} lines ".format(test_set_size))

    # Obtain embeddings for the lines in the test set.

    # NOTE: Pickle files might store lines images in a different 'order' than
    # the one corresponding to the list of images in the text file. This means
    # that the method that obtains the batches provides lines in different order
    # if executed in read_as_pickle mode or not. Therefore, the embeddings
    # obtained in the two cases will not be comparable, since, i.e., the first
    # embedding will correspond to a certain line in the read_as_pickle mode and
    # to another one in the text-files mode. Also, when the number of lines in
    # the test set is not a multiple of the batch size, for some lines the
    # embeddings will not be obtained. These lines without embeddings will be
    # different in the two reading modes for the reasons explained above and
    # therefore some lines might be visualized in one case but not in the other.
    for i in range(test_set_size / batch_size):
        print("Batch no. {}".format(i))
        (batch_input_img, batch_labels,
         batch_line_types) = test_generator.next_batch(batch_size)
        # Create dictionary of the values to feed to the tensors to run the
        # operation.
        feed_dict = {
            input_img: batch_input_img,
            line_types: batch_line_types,
            keep_prob: 1.
        }

        # To ensure backcompatibility, geometric information is fed into the
        # network only if the version of the network trained contains the
        # associated tensor.
        if read_as_pickle and geometric_info_found:
            # Retrieve geometric info depending on the type of line
            # parametrization used when training.
            batch_start_points = batch_labels[:, :3]
            batch_end_points = batch_labels[:, 3:6]
            if (geometric_info.shape[1] == 4):
                # Line parametrization: 'orthonormal'.
                batch_geometric_info = get_geometric_info(
                    start_points=batch_start_points,
                    end_points=batch_end_points,
                    line_parametrization='orthonormal')
                batch_geometric_info = np.array(batch_geometric_info).reshape(
                    -1, 4)
            elif (geometric_info.shape[1] == 6):
                # Line parametrization: 'direction_and_centerpoint'.
                batch_geometric_info = get_geometric_info(
                    start_points=batch_start_points,
                    end_points=batch_end_points,
                    line_parametrization='direction_and_centerpoint')
                batch_geometric_info = np.array(batch_geometric_info).reshape(
                    -1, 6)
            else:
                raise ValueError("The trained geometric_info Tensor should "
                                 "have shape[1] either equal to 4 (line "
                                 "parametrization 'orthonormal') or equal to 6 "
                                 "(line parametrization "
                                 "'direction_and_centerpoint').")
            feed_dict[geometric_info] = batch_geometric_info

        start_time = timer()
        output = sess.run(embeddings, feed_dict=feed_dict)
        end_time = timer()

        test_embeddings_all = np.vstack([test_embeddings_all, output])
        print('Time needed to retrieve descriptors for %d lines: %.3f seconds' %
              (batch_size, (end_time - start_time)))
        # Display every 5 steps.
        if i % 5 == 0:
            print("Embeddings got for {} lines ".format(
                test_embeddings_all.shape[0]))

    lines_total = test_embeddings_all.shape[0]
    print("In total, embeddings got for {} lines ".format(lines_total))

    print("Writing embeddings to file")
    np.save(embeddings_path, test_embeddings_all)

num_lines_with_embeddings = len(test_embeddings_all)

# Retrieve true instances.
if read_as_pickle:
    data_lines_world = np.array(test_generator.pickled_labels, dtype=np.float32)
else:
    from tools.visualization import get_lines_world_coordinates_with_instances
    data_lines_world = get_lines_world_coordinates_with_instances(
        trajectory=traj, frames=test, dataset_name='train')

# Only keep the lines for which an embedding was sought (since data is processed
# in batches only for a number of lines multiple of the batch size the embedding
# is computed, cf. above).
data_lines_world = data_lines_world[:num_lines_with_embeddings]
if (use_ground_truth_instance_labels):
    if (data_lines_world.shape[1] != 7):
        print("Ground-truth instance labels not found: entry 'labels' of the "
              "lines in the pickle file has not dimension 7. Please set "
              "'use_ground_truth_instance_labels' to False.")
        sys.exit()
    else:
        instance_labels = data_lines_world[:, -1]
        np.save(
            os.path.join(
                log_files_folder,
                'ground_truth_instances_{}.npy'.format(name_test_file)),
            instance_labels)

# Cluster lines.
if cluster_strategy == "kmeans":
    num_clusters = 32
    print("Clustering using K-means with {} clusters".format(num_clusters))
    cluster_labels = cluster_lines_kmeans(
        embeddings=test_embeddings_all, num_clusters=num_clusters)
elif cluster_strategy == "aggr_clustering":
    num_clusters = 32
    print("Clustering using Agglomerative Clustering with {} clusters".format(
        num_clusters))
    cluster_labels = cluster_lines_aggr_clustering(
        embeddings=test_embeddings_all, num_clusters=num_clusters)
elif cluster_strategy == "affinity_propagation":
    print("Clustering using Affinity Propagation")
    cluster_labels, num_clusters = cluster_lines_affinity_propagation(
        embeddings=test_embeddings_all)
    print("Found {} clusters.".format(num_clusters))
else:
    print("Invalid clustering strategy. Please use one of: 'kmeans', "
          "'aggr_clustering', 'affinity_propagation'.")
    sys.exit()

# To display frequencies.
if (use_ground_truth_instance_labels):
    count_map_cluster_labels_to_instance_labels = np.zeros(
        [max(cluster_labels) + 1,
         int(max(instance_labels)) + 1])
    for label_index in range(len(cluster_labels)):
        cluster_label = cluster_labels[label_index]
        corresponding_instance_label = int(instance_labels[label_index])
        count_map_cluster_labels_to_instance_labels[
            cluster_label, corresponding_instance_label] += 1

# Map real instances and instances from clustering and display lines with
# colours. Lines with the same colour belong to the same cluster.
if visualizer == 'open3d':
    from tools.visualization import plot_lines_with_open3d
    pcl_lines_open3d_cluster = pcl_lines_for_plot(
        data_lines_world, lines_color=cluster_labels, visualizer='open3d')
    print("Displaying scene with obtained instances")
    plot_lines_with_open3d(pcl_lines_open3d_cluster, "Clusterized instances")
    if (use_ground_truth_instance_labels):
        pcl_lines_open3d_ground_truth = pcl_lines_for_plot(
            data_lines_world,
            lines_color=np.int32(instance_labels),
            visualizer='open3d')
        print("Displaying scene with ground-truth instances")
        plot_lines_with_open3d(pcl_lines_open3d_ground_truth,
                               "Ground-truth instances")

elif visualizer == 'matplotlib':
    from tools.visualization import plot_lines_with_matplotlib
    pcl_lines_matplotlib_cluster = pcl_lines_for_plot(
        data_lines_world, lines_color=cluster_labels, visualizer='matplotlib')
    print("Displaying scene with obtained instances")
    plot_lines_with_matplotlib(pcl_lines_matplotlib_cluster,
                               "Clusterized instances")
    if (use_ground_truth_instance_labels):
        pcl_lines_matplotlib_ground_truth = pcl_lines_for_plot(
            data_lines_world,
            lines_color=np.int32(instance_labels),
            visualizer='matplotlib')
        print("Displaying scene with ground-truth instances")
        plot_lines_with_matplotlib(pcl_lines_matplotlib_ground_truth,
                                   "Ground-truth instances")

# Display embeddings in the feature space, coloured with instance label.
LOG_DIR = os.path.join(log_files_folder,
                       'embedding_logs_{}'.format(name_test_file))
if not os.path.isdir(LOG_DIR): os.makedirs(LOG_DIR)
metadata = os.path.join(LOG_DIR, 'embedding_metadata.tsv')
test_embeddings = tf.Variable(test_embeddings_all, name='test_embeddings')

with open(metadata, 'w') as metadata_file:
    if (use_ground_truth_instance_labels):
        for label in instance_labels:
            metadata_file.write('%d\n' % label)
    else:
        print("Flag 'use_ground_truth_instance_labels' is set to False, "
              "saving the embeddings all with the same label (0). Colouring "
              "in TensorBoard's Projector will therefore not be possible.")
        for _ in range(data_lines_world.shape[0]):
            metadata_file.write('0\n')

with tf.Session() as sess:
    saver = tf.train.Saver([test_embeddings])

    sess.run(test_embeddings.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'test_embeddings.ckpt'))

    config = tensorboard.plugins.projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = test_embeddings.name
    # Link this tensor to its metadata file (i.e., labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    tensorboard.plugins.projector.visualize_embeddings(
        tf.summary.FileWriter(LOG_DIR), config)
