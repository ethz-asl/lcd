"""
The datagenerator classes used for training with Keras.
"""
import numpy as np
import pandas as pd
import cv2
import os
import sys
import time
import pickle
from numba import jit, njit
from collections import Counter

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import Sequence


def load_frame(path):
    """
    Function to load preprocessed line files frame-wise.
    How the lines are saved can be viewed in tools/line_file_utils.py
    :param path: The path to the line file of the frame.
    :return: line_count: The number of lines N
             line_geometries: The geometric data of the line in the form of a numpy array with shape (N, 14),
                              dtype=float
             line_labels: The ground truth instance labels of each line, np array of shape (N, 1), dtype=int
             line_class_ids: The ground truth semantic class labels of each line, np array of shape (N, 1), dtype=int
             line_vci_paths: The paths to the virtual camera images of each line in the form of a python array (N)
             transform: The 4x4 transformation matrix of the camera view of this frame.
    """
    data_lines = pd.read_csv(path, sep=" ", header=None)
    line_vci_paths = data_lines.values[:, 0]
    line_geometries = data_lines.values[:, 1:15].astype(float)
    line_labels = data_lines.values[:, 15].astype(int)
    line_class_ids = data_lines.values[:, 17].astype(int)
    line_count = line_geometries.shape[0]

    cam_origin = data_lines.values[0, 19:22].astype(float)
    cam_rotation = data_lines.values[0, 22:26].astype(float)
    transform = get_world_transform(cam_origin, cam_rotation)

    return line_count, line_geometries, line_labels, line_class_ids, line_vci_paths, transform


def get_world_transform(cam_origin, cam_rotation):
    """
    Returns the 4x4 transformation matrix from the camera origin and camera rotation.
    :param cam_origin: Cartesian origin of the camera. Numpy array of shape (3, 1)
    :param cam_rotation: Quaternion of the rotation of the camera. Numpy array of shape (4, 1)
    :return: The 4x4 transformation matrix of the camera.
    """
    x = cam_origin[0]
    y = cam_origin[1]
    z = cam_origin[2]
    e0 = cam_rotation[3]
    e1 = cam_rotation[0]
    e2 = cam_rotation[1]
    e3 = cam_rotation[2]
    T = np.array([[e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2, 2 * e1 * e2 - 2 * e0 * e3, 2 * e0 * e2 + 2 * e1 * e3, x],
                  [2 * e0 * e3 + 2 * e1 * e2, e0 ** 2 - e1 ** 2 + e2 ** 2 - e3 ** 2, 2 * e2 * e3 - 2 * e0 * e1, y],
                  [2 * e1 * e3 - 2 * e0 * e2, 2 * e0 * e1 + 2 * e2 * e3, e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2, z],
                  [0, 0, 0, 1]])
    return T


def set_mean_zero(lines):
    """
    Offsets the lines by their mean to make their mean zero. The mean is the mean of all start and end points.
    :param lines: The geometry array of the lines, a numpy array of shape (N, 14)
    :return: The geometry array of the lines, offset by their mean, a numpy array of shape (N, 14)
    """
    mean = np.mean(np.vstack([lines[:, :3], lines[:, 3:6]]), axis=0)
    lines[:, :3] -= mean
    lines[:, 3:6] -= mean

    return lines


def load_image(path, img_shape):
    """
    Helper function to load and preprocess a virtual camera image.
    :param path: The path to the virtual camera image.
    :param img_shape: The shape the image needs to be resized to.
    :return: The image in the form of a numpy array (img_shape).
    """
    # Load the image with opencv.
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("WARNING: VIRTUAL CAMERA IMAGE NOT FOUND AT {}".format(path))
        return np.zeros(img_shape)
    else:
        # Convert the image from BGR to RGB.
        img = np.flip(img, axis=-1)
        # Preprocess the image for use with the pretrained VGG16 network.
        img = preprocess_input(img)
        # Linear resize of the image to a fixed size.
        img = cv2.resize(img, dsize=(img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
        return img


class Cluster:
    """
    A class to store lines and frame_id of a ground truth cluster.
    """
    def __init__(self, frame_id, frame_path_id, lines):
        """
        Initializes a Cluster object.
        :param frame_id: The frame_id as in the Scene object.
        :param frame_path_id: The frame id as in the name of the frame path. It is stored so that the frame image
                              can be found for visualization purposes.
        :param lines: The indices of the lines of this cluster in its corresponding frame.
        """
        self.lines = lines
        self.frame_id = frame_id
        self.frame_path_id = frame_path_id


class Scene:
    """
    A class to store the data of a scene.
    """
    def __init__(self, path, bg_classes, min_line_count, max_clusters, img_shape,
                 min_line_count_cluster=5, use_random_lighting=False):
        """
        Initializes a Scene object.
        :param path: The path to the scene directory with the preprocessed frame line files.
        :param bg_classes: The semantic classes to be classified as background.
        :param min_line_count: The minimum number of lines in each frame. Frames with less lines are removed.
        :param max_clusters: The maximum number of clusters that the neural network can cluster. Default is 15.
        :param img_shape: The shape of the images for neural network input.
        :param min_line_count_cluster: The minimum number of lines in a cluster used for training. Default is 5
        """
        self.scene_path = path
        self.name = path.split('/')[-1]
        # Find all files located in the scene directory.
        self.frame_paths = [os.path.join(path, name) for name in os.listdir(path)
                            if os.path.isfile(os.path.join(path, name))]
        if use_random_lighting:
            random_path = os.path.join("/", *path.split('/')[:-2], path.split('/')[-2] + "_random", self.name)
            # print(random_path)
            if os.path.exists(random_path):
                self.frame_paths += [os.path.join(random_path, name) for name in os.listdir(random_path)
                                     if os.path.isfile(os.path.join(random_path, name))]
                print("Found random lighting.")
            else:
                print("Random lighting does not exist for scene {}".format(path))
        self.frame_paths.sort(key=lambda x: int(x.split('_')[-1]))
        self.frame_count = len(self.frame_paths)
        self.bg_classes = bg_classes
        self.min_line_count = min_line_count
        self.max_clusters = max_clusters
        self.img_shape = img_shape
        self.valid_frames = []
        self.clusters = []
        self.cluster_labels = []
        self.cluster_classes = []
        self.min_line_count_cluster = min_line_count_cluster
        self.get_frames_info()

    def get_frames_info(self):
        """
        Load the lines from all frames and gather all clusters from each frames, with their corresponding label.
        """
        cluster_labels = []
        cluster_classes = []
        clusters = []

        for i, frame in enumerate(self.frame_paths):
            # Save the index of the frame as in the name so that the frame can be loaded for visualization.
            frame_idx = int(frame.split('_')[-1])
            # Load the lines from the line file.
            count, geometries, labels, classes, vci_paths, transform = load_frame(frame)
            # Find the clusters of all non background lines.
            frame_clusters = Counter(labels[get_non_bg_mask(classes, self.bg_classes)]).most_common()

            # Remove frames with not enough lines (only for cluster description training) or not enough instances.
            if count > self.min_line_count and len(frame_clusters) > 0:
                self.valid_frames.append(i)

            for cluster in frame_clusters:
                # Save clusters with a minimum line count for use in cluster descriptor training.
                cluster_line_count = cluster[1]
                if cluster_line_count >= self.min_line_count_cluster:
                    # Extract all lines, labels belonging to the cluster and the semantic class of the cluster.
                    cluster_label = cluster[0]
                    cluster_class = classes[np.where(labels == cluster_label)][0]
                    cluster_lines = np.where(labels == cluster_label)

                    if cluster_class in self.bg_classes:
                        cluster_label = 0

                    # Save this cluster in a Cluster object.
                    new_cluster = Cluster(i, frame_idx, cluster_lines)
                    # Group the clusters by their instance label.
                    if cluster_label in cluster_labels:
                        clusters[cluster_labels.index(cluster_label)].append(new_cluster)
                    else:
                        cluster_labels.append(cluster_label)
                        cluster_classes.append(cluster_class)
                        clusters.append([new_cluster])

        self.clusters = clusters
        self.cluster_labels = cluster_labels
        self.cluster_classes = cluster_classes

    def get_cluster(self, cluster_id, line_count, shuffle, blacklisted=-1, center=True, forced_choice=None):
        """
        Get the line geometries and images of a random cluster with a certain instance_id for use during training.
        :param cluster_id: The instance id of the cluster as saved in the clusters array.
        :param line_count: The maximum number of lines to return.
        :param shuffle: If the lines should be extracted randomly or deterministically.
        :param blacklisted: Exclude a certain cluster of this instance_id while extracting. For example, if the
                            anchor cluster was chosen, we don't want to reuse it as positive cluster.
        :param center: If the mean of the cluster should be set to zero.
        :param forced_choice: Provide a cluster index to force a cluster choice. If forced_choice=None, a random
                              cluster will be used.
        :return: line_count: The number of lines in the cluster.
                 lines: The geometry of the lines of the cluster, a numpy array of shape (N, 14).
                 cluster_label: The instance label of the cluster.
                 cluster_class: The semantic label of the cluster.
                 images: The virtual camera images of the lines.
                 cluster_choice: The chosen cluster id. This is forced_choice if forced_choice is specified.
        """
        # Choose one of the clusters of the desired instance label.
        if forced_choice is None:
            cluster_count_at_id = len(self.clusters[cluster_id])
            if shuffle:
                cluster_choice = np.random.choice([x for x in range(cluster_count_at_id) if x != blacklisted])
            else:
                cluster_choice = blacklisted + 1
        else:
            cluster_choice = forced_choice

        cluster = self.clusters[cluster_id][cluster_choice]
        cluster_label = self.cluster_labels[cluster_id]
        cluster_class = self.cluster_classes[cluster_id]
        frame_id = cluster.frame_id

        # Load the corresponding frame of the chosen cluster.
        tot_count, lines, labels, classes, vci_paths, t_1 = load_frame(self.frame_paths[frame_id])

        if cluster_class in self.bg_classes:
            # If we have a background label, all background lines are put into the cluster.
            indices = np.where(np.isin(classes, self.bg_classes))[0]
        else:
            indices = np.where(labels == cluster_label)[0]

        # Shuffle the lines if desired.
        if shuffle:
            np.random.shuffle(indices)

        # Choose lines so that the maximum number of lines is not violated.
        lines = lines[indices[:line_count], :]
        # Load and concatenate the images into one numpy array.
        images = np.concatenate([np.expand_dims(load_image(vci_paths[i], self.img_shape), 0)
                                 for i in indices[:line_count]])
        labels = labels[indices[:line_count]]
        classes = classes[indices[:line_count]]

        if cluster_class not in self.bg_classes:
            if np.unique(classes).shape[0] != 1:
                print("WARNING: MORE THAN 1 CLASS DETECTED.")
                print(classes)
                print(labels)
                print(self.frame_paths[frame_id])

        # Set the mean of the lines zero if desired.
        if center:
            lines = set_mean_zero(lines)

        return min(line_count, len(indices)), lines, cluster_label, cluster_class, images, cluster_choice

    def get_frame(self, frame_id, line_count, shuffle):
        """
        Get the line geometries and images of an entire frame.
        :param frame_id: The id of the frame as in the Scene object.
        :param line_count: The maximum number of lines to be loaded.
        :param shuffle: If the data should be randomized.
        :return: line_count: The number of lines in the frame.
                 lines: The geometry of the lines of the frame, a numpy array of shape (N, 14).
                 labels: The instance labels of the lines, a numpy array of shape (N).
                 classes: The semantic label of the cluster, a numpy array of shape (N).
                 images: The virtual camera images of the lines in the form of a numpy array of shape (N, img_shape)
        """
        # Load the lines from the frame file.
        tot_count, lines, labels, classes, vci_paths, t_1 = load_frame(self.frame_paths[frame_id])
        # Load the virtual camera images.
        vcis = [load_image(path, self.img_shape) for path in vci_paths]

        labels = np.stack(labels)
        lines = np.vstack(lines)
        classes = np.stack(classes)

        # Delete lines of small clusters so that the number of clusters is below maximum.
        cluster_line_counts = Counter(np.array(labels)[get_non_bg_mask(classes, self.bg_classes)]).most_common()
        for pair in cluster_line_counts[self.max_clusters:]:
            delete_idx = np.where(labels == pair[0])
            labels = np.delete(labels, delete_idx)
            lines = np.delete(lines, delete_idx, axis=0)
            classes = np.delete(classes, delete_idx)
            for i in np.sort(delete_idx[0])[::-1]:
                del(vcis[i])

        count = labels.shape[0]

        # Shuffle the lines randomly if desired.
        indices = np.arange(count)
        if shuffle:
            np.random.shuffle(indices)
        else:
            rand_state = np.random.get_state()
            np.random.seed(0)
            np.random.shuffle(indices)
            np.random.set_state(rand_state)
        indices = indices[:line_count]

        return count, lines[indices, :], labels[indices], classes[indices], \
            np.concatenate([np.expand_dims(img, axis=0) for img in np.array(vcis)[indices]], axis=0)


def get_non_bg_mask(classes, bg_classes):
    """
    Get the indices of the lines that are not background lines.
    :param classes: The semantic classes of each line in the form of a numpy array
    :param bg_classes: The background classes.
    :return: The indices of the lines that are not background lines.
    """
    return np.where(np.isin(classes, bg_classes, invert=True))


def load_scenes(files_dir, bg_classes, min_line_count, max_line_count, max_clusters, img_shape,
                min_line_count_cluster=5, use_random_lighting=False):
    """
    Loads all scenes of the dataset either from the line files or from a precomputed pickle file.
    When loading the scenes for the first time, this pickle file will be created.
    :param files_dir: The precomputed dataset directory containing the line files.
    :param bg_classes: The semantic labels that are to be classified as background.
    :param min_line_count: The minimum number of lines per frame.
    :param max_line_count: The maximum number of lines per frame.
    :param max_clusters: The maximum number of clusters that can be classified by the neural network.
    :param img_shape: The desired shape of the virtual camera images for the neural network.
    :param min_line_count_cluster: The minimum number of lines per cluster used during the training of the cluster
                                   descriptors.
    :return: A list with all the valid scenes from the dataset.
    """
    pickle_file_path = os.path.join(files_dir, "scenes_{}_{}_{}".format(min_line_count, max_line_count, max_clusters))
    if os.path.isfile(pickle_file_path):
        print("Opening existing scene file.")
        with open(pickle_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Creating new scene file.")
        scene_paths = [os.path.join(files_dir, name) for name in os.listdir(files_dir)
                       if os.path.isdir(os.path.join(files_dir, name))]
        scene_paths.sort()
        scenes = [Scene(path, bg_classes, min_line_count, max_clusters, img_shape,
                        min_line_count_cluster=min_line_count_cluster, use_random_lighting=use_random_lighting)
                  for path in scene_paths]
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(scenes, f)
        return scenes


def create_training_plan(scenes):
    """
    Creates a training plan from the scenes containing all frames from all scenes to facilitate loading.
    Used for the training of the clustering network.
    :param scenes: The list of Scene objects of the dataset.
    :return: A list of tuples containing the (scene id, frame id in scene)
    """
    plan = []
    for i, scene in enumerate(scenes):
        for frame_id in scene.valid_frames:
            plan.append((i, frame_id))

    return plan


class LineDataSequence(Sequence):
    """
    The data generator class for training of the clustering neural network, used for training with the Keras API.
    It inherits from the Sequence class.
    """
    def __init__(self, files_dir, batch_size, bg_classes, shuffle=False, data_augmentation=False,
                 mean=np.array([0., 0., 3.]), img_shape=(64, 96, 3), min_line_count=30, max_line_count=300,
                 max_cluster_count=30, load_images=True, training_mode=True):
        """
        Initializes a LineDataSequence object.
        :param files_dir: The directory of the preprocessed dataset with line files.
        :param batch_size: The batch size.
        :param bg_classes: The semantic labels that are to be classified as background.
        :param shuffle: If the dataset should be randomized for each epoch. For training purposes.
        :param data_augmentation: If data augmentation should be applied.
        :param mean: The mean of the start and end points of all lines in the training set. Default is [0., 0., 3.]
        :param img_shape: The shape of the images for the neural network.
        :param min_line_count: The minimum number of lines per frame.
        :param max_line_count: The maximum number of lines per frame.
        :param max_cluster_count: The maximum number of clusters that can be classified by the neural network.
        :param load_images: If images should be loaded. If not, the images will be left black.
        :param training_mode: If in training mode, lines are removed so that the number of clusters in each frame is
                              smaller than max_cluster_count.
        """
        self.files_dir = files_dir
        self.bg_classes = bg_classes
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.mean = mean
        self.img_shape = img_shape
        self.load_images = load_images
        self.min_line_count = min_line_count
        self.max_line_count = max_line_count
        self.max_cluster_count = max_cluster_count
        self.batch_size = batch_size
        if training_mode:
            # In training mode, the number of clusters should not exceed the network's max number of clusters.
            max_scene_cluster_count = max_cluster_count
        else:
            max_scene_cluster_count = 100000
        self.scenes = load_scenes(files_dir, bg_classes, min_line_count, max_line_count, max_scene_cluster_count,
                                  img_shape, use_random_lighting=True)
        self.training_plan = create_training_plan(self.scenes)
        self.frame_count = len(self.training_plan)
        self.frame_indices = np.arange(self.frame_count, dtype=int)
        self.shuffle_indices()

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        """
        Shuffles the training plan for every epoch if desired.
        """
        if self.shuffle:
            np.random.shuffle(self.frame_indices)

    def __getitem__(self, idx):
        """
        Get a batch of frames used by the neural network.
        :param idx: The index of the frame.
        :return: The data dictionary with the inputs for the neural network:
                'lines', 'labels', 'valid_input_mask', 'background_mask', 'images', 'unique_labels', 'cluster_count'
        """
        # Take elements from the training plan for this batch according to batch_size.
        training_plan = [self.training_plan[i] for i in
                         self.frame_indices[idx * self.batch_size:(idx + 1) * self.batch_size]]

        batch_geometries = []
        batch_labels = []
        batch_valid_mask = []
        batch_bg_mask = []
        batch_images = []
        batch_unique = []
        batch_counts = []

        # If randomizing is not desired, fix the random seed to the index so that it is deterministic.
        if not self.shuffle:
            rand_state = np.random.get_state()
            np.random.seed(idx)

        for element in training_plan:
            # Load the corresponding frame from the Scene class.
            line_count, line_geometries, line_labels, line_class_ids, line_images = \
                self.scenes[element[0]].get_frame(element[1], self.max_line_count, self.shuffle)

            line_geometries = subtract_mean(line_geometries, self.mean)

            # Augment data with slight rotation and image noise and blackout.
            if self.data_augmentation and np.random.binomial(1, 0.5):
                # augment_flip(line_geometries, line_images)
                # (Flip is not desired because the orientation of the lines is mostly deterministic - dependent on the
                #  gradient in the image)
                augment_global(line_geometries, np.radians(15.), 0.3)
            if self.data_augmentation and np.random.binomial(1, 0.5):
                augment_images(line_images)

            # Normalize the lines by dividing the start and end points by 2.
            line_geometries = normalize(line_geometries, 2.0)
            # Calculate and append the length to the data.
            line_geometries = add_length(line_geometries)

            # Zero padding of lines with no input.
            out_valid_mask = np.zeros((self.max_line_count,), dtype=bool)
            out_geometries = np.zeros((self.max_line_count, line_geometries.shape[1]))
            out_labels = np.zeros((self.max_line_count,), dtype=int)
            out_classes = np.zeros((self.max_line_count,), dtype=int)
            out_bg = np.zeros((self.max_line_count,), dtype=bool)
            out_valid_mask[:line_count] = True
            out_geometries[:line_count, :] = line_geometries
            out_classes[:line_count] = line_class_ids
            out_bg[line_count:self.max_line_count] = False
            out_bg[np.isin(out_classes, self.bg_classes)] = True

            # We reserve label 0 for background lines and move all other labels.
            for i, label in enumerate(np.unique(line_labels)):
                out_labels[np.where(line_labels == label)] = i + 1
            out_labels[out_bg] = 0

            if self.load_images:
                # Zero padding of the images.
                out_images = line_images
                if line_count < self.max_line_count:
                    out_images = np.concatenate([out_images,
                                                 np.zeros((self.max_line_count - line_count,
                                                           self.img_shape[0],
                                                           self.img_shape[1],
                                                           self.img_shape[2]))], axis=0)
            else:
                # Black out images if they are not desired.
                out_images = np.zeros((self.max_line_count, self.img_shape[0], self.img_shape[1], self.img_shape[2]))

            # Add the data to the batch.
            batch_labels.append(np.expand_dims(out_labels, axis=0))
            batch_geometries.append(np.expand_dims(out_geometries, axis=0))
            batch_valid_mask.append(np.expand_dims(out_valid_mask, axis=0))
            batch_bg_mask.append(np.expand_dims(out_bg, axis=0))
            batch_images.append(np.expand_dims(out_images, axis=0))

            # Compute the unique clusters for usage in metrics.
            unique_labels = np.unique(out_labels[np.logical_and(np.logical_not(out_bg), out_valid_mask)])
            cluster_count = unique_labels.shape[0]
            if cluster_count >= self.max_cluster_count:
                unique_labels = unique_labels[:self.max_cluster_count]
            else:
                unique_labels = np.pad(unique_labels, (0, self.max_cluster_count - cluster_count),
                                       mode='constant', constant_values=-1)
            if cluster_count == 0:
                print("WARNING: NO CLUSTERS IN FRAME.")
            unique_labels = np.expand_dims(unique_labels, axis=0)
            cluster_count = np.expand_dims(cluster_count, axis=0)
            batch_unique.append(unique_labels)
            batch_counts.append(cluster_count)

        # Combine the batches to feed into the neural network.
        geometries = np.concatenate(batch_geometries, axis=0)
        labels = np.concatenate(batch_labels, axis=0)
        valid_mask = np.concatenate(batch_valid_mask, axis=0)
        bg_mask = np.concatenate(batch_bg_mask, axis=0)
        images = np.concatenate(batch_images, axis=0)
        unique = np.concatenate(batch_unique, axis=0)
        counts = np.concatenate(batch_counts, axis=0)

        if not self.shuffle:
            np.random.set_state(rand_state)

        return {'lines': geometries,
                'labels': labels,
                'valid_input_mask': valid_mask,
                'background_mask': bg_mask,
                'images': images,
                'unique_labels': unique,
                'cluster_count': counts}, labels

    def __len__(self):
        return int(np.ceil(self.frame_count / self.batch_size))


def create_training_plan_clusters(scenes, min_num_clusters=2):
    """
    Create the training plan for the cluster descriptor training. This lists all clusters from all frames
    from all scenes, where another cluster of the same instance label is present in the scene.
    In short, we list all clusters that appear in different frames.
    :param scenes: The Scene objects of the dataset.
    :param min_num_clusters: The minimum number appearances of the cluster in the scene (Default: 2).
    :return: plan: The training plan containing a list of tuples (scene_id, instance_id, white_list, floor_id). The
                   white_list is used to determine which scene is not in the same floor.
             floor_ids: The floor ids of a scene. The floor_id is the id of the floor the scene is located in. In the
                        InteriorNet dataset, this is determined by the numbers before the first underscore. E.g.
                        3FO4IDEI1LAV_Dining_room and 3FO4IDEI1LAV_Bedroom are in the floor 3FO4IDEI1LAV.
    """
    class_histogram = []

    plan = []

    # The floor names are the name of the floors with multiple scenes, e.g. 3FO4IMXSHSVT
    floor_names = []
    curr_id = 0
    floor_ids = np.zeros((len(scenes),), dtype=int)
    for i, scene in enumerate(scenes):
        if len(scene.cluster_labels) > 0:
            floor_name = scene.name.split('_')[0]
            if floor_name in floor_names:
                floor_ids[i] = floor_names.index(floor_name)
            else:
                floor_names.append(floor_name)
                floor_ids[i] = curr_id
                curr_id += 1

            # The white list is the the list of scenes that are not the scene the current cluster is in,
            # to prevent trying to distinct between instances that are the same.
            white_list = []
            for j, other_scene in enumerate(scenes):
                other_floor_name = other_scene.name.split('_')[0]
                if len(other_scene.cluster_labels) > 0:
                    if floor_name != other_floor_name:
                        white_list.append(j)
            for j, cluster_label in enumerate(scene.cluster_classes):
                if len(scene.clusters[j]) >= min_num_clusters:
                    plan.append((i, j, white_list, floor_ids[i]))
                    class_histogram.append(scene.cluster_classes[j])

    print("Found {} scenes in {} floors.".format(len(scenes), len(floor_names)))

    return plan, floor_ids


class ClusterDataSequence(Sequence):
    """
    The data generator class for cluster descriptor training used for training with the Keras API.
    It inherits from the Sequence class.
    """
    def __init__(self, files_dir, batch_size, bg_classes, shuffle=False, data_augmentation=False,
                 img_shape=(64, 96, 3), min_line_count=5, max_line_count=50,
                 load_images=True, training_mode=True):
        """
        Initializes a ClusterDataSequence object.
        :param files_dir: The directory of the preprocessed dataset with line files.
        :param batch_size: The batch size.
        :param bg_classes: The semantic labels that are to be classified as background.
        :param shuffle: If the dataset should be randomized for each epoch. For training purposes.
        :param data_augmentation: If data augmentation should be applied.
        :param img_shape: The shape of the images for the neural network.
        :param min_line_count: The minimum number of lines per frame.
        :param max_line_count: The maximum number of lines per frame.
        :param load_images: If images should be loaded. If not, the images will be left black.
        :param training_mode: If in training mode, only clusters with two or more occurences are loaded. Otherwise
                              every cluster is loaded.
        """
        self.files_dir = files_dir
        self.bg_classes = bg_classes
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.img_shape = img_shape
        self.load_images = load_images
        self.min_line_count = min_line_count
        self.max_line_count = max_line_count
        self.batch_size = batch_size
        self.scenes = load_scenes(files_dir, bg_classes, min_line_count, max_line_count, 1000, img_shape,
                                  min_line_count_cluster=min_line_count, use_random_lighting=True)
        if training_mode:
            min_num_clusters = 2
        else:
            min_num_clusters = 1
        self.training_plan, self.floor_ids = create_training_plan_clusters(self.scenes, min_num_clusters)
        self.cluster_count = len(self.training_plan)
        self.cluster_indices = np.arange(self.cluster_count, dtype=int)
        self.shuffle_indices()

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        """
        Shuffles the training plan for every epoch if desired.
        """
        if self.shuffle:
            np.random.shuffle(self.cluster_indices)

    def process_cluster(self, line_count, cluster_lines, cluster_images):
        """
        Utility function to process a cluster with data augmentation, ready feed into the neural network.
        :param line_count: The number of lines in the cluster.
        :param cluster_lines: The geometry of the lines in the form of a numpy array with shape (N, 14)
        :param cluster_images: The images of each line in the form of a numpy array with shape (N, image_shape)
        :return: geometries: The zero padded geometry and augmented vector for the cluster (with length) with shape
                             (N_max, 15)
                 valid_mask: The mask with the valid lines with shape (N_max)
                 images: The zero padded images and augmented vector of the lines with shape (N_max, image_shape)
        """
        # Augment the data with random rotations and image blackout or noising.
        if self.data_augmentation and np.random.binomial(1, 0.5):
            augment_global(cluster_lines, np.radians(8.), 0.0)
            augment_images(cluster_images)

        # Normalize the data by dividing it by 2.
        line_geometries = normalize(cluster_lines, 2.0)
        # Add the length information to the geometry vector.
        line_geometries = add_length(line_geometries)

        # Zero padding of the geometries and creating the valid mask.
        out_valid_mask = np.zeros((self.max_line_count,), dtype=bool)
        out_geometries = np.zeros((self.max_line_count, line_geometries.shape[1]))
        out_valid_mask[:line_count] = True
        out_geometries[:line_count, :] = line_geometries

        if self.load_images:
            # Zero padding of the images vector.
            out_images = cluster_images
            if line_count < self.max_line_count:
                out_images = np.concatenate([out_images,
                                             np.zeros((self.max_line_count - line_count,
                                                       self.img_shape[0],
                                                       self.img_shape[1],
                                                       self.img_shape[2]))], axis=0)
        else:
            # Black out images if they are not desired.
            out_images = np.zeros((self.max_line_count, self.img_shape[0], self.img_shape[1], self.img_shape[2]))

        return np.expand_dims(out_geometries, axis=0), np.expand_dims(out_valid_mask, axis=0), \
               np.expand_dims(out_images, axis=0)

    def __getitem__(self, idx):
        """
        Get a batch of triplets of clusters used for training. The anchor is given and the positive and negative cluster
        are chosen at random (or deterministic if desired).
        :param idx: The index of training plan, containg the anchor clusters.
        :return: The triplet input dictionary for the neural network:
                 'lines_a', 'valid_input_mask_a', 'images_a', 'lines_p', 'valid_input_mask_p', 'images_p', 'lines_n'
                 'valid_input_mask_n', 'images_n', 'ones'
        """
        # Take elements from the training plan for this batch according to batch_size. Each element corresponds to the
        # anchor cluster.
        training_plan = [self.training_plan[i] for i in
                         self.cluster_indices[idx * self.batch_size:(idx + 1) * self.batch_size]]

        batch_geometries_a = []
        batch_valid_mask_a = []
        batch_images_a = []
        batch_geometries_p = []
        batch_valid_mask_p = []
        batch_images_p = []
        batch_geometries_n = []
        batch_valid_mask_n = []
        batch_images_n = []
        batch_ones = []

        # Make deterministic if desired.
        rand_state = np.random.get_state()
        if not self.shuffle:
            np.random.seed(idx)

        # Iterate over all batch elements.
        for element in training_plan:
            # The white_list is the list of scenes that are not on the same floor as the anchor cluster.
            white_list = element[2]

            # Obtain the lines, images, etc. of the anchor cluster.
            line_count, cluster_lines, cluster_label, cluster_class, cluster_images, cluster_id = \
                self.scenes[element[0]].get_cluster(element[1], self.max_line_count, self.shuffle, center=True)
            # Process the data of the anchor cluster for the neural network.
            geometries_a, valid_mask_a, images_a = self.process_cluster(line_count, cluster_lines, cluster_images)

            # Obtain the lines, images, etc. of the positive cluster. The cluster_id of the anchor cluster is
            # blacklisted.
            line_count_p, lines_p, label_p, class_p, images_p, id_p = \
                self.scenes[element[0]].get_cluster(element[1], self.max_line_count, True,
                                                    blacklisted=cluster_id, center=True)
            assert(id_p != cluster_id)
            # Process the data of the positive cluster for the neural network.
            geometries_p, valid_mask_p, images_p = self.process_cluster(line_count_p, lines_p, images_p)

            # Randomly choose another scene, and a cluster in that scene. This scene is chosen to not be on the same
            # floor as the scene containing the anchor cluster.
            other_scene_id = np.random.choice(white_list)
            other_scene = self.scenes[other_scene_id]
            # With probability 0.5, try to choose a cluster with the same class.
            if np.random.binomial(1, 0.5):
                same_classes_clusters = np.where(np.array(other_scene.cluster_classes) == cluster_class)[0]
                if len(same_classes_clusters) > 0:
                    other_cluster_id = np.random.choice(same_classes_clusters)
                else:
                    other_cluster_id = np.random.choice(np.arange(len(other_scene.cluster_labels)))
            else:
                other_cluster_id = np.random.choice(np.arange(len(other_scene.cluster_labels)))
            # Obtain the lines, images, etc. of the negative cluster.
            line_count_n, lines_n, label_n, class_n, images_n, id_n = \
                other_scene.get_cluster(other_cluster_id, self.max_line_count, True,
                                        center=True)
            # Process the data of the negative cluster for the neural network.
            geometries_n, valid_mask_n, images_n = self.process_cluster(line_count_n, lines_n, images_n)

            # Stack the data into a batch.
            batch_geometries_a.append(geometries_a)
            batch_valid_mask_a.append(valid_mask_a)
            batch_images_a.append(images_a)
            batch_geometries_p.append(geometries_p)
            batch_valid_mask_p.append(valid_mask_p)
            batch_images_p.append(images_p)
            batch_geometries_n.append(geometries_n)
            batch_valid_mask_n.append(valid_mask_n)
            batch_images_n.append(images_n)
            batch_ones.append(np.expand_dims(np.ones((1, 1)), axis=0))

        # Concatenate the batch data for input into the neural networks.
        geometries_a = np.concatenate(batch_geometries_a, axis=0)
        valid_mask_a = np.concatenate(batch_valid_mask_a, axis=0)
        images_a = np.concatenate(batch_images_a, axis=0)
        geometries_p = np.concatenate(batch_geometries_p, axis=0)
        valid_mask_p = np.concatenate(batch_valid_mask_p, axis=0)
        images_p = np.concatenate(batch_images_p, axis=0)
        geometries_n = np.concatenate(batch_geometries_n, axis=0)
        valid_mask_n = np.concatenate(batch_valid_mask_n, axis=0)
        images_n = np.concatenate(batch_images_n, axis=0)
        ones = np.concatenate(batch_ones, axis=0)

        if not self.shuffle:
            np.random.set_state(rand_state)

        return {'lines_a': geometries_a,
                'valid_input_mask_a': valid_mask_a,
                'images_a': images_a,
                'lines_p': geometries_p,
                'valid_input_mask_p': valid_mask_p,
                'images_p': images_p,
                'lines_n': geometries_n,
                'valid_input_mask_n': valid_mask_n,
                'images_n': images_n,
                'ones': ones}, ones

    def __len__(self):
        """
        :return: The number of batches in the dataset.
        """
        return int(np.ceil(self.cluster_count / self.batch_size))


def add_length(line_geometries):
    """
    Appends the length information to the geometry vector.
    :param line_geometries: The geometry vector of the lines without length, with shape (N, 14)
    :return: The geometry vector of the lines with length, with shape (N, 15)
    """
    return np.hstack([line_geometries, np.linalg.norm(line_geometries[:, 3:6] - line_geometries[:, 0:3], axis=1)
                     .reshape((line_geometries.shape[0], 1))])


def subtract_mean(line_geometries, mean):
    """
    Substracts the mean vector from all start and end points.
    :param line_geometries: The geometry vector of the lines without length, with shape (N, 14)
    :param mean: The vector to be subtracted from the start and end points.
    :return: The geometry vector of the lines without length with subtracted mean, with shape (N, 14)
    """
    # The mean is the mean of all start and end points.
    mean_vec = np.zeros((1, line_geometries.shape[1]))
    mean_vec[0, :3] = mean
    mean_vec[0, 3:6] = mean
    line_geometries = line_geometries - mean_vec

    return line_geometries


def normalize(line_geometries, std_dev):
    """
    Normalizes the start and end points by dividing them with the standard deviation.
    :param line_geometries: The geometry vector of the lines, with shape (N, 14)
    :param std_dev: The value the start and end points are to be divided by.
    :return:
    """
    line_geometries[:, 0:6] = line_geometries[:, 0:6] / std_dev

    return line_geometries


def augment_global(line_geometries, angle_deviation, offset_deviation):
    """
    Rotates all lines in the scene around a random rotation vector to simulate slight viewpoint changes.
    A random offset ist also applied.
    :param line_geometries: The geometry vector of the lines to be augmented, with shape (N, 14)
    :param angle_deviation: The standard deviation of the rotation vector.
    :param offset_deviation: The standard deviation of the offset vector.
    """

    theta = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, 2 * np.pi)
    x = np.sin(theta) * np.cos(psi)
    y = np.sin(theta) * np.sin(psi)
    z = np.cos(theta)
    angle = np.random.normal(0, angle_deviation)
    s = np.sin(angle)
    c = np.cos(angle)
    C = np.array([[x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                  [x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s],
                  [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c]])

    # Offset all start and end points.
    offset = np.random.normal([0, 0, 0], offset_deviation, (3,))

    # Rotate start points and end points
    line_geometries[:, :3] = np.transpose(C.dot(np.transpose(line_geometries[:, :3]))) + offset
    line_geometries[:, 3:6] = np.transpose(C.dot(np.transpose(line_geometries[:, 3:6]))) + offset

    # Rotate normals.
    line_geometries[:, 6:9] = np.transpose(C.dot(np.transpose(line_geometries[:, 6:9])))
    line_geometries[:, 9:12] = np.transpose(C.dot(np.transpose(line_geometries[:, 9:12])))


def augment_images(images):
    """
    Augments the images by applying one of the following:
    1. Increasing or decreasing the brightness of all images.
    2. Blacking out all images.
    3. Adding gaussian noise to all images.
    :param images: The virtual camera images vector of a frame or cluster to be augmented, with shape (N, image_shape)
    """
    if np.random.binomial(1, 0.1):
        darkness = np.random.normal(1., 0.2)
        images[:, :, :, :] = images * darkness
    elif np.random.binomial(1, 0.1):
        images[:, :, :, :] = 0.
    elif np.random.binomial(1, 0.1):
        images[:, :, :, :] = np.random.normal(0., 30., images.shape)


def augment_flip(line_geometries, images):
    """
    Augemnts the geometries and the images by randomly flipping the lines and images.
    :param line_geometries: The geometry vector of the lines to be augmented, with shape (N, 14)
    :param images: The virtual camera images vector of a frame or cluster to be augmented, with shape (N, image_shape)
    """
    for i in range(line_geometries.shape[0]):
        if np.random.binomial(1, 0.5):
            buffer_start = np.copy(line_geometries[i, :3])
            line_geometries[i, :3] = line_geometries[i, 3:6]
            line_geometries[i, 3:6] = buffer_start
            buffer_normal_1 = np.copy(line_geometries[i, 6:9])
            line_geometries[i, 6:9] = line_geometries[i, 9:12]
            line_geometries[i, 9:12] = buffer_normal_1
            buffer_open_start = np.copy(line_geometries[i, 12])
            line_geometries[i, 12] = line_geometries[i, 13]
            line_geometries[i, 13] = buffer_open_start

            images[i, :, :, :] = np.flip(images[i, :, :, :], axis=2)

