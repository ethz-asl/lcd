import numpy as np
import pandas as pd
import cv2
import os
import sys


def load_frame(path):
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        line_vci_paths = data_lines.values[:, 0]
        line_geometries = data_lines.values[:, 1:15].astype(float)
        line_labels = data_lines.values[:, 15]
        line_class_ids = data_lines.values[:, 17]
        line_count = line_geometries.shape[0]
    except pd.errors.EmptyDataError:
        line_geometries = 0
        line_labels = 0
        line_class_ids = 0
        line_count = 0
        line_vci_paths = 0

    return line_count, line_geometries, line_labels, line_class_ids, line_vci_paths


class Frame:
    def __init__(self, path):
        self.path = path
        self.line_count, self.line_geometries, self.line_labels, self.line_class_ids, self.line_vci_paths = load_frame(path)

    def get_batch(self, batch_size, img_shape, shuffle):
        count = min(batch_size, self.line_count)
        indices = np.arange(count)
        if shuffle:
            np.random.shuffle(indices)

        images = []
        for i in range(len(indices)):
            img = cv2.imread(self.line_vci_paths[i], cv2.IMREAD_UNCHANGED) / 255. * 2. - 1.
            images.append(np.expand_dims(cv2.resize(img, dsize=(img_shape[1], img_shape[0]),
                                                    interpolation=cv2.INTER_LINEAR), axis=0))
            #print(img.shape)
            #print(cv2.resize(img, dsize=(img_shape[1], img_shape[0]),
            #                                        interpolation=cv2.INTER_LINEAR).shape)
        images = np.concatenate(images, axis=0)

        return count, self.line_geometries[indices, :], \
            self.line_labels[indices], self.line_class_ids[indices], images


class LineDataGenerator:
    def __init__(self, files_dir, bg_classes,
                 shuffle=False, data_augmentation=False, mean=np.zeros((3,)), img_shape=(224, 224, 3)):
        # Initialize parameters.
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.pointer = 0
        self.bg_classes = bg_classes
        self.frame_count = 0
        self.frames = []
        self.frame_indices = []
        self.load_frames(files_dir)
        self.mean = mean
        self.img_shape = img_shape

        if self.shuffle:
            self.shuffle_data()

    def load_frames(self, files_dir):
        frame_paths = [os.path.join(files_dir, name) for name in os.listdir(files_dir)
                       if os.path.isfile(os.path.join(files_dir, name))]
        frame_paths.sort()

        self.frame_count = len(frame_paths)
        self.frames = [Frame(path) for path in frame_paths]
        self.frame_indices = np.arange(self.frame_count)

    def get_mean(self):
        mean = np.zeros((3,))
        count = 0
        for frame in self.frames:
            count = count + frame.line_count
            for i in range(frame.line_count):
                mean = mean + frame.line_geometries[i, :3] / 2.
                mean = mean + frame.line_geometries[i, 3:6] / 2.

        return mean / count

    def set_mean(self, mean):
        self.mean = mean

    def shuffle_data(self):
        """ Randomly shuffles the data stored.
        """
        np.random.shuffle(self.frame_indices)

    def reset_pointer(self):
        """ Resets internal pointer to point to the beginning of the stored
            data.
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def set_pointer(self, index):
        """ Sets the internal pointer to point to the given index.

        Args:
            index (int): Index to which the internal pointer should point.
        """
        self.pointer = index

    def next_batch(self, batch_size):
        if self.pointer == self.frame_count:
            self.reset_pointer()

        line_count, line_geometries, line_labels, line_class_ids, line_images = \
            self.frames[self.frame_indices[self.pointer]].get_batch(batch_size, self.img_shape, self.shuffle)
        self.pointer = self.pointer + 1

        # Subtract mean of start and end points.
        # Intuitively, the mean lies some where straight forward, i.e. [0., 0., 3.].
        line_geometries = subtract_mean(line_geometries, self.mean)
        line_geometries = add_length(line_geometries)

        if self.data_augmentation:
            augment_flip(line_geometries)
            augment_global(line_geometries, np.radians(20.), 0.5)

        valid_mask = np.zeros((batch_size,), dtype=bool)
        out_geometries = np.zeros((batch_size, line_geometries.shape[1]))
        out_labels = np.zeros((batch_size,), dtype=int)
        out_classes = np.zeros((batch_size,), dtype=int)
        out_bg = np.zeros((batch_size,), dtype=bool)

        valid_mask[:line_count] = True
        out_geometries[:line_count, :] = line_geometries
        out_labels[:line_count] = line_labels
        out_classes[:line_count] = line_class_ids
        out_bg[np.isin(out_classes, self.bg_classes)] = True

        out_images = line_images
        if line_count < batch_size:
            out_images = np.concatenate([out_images,
                                         np.zeros((batch_size - line_count,
                                                   self.img_shape[0],
                                                   self.img_shape[1],
                                                   self.img_shape[2]))],
                                        axis=0)

        return out_geometries, out_labels, valid_mask, out_bg, out_images


def add_length(line_geometries):
    return np.hstack([line_geometries, np.linalg.norm(line_geometries[:, 3:6] - line_geometries[:, 0:3], axis=1)
                     .reshape((line_geometries.shape[0], 1))])


def subtract_mean(line_geometries, mean):
    # The mean is the mean of all start and end points.
    mean_vec = np.zeros((1, line_geometries.shape[1]))
    mean_vec[0, :3] = mean
    mean_vec[0, 3:6] = mean
    line_geometries = line_geometries - mean_vec

    return line_geometries


def augment_global(line_geometries, angle_deviation, offset_deviation):
    # Rotates all lines in the scene around a random rotation vector.
    # Simulates slight viewpoint changes.

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

    offset = np.random.normal([0, 0, 0], offset_deviation, (3,))

    # Rotate start points and end points
    line_geometries[:, :3] = np.transpose(C.dot(np.transpose(line_geometries[:, :3]))) + offset
    line_geometries[:, 3:6] = np.transpose(C.dot(np.transpose(line_geometries[:, 3:6]))) + offset

    # Rotate normals.
    line_geometries[:, 6:9] = np.transpose(C.dot(np.transpose(line_geometries[:, 6:9])))
    line_geometries[:, 9:12] = np.transpose(C.dot(np.transpose(line_geometries[:, 9:12])))


def augment_local(line_geometries, offset_deviation, length_deviation, ):
    print("To be implemented.")


def augment_flip(line_geometries):
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

