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
        self.line_count, self.line_geometries, self.line_labels, self.line_class_ids, self.line_vci_paths = \
            load_frame(path)

    def get_batch(self, batch_size, img_shape, shuffle, load_images):
        count = min(batch_size, self.line_count)
        indices = np.arange(count)
        if shuffle:
            np.random.shuffle(indices)

        images = []

        if load_images:
            for i in range(len(indices)):
                img = cv2.imread(self.line_vci_paths[i], cv2.IMREAD_UNCHANGED)
                if img is None:
                    print("WARNING: VIRTUAL CAMERA IMAGE NOT FOUND AT {}".format(self.line_vci_paths[i]))
                    images.append(np.expand_dims(np.zeros(img_shape), axis=0))
                else:
                    images.append(np.expand_dims(cv2.resize(img / 255. * 2. - 1., dsize=(img_shape[1], img_shape[0]),
                                                            interpolation=cv2.INTER_LINEAR), axis=0))
            images = np.concatenate(images, axis=0)

        # print("Path: {}".format(self.line_vci_paths[0]))
        return count, self.line_geometries[indices, :], \
            self.line_labels[indices], self.line_class_ids[indices], images


class LineDataGenerator:
    def __init__(self, files_dir, bg_classes,
                 shuffle=False, data_augmentation=False, mean=np.zeros((3,)), img_shape=(224, 224, 3),
                 sort=False, min_line_count=30, max_cluster_count=15):
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
        self.sort = sort
        self.cluster_counts = []
        self.line_counts = []
        self.cluster_count_file = "cluster_counts"
        self.line_count_file = "line_counts"
        self.skipped_frames = 0
        self.min_line_count = min_line_count
        self.max_cluster_count = max_cluster_count

        if self.shuffle:
            self.shuffle_data()

    def load_frames(self, files_dir):
        frame_paths = [os.path.join(files_dir, name) for name in os.listdir(files_dir)
                       if os.path.isfile(os.path.join(files_dir, name))]
        if not self.shuffle:
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

        print("Dataset completed. Number of skipped frames: {}".format(self.skipped_frames))
        self.skipped_frames = 0
        np.save(self.cluster_count_file, np.array(self.cluster_counts))
        np.save(self.line_count_file, np.array(self.line_counts))

    def set_pointer(self, index):
        """ Sets the internal pointer to point to the given index.

        Args:
            index (int): Index to which the internal pointer should point.
        """
        self.pointer = index

    def next_batch(self, batch_size, load_images):
        if self.pointer == self.frame_count:
            self.reset_pointer()

        line_count, line_geometries, line_labels, line_class_ids, line_images = \
            self.frames[self.frame_indices[self.pointer]].get_batch(batch_size,
                                                                    self.img_shape,
                                                                    self.shuffle,
                                                                    load_images)
        self.pointer = self.pointer + 1

        cluster_count = len(np.unique(line_labels[np.isin(line_class_ids, self.bg_classes, invert=True)]))

        # Write line counts and cluster counts for histogram:
        self.cluster_counts.append(cluster_count)
        self.line_counts.append(line_count)

        if line_count < self.min_line_count:
            # print("Skipping frame because it does not have enough lines")
            self.skipped_frames += 1
            return self.next_batch(batch_size, load_images)

        if cluster_count == 0 or cluster_count > self.max_cluster_count:
            # print("Skipping frame because it has 0 or too many instances.")
            self.skipped_frames += 1
            return self.next_batch(batch_size, load_images)

        out_k = np.zeros((31,))
        out_k[min(30, cluster_count)] = 1.

        # Subtract mean of start and end points.
        # Intuitively, the mean lies some where straight forward, i.e. [0., 0., 3.].
        line_geometries = subtract_mean(line_geometries, self.mean)
        line_geometries = normalize(line_geometries, 2.0)
        line_geometries = add_length(line_geometries)

        if self.data_augmentation and np.random.binomial(1, 0.5):
            augment_flip(line_geometries)
            augment_global(line_geometries, np.radians(15.), 0.2)

        # Sort by x value of leftest point:
        if self.sort:
            sorted_ids = np.argsort(np.min(line_geometries[:, [0, 3]], axis=1))
            line_geometries = line_geometries[sorted_ids, :]
            line_labels = line_labels[sorted_ids]
            line_class_ids = line_class_ids[sorted_ids]
            if load_images:
                line_images = line_images[sorted_ids, :, :, :]

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
        out_bg[line_count:batch_size] = False

        if load_images:
            # TODO: Sort images too.
            out_images = line_images
            if line_count < batch_size:
                out_images = np.concatenate([out_images,
                                             np.zeros((batch_size - line_count,
                                                       self.img_shape[0],
                                                       self.img_shape[1],
                                                       self.img_shape[2]))],
                                            axis=0)
        else:
            out_images = np.zeros((batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]))

        return out_geometries, out_labels, valid_mask, out_bg, out_images, out_k


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


def normalize(line_geometries, std_dev):
    line_geometries[:, 0:6] = line_geometries[:, 0:6] / std_dev

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


def kl_loss_np(prediction, labels, val_mask, bg_mask):
    h_labels = np.expand_dims(labels, axis=-1)
    v_labels = np.transpose(h_labels, axes=(1, 0))

    mask_equal = np.equal(h_labels, v_labels)
    mask_not_equal = np.logical_not(mask_equal)
    # mask_equal = np.expand_dims(mask_equal, axis=-1).astype(float)
    # mask_not_equal = np.expand_dims(mask_not_equal, axis=-1).astype(float)
    mask_equal = mask_equal.astype(float)
    mask_not_equal = mask_not_equal.astype(float)

    h_bg = np.expand_dims(np.logical_not(bg_mask), axis=-1).astype(float)
    v_bg = np.transpose(h_bg, axes=(1, 0))
    mask_not_bg_unexpanded = h_bg * v_bg
    # mask_not_bg = np.expand_dims(h_bg * v_bg, -1)
    mask_not_bg = mask_not_bg_unexpanded

    h_val = np.expand_dims(val_mask, axis=-1).astype(float)
    v_val = np.transpose(h_val, axes=(1, 0))
    mask_val = h_val * v_val
    mask_val_unexpanded = np.copy(mask_val)
    np.fill_diagonal(mask_val_unexpanded, 0.)
    # mask_val = np.expand_dims(mask_val_unexpanded, axis=-1)
    mask_val = mask_val_unexpanded

    h_pred = np.expand_dims(prediction, axis=0)
    v_pred = np.transpose(h_pred, axes=(1, 0, 2))

    d = h_pred * np.log(np.nan_to_num(h_pred / v_pred) + 0.000001)
    d = np.sum(d, axis=-1, keepdims=False)

    equal_loss_layer = mask_equal * d
    not_equal_loss_layer = mask_not_equal * np.maximum(0., 2.0 - d)
    # print(np.max(loss_layer * mask_val * mask_not_bg))
    # print("Compare:")
    # print(np.sum(not_equal_loss_layer * mask_val * mask_not_bg) + np.sum(equal_loss_layer * mask_val * mask_not_bg))


def get_fake_instancing(labels, valid_mask, bg_mask):
    valid_count = np.where(valid_mask == 1)[0].shape[0]
    unique_labels = np.unique(labels[np.where(np.logical_and(valid_mask, np.logical_not(bg_mask)))])
    out = np.zeros((150, 15), dtype=float)

    for i, label in enumerate(unique_labels):
        out[np.where(labels == label), i] = 1.

    kl_loss_np(out, labels, valid_mask, bg_mask)

    return np.expand_dims(out, axis=0)


def generate_data(image_data_generator, max_line_count, line_num_attr, batch_size=1):
    batch_geometries = []
    batch_labels = []
    batch_valid_mask = []
    batch_bg_mask = []
    batch_images = []
    batch_k = []
    batch_fake = []
    batch_unique = []
    batch_counts = []

    for i in range(batch_size):
        # image_data_generator.reset_pointer()
        geometries, labels, valid_mask, bg_mask, images, k = image_data_generator.next_batch(max_line_count,
                                                                                             load_images=False)
        batch_labels.append(labels.reshape((1, max_line_count)))
        batch_geometries.append(geometries.reshape((1, max_line_count, line_num_attr)))
        batch_valid_mask.append(valid_mask.reshape((1, max_line_count)))
        batch_bg_mask.append(bg_mask.reshape((1, max_line_count)))
        batch_images.append(np.expand_dims(images, axis=0))
        batch_k.append(np.expand_dims(k, axis=0))
        batch_fake.append(np.zeros((1, max_line_count, 15)))

        unique_labels = np.unique(labels[np.logical_and(np.logical_not(bg_mask), valid_mask)])
        cluster_count = unique_labels.shape[0]
        if cluster_count >= 15:
            unique_labels = unique_labels[:15]
        else:
            unique_labels = np.pad(unique_labels, (0, 15 - cluster_count), mode='constant', constant_values=-1)
        unique_labels = np.expand_dims(unique_labels, axis=0)
        cluster_count = np.expand_dims(cluster_count, axis=0)
        batch_unique.append(unique_labels)
        batch_counts.append(cluster_count)

    geometries = np.concatenate(batch_geometries, axis=0)
    labels = np.concatenate(batch_labels, axis=0)
    valid_mask = np.concatenate(batch_valid_mask, axis=0)
    bg_mask = np.concatenate(batch_bg_mask, axis=0)
    images = np.concatenate(batch_images, axis=0)
    batch_k = np.concatenate(batch_k, axis=0)
    batch_fake = np.concatenate(batch_fake, axis=0)

    batch_unique = np.concatenate(batch_unique, axis=0)
    batch_counts = np.concatenate(batch_counts, axis=0)

    return {'lines': geometries,
            'labels': labels,
            'valid_input_mask': valid_mask,
            'background_mask': bg_mask,
            # 'images': images,
            'unique_labels': batch_unique,
            'cluster_count': batch_counts,
            'fake': batch_fake}, batch_k


def data_generator(image_data_generator, max_line_count, line_num_attr, batch_size=1):
    while True:
        yield generate_data(image_data_generator, max_line_count, line_num_attr, batch_size)
