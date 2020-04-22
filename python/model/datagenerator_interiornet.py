import numpy as np
import pandas as pd
import cv2
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'tools'))


class ImageDataGenerator:
    def __init__(self,
                 files_list,
                 mean,
                 horizontal_flip=False,
                 shuffle=False,
                 image_type='bgr',
                 scale_size=(227, 227),
                 read_as_pickle=False):
        # Initialize parameters.
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle
        self.image_type = image_type
        if image_type not in ['bgr', 'bgr-d']:
            raise ValueError("Images should be 'bgr' or 'bgr-d'")
        elif image_type == 'bgr':
            assert (mean.shape[0] == 3)
        elif image_type == 'bgr-d':
            assert (mean.shape[0] == 4)
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_files(files_list)

        if self.shuffle:
            self.shuffle_data()

    def read_files(self, files_list):
        # Read all files in list and assign them a trajectory number each.

        class Frame:
            pass

        self.trajectories = []
        self.lines_image_paths = []

        line_count = 0
        line_geometry_data = ()
        line_labels_data = ()

        for i in range(len(files_list)):
            # Read line file.
            try:
                data_lines = pd.read_csv(files_list[i], sep=" ", header=None)
                data_lines = data_lines.values
                file_line_count = data_lines.shape[0]
            except pd.errors.EmptyDataError:
                print("WARNING: File {} not found.".format(files_list[i]))
                file_line_count = 0

            # Extract data from line file.
            line_geometry_data += (data_lines[:, [1,2,3,4,5,6,10,11,12,13,14,15,16,17]], )
            line_labels_data += (data_lines[:, [1,2,3,8]], )
            trajectory = Frame()
            trajectory.line_indices = np.arange(line_count, line_count+file_line_count)
            trajectory.pointer = 0
            self.trajectories.append(trajectory)
            self.lines_image_paths += data_lines[:, 0].tolist()
            line_count += file_line_count

        # Combine gathered line data. The trajectory objects store the indices
        # to their lines.
        self.lines_geometry = np.vstack(line_geometry_data)
        self.lines_labels = np.vstack(line_labels_data)
        # TODO: multiple trajectories.
        self.data_size = self.lines_labels.shape[0]

    def shuffle_data(self):
        """ Randomly shuffles the data stored.
        """
        for trajectory in self.trajectories:
            np.random.shuffle(trajectory.line_indices)

    def reset_pointer(self):
        """ Resets internal pointer to point to the beginning of the stored
            data.
        """
        self.trajectories[0].pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def set_pointer(self, index):
        """ Sets the internal pointer to point to the given index.

        Args:
            index (int): Index to which the internal pointer should point.
        """
        self.trajectories[0].pointer = index

    def next_batch(self, batch_size):
        """ Forms a batch of size batch_size, returning for each line in the
            batch its virtual camera image, its label ([Center point 3x],
            [Instance label 1x]) and its line type.

        Args:
            batch_size (int): Size of the batch to generate.

        Returns:
            images (numpy array of shape (batch_size, self.scale_size[0],
                self.scale_size[1], num_channels), where num_channels is 3 if
                self.image_type is 'bgr' and 4 if self.image_type is 'bgr-d'):
                images[i, :, :, :] contains the virtual image associated to the
                i-th line in the batch.
            labels (numpy array of shape (batch_size, 4) and dtype np.float32):
                labels[i, :] contains the label ([Center point 3x],
                [Instance label 1x]) of the i-th line in the batch.
            line_types (numpy array of shape (batch_size, 1) and dtype
                np.float32): line_types[i, :] contains the line type of the i-th
                line in the batch (0.: Discontinuity line, 1.: Planar line,
                2.: Edge line, 3.: Intersection line).
        """
        if self.data_size - self.trajectories[0].pointer < batch_size:
            self.reset_pointer()

        # Allocate memory for the batch of images.
        if self.image_type == 'bgr':
            images = np.ndarray(
                [batch_size, self.scale_size[1], self.scale_size[0], 3])
        elif self.image_type == 'bgr-d':
            images = np.ndarray(
                [batch_size, self.scale_size[1], self.scale_size[0], 4])

        # Get next batch of image (by retrieving its path), labels and line
        # types.
        indices = self.trajectories[0].line_indices[
            self.trajectories[0].pointer:self.trajectories[0].pointer + batch_size]

        paths = [self.lines_image_paths[i] for i in indices]
        labels = [self.lines_labels[i, :] for i in indices]
        line_types = [self.lines_geometry[i, -1] for i in indices]
        geometries = [np.array(self.lines_geometry[i, :], dtype=np.float32) for i in indices]

        #print("Number of indices: {}".format(len(indices)))
        #print("Batch size: {}".format(batch_size))
        #print("Pointer location: {}".format(self.trajectories[0].pointer))

        #print("WARNING, IMAGES NOT LOADED!")

        # Read images.
        for i in range(len(paths)):
            if self.image_type == 'bgr':
                # BGR image.
                img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
            elif self.image_type == 'bgr-d':
                path_rgb = paths[i]
                path_depth = path_rgb.replace('rgb', 'depth')
                #img_bgr = cv2.imread(path_rgb, cv2.IMREAD_UNCHANGED)
                #img_depth = cv2.imread(path_depth, cv2.IMREAD_UNCHANGED)
                img_bgr = np.zeros((90, 60, 3))
                img_depth = np.zeros((90, 60, 1))

                # BGR-D image.
                img = np.dstack([img_bgr, img_depth])

            # Flip image at random if flag is selected.
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            # Rescale image so as to match desired scale size.
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)

            # Subtract mean of training set.
            img -= self.mean

            # Append image to the output.
            images[i] = img

        # Update pointer.
        self.trajectories[0].pointer += batch_size

        # Convert labels and line types to numpy arrays.
        labels = np.array(labels, dtype=np.int32)
        #line_types = np.asarray(
        #    line_types, dtype=np.float32).reshape(batch_size, -1)
        #geometries = np.array(geometries, dtype=np.float32)

        #assert labels.shape == (batch_size, 4)
        #assert geometries.shape == (batch_size, 14)

        # Return array of images, labels and line types.
        return images, labels[:, -1], line_types, geometries
