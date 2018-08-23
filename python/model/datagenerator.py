import numpy as np
import cv2

"""
Adapted from https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/datagenerator.py
"""


class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=False, image_type='bgr', mean=np.array([22.47429166, 20.13914579, 5.62511388]), scale_size=(227, 227)):

        # Init params
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle
        self.image_type = image_type
        if image_type not in ['bgr', 'bgr-d']:
            raise ValueError("Images should be 'bgr' or 'bgr-d'")
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_class_list(class_list)

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.labels.append([float(i) for i in items[1:]])

            # store total number of data
            self.data_size = len(self.labels)

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = list(self.images)
        labels = list(self.labels)
        self.images = []
        self.labels = []

        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def set_pointer(self, index):
        """
        set pointer to index of the list
        """
        self.pointer = index

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]

        # update pointer
        self.pointer += batch_size

        # Read images
        if self.image_type == 'bgr':
            images = np.ndarray(
                [batch_size, self.scale_size[0], self.scale_size[1], 3])
        elif self.image_type == 'bgr-d':
            images = np.ndarray(
                [batch_size, self.scale_size[0], self.scale_size[1], 4])
        for i in range(len(paths)):
            if self.image_type == 'bgr':
                # bgr image
                img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
            elif self.image_type == 'bgr-d':
                path_rgb = paths[i]
                path_depth = path_rgb.replace('rgb', 'depth')
                img_bgr = cv2.imread(path_rgb, cv2.IMREAD_UNCHANGED)
                img_depth = cv2.imread(path_depth, cv2.IMREAD_UNCHANGED)
                # bgr-d image
                img = np.dstack([img_bgr, img_depth])

            # flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            # rescale image
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)

            # subtract mean
            img -= self.mean

            images[i] = img

        # To numpy array
        labels = np.array(labels, dtype=np.float32)
        assert labels.shape == (batch_size, 4)

        # return array of images and labels
        return images, labels
