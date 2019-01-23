import argparse
import cv2
import os
import numpy as np

from pickle_dataset import merge_pickled_dictionaries
from sklearn.externals import joblib


def get_train_set_mean(file_path, image_type, read_as_pickle):
    """ Get the train set mean.
    """
    if read_as_pickle:
        if image_type != 'bgr' and image_type != 'bgr-d':
            raise ValueError("Image type should be 'bgr' or 'bgr-d'")
        blue_mean = 0
        green_mean = 0
        red_mean = 0
        depth_mean = 0
        total_images = 0
        pickled_dict = {}
        # Merge dictionaries extracted from all the pickle files in file_path.
        for file_ in file_path:
            temp_dict = joblib.load(file_)
            merge_pickled_dictionaries(pickled_dict, temp_dict)
        for dataset_name in pickled_dict.keys():
            dataset_name_dict = pickled_dict[dataset_name]
            for trajectory_number in dataset_name_dict.keys():
                trajectory_number_dict = dataset_name_dict[trajectory_number]
                for frame_number in trajectory_number_dict.keys():
                    frame_number_dict = trajectory_number_dict[frame_number]
                    for line_number in frame_number_dict['rgb'].keys():
                        total_images += 1
                        # BGR image.
                        img_bgr = frame_number_dict['rgb'][line_number]['img']
                        blue_mean += np.mean(img_bgr[:, :, 0])
                        green_mean += np.mean(img_bgr[:, :, 1])
                        red_mean += np.mean(img_bgr[:, :, 2])
                        # Depth image.
                        img_depth = frame_number_dict['depth'][line_number][
                            'img']
                        depth_mean += np.mean(img_depth)
        if image_type == 'bgr':
            image_mean = np.array([blue_mean, green_mean, red_mean
                                  ]) / total_images
        elif image_type == 'bgr-d':
            image_mean = np.array([blue_mean, green_mean, red_mean, depth_mean
                                  ]) / total_images
        else:
            raise ValueError("Image type should be 'bgr' or 'bgr-d'")
    else:
        image_paths = []
        # In non-pickle mode only consider first file in list.
        with open(file_path[0]) as f:
            lines = f.readlines()
            for l in lines:
                items = l.split()
                image_paths.append(items[0])

        if image_type == 'bgr':
            blue_mean = 0
            green_mean = 0
            red_mean = 0
            for path_rgb in image_paths:
                img_bgr = cv2.imread(path_rgb, cv2.IMREAD_UNCHANGED)
                blue_mean += np.mean(img_bgr[:, :, 0])
                green_mean += np.mean(img_bgr[:, :, 1])
                red_mean += np.mean(img_bgr[:, :, 2])

            image_mean = np.array([blue_mean, green_mean, red_mean
                                  ]) / len(image_paths)
        elif image_type == 'bgr-d':
            blue_mean = 0
            green_mean = 0
            red_mean = 0
            depth_mean = 0
            for path_rgb in image_paths:
                img_bgr = cv2.imread(path_rgb, cv2.IMREAD_UNCHANGED)
                path_depth = path_rgb.replace('rgb', 'depth')
                img_depth = cv2.imread(path_depth, cv2.IMREAD_UNCHANGED)
                blue_mean += np.mean(img_bgr[:, :, 0])
                green_mean += np.mean(img_bgr[:, :, 1])
                red_mean += np.mean(img_bgr[:, :, 2])
                depth_mean += np.mean(img_depth)

            image_mean = np.array([blue_mean, green_mean, red_mean, depth_mean
                                  ]) / len(image_paths)
        else:
            raise ValueError("Image type should be 'bgr' or 'bgr-d'")
    return image_mean


def print_batch_triplets_statistics(triplet_strategy,
                                    batch_index,
                                    instance_labels,
                                    pairwise_dist,
                                    mask_anchor_positive=None,
                                    mask_anchor_negative=None,
                                    valid_triplets=None,
                                    hardest_positive_dist=None,
                                    hardest_negative_dist=None,
                                    hardest_positive_element=None,
                                    hardest_negative_element=None,
                                    write_folder=None):
    """ Print statistics about values extracted from the triplets in a training
        batch.
    Args:
        triplet_strategy (string): Either 'batch_all' or 'batch_hard', strategy
            for the selection of the triplets.
        batch_index (int): Index of the batch.
        instance_labels (Numpy array of shape (batch_size, ) and dtype
            np.float32): Instance labels associated to the elements in the
            training batch.
        pairwise_dist (Numpy array of shape (batch_size, batch_size) and dtype
            np.float32): Matrix of the pairwise distances between the embeddings
            of the elements in the batch (e.g. pairwise_dist[i, j] contains the
            distance between the embedding of the i-th element and the embedding
            of the j-th element in the batch).
        mask_anchor_positive (Numpy array of shape (batch_size, batch_size) and
            dtype bool): mask_anchor_positive[i, j] is True if a valid
            anchor-positive pair can be formed by taking the i-th element in the
            batch as anchor and the j-th element in the batch as positive, False
            otherwise. Must be not None if triplet strategy is 'batch_hard'.
        mask_anchor_negative (Numpy array of shape (batch_size, batch_size) and
            dtype bool): mask_anchor_negative[i, j] is True if a valid
            anchor-negative pair can be formed by taking the i-th element in the
            batch as anchor and the j-th element in the batch as negative, False
            otherwise. Must be not None if triplet strategy is 'batch_hard'.
        valid_triplets (Numpy array of shape (batch_size, batch_size,
            batch_size) and dtype bool): valid_triplets[i, j, k] is True if a
            valid triplet can be formed by taking the i-th element in the batch
            as anchor, the j-th element in the batch as positive and the k-th
            element in the batch as negative, and the triplet is not an easy
            one, i.e. d(a, p) - d(a,n) + margin > 0. It is False otherwise. Must
            be not None if triplet strategy is 'batch_all'.
        hardest_positive_dist (Numpy array of shape (batch_size, 1) and dtype
            np.float32): hardest_positive_dist[i] contains the distance of the
            element in the batch (WLOG, with index j) that is furthest away from
            the i-th element in the batch, with (i, j) being an anchor-positive
            pair. Must be not None if triplet strategy is 'batch_hard'.
        hardest_negative_dist (Numpy array of shape (batch_size, 1) and dtype
            np.float32): hardest_negative_dist[i] contains the distance of the
            element in the batch (WLOG, with index j) that is closest to the
            i-th element in the batch, with (i, j) being an anchor-negative
            pair. Must be not None if triplet strategy is 'batch_hard'.
        hardest_positive_element (Numpy array of shape (batch_size, ) and dtype
            np.int64): hardest_positive_element[i] contains the index of the
            element in the batch that is furthest away from the i-th element in
            the batch, with (i, hardest_positive_element[i]) being an
            anchor-positive pair. Must be not None if triplet strategy is
            'batch_hard'.
        hardest_negative_element (Numpy array of shape (batch_size, ) and dtype
            np.int64): hardest_negative_element[i] contains the index of the
            element in the batch that is closest to the i-th element in
            the batch, with (i, hardest_negative_element[i]) being an
            anchor-negative pair. Must be not None if triplet strategy is
            'batch_hard'.
        write_folder (string): Folder where to output the statistics file.
    """
    if (triplet_strategy not in ['batch_all', 'batch_hard']):
        print("Triplet strategy must be either 'batch_all' or 'batch_hard'.")
        return
    elif (triplet_strategy == 'batch_all'):
        if (valid_triplets is None):
            print("Please pass a valid 'valid_triplets' argument when using "
                  "triplet strategy 'batch_all'")
            return
        if (hardest_positive_dist is None or hardest_negative_dist is None):
            print("Please pass valid 'hardest_positive_dist' and " +
                  "'hardest_negative_dist argument' when using triplet " +
                  "strategy 'batch_all'")
            return
        assert(instance_labels.shape[0] == pairwise_dist.shape[0] == \
               pairwise_dist.shape[1] == valid_triplets.shape[0] == \
               valid_triplets.shape[1] == valid_triplets.shape[2])

        with open(
                os.path.join(write_folder,
                             'batch_{}_stats.txt'.format(batch_index)),
                'w') as f:
            # Print instance labels.
            f.write("***** Labels in batch {} ".format(batch_index) +
                    "[Center point of line (3x), instance label]*****\n")
            for idx, label in enumerate(instance_labels):
                f.write("{0}: {1}\n".format(idx, label))
            # Print valid triplets.
            batch_size = instance_labels.shape[0]
            f.write("***** Valid triplets *****\n")
            for i in range(batch_size):
                for j in range(batch_size):
                    for k in range(batch_size):
                        if (valid_triplets[i, j, k] == True):
                            f.write(
                                "* Triplet ({0}, {1}, {2}) ".format(i, j, k) +
                                "is a valid non-easy anchor-positive-negative "
                                "triplet.\n")
                            f.write(
                                "  d(a, p) = {}".format(pairwise_dist[i, j]) +
                                "  d(a, n) = {}\n".format(pairwise_dist[i, k]))
    elif (triplet_strategy == 'batch_hard'):
        if (mask_anchor_positive is None or mask_anchor_negative is None):
            print("Please pass valid 'mask_anchor_positive' and "
                  "'mask_anchor_negative' arguments when using triplet "
                  "'batch_hard'")
            return
        assert(instance_labels.shape[0] == pairwise_dist.shape[0] == \
               pairwise_dist.shape[1] == mask_anchor_positive.shape[0] == \
               mask_anchor_positive.shape[1] == \
               mask_anchor_negative.shape[0] == \
               mask_anchor_negative.shape[1] == \
               hardest_positive_dist.shape[0] == \
               hardest_negative_dist.shape[0] == \
               hardest_positive_element.shape[0] == \
               hardest_negative_element.shape[0])

        batch_size = instance_labels.shape[0]
        hardest_positive_dist = hardest_positive_dist.reshape(-1)
        hardest_negative_dist = hardest_negative_dist.reshape(-1)
        with open(
                os.path.join(write_folder,
                             'batch_{}_stats.txt'.format(batch_index)),
                'w') as f:
            # Print instance labels.
            f.write("***** Instance labels in the batch *****\n")
            for idx, label in enumerate(instance_labels):
                f.write("{0}: {1}\n".format(idx, label))
            # Print valid anchor-positive pairs.
            f.write("***** Valid anchor-positive pairs *****\n")
            for i in range(batch_size):
                for j in range(batch_size):
                    if (mask_anchor_positive[i, j] == True):
                        f.write("Pair ({0}, {1}) ".format(i, j) +
                                "is a valid " + "anchor-positive pair.\n")
                        f.write("  d(a, p) = {}\n".format(pairwise_dist[i, j]))
            # Print valid anchor-negative pairs.
            f.write("***** Valid anchor-negative pairs *****\n")
            for i in range(batch_size):
                for j in range(batch_size):
                    if (mask_anchor_negative[i, j] == True):
                        f.write("Pair ({0}, {1}) ".format(i, j) +
                                "is a valid " + "anchor-negative pair.\n")
                        f.write("  d(a, n) = {}\n".format(pairwise_dist[i, j]))
            # Print hardest positive and negative distance.
            f.write("***** Hardest positive/negative distance *****\n")
            for i in range(batch_size):
                f.write("* max_p d({0}, p) = d({0}, {1}) =  {2}, ".format(
                    i, hardest_positive_element[i], hardest_positive_dist[i]
                ) + "min_n d({0}, n) = d({0}, {1}) = {2}\n".format(
                    i, hardest_negative_element[i], hardest_negative_dist[i]))
