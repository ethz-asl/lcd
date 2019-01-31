import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.externals import joblib

from pickle_dataset import merge_pickled_dictionaries


def get_train_set_mean(file_path, image_type, read_as_pickle):
    """ Gets the mean of the training set.
    Args:
        file_path (string):
            * If read_as_pickle == True: List of the paths of the pickle files
                from which the data should be loaded.
            * If read_as_pickle == False: List of the paths of the text files
                that contain the location and other information of the data that
                should be loaded.
        image_type (string): Either 'bgr' or 'bgr-d'. Type of the input images.
        read_as_pickle (bool): If True, the input data is read from a set of
            pickle files, otherwise it is read by loading the images from disk
            based on the locations specified in files_list.

    Returns:
        image_mean (numpy array of shape (num_channels, ) and dtype=np.float32,
            where num_channels is 3 if image_type is 'bgr' and 4 if image_type
            is 'bgr-d'): Channelwise mean of the images in the dataset.
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
            raise ValueError("Image type should be 'bgr' or 'bgr-d'.")

    return image_mean


def write_triplet_image_to_file(
        image_anchor, image_positive, image_negative, anchor_index,
        positive_index, negative_index, anchor_positive_dist,
        anchor_negative_dist, set_mean, batch_index, epoch_index, write_folder):
    """ Writes to file an image showing the virtual camera images of the lines
        in the triplet selected and short statistics about the triplet.
    Args:
        image_anchor (numpy array of shape (scale_size[0], scale_size[1],
            num_channels), where scale_size is defined when the batches are
            generated in model/datagenerator.py and num_channels is either 3 or
            4, depending on whether the input images are respectively BGR or
            BGR-D): Virtual camera image of the line that is the anchor element
            in the input triplet.
        image_positive (numpy array of shape (scale_size[0], scale_size[1],
            num_channels), where scale_size is defined when the batches are
            generated in model/datagenerator.py and num_channels is either 3 or
            4, depending on whether the input images are respectively BGR or
            BGR-D): Virtual camera image of the line that is the positive
            element in the input triplet.
        image_negative (numpy array of shape (scale_size[0], scale_size[1],
            num_channels), where scale_size is defined when the batches are
            generated in model/datagenerator.py and num_channels is either 3 or
            4, depending on whether the input images are respectively BGR or
            BGR-D): Virtual camera image of the line that is the negative
            element in the input triplet.
        anchor_index (int): Index of the anchor element in the batch.
        positive_index (int): Index of the positive element in the batch.
        negative_index (int): Index of the negative element in the batch.
        anchor_positive_dist (float): Distance between the anchor and the
            positive element in the triplet.
        anchor_negative_dist (float): Distance between the anchor and the
            negative element in the triplet.
        set_mean (numpy array of shape (num_channels, ) and dtype np.float32,
            where num_channels is either 3 or 4, depending on whether the input
            images are respectively BGR or BGR-D): Channelwise mean of all the
            images in the dataset from which the batch was extracted.
        batch_index (int): Index of the batch.
        epoch_index (int): Index of the epoch.
        write_folder (string): Folder where to output the image. The latter will
            be created in the subfolder
            'epoch_<epoch_index>/batch_<batch_index>'.
    """
    assert (image_anchor.shape == image_positive.shape == image_negative.shape)
    assert (image_anchor.shape[2] == set_mean.shape[0])
    # Create directory if nonexistent (based on
    # https://stackoverflow.com/a/12517490).
    file_path = os.path.join(write_folder,
                             'epoch_{0}/batch_{1}/a_{2}-p_{3}-n_{4}.png'.format(
                                 epoch_index, batch_index, anchor_index,
                                 positive_index, negative_index))
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    # Make the local variable images copy of the passed arguments, to avoid
    # modifying them if using +=, /=, etc., which get evaluated to __iadd__,
    # __itruediv__, etc. methods, which modify the original referenced arrays.
    image_anchor = np.copy(image_anchor)
    image_positive = np.copy(image_positive)
    image_negative = np.copy(image_negative)
    # Since matplotlib requires the images to be RGB and with intensities
    # normalized between 0 and 1, the mean of the dataset - previously
    # subtracted - is added to the images, and the latter are normalized.
    image_anchor += set_mean
    image_positive += set_mean
    image_negative += set_mean
    image_anchor[:, :, :3] /= 255.
    image_positive[:, :, :3] /= 255.
    image_negative[:, :, :3] /= 255.
    # Make BGR/BGR-D images RGB.
    image_anchor = image_anchor[:, :, [2, 1, 0]]
    image_positive = image_positive[:, :, [2, 1, 0]]
    image_negative = image_negative[:, :, [2, 1, 0]]
    # Draw image.
    plt.figure()
    ax_anchor = plt.subplot(1, 3, 1)
    ax_anchor.imshow(image_anchor)
    ax_anchor.set_title('Anchor = {}'.format(anchor_index))
    ax_anchor.axis('off')
    ax_positive = plt.subplot(1, 3, 2)
    ax_positive.imshow(image_positive)
    ax_positive.set_title('Positive = {}'.format(positive_index))
    ax_positive.axis('off')
    ax_negative = plt.subplot(1, 3, 3)
    ax_negative.imshow(image_negative)
    ax_negative.set_title('Negative = {}'.format(negative_index))
    ax_negative.axis('off')
    ax_positive.text(0, image_positive.shape[0] * 1.2,
                     'd(a, p) = {:.5f}'.format(anchor_positive_dist))
    ax_negative.text(0, image_negative.shape[0] * 1.2,
                     'd(a, n) = {:.5f}'.format(anchor_negative_dist))
    # Save image to file.
    plt.savefig(file_path)
    # Close figure.
    plt.close()


def print_batch_triplets_statistics(
        triplet_strategy,
        images,
        set_mean,
        batch_index,
        epoch_index,
        write_folder,
        labels,
        pairwise_dist,
        loss,
        mask_anchor_positive=None,
        mask_anchor_negative=None,
        valid_positive_triplets=None,
        sum_valid_positive_triplets_anchor_positive_dist=None,
        num_anchor_positive_pairs_with_valid_positive_triplets=None,
        lambda_regularization=None,
        regularization_term=None,
        hardest_positive_dist=None,
        hardest_negative_dist=None,
        hardest_positive_element=None,
        hardest_negative_element=None):
    """ Prints statistics about values extracted from the triplets in a training
        batch and writes to file images containing the virtual images of the
        lines in the triplets selected.
    Args:
        triplet_strategy (string): Either 'batch_all', 'batch_hard' or
            'batch_all_wohlhart_lepetit', strategy for the selection of the
            triplets.
        images (numpy array of shape (batch_size, scale_size[0], scale_size[1],
            num_channels), where scale_size is defined when the batches are
            generated in model/datagenerator.py and num_channels is either 3 or
            4, depending on whether the input images are respectively BGR or
            BGR-D): Virtual camera images in the batch.
        set_mean (numpy array of shape (num_channels, ) and dtype np.float32,
            where num_channels is either 3 or 4, depending on whether the input
            images are respectively BGR or BGR-D): Channelwise mean of all the
            images in the dataset from which the batch was extracted.
        batch_index (int): Index of the batch.
        epoch_index (int): Index of the epoch.
        write_folder (string): Folder where to output the statistics file. The
            latter will be created in the subfolder
            'epoch_<epoch_index>/batch_<batch_index>'.
        labels (numpy array of shape (batch_size, 7) and dtype np.float32):
            Labels associated to the elements in the training batch. The labels
            are in the format
                [Start point (3x), End point (3x), Instance label (1x)].
        pairwise_dist (numpy array of shape (batch_size, batch_size) and dtype
            np.float32): Matrix of the pairwise distances between the embeddings
            of the elements in the batch (e.g. pairwise_dist[i, j] contains the
            distance between the embedding of the i-th element and the embedding
            of the j-th element in the batch).
        loss (float): Loss obtained for the batch.
        mask_anchor_positive (numpy array of shape (batch_size, batch_size) and
            dtype bool): mask_anchor_positive[i, j] is True if a valid
            anchor-positive pair can be formed by taking the i-th element in the
            batch as anchor and the j-th element in the batch as positive, False
            otherwise. Must be not None if triplet strategy is 'batch_hard'.
        mask_anchor_negative (numpy array of shape (batch_size, batch_size) and
            dtype bool): mask_anchor_negative[i, j] is True if a valid
            anchor-negative pair can be formed by taking the i-th element in the
            batch as anchor and the j-th element in the batch as negative, False
            otherwise. Must be not None if triplet strategy is 'batch_hard'.
        valid_positive_triplets (numpy array of shape (batch_size, batch_size,
            batch_size) and dtype bool): valid_positive_triplets[i, j, k] is
            True if a valid triplet can be formed by taking the i-th element in
            the batch as anchor, the j-th element in the batch as positive and
            the k-th element in the batch as negative, and the triplet is not an
            easy one, i.e. d(a, p) - d(a,n) + margin is not less than 0. It is
            False otherwise. Must be not None if triplet strategy is
            'batch_all' or 'batch_all_wohlhart_lepetit'.
        sum_valid_positive_triplets_anchor_positive_dist (float): Sum of the
            anchor-positive distances d(a, p) over all anchor-positive pairs
            such that there exists at least one element n such that (a, p, n) is
            a valid positive (i.e., non-easy) triplet. Must be not None if
            triplet strategy is 'batch_all' or 'batch_all_wohlhart_lepetit'.
        num_anchor_positive_pairs_with_valid_positive_triplets (float): Number
            of anchor-positive pairs (a, p) such that there exists at least one
            element n such that (a, p, n) is a valid positive (i.e., non-easy)
            triplet. Must be not None if triplet strategy is 'batch_all' or
            'batch_all_wohlhart_lepetit'.
        lambda_regularization (float): Regularization parameter in the
            'batch-all' loss. Must be not None if triplet strategy is
            'batch_all' or 'batch_all_wohlhart_lepetit'.
        regularization_term (float): Regularization term in the loss for the
            current batch. Must be not None if triplet strategy is 'batch_all'
            or 'batch_all_wohlhart_lepetit'.
        hardest_positive_dist (numpy array of shape (batch_size, 1) and dtype
            np.float32): hardest_positive_dist[i] contains the distance of the
            element in the batch (WLOG, with index j) that is furthest away from
            the i-th element in the batch, with (i, j) being an anchor-positive
            pair. Must be not None if triplet strategy is 'batch_hard'.
        hardest_negative_dist (numpy array of shape (batch_size, 1) and dtype
            np.float32): hardest_negative_dist[i] contains the distance of the
            element in the batch (WLOG, with index j) that is closest to the
            i-th element in the batch, with (i, j) being an anchor-negative
            pair. Must be not None if triplet strategy is 'batch_hard'.
        hardest_positive_element (numpy array of shape (batch_size, ) and dtype
            np.int64): hardest_positive_element[i] contains the index of the
            element in the batch that is furthest away from the i-th element in
            the batch, with (i, hardest_positive_element[i]) being an
            anchor-positive pair. Must be not None if triplet strategy is
            'batch_hard'.
        hardest_negative_element (numpy array of shape (batch_size, ) and dtype
            np.int64): hardest_negative_element[i] contains the index of the
            element in the batch that is closest to the i-th element in
            the batch, with (i, hardest_negative_element[i]) being an
            anchor-negative pair. Must be not None if triplet strategy is
            'batch_hard'.
    """
    if (triplet_strategy not in [
            'batch_all', 'batch_hard', 'batch_all_wohlhart_lepetit'
    ]):
        print("Triplet strategy must be either 'batch_all', 'batch_hard' or "
              "'batch_all_wohlhart_lepetit'.")
        return
    elif (triplet_strategy == 'batch_all' or
          triplet_strategy == 'batch_all_wohlhart_lepetit'):
        if (valid_positive_triplets is None):
            print("Please pass a valid 'valid_positive_triplets' argument when "
                  "using triplet strategy 'batch_all' or "
                  "'batch_all_wohlhart_lepetit'.")
            return
        if (sum_valid_positive_triplets_anchor_positive_dist is None):
            print("Please pass a valid "
                  "'sum_valid_positive_triplets_anchor_positive_dist' argument "
                  "when using triplet strategy 'batch_all' or "
                  "'batch_all_wohlhart_lepetit'.")
            return
        if (num_anchor_positive_pairs_with_valid_positive_triplets is None):
            print("Please pass a valid "
                  "'num_anchor_positive_pairs_with_valid_positive_triplets' "
                  "argument when using triplet strategy 'batch_all' or "
                  "'batch_all_wohlhart_lepetit'.")
            return
        if (lambda_regularization is None or regularization_term is None):
            print("Please pass valid 'lambda_regularization' and "
                  "'regularization_term' arguments when using triplet strategy "
                  "'batch_all' or 'batch_all_wohlhart_lepetit'.")
            return
        assert(images.shape[0] == labels.shape[0] == pairwise_dist.shape[0] == \
               pairwise_dist.shape[1] == valid_positive_triplets.shape[0] == \
               valid_positive_triplets.shape[1] == \
               valid_positive_triplets.shape[2])
        assert (labels.shape[1] == 7)

        # Create directory if nonexistent (based on
        # https://stackoverflow.com/a/12517490).
        file_path = os.path.join(write_folder,
                                 'epoch_{0}/batch_{1}/stats.txt'.format(
                                     epoch_index, batch_index))
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        with open(file_path, 'w') as f:
            # Print coordinates of endpoints and instance labels.
            f.write("***** Lines in batch {} *****\n".format(batch_index))
            for idx, label in enumerate(labels):
                f.write("{0}: Start point {1}\n".format(idx, label[:3]))
                f.write("{0}End point {1}\n".format(" " * (len(str(idx)) + 2),
                                                    label[3:6]))
                f.write("{0}Instance label {1}\n".format(
                    " " * (len(str(idx)) + 2), label[-1]))
            # Print loss and auxiliary information.
            f.write("***** Loss *****\n")
            f.write("The loss obtained for the batch is {}.\n".format(loss))
            f.write("The regularization term for the loss, with a ")
            f.write("regularization hyperparameter of {}, is ".format(
                lambda_regularization))
            f.write("{}.\n".format(regularization_term))
            f.write("The sum of all anchor-positive distances d(a, p) over ")
            f.write("all the anchor-positive pairs (a, p) such that there ")
            f.write("exists at least one element n such that (a, p, n) is a ")
            f.write("valid positive (i.e., non-easy) triplet is {}.\n".format(
                sum_valid_positive_triplets_anchor_positive_dist))
            f.write("The number of anchor-positive pairs (a, p) such that ")
            f.write("there exists at least one element n such that (a, p, n) ")
            f.write("is a valid positive (i.e., non-easy) triplet is ")
            f.write("{}.\n".format(
                int(num_anchor_positive_pairs_with_valid_positive_triplets)))
            # Print valid triplets.
            batch_size = labels.shape[0]
            f.write("***** Valid triplets *****\n")
            for i in range(batch_size):
                for j in range(batch_size):
                    for k in range(batch_size):
                        if (valid_positive_triplets[i, j, k] == True):
                            f.write(
                                "* Triplet ({0}, {1}, {2}) ".format(i, j, k) +
                                "is a valid non-easy anchor-positive-negative "
                                "triplet.\n")
                            f.write(
                                "  d(a, p) = {}".format(pairwise_dist[i, j]) +
                                "  d(a, n) = {}\n".format(pairwise_dist[i, k]))
                            # Generate image that displays the triplet
                            # considered.
                            write_triplet_image_to_file(
                                image_anchor=images[i],
                                image_positive=images[j],
                                image_negative=images[k],
                                anchor_index=i,
                                positive_index=j,
                                negative_index=k,
                                anchor_positive_dist=pairwise_dist[i, j],
                                anchor_negative_dist=pairwise_dist[i, k],
                                set_mean=set_mean,
                                batch_index=batch_index,
                                epoch_index=epoch_index,
                                write_folder=write_folder)
    elif (triplet_strategy == 'batch_hard'):
        if (mask_anchor_positive is None or mask_anchor_negative is None):
            print("Please pass valid 'mask_anchor_positive' and "
                  "'mask_anchor_negative' arguments when using triplet "
                  "'batch_hard'.")
            return
        if (hardest_positive_dist is None or hardest_negative_dist is None or
                hardest_positive_element is None or
                hardest_negative_element is None):
            print("Please pass valid 'hardest_positive_dist', " +
                  "'hardest_negative_dist', 'hardest_positive_element', " +
                  "'hardest_negative_element' arguments when using triplet " +
                  "strategy 'batch_hard'.")
            return
        assert(images.shape[0] == labels.shape[0] == pairwise_dist.shape[0] == \
               pairwise_dist.shape[1] == \
               mask_anchor_positive.shape[0] == \
               mask_anchor_positive.shape[1] == \
               mask_anchor_negative.shape[0] == \
               mask_anchor_negative.shape[1] == \
               hardest_positive_dist.shape[0] == \
               hardest_negative_dist.shape[0] == \
               hardest_positive_element.shape[0] == \
               hardest_negative_element.shape[0])
        assert (labels.shape[1] == 7)

        batch_size = labels.shape[0]
        hardest_positive_dist = hardest_positive_dist.reshape(-1)
        hardest_negative_dist = hardest_negative_dist.reshape(-1)

        # Create directory if nonexistent (based on
        # https://stackoverflow.com/a/12517490).
        file_path = os.path.join(write_folder,
                                 'epoch_{0}/batch_{1}/stats.txt'.format(
                                     epoch_index, batch_index))
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        with open(file_path, 'w') as f:
            # Print coordinates of endpoints and instance labels.
            f.write("***** Lines in batch {} *****\n".format(batch_index))
            for idx, label in enumerate(labels):
                f.write("{0}: Start point {1}\n".format(idx, label[:3]))
                f.write("{0}End point {1}\n".format(" " * (len(str(idx)) + 2),
                                                    label[3:6]))
                f.write("{0}Instance label {1}\n".format(
                    " " * (len(str(idx)) + 2), label[-1]))
            # Print loss.
            f.write("***** Loss *****\n")
            f.write("The loss obtained for the batch is {}".format(loss))
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
                # Generate image that displays the triplet considered.
                write_triplet_image_to_file(
                    image_anchor=images[i],
                    image_positive=images[hardest_positive_element[i]],
                    image_negative=images[hardest_negative_element[i]],
                    anchor_index=i,
                    positive_index=hardest_positive_element[i],
                    negative_index=hardest_negative_element[i],
                    anchor_positive_dist=hardest_positive_dist[i],
                    anchor_negative_dist=hardest_negative_dist[i],
                    set_mean=set_mean,
                    batch_index=batch_index,
                    epoch_index=epoch_index,
                    write_folder=write_folder)
