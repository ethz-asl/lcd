import tensorflow as tf


def iou_metric(labels, unique_labels, cluster_counts, bg_mask, valid_mask, max_clusters):
    """
    The metric function used for the iou metric during clustering. For future use, the NMI metric should be implemented
    and used directly here.
    :param labels: The ground truth label tensor of the lines.
    :param unique_labels: The unique labels tensor that contains all the unique labels in the labels tensor.
    :param cluster_counts: The number of clusters per batch element.
    :param bg_mask: The background masks.
    :param valid_mask: The valid mask specifying the lines with valid inputs.
    :param max_clusters: The maximum number of clusters.
    :return: The iou metric function for use with the Keras API.
    """
    def iou(y_true, y_pred):
        # Choose only instance lines. This has to be updated to include all background lines as well.
        mask = tf.logical_and(tf.logical_not(bg_mask), valid_mask)
        mask = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1)

        # Expand the dimensions of the ground truth labels and the unique labels so that they can be compared directly
        # through logical operations.
        gt_labels = tf.expand_dims(tf.expand_dims(labels, axis=-1), axis=-1)
        unique_gt_labels = tf.expand_dims(tf.expand_dims(unique_labels, axis=1), axis=-1)
        # Repeat with the predicted labels.
        pred_labels = tf.expand_dims(tf.expand_dims(tf.argmax(y_pred[:, :, :], axis=-1, output_type=tf.dtypes.int32),
                                                    axis=-1), axis=-1)
        # Available instances are 1 to 16, 0 is background.
        unique_pred_labels = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(1, max_clusters + 1, dtype='int32'),
                                                                          axis=0), axis=0), axis=0)

        # Create a representation of the ground truth and predicted labels that is invariant to the sorting.
        gt_matrix = tf.equal(gt_labels, unique_gt_labels)
        pred_matrix = tf.equal(pred_labels, unique_pred_labels)

        # Compute the intersections of every ground truth cluster with every predicted cluster.
        intersections = tf.cast(tf.logical_and(tf.logical_and(gt_matrix, pred_matrix), mask), dtype='float32')

        # Compute the unions of every ground truth cluster with every predicted cluster.
        unions = tf.cast(tf.logical_and(tf.logical_or(gt_matrix, pred_matrix), mask), dtype='float32')
        intersections = tf.reduce_sum(intersections, axis=1)
        unions = tf.reduce_sum(unions, axis=1)

        # Compute the intersection over union of every ground truth cluster with every predicted cluster.
        # The maximum intersection over union is chosen for each ground truth cluster.
        iou_out = tf.reduce_max(tf.math.divide_no_nan(intersections, unions), axis=-1)
        # The arithmetic mean of this intersection over union is used as metric.
        iou_out = tf.math.divide_no_nan(tf.reduce_sum(iou_out, axis=-1, keepdims=True),
                                        tf.cast(cluster_counts, dtype='float32'))

        return tf.reduce_mean(iou_out)

    return iou


def bg_accuracy_metrics(bg_mask, valid_mask, threshold=0.3):
    """
    Create a metric used to measure the background classification performance.
    :param bg_mask: The ground truth background mask.
    :param valid_mask: The mask of lines containing valid input.
    :param threshold: The decision threshold for background classification (currently not used, because the argmax
                      determines if it is a background line or not)
    :return: The function for the background accuracy metric used in the Keras API.
    """
    num_valid_1d = tf.reduce_sum(tf.cast(valid_mask, dtype='float32'), axis=-1)

    def bg_tp(y_true, y_pred):
        bg_tensor = y_pred[:, :, 0]
        bg_correct = tf.logical_and(valid_mask, tf.logical_and(bg_mask, tf.greater(bg_tensor, threshold)))
        num_correct = tf.reduce_sum(tf.cast(bg_correct, dtype='float32'), axis=-1)
        ratio_correct = tf.math.divide_no_nan(num_correct, num_valid_1d)

        return tf.reduce_mean(ratio_correct)

    def bg_fp(y_true, y_pred):
        bg_tensor = y_pred[:, :, 0]
        bg_fp = tf.logical_and(valid_mask, tf.logical_and(tf.logical_not(bg_mask), tf.greater(bg_tensor, threshold)))
        num_fp = tf.reduce_sum(tf.cast(bg_fp, dtype='float32'), axis=-1)
        ratio_fp = tf.math.divide_no_nan(num_fp, num_valid_1d)

        return tf.reduce_mean(ratio_fp)

    def bg_acc(y_true, y_pred):
        pred_bg = tf.argmax(y_pred, axis=-1)
        pred_bg = tf.equal(pred_bg, 0)
        corrects = tf.logical_and(valid_mask, tf.equal(pred_bg, bg_mask))
        num_corrects = tf.reduce_sum(tf.cast(corrects, dtype='float32'), axis=-1)

        return tf.reduce_mean(num_corrects)

    return [bg_acc]


def get_kl_losses_and_metrics(labels_tensor, valid_mask, bg_mask, num_lines):
    """
    Get the KL divergence based clustering losses which consist of the background classification loss and the
    pairwise instancing losses.
    :param labels_tensor: The ground truth label tensor of the lines.
    :param valid_mask: The mask of lines containing valid input.
    :param bg_mask: The ground truth background mask.
    :param num_lines: The maximum number of lines.
    :return:
    """
    # Expand the label tensor in two dimensions to obtain a cross comparison matrix.
    h_labels = tf.expand_dims(labels_tensor, axis=-1)
    v_labels = tf.transpose(h_labels, perm=(0, 2, 1))

    mask_equal = tf.equal(h_labels, v_labels)
    mask_not_equal = tf.not_equal(h_labels, v_labels)

    h_bg = tf.expand_dims(tf.logical_not(bg_mask), axis=-1)
    v_bg = tf.transpose(h_bg, perm=(0, 2, 1))
    mask_not_bg = tf.logical_and(h_bg, v_bg)

    # Mask to remove non valid lines from adding to the loss.
    h_val = tf.expand_dims(valid_mask, axis=-1)
    v_val = tf.transpose(h_val, perm=(0, 2, 1))
    mask_val = tf.logical_and(h_val, v_val)
    mask_val = tf.linalg.set_diag(mask_val, tf.zeros(tf.shape(mask_val)[0:-1], dtype='bool'))

    loss_mask = tf.logical_and(mask_val, mask_not_bg)

    num_valid_1d = tf.reduce_sum(tf.cast(valid_mask, dtype='float32'), axis=-1, keepdims=True)

    def cluster_loss(y_true, y_pred):
        # Extract the tensor for instancing.
        cluster_tensor = y_pred[:, :, 1:]
        extended_pred = tf.expand_dims(cluster_tensor, axis=2)
        h_pred = extended_pred
        v_pred = tf.transpose(extended_pred, perm=(0, 2, 1, 3))
        # Compute the KL divergence for all pairs.
        d = h_pred * tf.math.log(tf.math.divide_no_nan(h_pred, v_pred + 1e-10) + 1e-10)
        d = tf.reduce_sum(d, axis=-1, keepdims=False)

        # Apply the KL divergence loss according to the formula in the paper.
        equal_loss = tf.where(tf.logical_and(mask_equal, loss_mask), d, 0.)
        not_equal_loss = tf.where(tf.logical_and(mask_not_equal, loss_mask),
                                  tf.maximum(0., 2.0 - d), 0.)
        output = equal_loss + not_equal_loss
        output = tf.reduce_mean(output, axis=-1)
        output = tf.math.divide_no_nan(output, num_valid_1d) * num_lines
        return output

    def bg_loss(y_true, y_pred):
        # Extract the tensor responsible for the background label.
        bg_tensor = y_pred[:, :, 0]
        # Calculate the binary cross entropy for background classification. This needs to be changed in the future
        # to correspond to the formula in the paper. Although this is equivalent, the resulting gradient will be
        # different. (1. - bg_tensor) should be replaced by the sum of the instancing bins.
        d = tf.where(bg_mask, -tf.math.log(bg_tensor), -tf.math.log(1. - bg_tensor))
        d = tf.where(valid_mask, d, 0.)
        return d

    def loss(y_true, y_pred):
        # Add the losses to obtain the final loss.
        return cluster_loss(y_true, y_pred) + bg_loss(y_true, y_pred)

    return loss, [cluster_loss, bg_loss]


def triplet_metrics(embedding_a, embedding_p, embedding_n, margin=0.2):
    """
    Get metrics used during the training of the cluster descriptor network. These are the ratio of hard triplets,
    the ratio of semi hard triplets and the ratio of easy triplets.
    :param embedding_a: The embedding tensor of the anchor descriptor.
    :param embedding_p: The embedding tensor of the positive descriptor.
    :param embedding_n: The embedding tensor of the negative descriptor.
    :param margin: The margin used in the training of the triplets.
    :return: A list of the triplet metric functions.
    """
    d_a_p = tf.norm(embedding_p - embedding_a, axis=-1)
    d_a_n = tf.norm(embedding_n - embedding_a, axis=-1)
    term = d_a_p - d_a_n + margin

    def hard_triplets(y_true, y_pred):
        return tf.cast(tf.logical_and(tf.greater(term, 0.), tf.greater(d_a_p, d_a_n)), dtype='float32')

    def semi_hard_triplets(y_true, y_pred):
        return tf.cast(tf.logical_and(tf.greater(term, 0.), tf.greater(d_a_n, d_a_p)), dtype='float32')

    def easy_triplets(y_true, y_pred):
        return tf.cast(tf.greater(0., term), dtype='float32')

    return [hard_triplets, semi_hard_triplets, easy_triplets]


def triplet_loss(embedding_a, embedding_p, embedding_n, margin=0.2):
    """
    Get the triplet loss.
    :param embedding_a: The embedding tensor of the anchor descriptor.
    :param embedding_p: The embedding tensor of the positive descriptor.
    :param embedding_n: The embedding tensor of the negative descriptor.
    :param margin: The margin used in the training of the triplets.
    :return: The triplet loss function.
    """
    def loss(y_true, y_pred):
        out = tf.maximum(tf.norm(embedding_p - embedding_a, axis=-1) -
                         tf.norm(embedding_n - embedding_a, axis=-1) + margin, 0.)
        return out

    return loss


def debug_metrics(tensor):
    """
    Metrics used for debug purposes. Currently not in use.
    """
    def d_sum(y_true, y_pred):
        return tf.reduce_sum(tensor, axis=-1)

    def d_l1(y_true, y_pred):
        return tf.reduce_sum(tf.abs(tensor), axis=-1)

    def d_std(y_true, y_pred):
        return tf.math.reduce_std(tensor, axis=-1)

    def d_max(y_true, y_pred):
        return tf.reduce_max(tensor, axis=-1)

    def d_min(y_true, y_pred):
        return tf.reduce_min(tensor, axis=-1)

    return [d_sum, d_l1, d_std, d_max, d_min]
