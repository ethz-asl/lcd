"""
Adapted from https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
"""

import tensorflow as tf


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: Tensor of shape (batch_size, embed_dim).
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: Tensor of shape (batch_size, batch_size).
    """
    # Get the dot product between all embeddings
    # Shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of
    # `dot_product`. This also provides more numerical stability (the diagonal
    # of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # Shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * \
        dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put
    # everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on
        # the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly
        # 0.0.
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels, use_dist=False, max_dist=1.0):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct, have
       same instance label and - if use_dist is set to true - are close in
       space.
    Args:
        labels: tf.float32 `Tensor` with shape [batch_size, 4].
        use_dist: True if two the distance between the center points of the line
                  should be used to determine whether a pair is an
                  anchor-positive pair.
        max_dist: max distance between the centers of the lines for the lines to
                  be considered as close in space.
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size].
    """
    # Check that i and j are distinct.
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i, -1] == labels[j, -1]
    # Use broadcasting where the 1st argument has shape (1, batch_size) and the
    # 2nd (batch_size, 1).
    instance_labels = tf.cast(labels[:, -1], tf.int32)
    instance_equal = tf.equal(
        tf.expand_dims(instance_labels, 0), tf.expand_dims(instance_labels, 1))

    # Return the boolean mask where a pair is considered anchor-positive if
    # the indices are different (different elements) and the instances are
    # equal.
    mask = tf.logical_and(indices_not_equal, instance_equal)

    if (use_dist):
        # Also the distance information should be used.
        # Check if i and j are close in space (i.e., if the center points of the
        # lines are close in space).
        distances = _pairwise_distances(labels[:, :3])
        close_in_space = tf.less_equal(distances, max_dist)

        # Combine the previous mask and the distance-related mask.
        mask = tf.logical_and(mask, close_in_space)

    return mask


def _get_anchor_negative_triplet_mask(labels, use_dist=False, max_dist=1.0):
    """Return a 2D mask where mask[a, n] is True if a and n have distinct labels
       or - if use_dist is set to true - a and n are far away in space.
    Args:
        labels: tf.float32 `Tensor` with shape [batch_size, 4].
        use_dist: True if two the distance between the center points of the line
                  should be used to determine whether a pair is an
                  anchor-negative pair.
        max_dist: max distance between the centers of the lines for the lines to
                  be considered as close in space.
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size].
    """
    # Check if labels[i, -1] != labels[k, -1]
    # Use broadcasting where the 1st argument has shape (1, batch_size) and the
    # 2nd (batch_size, 1).
    instance_labels = tf.cast(labels[:, -1], tf.int32)
    instance_equal = tf.equal(
        tf.expand_dims(instance_labels, 0), tf.expand_dims(instance_labels, 1))
    instance_not_equal = tf.logical_not(instance_equal)

    # Return the boolean mask where a pair is considered anchor-negative if
    # the instances are not equal.
    mask = instance_not_equal

    if (use_dist):
        # Also the distance information should be used.
        # Check if i and j are close in space (i.e., if the center points of the
        # lines are close in space).
        distances = _pairwise_distances(labels[:, :3])
        close_in_space = tf.less_equal(distances, max_dist)
        far_away_in_space = tf.logical_not(close_in_space)

        # Combine the previous mask and the distance-related mask.
        mask = tf.logical_or(mask, far_away_in_space)

    return mask


def _get_triplet_mask(labels, use_dist=False, max_dist=1.0):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is
    valid.
    A triplet (i, j, k) is valid if:
        1. i, j, k are distinct
        2. (i, j) positive, (i, k) negative
    Args:
        labels: tf.float32 `Tensor` with shape [batch_size, 4].
        use_dist: True if two the distance between the center points of the
                  lines should be used to determine a pair whether an
                  anchor-positive/anchor-negative pair.
        max_dist: max distance between the centers of the lines for the lines to
                  be considered as close in space.
    """
    # Check that i, j and k are distinct.
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(
        tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i, -1] == labels[j, -1]
    # Use broadcasting where the 1st argument has shape (1, batch_size) and the
    # 2nd (batch_size, 1).
    i_j_positive = _get_anchor_positive_triplet_mask(
        labels, use_dist=use_dist, max_dist=max_dist)
    i_k_negative = _get_anchor_negative_triplet_mask(
        labels, use_dist=use_dist, max_dist=max_dist)

    i_j_positive_mask = tf.expand_dims(i_j_positive, 2)
    i_k_negative_mask = tf.expand_dims(i_k_negative, 1)
    valid_mask = tf.logical_and(i_j_positive_mask, i_k_negative_mask)

    mask = tf.logical_and(distinct_indices, valid_mask)

    return mask


def batch_hardest_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a
    triplet.
    Args:
        labels: Labels of the batch, of size (batch_size, 4).
        embeddings: Tensor of shape (batch_size, embed_dim).
        margin: Margin for triplet loss.
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix. If false, output is the pairwise euclidean
                 distance matrix.
    Returns:
        triplet_loss: Scalar tensor containing the triplet loss.
    """
    # The label for each line must be of shape 4: 3 values for the center of the
    # line and one for the instance.
    assert labels.shape[1] == 4, "{}".format(labels.shape)

    # Get the pairwise distance matrix.
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive.
    # First, we need to get a mask for every valid positive (they should have
    # same label).
    mask_anchor_positive_bool = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive_bool)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and
    # label(a) == label(p)).
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # Shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(
        anchor_positive_dist, axis=1, keepdims=True)
    hardest_positive_element = tf.argmax(anchor_positive_dist, axis=1)
    tf.summary.scalar("hardest_positive_dist",
                      tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative.
    # First, we need to get a mask for every valid negative (they should have
    # different labels).
    mask_anchor_negative_bool = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative_bool)

    # We add the maximum value in each row to the invalid negatives
    # (label(a) == label(n)).
    max_anchor_negative_dist = tf.reduce_max(
        pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + \
        max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # Shape (batch_size,).
    hardest_negative_dist = tf.reduce_min(
        anchor_negative_dist, axis=1, keepdims=True)
    hardest_negative_element = tf.argmin(anchor_negative_dist, axis=1)
    tf.summary.scalar("hardest_negative_dist",
                      tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss.
    triplet_loss = tf.maximum(
        hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss.
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss, mask_anchor_positive_bool, mask_anchor_negative_bool, \
           hardest_positive_dist, hardest_negative_dist, \
           hardest_positive_element, hardest_negative_element, pairwise_dist


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive
    ones.
    Args:
        labels: Labels of the batch, of size (batch_size, 4)
        embeddings: Tensor of shape (batch_size, embed_dim)
        margin: Margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix. If false, output is the pairwise euclidean
                 distance matrix.
    Returns:
        triplet_loss: Scalar tensor containing the triplet loss.
    """
    # The label for each line must be of shape 4: 3 values for the center of the
    # line and one for the instance.
    assert labels.shape[1] == 4, "{}".format(labels.shape)

    # Get the pairwise distance matrix.
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # Shape (batch_size, batch_size, 1).
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(
        anchor_positive_dist.shape)
    # Shape (batch_size, 1, batch_size).
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(
        anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size).
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i,
    # positive=j, negative=k.
    # Use broadcasting where the 1st argument has shape
    # (batch_size, batch_size, 1) and the 2nd (batch_size, 1, batch_size).
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets (where label(a) != label(p) or
    # label(n) == label(a) or a == p).
    mask = tf.to_float(_get_triplet_mask(labels))
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets).
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0).
    valid_triplets_bool = tf.greater(triplet_loss, 1e-16)
    valid_triplets = tf.to_float(valid_triplets_bool)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    tf.summary.scalar("num_valid_triplets", num_valid_triplets)

    fraction_positive_triplets = num_positive_triplets / \
        (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets.
    triplet_loss = tf.reduce_sum(triplet_loss) / \
        (num_positive_triplets + 1e-16)

    return (triplet_loss, fraction_positive_triplets, valid_triplets_bool,
            pairwise_dist)
