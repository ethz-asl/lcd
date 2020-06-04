import tensorflow as tf


def debug_metrics(tensor):
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


def iou_metric(labels, unique_labels, cluster_counts, bg_mask, valid_mask, max_clusters):
    def iou(y_true, y_pred):
        mask = tf.logical_and(tf.logical_not(bg_mask), valid_mask)
        mask = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1)

        gt_labels = tf.expand_dims(tf.expand_dims(labels, axis=-1), axis=-1)
        unique_gt_labels = tf.expand_dims(tf.expand_dims(unique_labels, axis=1), axis=-1)
        pred_labels = tf.expand_dims(tf.expand_dims(tf.argmax(y_pred[:, :, 1:], axis=-1, output_type=tf.dtypes.int32),
                                                    axis=-1), axis=-1)
        unique_pred_labels = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(0, 15, dtype='int32'),
                                                                          axis=0), axis=0), axis=0)

        gt_matrix = tf.equal(gt_labels, unique_gt_labels)
        pred_matrix = tf.equal(pred_labels, unique_pred_labels)

        intersections = tf.cast(tf.logical_and(tf.logical_and(gt_matrix, pred_matrix), mask), dtype='float32')

        unions = tf.cast(tf.logical_and(tf.logical_or(gt_matrix, pred_matrix), mask), dtype='float32')
        intersections = tf.reduce_sum(intersections, axis=1)
        unions = tf.reduce_sum(unions, axis=1)

        iou_out = tf.reduce_max(tf.math.divide_no_nan(intersections, unions), axis=-1)
        iou_out = tf.math.divide_no_nan(tf.reduce_sum(iou_out, axis=-1, keepdims=True),
                                        tf.cast(cluster_counts, dtype='float32'))

        return tf.reduce_mean(iou_out)

    return iou


def bg_accuracy_metrics(bg_mask, valid_mask, threshold=0.3):
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


def get_kl_losses_and_metrics(instancing_tensor, labels_tensor, valid_mask, bg_mask, num_lines):
    h_labels = tf.expand_dims(labels_tensor, axis=-1)
    v_labels = tf.transpose(h_labels, perm=(0, 2, 1))

    mask_equal = tf.equal(h_labels, v_labels)
    mask_not_equal = tf.not_equal(h_labels, v_labels)

    h_bg = tf.expand_dims(tf.logical_not(bg_mask), axis=-1)
    v_bg = tf.transpose(h_bg, perm=(0, 2, 1))
    mask_not_bg = tf.logical_and(h_bg, v_bg)

    h_val = tf.expand_dims(valid_mask, axis=-1)
    v_val = tf.transpose(h_val, perm=(0, 2, 1))
    mask_val = tf.logical_and(h_val, v_val)
    mask_val = tf.linalg.set_diag(mask_val, tf.zeros(tf.shape(mask_val)[0:-1], dtype='bool'))

    loss_mask = tf.logical_and(mask_val, mask_not_bg)

    num_valid = tf.reduce_sum(tf.cast(loss_mask, dtype='float32'), axis=(1, 2), keepdims=True)
    num_valid_1d = tf.reduce_sum(tf.cast(valid_mask, dtype='float32'), axis=-1, keepdims=True)

    # cluster_tensor = instancing_tensor[:, :, 1:]
    # bg_tensor = instancing_tensor[:, :, 0]

    def cluster_loss(y_true, y_pred):
        cluster_tensor = y_pred[:, :, 1:]
        extended_pred = tf.expand_dims(cluster_tensor, axis=2)
        h_pred = extended_pred
        v_pred = tf.transpose(extended_pred, perm=(0, 2, 1, 3))
        d = h_pred * tf.math.log(tf.math.divide_no_nan(h_pred, v_pred + 1e-10) + 1e-10)
        d = tf.reduce_sum(d, axis=-1, keepdims=False)

        equal_loss = tf.where(tf.logical_and(mask_equal, loss_mask), d, 0.)
        not_equal_loss = tf.where(tf.logical_and(mask_not_equal, loss_mask),
                                  tf.maximum(0., 2.0 - d), 0.)
        output = equal_loss + not_equal_loss
        output = tf.reduce_mean(output, axis=-1)
        output = tf.math.divide_no_nan(output, num_valid_1d) * num_lines
        return output

    def bg_loss(y_true, y_pred):
        bg_tensor = y_pred[:, :, 0]
        d = tf.where(bg_mask, -tf.math.log(bg_tensor), -tf.math.log(1. - bg_tensor))
        d = tf.where(valid_mask, d, 0.)
        # output = tf.math.divide_no_nan(d, num_valid_1d) * 150.
        return d

    def loss(y_true, y_pred):
        return cluster_loss(y_true, y_pred) + bg_loss(y_true, y_pred)

    pred_labels = tf.argmax(instancing_tensor, axis=-1)
    h_pred_labels = tf.expand_dims(pred_labels, axis=-1)
    v_pred_labels = tf.transpose(h_pred_labels, perm=(0, 2, 1))

    pred_equals = tf.equal(h_pred_labels, v_pred_labels)
    pred_not_equals = tf.not_equal(h_pred_labels, v_pred_labels)

    true_p = tf.cast(tf.logical_and(pred_equals, tf.logical_and(loss_mask, mask_equal)), dtype='float32')
    true_p = tf.reduce_sum(true_p, axis=(1, 2))

    true_n = tf.cast(tf.logical_and(pred_not_equals, tf.logical_and(loss_mask, mask_not_equal)), dtype='float32')
    true_n = tf.reduce_sum(true_n, axis=(1, 2))

    gt_p = tf.cast(tf.logical_and(loss_mask, mask_equal), dtype='float32')
    gt_p = tf.reduce_sum(gt_p, axis=(1, 2))

    gt_n = tf.cast(tf.logical_and(loss_mask, mask_not_equal), dtype='float32')
    gt_n = tf.reduce_sum(gt_n, axis=(1, 2))

    pred_p = tf.cast(tf.logical_and(pred_equals, loss_mask), dtype='float32')
    pred_p = tf.reduce_sum(pred_p, axis=(1, 2))

    pred_n = tf.cast(tf.logical_and(pred_not_equals, loss_mask), dtype='float32')
    pred_n = tf.reduce_sum(pred_n, axis=(1, 2))

    def tp_gt_p(y_true, y_pred):
        return tf.reduce_mean(tf.math.divide_no_nan(true_p, gt_p))

    def tn_gt_n(y_true, y_pred):
        return tf.reduce_mean(tf.math.divide_no_nan(true_n, gt_n))

    def tp_pd_p(y_true, y_pred):
        return tf.reduce_mean(tf.math.divide_no_nan(true_p, pred_p))

    def tn_pd_n(y_true, y_pred):
        return tf.reduce_mean(tf.math.divide_no_nan(true_n, pred_n))

    return loss, [cluster_loss, bg_loss]# [tp_gt_p, tn_gt_n, tp_pd_p, tn_pd_n]


def triplet_metrics(embedding_a, embedding_p, embedding_n, margin=0.2):
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
    def loss(y_true, y_pred):
        out = tf.maximum(tf.norm(embedding_p - embedding_a, axis=-1) -
                         tf.norm(embedding_n - embedding_a, axis=-1) + margin, 0.)
        return out

    return loss
