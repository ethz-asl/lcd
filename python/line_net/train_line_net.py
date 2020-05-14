import time
import datetime
import os

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from keras.models import load_model

import tensorflow as tf

import numpy as np
np.random.seed(123)

from datagenerator_framewise import LineDataGenerator


def initialize_bias(shape, dtype=float, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def drop_diagonal_and_mask(shape, validity_mask):
    eye = K.expand_dims(tf.eye(num_rows=shape[0], dtype='float32'), axis=-1)

    validity_expanded = K.expand_dims(K.expand_dims(validity_mask, axis=-1), axis=-1)
    h_mask = K.cast(K.repeat_elements(validity_expanded, shape[0], -2), dtype='float32')
    v_mask = K.permute_dimensions(h_mask, (0, 2, 1, 3))

    def layer(tensor):
        return (1. - eye) * tensor * h_mask * v_mask

    return layer


def get_n_max_neighbors(norm_tensor, num_lines, n=3):
    n = tf.constant(n)

    def layer(feature_tensor):
        indices = tf.argsort(tf.squeeze(norm_tensor, axis=-1), axis=-1, direction='DESCENDING')
        indices = tf.reshape(indices[:, :, :n], (-1, num_lines, n))
        # Get coordinate grid for indexing with argsort indices.
        # This might be an issue with a batchsize greater than 1.
        # Edit: I think i fixed it now.
        index_grid = tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(tf.range(0, num_lines), axis=-1), repeats=n, axis=-1), axis=0), repeats=tf.shape(indices)[0], axis=0)
        indices = tf.stack([indices, index_grid], axis=-1)
        max_features = tf.gather_nd(feature_tensor, indices, batch_dims=1)
        tensors = tf.unstack(max_features, axis=-2)
        return tf.concat(tensors, axis=-1)

    return layer


def get_losses_and_metrics(norm_tensor, norm_tensor_2, instancing_tensor,
                           labels_tensor, validity_mask, bg_mask, num_lines, margin):
    # eye = K.expand_dims(K.eye(size=(num_lines, num_lines), dtype='float32'), axis=-1)

    labels = K.expand_dims(K.expand_dims(labels_tensor, axis=-1), axis=-1)
    labels = K.repeat_elements(K.cast(labels, dtype='int32'), num_lines, -2)

    h_mask = labels
    v_mask = K.permute_dimensions(h_mask, (0, 2, 1, 3))

    # Background interconnections are loss free, to prevent trying to fit on background clusters. These are
    # typically very far apart and obviously hard to cluster.
    # In the future, maybe these lines can be classified as background lines.
    bg = K.expand_dims(K.expand_dims(bg_mask, axis=-1), axis=-1)
    bg = K.repeat_elements(K.cast(bg, dtype='float32'), num_lines, -2)
    h_bg = bg
    v_bg = K.permute_dimensions(h_bg, (0, 2, 1, 3))
    mask_non_bg = 1. - h_bg * v_bg

    mask_equal = K.cast(K.equal(h_mask, v_mask), dtype='float32')
    mask_not_equal = (1. - mask_equal) * mask_non_bg
    mask_equal = drop_diagonal_and_mask((num_lines, num_lines), validity_mask)(mask_equal) * mask_non_bg

    num_total = tf.constant(num_lines, dtype='float32')
    num_valid = K.sum(tf.cast(validity_mask, dtype='float'), axis=-1)

    # norms = K.expand_dims(K.sqrt(K.sum(K.square(compare_tensor), axis=3)), -1)

    # Losses:

    def compare_norm_loss(y_true, y_pred):
        #loss_layer = K.maximum(0., K.square(1. - norm_tensor) - 0.04) * mask_equal + \
        #             K.maximum(0., K.square(norm_tensor - 0.) - 0.04) * mask_not_equal
        loss_layer = tf.nn.sigmoid_cross_entropy_with_logits(K.ones_like(norm_tensor), norm_tensor) * mask_equal * 10. + \
                     tf.nn.sigmoid_cross_entropy_with_logits(K.zeros_like(norm_tensor), norm_tensor) * mask_not_equal
        return loss_layer * num_total / num_valid  # tf.reduce_max(loss_layer, axis=-1) +

    def compare_norm_loss_2(y_true, y_pred):
        loss_layer = K.maximum(0., K.square(1. - norm_tensor_2) - 0.04) * mask_equal + \
                     K.maximum(0., K.square(norm_tensor_2 - 0.) - 0.04) * mask_not_equal
        return loss_layer  # tf.reduce_max(loss_layer, axis=-1) +

    def kl_cross_div_loss(y_true, y_pred):
        repeated_pred = K.repeat_elements(instancing_tensor, num_lines, 1)
        h_pred = K.permute_dimensions(repeated_pred, (0, 1, 2, 3))
        v_pred = K.permute_dimensions(repeated_pred, (0, 2, 1, 3))
        d = h_pred * K.log(h_pred / v_pred)
        loss_layer = mask_equal * d + mask_not_equal * K.maximum(0., margin - d)
        return K.identity(K.sum(loss_layer, axis=3) / 32., name='kl_cross_div_loss')

    def loss(y_true, y_pred):
        return compare_norm_loss(y_true, y_pred)#  + compare_norm_loss_2(y_true, y_pred)
        # + kl_cross_div_loss(y_true, y_pred) * 0.

    # Metrics:

    positives = K.cast(K.greater(K.sigmoid(norm_tensor), 0.5), dtype='float32')
    negatives = K.cast(K.greater(0.5, K.sigmoid(norm_tensor)), dtype='float32')

    num_equal = K.sum(mask_equal, axis=(1, 2, 3))
    num_not_equal = K.sum(mask_not_equal, axis=(1, 2, 3))

    num_positives = K.sum(positives, axis=(1, 2, 3))
    num_negatives = K.sum(negatives, axis=(1, 2, 3))

    true_positives = K.sum(mask_equal * positives, axis=(1, 2, 3))
    false_positives = K.sum(mask_not_equal * positives, axis=(1, 2, 3))
    false_negatives = K.sum(mask_equal * negatives, axis=(1, 2, 3))
    true_negatives = K.sum(mask_not_equal * negatives, axis=(1, 2, 3))

    # Ratio true positives: true_positives / num_equal
    # Ratio true negatives: true_negatives / num_not_equal
    # Ratio false positives: false_positives / num_not_equal
    # Ratio false negatives: false_negatives / num_equal

    def tp_over_pred_p(y_true, y_pred):
        return tf.math.divide_no_nan(true_positives, num_positives)

    def tn_over_pred_n(y_ture, y_pred):
        return K.identity(true_negatives / num_negatives, name="tn_over_pred_n")

    def tp_over_gt_p(y_true, y_pred):
        return tf.math.divide_no_nan(true_positives, num_equal)

    def tn_over_gt_n(y_true, y_pred):
        return K.identity(true_negatives / num_not_equal, name="tn_over_gt_n")

    return loss, [compare_norm_loss, tp_over_pred_p, tp_over_gt_p]
    # [compare_norm_loss, kl_cross_div_loss, tp_over_pred_p, tn_over_pred_n, tp_over_gt_p, tn_over_gt_n]


def get_kl_losses_and_metrics(instancing_tensor, labels_tensor, validity_mask, bg_mask, num_lines):
    # labels = K.expand_dims(K.expand_dims(labels_tensor, axis=-1), axis=-1)
    # labels = K.repeat_elements(K.cast(labels, dtype='int32'), num_lines, -2)

    h_labels = K.expand_dims(labels_tensor, axis=-1)
    v_labels = tf.transpose(h_labels, perm=(0, 2, 1))

    # Background interconnections are loss free, to prevent trying to fit on background clusters. These are
    # typically very far apart and obviously hard to cluster.
    # In the future, maybe these lines can be classified as background lines.
    # bg = K.expand_dims(K.expand_dims(bg_mask, axis=-1), axis=-1)
    # bg = K.repeat_elements(K.cast(bg, dtype='float32'), num_lines, -2)
    # h_bg = bg
    # v_bg = K.permute_dimensions(h_bg, (0, 2, 1, 3))
    # mask_non_bg = 1. - h_bg * v_bg

    mask_equal = tf.equal(h_labels, v_labels)
    mask_not_equal = tf.not_equal(h_labels, v_labels)
    # mask_equal = tf.cast(mask_equal, dtype='float32')
    print("EQUAL SHAPE")
    print(mask_equal.shape)
    # mask_not_equal = tf.cast(mask_not_equal, dtype='float32')
    print("NOT EQUAL SHAPE")
    print(mask_not_equal.shape)

    h_bg = tf.expand_dims(tf.logical_not(bg_mask), axis=-1)
    v_bg = tf.transpose(h_bg, perm=(0, 2, 1))
    mask_not_bg = tf.logical_and(h_bg, v_bg)

    h_val = tf.expand_dims(validity_mask, axis=-1)
    v_val = tf.transpose(h_val, perm=(0, 2, 1))
    mask_val = tf.logical_and(h_val, v_val)
    mask_val = tf.linalg.set_diag(mask_val, tf.zeros(tf.shape(mask_val)[0:-1], dtype='bool'))
    print("VAL SHAPE")
    print(mask_val.shape)

    loss_mask = tf.logical_and(mask_val, mask_not_bg)

    num_valid = tf.reduce_sum(tf.cast(loss_mask, dtype='float32'), axis=(1, 2), keepdims=True)
    # num_total = tf.ones_like(num_valid) * num_lines * num_lines

    #def k_cross_div_same(y_true, y_pred):

    def kl_cross_div_loss(y_true, y_pred):
        # Drop diagonal and mask:#  * mask_equal * h_mask_drop * v_mask_drop * mask_non_bg
        # eye = K.expand_dims(tf.eye(num_rows=num_lines, dtype='float32'), axis=-1)
        # validity_expanded = K.expand_dims(K.expand_dims(K.cast(validity_mask, dtype='float32'), axis=-1), axis=-1)
        # h_mask_drop = K.repeat_elements(validity_expanded, num_lines, axis=-2)
        # v_mask_drop = tf.transpose(h_mask, perm=(0, 2, 1, 3))
        # mask_equal_ = tf.linalg.set_diag(mask_equal, tf.zeros(mask_equal.shape[1:-1]))
        # End drop diagonal and mask.

        # num_total = tf.constant(num_lines, dtype='float32')
        # num_valid = K.sum(tf.cast(validity_mask, dtype='float'), axis=-1)

        # num_equal = K.sum(mask_equal, axis=(1, 2))
        # num_not_equal = K.sum(mask_not_equal, axis=(1, 2))

        extended_pred = K.expand_dims(instancing_tensor, axis=2)
        h_pred = extended_pred  # K.permute_dimensions(extended_pred, (0, 1, 2, 3))
        v_pred = tf.transpose(extended_pred, perm=(0, 2, 1, 3))
        d = h_pred * tf.math.log(tf.math.divide_no_nan(h_pred, v_pred + 1e-100) + 1e-100)
        d = tf.reduce_sum(d, axis=-1, keepdims=False)
        print("D SHAPE")
        print(d.shape)
        equal_loss = tf.where(tf.logical_and(mask_equal, loss_mask), d, 0.)
        not_equal_loss = tf.where(tf.logical_and(mask_not_equal, loss_mask),
                                  tf.maximum(0., 2.0 - d), 0.)
        print("FINAL LOSS SHAPE")
        # print(out_loss.shape)
        return tf.math.divide_no_nan((equal_loss + not_equal_loss), num_valid) * 150. * 150.  # equal_loss * mask_val * mask_not_bg + not_equal_loss * mask_val * mask_not_bg

    pred_labels = tf.argmax(instancing_tensor, axis=-1)
    print("PRED_LABELS SHAPE")
    print(pred_labels.shape)
    h_pred_labels = tf.expand_dims(pred_labels, axis=-1)
    print("PRED_LABELS_EXP SHAPE")
    print(h_pred_labels.shape)
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

    return kl_cross_div_loss, [tp_gt_p, tn_gt_n, tp_pd_p, tn_pd_n]


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


def expand_lines(num_lines):
    def expand_lines_layer(line_inputs):
        expanded = tf.expand_dims(line_inputs, axis=-1)
        repeated = tf.repeat(expanded, num_lines, -1)
        h_input = tf.transpose(repeated, perm=(0, 1, 3, 2))
        v_input = tf.transpose(repeated, perm=(0, 3, 1, 2))
        return tf.concat([h_input, v_input], axis=-1)

    return expand_lines_layer


def img_compare_features(num_lines):
    def img_compare_layer(img_features):
        expanded = tf.expand_dims(img_features, axis=-2)
        expanded = tf.repeat(expanded, num_lines, axis=-2)
        x1 = expanded
        x2 = tf.transpose(expanded, perm=(0, 2, 1, 3))

        f1 = tf.square(x1 - x2)
        f2 = tf.square(x1) - tf.square(x2)

        feature_layer = tf.concat([f1, f2], axis=-1)

        return [f1, f2]

    return img_compare_layer


def cluster_count_metrics():
    def diff(y_true, y_pred):
        return K.abs(K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1))

    def gt_mean(y_true, y_pred):
        return K.argmax(y_true, axis=-1)

    def pred_mean(y_true, y_pred):
        return K.argmax(y_pred, axis=-1)

    return [diff, gt_mean, pred_mean, 'categorical_accuracy']


def normalize_embeddings(input):
    return K.l2_normalize(input, axis=3)


def masked_binary_cross_entropy(bg_mask, validity_mask, num_lines):
    def loss(y_true, y_pred):
        out = K.binary_crossentropy(y_true, y_pred, from_logits=True)
        validity = K.cast(validity_mask, dtype='float32')
        bg = K.cast(bg_mask, dtype='float32')
        num_bg = K.sum(bg, axis=1, keepdims=True)
        num_valid = K.sum(validity, axis=1, keepdims=True)
        num_total = K.sum(K.ones_like(validity), axis=1, keepdims=True)
        # num_bg * (num_valid - num_bg)
        weights = bg / 0.25 + (1. - bg)
        return out * validity * num_total / num_valid * weights

    return loss


def masked_binary_accuracy(validity_mask, threshold=0.5):
    def binary_accuracy(y_true, y_pred):
        validity = K.cast(validity_mask, dtype='float32')
        num_lines = K.sum(validity, axis=1)
        return K.sum(K.cast(K.equal(y_true, K.cast(K.greater(y_pred, threshold), dtype='float32')),
                            dtype='float32') * validity, axis=1) / num_lines

    return binary_accuracy


def bg_percentage_metric(bg_mask, validity_mask):
    def bg_percentage(y_true, y_pred):
        return K.sum(K.cast(bg_mask, dtype='float32'), axis=1) / \
               K.sum(K.cast(validity_mask, dtype='float32'), axis=1)

    return bg_percentage


def debug_metrics(tensor):
    def d_sum(y_true, y_pred):
        return K.sum(tensor, axis=-1)

    def d_l1(y_true, y_pred):
        return K.sum(K.abs(tensor), axis=-1)

    def d_std(y_true, y_pred):
        return K.std(tensor, axis=-1)

    def d_max(y_true, y_pred):
        return K.max(tensor, axis=-1)

    def d_min(y_true, y_pred):
        return K.min(tensor, axis=-1)

    return [d_sum, d_l1, d_std, d_max, d_min]


def argmax_metric():
    def argmax(y_true, y_pred):
        return K.argmax(y_pred, axis=-1)

    return argmax


def iou_metric(labels, unique_labels, cluster_counts, bg_mask, valid_mask, max_clusters):
    def iou(y_true, y_pred):
        mask = tf.logical_and(tf.logical_not(bg_mask), valid_mask)
        mask = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1)

        gt_labels = tf.expand_dims(tf.expand_dims(labels, axis=-1), axis=-1)
        unique_gt_labels = tf.expand_dims(tf.expand_dims(unique_labels, axis=1), axis=-1)
        pred_labels = tf.expand_dims(tf.expand_dims(tf.argmax(y_pred, axis=-1, output_type=tf.dtypes.int32),
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
        iou_out = tf.reduce_sum(iou_out, axis=-1, keepdims=True) / tf.cast(cluster_counts, dtype='float32')

        return tf.reduce_mean(iou_out)

    return iou


def line_net_model(line_num_attr, num_lines, img_shape, margin):
    """
    Model architecture
    """

    # Some attributes for quick changing:
    first_order_embedding_size = 1
    first_order_size = 32
    output_embedding_size = 32
    n_neighbors = 5

    # Inputs for geometric line information.
    img_inputs = kl.Input(shape=(num_lines, img_shape[0], img_shape[1], img_shape[2]), dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    validity_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
    fake_input = kl.Input(shape=(num_lines, 15), dtype='float32', name='fake')

    # img_inputs = kl.ZeroPadding3D(padding=(0, 0, 0))

    # expanded_val = kl.Lambda(lambda x: K.cast(K.expand_dims(x, axis=-1), dtype='float32'))(validity_input)

    # RBDLSTM tests.
    # line_features = kl.Dense(1000, activation='relu')(line_inputs)
    # line_features = kl.Multiply()([line_features, expanded_val])

    line_features = line_inputs
    # line_features = kl.Lambda(lambda x: K.expand_dims(K.cast(x, dtype='float32') / 300., axis=-1))(label_input)
    debug = fake_input
    # line_features = fake_input
    line_features = kl.Masking(mask_value=0.0)(line_features)
    line_features = kl.TimeDistributed(kl.Dense(1000, activation=None, name='embedding_layer'),
                                       input_shape=(150, 15))(line_features)
    line_features = kl.LeakyReLU()(line_features)
    print("LINE FEATURES SHAPE")
    print(line_features.shape)
    line_f1 = kl.Bidirectional(kl.LSTM(500, return_sequences=True, return_state=False, name="lstm_1"),
                              merge_mode='concat')(line_features)
    # line_f1 = kl.TimeDistributed(kl.BatchNormalization())(line_f1)
    line_f1 = kl.Add()([line_features, line_f1])
    line_f2 = kl.Bidirectional(kl.LSTM(500, return_sequences=True, return_state=False, name="lstm_2"),
                               merge_mode='concat')(line_f1)
    # line_f2 = kl.TimeDistributed(kl.BatchNormalization())(line_f2)
    line_f2 = kl.Add()([line_f2, line_f1])
    # line_f3 = kl.Bidirectional(kl.LSTM(500, return_sequences=True, return_state=False),
    #                            merge_mode='concat')(line_f2)
    # line_f3 = kl.Add()([line_f3, line_f2])
    # line_idist is the sequence output for clustering, line_ic is the output for the cluster count.
    # line_idist, _, line_ic_0, _, line_ic_1 = \
    #     kl.Bidirectional(kl.LSTM(500, return_sequences=True, return_state=True, name="lstm_1"),
    #                      merge_mode='concat')(line_f3)
    # line_ic = kl.Concatenate()([line_ic_0, line_ic_1])

    # line_idist = kl.TimeDistributed(kl.BatchNormalization())(line_idist)
    line_idist = line_f2  # kl.Add()([line_idist, line_f3])
    # line_idist = kl.TimeDistributed(kl.Dense(500, use_bias=False))(line_idist)
    # line_idist = kl.TimeDistributed(kl.BatchNormalization())(line_idist)
    # line_idist = kl.TimeDistributed(kl.Activation('relu'))(line_idist)
    # debug = line_idist
    line_idist = kl.TimeDistributed(kl.BatchNormalization())(line_idist)
    line_idist = kl.TimeDistributed(kl.Dense(15, activation=None))(line_idist)
    line_idist = kl.TimeDistributed(kl.BatchNormalization())(line_idist)
    debug = line_idist
    line_idist = kl.TimeDistributed(kl.Softmax())(line_idist)
    line_idist_logits = line_idist  # kl.TimeDistributed(kl.Softmax(axis=-1), name='cl_dis')(line_idist)
    print("line_idist")
    print(line_idist_logits.shape)

    # line_ic = kl.Dense(500, use_bias=False)(line_ic)
    # line_ic = kl.BatchNormalization()(line_ic)
    # line_ic = kl.Activation('relu')(line_ic)
    # line_ic = kl.Dense(31, activation=None)(line_ic)
    # line_ic = kl.BatchNormalization()(line_ic)
    # line_ic_logits = kl.Softmax(name='cl_cnt')(line_ic)

    # Image classifier:
    # img_shape = Input(shape=(224, 224, 3), dtype='float32')
    back_bone = keras.applications.VGG16(include_top=False, weights='imagenet',
                                         input_shape=(img_shape[0], img_shape[1], img_shape[2]))
    img_emb = kl.Concatenate(axis=-1)([kl.GlobalMaxPool2D()(back_bone.output),
                                       kl.GlobalAveragePooling2D()(back_bone.output)])
    img_emb = kl.Dense(100, activation='relu')(img_emb)
    img_model = Model(inputs=back_bone.input, outputs=img_emb)
    # img_model.summary()

    img_features = kl.TimeDistributed(img_model)(img_inputs)
    img_f1, img_f2 = kl.Lambda(img_compare_features(num_lines), name='img_features')(img_features)
    img_features = kl.Concatenate()([img_f1, img_f2])
    img_features = kl.Dense(100, activation='relu')(img_features)
    img_norms = kl.Dense(1, activation=None)(img_features)
    img_logits = kl.Activation('sigmoid')(img_norms)

    expanded_input = kl.Lambda(expand_lines(num_lines), name='expand_input')(line_inputs)

    # NN for line distance metrics.
    # Output embedding size is 64.
    compare_model = Sequential(name='first_order_compare')
    compare_model.add(kl.Conv2D(1024, kernel_size=(1, 1),
                                input_shape=(num_lines, num_lines, line_num_attr * 2), activation='relu'))
    # compare_model.add(kl.Conv2D(1024, kernel_size=(1, 1), activation='relu'))
    # compare_model.add(kl.Conv2D(1024, kernel_size=(1, 1), activation='relu'))
    compare_model.add(kl.Conv2D(first_order_size, kernel_size=(1, 1), activation='relu'))

    first_order_norm_model = Sequential(name='first_order_norm')
    first_order_norm_model.add(kl.Conv2D(first_order_embedding_size, kernel_size=(1, 1),
                                         input_shape=(num_lines, num_lines, first_order_size),
                                         activation='sigmoid'))
    # first_order_norm_model.add(kl.Lambda(drop_diagonal_and_mask((num_lines, num_lines, first_order_embedding_size),
    #                                                             validity_input)))

    compare_model_2 = Sequential(name='second_order_compare')
    compare_model_2.add(kl.Conv2D(2096, kernel_size=(1, 1),
                                  input_shape=(num_lines, num_lines, first_order_size*(n_neighbors*2 + 1)),
                                  activation='relu'))
    # compare_model_2.add(kl.Conv2D(600, kernel_size=(1, 1), activation='relu'))
    # compare_model_2.add(kl.Conv2D(600, kernel_size=(1, 1), activation='relu'))
    # compare_model_2.add(kl.Conv2D(1024, kernel_size=(1, 1), activation='relu'))
    # compare_model_2.add(kl.Conv2D(32, kernel_size=(1, 1), activation='relu'))
    compare_model_2.add(kl.Conv2D(1, kernel_size=(1, 1), activation='sigmoid'))

    # NN for instancing distributions.
    # Output embedding size (max number of instances) is 32.
    instancing_model = Sequential(name='instancing')
    instancing_model.add(kl.AveragePooling2D((num_lines, 1)))
    instancing_model.add(kl.Lambda(normalize_embeddings))
    # instancing_model.add(kl.Convolution2D(50, kernel_size=(1, 1),
    #                                      input_shape=(num_lines, 1, first_order_embedding_size), activation='relu'))
    instancing_model.add(kl.Convolution2D(output_embedding_size, kernel_size=(1, 1), activation='sigmoid'))
    instancing_model.add(kl.Softmax(axis=-1))

    compare_embeddings = compare_model(expanded_input)
    first_order_norm = first_order_norm_model(compare_embeddings)
    line_embeddings = instancing_model(compare_embeddings)
    first_order_max_neighbors = kl.Lambda(get_n_max_neighbors(first_order_norm, num_lines, n=n_neighbors))(compare_embeddings)
    expanded_neighbours = kl.Lambda(expand_lines(num_lines), name="expand_neighbors")(first_order_max_neighbors)
    second_order_input = kl.Concatenate(axis=-1)([compare_embeddings, expanded_neighbours])
    second_order_norm = compare_model_2(second_order_input)

    losses, metrics = get_losses_and_metrics(#first_order_norm,
                                             img_norms,
                                             second_order_norm,
                                             line_embeddings,
                                             label_input,
                                             validity_input,
                                             bg_input,
                                             num_lines,
                                             margin)

    kl_losses, real_metrics = get_kl_losses_and_metrics(line_idist_logits, label_input, validity_input, bg_input, num_lines)
    # kl_test_metric, more_test_metrics = get_kl_losses_and_metrics(fake_input, label_input, validity_input, bg_input, num_lines)
    # kl_losses = get_losses_test(line_idist_logits)

    line_model = Model(inputs=[img_inputs, line_inputs, label_input, validity_input, bg_input, fake_input],
                       outputs=line_idist_logits,# [line_ic_logits, line_idist_logits], #img_logits, #second_order_norm,  # line_embeddings,
                       name='line_net_model')
    opt = SGD(lr=0.0005, momentum=0.9)
    # opt = keras.optimizers.RMSprop(learning_rate=0.018)
    line_model.compile(loss=kl_losses,# {'cl_cnt': 'categorical_crossentropy', 'cl_dis': kl_losses},#losses,#masked_binary_cross_entropy(bg_input, validity_input, num_lines), #losses,
                       optimizer=opt,
                       metrics=real_metrics + debug_metrics(debug),# [kl_test_metric] +
                       # metrics={'cl_cnt': cluster_count_metrics()},
                       experimental_run_tf_function=False)#[masked_binary_accuracy(validity_input), bg_percentage_metric(bg_input, validity_input)])

    return line_model


def line_net_model_2(line_num_attr, num_lines, img_shape):
    # Inputs for geometric line information.
    img_inputs = kl.Input(shape=(num_lines, img_shape[0], img_shape[1], img_shape[2]), dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    validity_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
    fake_input = kl.Input(shape=(num_lines, 15), dtype='float32', name='fake')

    # The embedding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(100))(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = kl.LeakyReLU()(line_embeddings)

    line_f1 = kl.Bidirectional(kl.LSTM(100, return_sequences=True),
                               merge_mode='concat', name='bdlstm_1')(line_embeddings)
    line_f1 = kl.TimeDistributed(kl.BatchNormalization(center=False))(line_f1)
    line_f1 = kl.Concatenate()([line_embeddings, line_f1])

    line_f2 = kl.Bidirectional(kl.LSTM(200, return_sequences=True),
                               merge_mode='concat', name='bdlstm_2')(line_f1)
    line_f2 = kl.TimeDistributed(kl.BatchNormalization())(line_f2)
    line_f2 = kl.Concatenate()([line_f1, line_f2])

    line_ins = kl.Bidirectional(kl.LSTM(500, return_sequences=True),
                                merge_mode='concat', name='bdlstm_3')(line_f2)
    line_ins = kl.TimeDistributed(kl.Dense(15))(line_ins)  # , activity_regularizer=kr.l1(l=0.001)
    line_ins = kl.TimeDistributed(kl.BatchNormalization(center=False))(line_ins)
    debug_layer = line_ins
    line_ins = kl.Softmax(name='instance_distribution')(line_ins)

    loss, metrics = get_kl_losses_and_metrics(line_ins, label_input, validity_input, bg_input, num_lines)
    opt = SGD(lr=0.0005, momentum=0.9)
    line_model = Model(inputs=[img_inputs, line_inputs, label_input, validity_input, bg_input, fake_input],
                       outputs=line_ins,
                       name='line_net_model')
    line_model.compile(loss=loss,
                       optimizer=opt,
                       metrics=metrics + debug_metrics(debug_layer),
                       experimental_run_tf_function=False)
    return line_model


def get_multi_head_attention_model(input_shape, dropout=0.2, idx=0, key_size=128, n_multi=2, n_add=2):
    assert n_multi + n_add > 0

    output_size = input_shape[1]

    model_input = kl.Input(shape=input_shape)

    outputs_multi = []
    for i in range(n_multi):
        # More layers can be added here.
        keys = kl.TimeDistributed(kl.Dense(key_size))(model_input)
        keys = kl.TimeDistributed(kl.BatchNormalization())(keys)
        # keys = kl.LeakyReLU()(keys)
        queries = kl.TimeDistributed(kl.Dense(key_size))(model_input)
        queries = kl.TimeDistributed(kl.BatchNormalization())(queries)
        # queries = kl.LeakyReLU()(queries)
        output = kl.Attention(use_scale=True)([queries, keys])
        outputs_multi.append(output)

    outputs_add = []
    for i in range(n_add):
        # More layers can be added here.
        keys = kl.TimeDistributed(kl.Dense(key_size))(model_input)
        keys = kl.TimeDistributed(kl.BatchNormalization())(keys)
        # keys = kl.LeakyReLU()(keys)
        queries = kl.TimeDistributed(kl.Dense(key_size))(model_input)
        queries = kl.TimeDistributed(kl.BatchNormalization())(queries)
        # Multiply with -1 so that it is actually a subtractive attention.
        # queries = kl.Lambda(lambda x: -x)(queries)
        # queries = kl.LeakyReLU()(queries)
        output = kl.AdditiveAttention(use_scale=True)([queries, keys])
        outputs_add.append(output)

    outputs = outputs_multi + outputs_add
    if len(outputs) > 1:
        output = kl.Concatenate()(outputs)
    else:
        output = outputs[0]

    # Should I add one more dense layer here?
    output = kl.TimeDistributed(kl.Dense(output_size))(output)
    output = kl.TimeDistributed(kl.Dropout(dropout))(output)
    output = kl.TimeDistributed(kl.BatchNormalization())(output)
    output = kl.LeakyReLU()(output)

    return Model(inputs=model_input, outputs=output, name='multi_head_attention_{}'.format(idx))


def get_inter_attention_layer(input_number, head_units=256, hidden_units=1024,
                              idx=0, dropout=0.2, key_size=128, n_multi_heads=2, n_add_heads=2):
    model_input = kl.Input(shape=(input_number, head_units))

    layer = get_multi_head_attention_model((input_number, head_units),
                                           idx=idx,
                                           key_size=key_size,
                                           n_multi=n_multi_heads,
                                           n_add=n_add_heads)(model_input)
    layer = kl.Add()([layer, model_input])

    # Two layers of dense connections running in parallel
    layer_2 = kl.TimeDistributed(kl.Dense(hidden_units))(layer)
    layer_2 = kl.TimeDistributed(kl.Dropout(dropout))(layer_2)
    layer_2 = kl.TimeDistributed(kl.BatchNormalization())(layer_2)
    layer_2 = kl.LeakyReLU()(layer_2)
    layer_2 = kl.TimeDistributed(kl.Dense(head_units))(layer_2)
    layer_2 = kl.TimeDistributed(kl.Dropout(dropout))(layer_2)
    layer_2 = kl.TimeDistributed(kl.BatchNormalization())(layer_2)
    layer_2 = kl.LeakyReLU()(layer_2)
    layer = kl.Add()([layer, layer_2])

    return Model(inputs=model_input, outputs=layer, name='inter_attention_{}'.format(idx))


def line_net_model_3(line_num_attr, num_lines, img_shape):
    # Inputs for geometric line information.
    # img_inputs = kl.Input(shape=(num_lines, img_shape[0], img_shape[1], img_shape[2]), dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
    fake_input = kl.Input(shape=(num_lines, 15), dtype='float32', name='fake')
    unique_label_input = kl.Input(shape=(15,), dtype='int32', name='unique_labels')
    cluster_count_input = kl.Input(shape=(1,), dtype='int32', name='cluster_count')

    head_units = 256

    # The embedding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(head_units))(line_embeddings)
    line_embeddings = kl.Dropout(0.1)(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = kl.LeakyReLU()(line_embeddings)

    # Build 5 multi head attention layers. Hopefully this will work.
    layer = get_inter_attention_layer(num_lines, idx=0)(line_embeddings)
    layer = get_inter_attention_layer(num_lines, idx=1)(layer)
    layer = get_inter_attention_layer(num_lines, idx=2)(layer)
    layer = get_inter_attention_layer(num_lines, idx=3)(layer)
    layer = get_inter_attention_layer(num_lines, idx=4)(layer)
    layer = get_inter_attention_layer(num_lines, idx=5)(layer)

    line_ins = kl.TimeDistributed(kl.Dense(15))(layer)
    line_ins = kl.BatchNormalization()(line_ins)
    debug_layer = line_ins
    line_ins = kl.TimeDistributed(kl.Softmax(name='instance_distribution'))(line_ins)

    loss, metrics = get_kl_losses_and_metrics(line_ins, label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, 15)
    opt = SGD(lr=0.0015, momentum=0.9)
    line_model = Model(inputs=[line_inputs, label_input, valid_input, bg_input, fake_input,
                               unique_label_input, cluster_count_input],
                       outputs=line_ins,
                       name='line_net_model')
    line_model.compile(loss=loss,
                       optimizer='adam',
                       metrics=[iou] + metrics + debug_metrics(debug_layer),
                       experimental_run_tf_function=False)
    return line_model


def get_fake_instancing(labels, valid_mask, bg_mask):
    valid_count = np.where(valid_mask == 1)[0].shape[0]
    unique_labels = np.unique(labels[np.where(np.logical_and(valid_mask, np.logical_not(bg_mask)))])
    out = np.zeros((150, 15), dtype=float)

    for i, label in enumerate(unique_labels):
        out[np.where(labels == label), i] = 1.

    # out[:, :] = 0.
    # out[:, 0] = 1.

    kl_loss_np(out, labels, valid_mask, bg_mask)

    return np.expand_dims(out, axis=0)


def train():
    # Paths to line files.
    train_files = "/nvme/line_ws/train"
    val_files = "/nvme/line_ws/val"
    test_files = "/nvme/line_ws/test"

    # The length of the geometry vector of a line.
    line_num_attr = 15
    img_shape = (120, 180, 3)
    max_line_count = 150
    batch_size = 20
    num_epochs = 50
    bg_classes = [0, 1, 2, 20, 22]

    # Create line net Keras model.
    # line_model = line_net_model(line_num_attr, max_line_count, img_shape, margin)
    line_model = line_net_model_3(line_num_attr, max_line_count, img_shape)
    line_model.summary()

    # log_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/120520_2010"
    # line_model.load_weights("/home/felix/line_ws/src/line_tools/python/line_net/logs/130520_2315/weights.20.hdf5",
    #                         by_name=True)
    train_set_mean = np.array([-0.00246431839, 0.0953982015,  3.15564408])

    train_ = True
    if train_:

        train_data_generator = LineDataGenerator(train_files, bg_classes,
                                                 shuffle=True,
                                                 data_augmentation=False,
                                                 img_shape=img_shape,
                                                 sort=True)
        # train_set_mean = train_data_generator.get_mean()
        train_data_generator.set_mean(train_set_mean)
        print("Train set mean is: {}".format(train_set_mean))
        val_data_generator = LineDataGenerator(val_files, bg_classes, mean=train_set_mean, img_shape=img_shape, sort=True)

        train_generator = data_generator(train_data_generator, max_line_count, line_num_attr, batch_size)
        val_generator = data_generator(val_data_generator, max_line_count, line_num_attr, batch_size)

        log_path = "./logs/{}".format(datetime.datetime.now().strftime("%d%m%y_%H%M"))
        save_weights_callback = keras.callbacks.ModelCheckpoint(os.path.join(log_path, "weights.{epoch:02d}.hdf5"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_path)

        line_model.fit_generator(generator=train_generator,
                                 verbose=1,
                                 max_queue_size=1,
                                 workers=1,
                                 use_multiprocessing=False,
                                 epochs=num_epochs,
                                 steps_per_epoch=np.floor((train_data_generator.frame_count - 3093) / batch_size),
                                 validation_data=val_generator,
                                 validation_steps=np.floor((val_data_generator.frame_count - 558) / batch_size),
                                 callbacks=[save_weights_callback, tensorboard_callback])


if __name__ == '__main__':
    train()



