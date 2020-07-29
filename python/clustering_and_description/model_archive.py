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


def argmax_metric():
    def argmax(y_true, y_pred):
        return K.argmax(y_pred, axis=-1)

    return argmax


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

    head_units = 384
    hidden_units = head_units * 4

    # The embedding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(head_units))(line_embeddings)
    line_embeddings = kl.Dropout(0.1)(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = kl.LeakyReLU()(line_embeddings)

    # Build 5 multi head attention layers. Hopefully this will work.
    layer = get_inter_attention_layer(num_lines, head_units=head_units, idx=0)(line_embeddings)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, idx=1)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, idx=2)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, idx=3)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, idx=4)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, idx=5)(layer)

    line_ins = kl.TimeDistributed(kl.Dense(15))(layer)
    line_ins = kl.BatchNormalization()(line_ins)
    debug_layer = line_ins
    line_ins = kl.TimeDistributed(kl.Softmax(name='instance_distribution'))(line_ins)

    loss, metrics = get_kl_losses_and_metrics(line_ins, label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, 15)
    opt = SGD(lr=0.0015, momentum=0.9)
    line_model = km.Model(inputs=[line_inputs, label_input, valid_input, bg_input, fake_input,
                               unique_label_input, cluster_count_input],
                          outputs=line_ins,
                          name='line_net_model')
    line_model.compile(loss=loss,
                       optimizer='adam',
                       metrics=[iou] + metrics + debug_metrics(debug_layer),
                       experimental_run_tf_function=False)
    return line_model


# NOT USED
def cluster_class_model(line_num_attr, num_lines, num_classes, img_shape):
    # Inputs for geometric line information.
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    one_input = kl.Input(shape=(1, 1), dtype='float32', name='ones')

    img_feature_size = 128
    geometric_size = 384
    head_units = 512
    hidden_units = head_units * 4

    num_multis = 4
    num_adds = 4
    key_size = 64

    dropout = 0.0

    # The geometric embedding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(geometric_size, kernel_constraint=kc.max_norm(3)),
                                         name='geometric_features')(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.Dropout(dropout))(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = kl.LeakyReLU()(line_embeddings)

    # The virtual camera image feature cnn:
    line_img_features = kl.Masking(mask_value=0.0)(img_inputs)
    line_img_features = get_img_model(img_input_shape, img_feature_size, dropout=0.,
                                      trainable=False)(line_img_features)

    line_embeddings = kl.Concatenate()([line_img_features, line_embeddings])

    # Build 5 multi head attention layers. Hopefully this will work.
    layer = get_inter_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                      hidden_units=hidden_units, idx=0, n_multi_heads=num_multis,
                                      n_add_heads=num_adds)(line_embeddings)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                      hidden_units=hidden_units, idx=1, n_multi_heads=num_multis,
                                      n_add_heads=num_adds)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                      hidden_units=hidden_units, idx=2, n_multi_heads=num_multis,
                                      n_add_heads=num_adds)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                      hidden_units=hidden_units, idx=3, n_multi_heads=num_multis,
                                      n_add_heads=num_adds)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                      hidden_units=hidden_units, idx=4, n_multi_heads=num_multis,
                                      n_add_heads=num_adds)(layer)
    output = get_global_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                        hidden_units=hidden_units, n_multi_heads=num_multis,
                                        n_add_heads=num_adds)([layer, one_input])
    output = kl.Dense(num_classes)(output)
    output = kl.BatchNormalization()(output)
    output = kl.Softmax(name='class_logits')(output)

    model = km.Model(inputs=[line_inputs, valid_input, img_inputs, one_input],
                     outputs=output,
                     name='cluster_model')

    opt = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                       optimizer=opt,
                       metrics=['categorical_accuracy'],
                       experimental_run_tf_function=False)

    return model