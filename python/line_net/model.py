import os

import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
import tensorflow.keras.constraints as kc
from tensorflow.keras.optimizers import SGD, Adam

import tensorflow as tf

from losses_and_metrics import get_kl_losses_and_metrics
from losses_and_metrics import iou_metric
from losses_and_metrics import bg_accuracy_metrics
from losses_and_metrics import debug_metrics
import losses_and_metrics


class MaskLayer(kl.Layer):
    def __init__(self):
        super().__init__()
        self._expects_mask_arg = True

    def compute_mask(self, inputs, mask=None):
        print(mask.shape)
        return mask

    def call(self, inputs, mask=None):
        print(mask.shape)
        num_valid = tf.reduce_sum(tf.cast(mask, dtype='float32'))
        return inputs, num_valid


class CustomMask(kl.Layer):
    def __init__(self, custom_mask):
        super().__init__()
        self._expects_mask_arg = True
        self.custom_mask = custom_mask

    def compute_mask(self, inputs, mask=None):
        return self.custom_mask

    def call(self, inputs, mask=None):
        return inputs


class MaskedReLu(kl.ReLU):
    def __init__(self):
        super().__init__()
        self._expects_mask_arg = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        return super.call()


def get_img_model(input_shape, output_dim, dropout=0.3, trainable=True):
    input_imgs = kl.Input(input_shape)

    # First 7 layers of vgg16.
    model = km.Sequential(name="vgg16_features")
    model.add(kl.TimeDistributed(
        kl.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block1_conv1", trainable=False
    ))
    # model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block1_conv2", trainable=False
    ))
    # model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        name="block2_conv1", trainable=False
    ))
    # model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block2_conv2", trainable=False
    ))
    # model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))))
    # model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block3_conv1", trainable=False
    ))
    # model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block3_conv2", trainable=False
    ))
    # model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block3_conv3", trainable=False
    ))
    model.add(kl.TimeDistributed(kl.MaxPool2D(pool_size=(4, 4), strides=(4, 4))))
    model.add(kl.TimeDistributed(kl.Flatten()))

    cnn_output = model(input_imgs)
    features = kl.TimeDistributed(kl.Dense(output_dim * 8, ), name='img_dense1')(cnn_output)
    features = kl.TimeDistributed(kl.BatchNormalization(), name='img_bn1')(features)
    features = kl.Activation('relu')(features)
    features = kl.TimeDistributed(kl.Dropout(dropout))(features)
    features = kl.TimeDistributed(kl.Dense(output_dim), name='img_dense2')(features)
    features = kl.TimeDistributed(kl.BatchNormalization(), name='img_bn2')(features)
    features = kl.Activation('relu')(features)

    return km.Model(inputs=input_imgs, outputs=features, name="image_features", trainable=trainable)


def get_multi_head_attention_model(input_shape, dropout=0.2, idx=0, key_size=128, n_multi=2, n_add=2):
    assert n_multi + n_add >= 0

    output_size = input_shape[1]

    model_input = kl.Input(shape=input_shape)

    outputs_multi = []
    for i in range(n_multi):
        # More layers can be added here.
        keys = kl.TimeDistributed(kl.Dense(key_size))(model_input)
        keys = kl.TimeDistributed(kl.BatchNormalization())(keys)
        queries = kl.TimeDistributed(kl.Dense(key_size))(model_input)
        queries = kl.TimeDistributed(kl.BatchNormalization())(queries)
        output = kl.Attention(use_scale=True)([queries, keys])
        outputs_multi.append(output)

    outputs_add = []
    for i in range(n_add):
        # More layers can be added here.
        keys = kl.TimeDistributed(kl.Dense(key_size))(model_input)
        keys = kl.TimeDistributed(kl.BatchNormalization())(keys)
        queries = kl.TimeDistributed(kl.Dense(key_size))(model_input)
        queries = kl.TimeDistributed(kl.BatchNormalization())(queries)
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

    return km.Model(inputs=model_input, outputs=output, name='multi_head_attention_{}'.format(idx))


def get_inter_attention_layer(input_number, head_units=256, hidden_units=1024,
                              idx=0, dropout=0.2, key_size=128, n_multi_heads=2, n_add_heads=2):
    model_input = kl.Input(shape=(input_number, head_units))

    if n_multi_heads + n_add_heads > 0:
        layer = get_multi_head_attention_model((input_number, head_units),
                                               idx=idx,
                                               dropout=dropout,
                                               key_size=key_size,
                                               n_multi=n_multi_heads,
                                               n_add=n_add_heads)(model_input)
        layer = kl.Add()([layer, model_input])
    else:
        layer = model_input
    layer = get_non_linearity()(layer)

    # Two layers of dense connections running in parallel
    layer_2 = kl.TimeDistributed(kl.Dense(hidden_units))(layer)
    layer_2 = kl.TimeDistributed(kl.Dropout(dropout))(layer_2)
    layer_2 = kl.TimeDistributed(kl.BatchNormalization())(layer_2)
    layer_2 = get_non_linearity()(layer_2)
    layer_2 = kl.TimeDistributed(kl.Dense(head_units))(layer_2)
    layer_2 = kl.TimeDistributed(kl.Dropout(dropout))(layer_2)
    layer_2 = kl.TimeDistributed(kl.BatchNormalization())(layer_2)
    layer = kl.Add()([layer, layer_2])
    layer = get_non_linearity()(layer)

    return km.Model(inputs=model_input, outputs=layer, name='inter_attention_{}'.format(idx))


def get_global_attention_layer(input_number, one_input_count=1, idx=0,
                               head_units=512, end_head_units=1024, hidden_units=1024, dropout=0.2,
                               key_size=128, n_multi_heads=2, n_add_heads=2):
    assert n_multi_heads + n_add_heads >= 0

    output_size = end_head_units

    model_input = kl.Input(shape=(input_number, head_units))
    one_input = kl.Input(shape=(1, one_input_count))

    use_bias = not one_input_count == 1

    outputs_multi = []
    for i in range(n_multi_heads):
        # More layers can be added here.
        # For the keys, bias should actually be added.
        keys = kl.TimeDistributed(kl.Dense(key_size, use_bias=use_bias))(model_input)
        keys = kl.TimeDistributed(kl.BatchNormalization())(keys)
        queries = kl.TimeDistributed(kl.Dense(key_size, use_bias=use_bias))(one_input)
        queries = kl.TimeDistributed(kl.BatchNormalization())(queries)
        output = kl.Attention(use_scale=True)([queries, keys])
        outputs_multi.append(output)

    outputs_add = []
    for i in range(n_add_heads):
        # More layers can be added here.
        # For the keys, bias should actually be added.
        keys = kl.TimeDistributed(kl.Dense(key_size, use_bias=use_bias))(model_input)
        keys = kl.TimeDistributed(kl.BatchNormalization())(keys)
        queries = kl.TimeDistributed(kl.Dense(key_size, use_bias=use_bias))(one_input)
        queries = kl.TimeDistributed(kl.BatchNormalization())(queries)
        output = kl.AdditiveAttention(use_scale=True)([queries, keys])
        outputs_add.append(output)

    outputs = outputs_multi + outputs_add
    if len(outputs) > 1:
        output = kl.Concatenate()(outputs)
    else:
        output = outputs[0]

    output = kl.Dense(output_size)(output)
    output = kl.Dropout(dropout)(output)
    output = kl.LeakyReLU()(output)
    output = kl.BatchNormalization()(output)

    # Add a residual fully connected layer at the end.
    output_2 = kl.Dense(hidden_units)(output)
    output_2 = kl.BatchNormalization()(output_2)
    output_2 = kl.LeakyReLU()(output_2)
    output_2 = kl.Dense(output_size)(output_2)
    output_2 = kl.BatchNormalization()(output_2)

    output = kl.Add()([output, output_2])
    output = kl.LeakyReLU()(output)

    model = km.Model(inputs=[model_input, one_input], outputs=output, name='global_attention_{}'.format(idx))
    return model


def get_non_linearity():
    return kl.LeakyReLU()


def image_pretrain_model(line_num_attr, num_lines, img_shape):
    # Inputs for geometric line information.
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
    fake_input = kl.Input(shape=(num_lines, 15), dtype='float32', name='fake')
    unique_label_input = kl.Input(shape=(15,), dtype='int32', name='unique_labels')
    cluster_count_input = kl.Input(shape=(1,), dtype='int32', name='cluster_count')

    head_units = 128
    hidden_units = head_units * 4
    key_size = 64
    num_multi = 4
    num_add = 0

    # The virtual camera image feature cnn:
    line_img_features = kl.Masking(mask_value=0.0)(img_inputs)
    line_img_features = get_img_model(img_input_shape, head_units)(line_img_features)

    line_embeddings = line_img_features

    # Build 5 multi head attention layers. Hopefully this will work.
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=0, key_size=key_size, n_multi_heads=num_multi,
                                      n_add_heads=num_add)(line_embeddings)
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=1, key_size=key_size, n_multi_heads=num_multi,
                                      n_add_heads=num_add)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=2, key_size=key_size, n_multi_heads=num_multi,
                                      n_add_heads=num_add)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=3, key_size=key_size, n_multi_heads=num_multi,
                                      n_add_heads=num_add)(layer)

    line_ins = kl.TimeDistributed(kl.Dense(16))(layer)
    line_ins = kl.TimeDistributed(kl.BatchNormalization())(line_ins)
    debug_layer = line_ins
    line_ins = kl.TimeDistributed(kl.Softmax(name='instance_distribution'))(line_ins)
    line_model = km.Model(inputs=[line_inputs, label_input, valid_input, bg_input, fake_input, img_inputs,
                                  unique_label_input, cluster_count_input],
                          outputs=line_ins,
                          name='line_net_model')

    # Set pretrained imagenet weights. This hack is necessary because vgg16 somehow does not work with the
    # time distributed layer...
    backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                 input_shape=img_input_shape[1:])
    line_model.get_layer("image_features").get_layer("vgg16_features").summary()
    backbone.summary()

    transfer_layers = ["block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2", "block3_conv1",
                       "block3_conv2", "block3_conv3"]

    for layer_name in transfer_layers:
        line_model.get_layer("image_features").get_layer("vgg16_features").get_layer(layer_name).set_weights(
            backbone.get_layer(layer_name).get_weights())

    loss, loss_metrics = get_kl_losses_and_metrics(line_ins, label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, 15)
    bg_acc = bg_accuracy_metrics(bg_input, valid_input)
    metrics = loss_metrics + [iou] + bg_acc
    opt = SGD(lr=0.0015, momentum=0.9)
    opt = Adam(learning_rate=0.0002)
    line_model.compile(loss=loss,
                       optimizer=opt,
                       metrics=metrics,
                       experimental_run_tf_function=False)

    return line_model, loss, opt, metrics


def line_net_model_fc(line_num_attr, num_lines, max_clusters, img_shape):
    # Inputs for geometric line information.
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
    # fake_input = kl.Input(shape=(num_lines, 15), dtype='float32', name='fake')
    unique_label_input = kl.Input(shape=(max_clusters,), dtype='int32', name='unique_labels')
    cluster_count_input = kl.Input(shape=(1,), dtype='int32', name='cluster_count')

    img_feature_size = 128
    geometric_size = 472
    head_units = 600
    hidden_units = head_units * 4

    num_multis = 0
    num_adds = 0
    key_size = 150

    dropout = 0.0

    # , kernel_constraint=kc.max_norm(3)

    # The geometric embedding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(geometric_size), name='geometric_features')(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.Dropout(dropout))(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = get_non_linearity()(line_embeddings)

    # line_embeddings = kl.Add()([line_embeddings, line_embeddings_2])
    # line_embeddings = get_non_linearity()(line_embeddings)

    # The virtual camera image feature cnn:
    line_img_features = kl.Masking(mask_value=0.0)(img_inputs)
    line_img_features = get_img_model(img_input_shape, img_feature_size, dropout=0.,
                                      trainable=False)(line_img_features)

    line_embeddings = kl.Concatenate()([line_img_features, line_embeddings])
    # line_embeddings = CustomMask(valid_input)(line_embeddings)
    # line_embeddings = get_non_linearity()(line_embeddings)

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
    layer = get_inter_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                      hidden_units=hidden_units, idx=5, n_multi_heads=num_multis,
                                      n_add_heads=num_adds)(layer)

    line_ins = kl.TimeDistributed(kl.Dense(max_clusters + 1), name='instancing_layer')(layer)
    line_ins = kl.TimeDistributed(kl.BatchNormalization())(line_ins)
    _, num_mask = MaskLayer()(line_ins)
    line_ins = kl.TimeDistributed(kl.Softmax(name='instance_distribution'))(line_ins)

    line_model = km.Model(inputs=[line_inputs, label_input, valid_input, bg_input, img_inputs,
                                  unique_label_input, cluster_count_input],
                          outputs=line_ins,
                          name='line_net_model')

    loss, loss_metrics = get_kl_losses_and_metrics(line_ins, label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, max_clusters)
    bg_acc = bg_accuracy_metrics(bg_input, valid_input)

    metrics = loss_metrics + [iou] + bg_acc# + [num_mask]

    # opt = SGD(lr=0.0015, momentum=0.9)
    opt = Adam(learning_rate=0.00005)
    line_model.compile(loss=loss,
                       optimizer=opt,
                       metrics=metrics,
                       experimental_run_tf_function=False)

    return line_model, loss, opt, metrics


def line_net_model_4(line_num_attr, num_lines, max_clusters, img_shape):
    # Inputs for geometric line information.
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
    # fake_input = kl.Input(shape=(num_lines, 15), dtype='float32', name='fake')
    unique_label_input = kl.Input(shape=(max_clusters,), dtype='int32', name='unique_labels')
    cluster_count_input = kl.Input(shape=(1,), dtype='int32', name='cluster_count')

    img_feature_size = 128
    geometric_size = 472
    head_units = 600
    hidden_units = head_units * 4

    num_multis = 4
    num_adds = 4
    key_size = 150

    dropout = 0.0

    # , kernel_constraint=kc.max_norm(3)

    # The geometric embedding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(geometric_size), name='geometric_features')(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.Dropout(dropout))(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = get_non_linearity()(line_embeddings)

    # Residual FC for geometric input.
    line_embeddings_2 = kl.TimeDistributed(kl.Dense(geometric_size*4))(line_embeddings)
    line_embeddings_2 = kl.TimeDistributed(kl.Dropout(dropout))(line_embeddings_2)
    line_embeddings_2 = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings_2)
    line_embeddings_2 = get_non_linearity()(line_embeddings_2)
    line_embeddings_2 = kl.TimeDistributed(kl.Dense(geometric_size))(line_embeddings_2)
    line_embeddings_2 = kl.TimeDistributed(kl.Dropout(dropout))(line_embeddings_2)
    line_embeddings_2 = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings_2)

    # line_embeddings = kl.Add()([line_embeddings, line_embeddings_2])
    # line_embeddings = get_non_linearity()(line_embeddings)

    # The virtual camera image feature cnn:
    line_img_features = kl.Masking(mask_value=0.0)(img_inputs)
    line_img_features = get_img_model(img_input_shape, img_feature_size, dropout=0.,
                                      trainable=False)(line_img_features)

    line_embeddings = kl.Concatenate()([line_img_features, line_embeddings])
    # line_embeddings = CustomMask(valid_input)(line_embeddings)
    # line_embeddings = get_non_linearity()(line_embeddings)

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
    layer = get_inter_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                      hidden_units=hidden_units, idx=5, n_multi_heads=num_multis,
                                      n_add_heads=num_adds)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units, dropout=dropout, key_size=key_size,
                                      hidden_units=hidden_units, idx=6, n_multi_heads=num_multis,
                                      n_add_heads=num_adds)(layer)

    line_ins = kl.TimeDistributed(kl.Dense(max_clusters + 1), name='instancing_layer')(layer)
    line_ins = kl.TimeDistributed(kl.BatchNormalization())(line_ins)
    _, num_mask = MaskLayer()(line_ins)
    line_ins = kl.TimeDistributed(kl.Softmax(name='instance_distribution'))(line_ins)

    line_model = km.Model(inputs=[line_inputs, label_input, valid_input, bg_input, img_inputs,
                                  unique_label_input, cluster_count_input],
                          outputs=line_ins,
                          name='line_net_model')

    loss, loss_metrics = get_kl_losses_and_metrics(line_ins, label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, max_clusters)
    bg_acc = bg_accuracy_metrics(bg_input, valid_input)

    metrics = loss_metrics + [iou] + bg_acc# + [num_mask]

    # opt = SGD(lr=0.0015, momentum=0.9)
    opt = Adam(learning_rate=0.00005)
    line_model.compile(loss=loss,
                       optimizer=opt,
                       metrics=metrics,
                       experimental_run_tf_function=False)

    return line_model, loss, opt, metrics


def save_line_net_model(model, path):
    for layer in model.layers:
        layer.trainable = False
    model.save_weights(path)
    for layer in model.layers:
        layer.trainable = True
    model.get_layer("image_features").trainable = False


def load_line_net_model(path, line_num_attr, max_line_count, max_clusters, img_shape):
    model, _, _, _ = line_net_model_4(line_num_attr, max_line_count, max_clusters, img_shape)
    for layer in model.layers:
        layer.trainable = False
    model.load_weights(path, by_name=True)
    for layer in model.layers:
        layer.trainable = True
    model.get_layer("image_features").trainable = False

    return model


def cluster_embedding_model(line_num_attr, num_lines, embedding_dim, img_shape):
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')
    one_input = kl.Input(shape=(1, 1), dtype='float32', name='ones_model')

    img_feature_size = 128
    geometric_size = 384
    head_units = 512
    hidden_units = head_units * 4

    num_multis = 4
    num_adds = 4
    key_size = 128

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

    num_end_multis = 8
    num_end_adds = 8
    end_head_units = 1024
    end_hidden_units = 4 * end_head_units
    end_key_size = 64
    output = get_global_attention_layer(num_lines, one_input_count=1, idx=0,
                                        head_units=head_units, end_head_units=end_head_units,
                                        dropout=dropout, key_size=end_key_size,
                                        hidden_units=end_hidden_units, n_multi_heads=num_end_multis,
                                        n_add_heads=num_end_adds)([layer, one_input])
    output = get_global_attention_layer(num_lines, one_input_count=end_head_units, idx=1,
                                        head_units=head_units, end_head_units=end_head_units,
                                        dropout=dropout, key_size=end_key_size,
                                        hidden_units=end_hidden_units, n_multi_heads=num_end_multis,
                                        n_add_heads=num_end_adds)([layer, output])
    output = kl.Flatten()(output)
    output = kl.Dense(embedding_dim)(output)
    output = kl.BatchNormalization()(output)
    output = kl.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='embeddings')(output)

    model = km.Model(inputs=[line_inputs, valid_input, img_inputs, one_input],
                     outputs=output,
                     name='cluster_embedding_model')
    return model


def cluster_triplet_loss_model(line_num_attr, num_lines, embedding_dim, img_shape, margin):
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    line_inputs_a = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines_a')
    valid_input_a = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask_a')
    img_inputs_a = kl.Input(shape=img_input_shape, dtype='float32', name='images_a')
    line_inputs_p = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines_p')
    valid_input_p = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask_p')
    img_inputs_p = kl.Input(shape=img_input_shape, dtype='float32', name='images_p')
    line_inputs_n = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines_n')
    valid_input_n = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask_n')
    img_inputs_n = kl.Input(shape=img_input_shape, dtype='float32', name='images_n')
    one_input = kl.Input(shape=(1, 1), dtype='float32', name='ones')

    embedding_model = cluster_embedding_model(line_num_attr, num_lines, embedding_dim, img_shape)

    embedding_a = embedding_model([line_inputs_a, valid_input_a, img_inputs_a, one_input])
    embedding_p = embedding_model([line_inputs_p, valid_input_p, img_inputs_p, one_input])
    embedding_n = embedding_model([line_inputs_n, valid_input_n, img_inputs_n, one_input])

    model = km.Model(inputs=[line_inputs_a, valid_input_a, img_inputs_a,
                             line_inputs_p, valid_input_p, img_inputs_p,
                             line_inputs_n, valid_input_n, img_inputs_n, one_input],
                     outputs=embedding_a,
                     name='cluster_model')

    loss = losses_and_metrics.triplet_loss(embedding_a, embedding_p, embedding_n, margin=margin)
    metrics = losses_and_metrics.triplet_metrics(embedding_a, embedding_p, embedding_n, margin=margin)

    opt = Adam(learning_rate=0.0001)
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics,
                  experimental_run_tf_function=False)

    return model


def save_cluster_model(model, path):
    for layer in model.layers:
        layer.trainable = False
    model.save_weights(path)
    for layer in model.layers:
        layer.trainable = True
    model.get_layer("cluster_embedding_model").get_layer("image_features").trainable = False


def load_cluster_embedding_model(path, line_num_attr, num_lines, embedding_dim, img_shape, margin):
    triplet_model = load_cluster_triplet_model(path, line_num_attr, num_lines, embedding_dim, img_shape, margin)
    return triplet_model.get_layer('cluster_embedding_model')


def load_cluster_triplet_model(path, line_num_attr, num_lines, embedding_dim, img_shape, margin):
    model = cluster_triplet_loss_model(line_num_attr, num_lines, embedding_dim, img_shape, margin)
    for layer in model.layers:
        layer.trainable = False
    model.load_weights(path, by_name=True)
    for layer in model.layers:
        layer.trainable = True
    model.get_layer('cluster_embedding_model').get_layer("image_features").trainable = False

    return model


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

