"""
The file containing all models used during place recognition: The descriptor model and the clustering model.
Additionally, a model for pretraining of the image weights.
"""

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
    """
    A debug layer to check if the mask is passed through.
    """
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
    """
    A layer to add a custom mask.
    """
    def __init__(self, custom_mask):
        super().__init__()
        self._expects_mask_arg = True
        self.custom_mask = custom_mask

    def get_config(self):
        return super().get_config()

    def compute_mask(self, inputs, mask=None):
        return self.custom_mask

    def call(self, inputs, mask=None):
        return inputs


def get_img_model(input_shape, output_dim, dropout=0.3, trainable=True):
    """
    Returns the image encoder model, consisting of the first 7 layers of the VGG16 network.
    :param input_shape: The shape of the image inputs.
    :param output_dim: The dimension of the image encoding.
    :param dropout: Dropout to be applied to the activations of the fully connected layers.
    :param trainable: Set to False if the image encoding weights should be frozen.
    :return: A Keras model object containing the image encoder model.
    """
    input_imgs = kl.Input(input_shape)

    # First 7 layers of vgg16.
    model = km.Sequential(name="vgg16_features")
    model.add(kl.TimeDistributed(
        kl.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block1_conv1", trainable=False
    ))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block1_conv2", trainable=False
    ))
    model.add(kl.TimeDistributed(kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        name="block2_conv1", trainable=False
    ))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block2_conv2", trainable=False
    ))
    model.add(kl.TimeDistributed(kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block3_conv1", trainable=False
    ))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation="relu"),
        name="block3_conv2", trainable=False
    ))
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


def get_multi_head_attention_model(input_shape, output_size, dropout=0.2, idx=0, key_size=128, n_multi=2, n_add=2):
    """
    Create the scaled multi head attention model as described in the paper.
    :param input_shape: The shape of the inputs of this layer.
    :param output_size: The shape of the outputs of this layer.
    :param dropout: The drop_out ratio of the fully connected layer.
    :param idx: The unique index of this layer (only for the name).
    :param key_size: The dimensionality of the heads.
    :param n_multi: The number of dot product attention heads.
    :param n_add: The number of additive attention heads.
    :return: A Keras model of the scaled multi head attention layer.
    """
    assert n_multi + n_add >= 0

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


def get_inter_attention_layer(input_number, input_shape=0, head_units=256, hidden_units=1024,
                              idx=0, dropout=0.2, key_size=128, n_multi_heads=2, n_add_heads=2):
    """
    Create the residual scaled multi head attention module with residual fully connected modules as described in the
    paper.
    :param input_number: The maximum number of lines.
    :param input_shape: The dimensionality of the line embeddings.
    :param head_units: The dimensionality of the output line embeddings.
    :param hidden_units: The dimensionality of the hidden layer in the residual fully connected modules.
    :param idx: The unique index of this layer (only for the name).
    :param dropout: The drop_out ratio of the fully connected layers.
    :param key_size: The dimensionality of the attention heads.
    :param n_multi_heads: The number of dot product attention heads.
    :param n_add_heads: The number of additive attention heads.
    :return: A Keras model of the residual scaled multi head attention module with residual fully connected modules.
    """
    if input_shape == 0:
        input_shape = head_units

    model_input = kl.Input(shape=(input_number, input_shape))

    if n_multi_heads + n_add_heads > 0:
        layer = get_multi_head_attention_model((input_number, input_shape),
                                               head_units,
                                               idx=idx,
                                               dropout=dropout,
                                               key_size=key_size,
                                               n_multi=n_multi_heads,
                                               n_add=n_add_heads)(model_input)
        # For the residual connections.
        layer = kl.Add()([layer, model_input])
    else:
        layer = model_input
    layer = get_non_linearity()(layer)

    # Two layers of dense connections running in parallel after the attention layer.
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
    """
    Get the global attention modules with residual fully connected modules for the descriptor network as described
    in the paper. These layers gather information over all lines and output them in an embedding.
    Multiple layers can be connected at the end.
    :param input_number: The maximum number of lines.
    :param one_input_count: The input dimension of the global attention layer. If this is one, the network will
                            learn the query, if not, the network will transform the one_input into a query with a
                            learned layer.
    :param idx: The unique index of this layer (only for the name).
    :param head_units: The dimensionality of the output line embeddings.
    :param end_head_units: The dimensionality of the output of this global attention layer.
    :param hidden_units: The dimensionality of the hidden layer in the residual fully connected modules.
    :param dropout: The drop_out ratio of the fully connected layers.
    :param key_size: The dimensionality of the attention heads.
    :param n_multi_heads: The number of dot product attention heads.
    :param n_add_heads: The number of additive attention heads.
    :return: A Keras model of the global attention module with residual fully connected modules.
    """
    assert n_multi_heads + n_add_heads >= 0

    output_size = end_head_units

    model_input = kl.Input(shape=(input_number, head_units))
    one_input = kl.Input(shape=(1, one_input_count))

    use_bias = not one_input_count == 1

    outputs_multi = []
    for i in range(n_multi_heads):
        # More layers can be added here.
        # For the keys, bias should actually be added. This should be changed in the future.
        keys = kl.TimeDistributed(kl.Dense(key_size, use_bias=use_bias))(model_input)
        keys = kl.TimeDistributed(kl.BatchNormalization())(keys)
        queries = kl.TimeDistributed(kl.Dense(key_size, use_bias=use_bias))(one_input)
        queries = kl.TimeDistributed(kl.BatchNormalization())(queries)
        output = kl.Attention(use_scale=True)([queries, keys])
        outputs_multi.append(output)

    outputs_add = []
    for i in range(n_add_heads):
        # More layers can be added here.
        # For the keys, bias should actually be added. This should be changed in the future.
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
    """
    Specify the type of non linearity used after the addition of residual layers.
    :return: The non_linearity layer. We use default LeakyReLu here.
    """
    return kl.LeakyReLU()


def image_pretrain_model(line_num_attr, num_lines, img_shape):
    """
    Creates a model used to pretrain the image encoding weights. It is the same as the clustering_and_description model just without
    any geometric data, and reduced dimensionality of all layers.
    :param line_num_attr: The dimensionality of the geometric input data.
    :param num_lines: The maximum number of lines.
    :param img_shape: The shape of the image inputs.
    :return: A ready to train Keras model to pretrain the image encoding weights.
    """
    # Inputs for geometric and visual line information.
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
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
    line_img_features = CustomMask(valid_input)(line_img_features)

    line_embeddings = line_img_features

    # Build 5 multi head attention modules.
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
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=4, key_size=key_size, n_multi_heads=num_multi,
                                      n_add_heads=num_add)(layer)

    line_ins = kl.TimeDistributed(kl.Dense(16))(layer)
    line_ins = kl.TimeDistributed(kl.BatchNormalization())(line_ins)
    line_ins = kl.TimeDistributed(kl.Softmax(name='instance_distribution'))(line_ins)
    line_model = km.Model(inputs=[line_inputs, label_input, valid_input, bg_input, img_inputs,
                                  unique_label_input, cluster_count_input],
                          outputs=line_ins,
                          name='line_net_model')

    # Set pretrained imagenet weights. This hack is necessary because vgg16 somehow does not work with the
    # time distributed layer in Keras...
    backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                 input_shape=img_input_shape[1:])
    line_model.get_layer("image_features").get_layer("vgg16_features").summary()
    backbone.summary()

    transfer_layers = ["block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2", "block3_conv1",
                       "block3_conv2", "block3_conv3"]

    for layer_name in transfer_layers:
        line_model.get_layer("image_features").get_layer("vgg16_features").get_layer(layer_name).set_weights(
            backbone.get_layer(layer_name).get_weights())

    loss, loss_metrics = get_kl_losses_and_metrics(label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, 15)
    bg_acc = bg_accuracy_metrics(bg_input, valid_input)
    metrics = loss_metrics + [iou] + bg_acc
    opt = Adam(learning_rate=0.0005)
    line_model.compile(loss=loss,
                       optimizer=opt,
                       metrics=metrics,
                       experimental_run_tf_function=False)

    return line_model, loss, opt, metrics


def line_net_model_fc(line_num_attr, num_lines, max_clusters, img_shape):
    """
    The clustering model without the attention layers used for comparison.
    :param line_num_attr: The dimensionality of the geometric input data.
    :param num_lines: The maximum number of lines.
    :param max_clusters: The maximum number of clusters distinguishable by the network.
    :param img_shape: The shape of the image inputs.
    :return: A ready to train Keras model for clustering, without attention layers.
    """
    # Inputs for geometric line information.
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
    unique_label_input = kl.Input(shape=(max_clusters,), dtype='int32', name='unique_labels')
    cluster_count_input = kl.Input(shape=(1,), dtype='int32', name='cluster_count')

    img_feature_size = 128
    geometric_size = 472
    head_units = 600
    hidden_units = head_units * 4

    # The number of attention heads is zero because we want a vanilla fully connected network here.
    num_multis = 0
    num_adds = 0
    key_size = 150

    dropout = 0.0

    # , kernel_constraint=kc.max_norm(3)

    # The geometric encoding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(geometric_size), name='geometric_features')(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.Dropout(dropout))(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = get_non_linearity()(line_embeddings)

    # The image encoding layer:
    line_img_features = kl.Masking(mask_value=0.0)(img_inputs)
    line_img_features = get_img_model(img_input_shape, img_feature_size, dropout=0.,
                                      trainable=False)(line_img_features)

    # Concatenate visual and geometric data.
    line_embeddings = kl.Concatenate()([line_img_features, line_embeddings])

    # Build 6 multi head attention layers.
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

    loss, loss_metrics = get_kl_losses_and_metrics(label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, max_clusters)
    bg_acc = bg_accuracy_metrics(bg_input, valid_input)

    metrics = loss_metrics + [iou] + bg_acc

    opt = Adam(learning_rate=0.00005)
    line_model.compile(loss=loss,
                       optimizer=opt,
                       metrics=metrics,
                       experimental_run_tf_function=False)

    return line_model, loss, opt, metrics


def line_net_model_4(line_num_attr, num_lines, max_clusters, img_shape):
    """
    The line clustering network used in the paper. It consists of 7 residual multi head attention modules with
    residual fully connected layers inbetween.
    :param line_num_attr: The dimensionality of the geometric input data.
    :param num_lines: The maximum number of lines.
    :param max_clusters: The maximum number of clusters distinguishable by the network.
    :param img_shape: The shape of the image inputs.
    :return: A ready to train Keras model for clustering.
    """
    # Inputs for geometric line information.
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = kl.Input(shape=(num_lines,), dtype='int32', name='labels')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = kl.Input(shape=(num_lines,), dtype='bool', name='background_mask')
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

    # The geometric encoding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(geometric_size), name='geometric_features')(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.Dropout(dropout))(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = get_non_linearity()(line_embeddings)

    # The visual encoding layer:
    line_img_features = kl.Masking(mask_value=0.0)(img_inputs)
    line_img_features = get_img_model(img_input_shape, img_feature_size, dropout=0.,
                                      trainable=False)(line_img_features)

    # Concatenate visual and geometric data.
    line_embeddings = kl.Concatenate()([line_img_features, line_embeddings])

    # Build 7 multi head attention layers. Hopefully this will work.
    layer = get_inter_attention_layer(num_lines, input_shape=head_units,
                                      head_units=head_units, dropout=dropout, key_size=key_size,
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

    loss, loss_metrics = get_kl_losses_and_metrics(label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, max_clusters)
    bg_acc = bg_accuracy_metrics(bg_input, valid_input)

    metrics = loss_metrics + [iou] + bg_acc

    opt = Adam(learning_rate=0.00005)
    line_model.compile(loss=loss,
                       optimizer=opt,
                       metrics=metrics,
                       experimental_run_tf_function=False)

    return line_model, loss, opt, metrics


def save_line_net_model(model, path):
    """
    Saves the clustering_and_description model weights during training.
    :param model: The model to be saved.
    :param path: The path the weights should be saved to.
    """
    # Somehow layers need to be set untrainable to save and load weights. This is due to a Keras bug.
    for layer in model.layers:
        layer.trainable = False
    model.save_weights(path)
    for layer in model.layers:
        layer.trainable = True
    # Freeze the image encoding network.
    model.get_layer("image_features").trainable = False


def load_line_net_model(path, line_num_attr, max_line_count, max_clusters, img_shape):
    """
    Loads the clustering_and_description model weights.
    :param path: The path to the file containing the weights.
    :param line_num_attr: The dimensionality of the geometric input data.
    :param max_line_count: The maximum number of lines.
    :param max_clusters: The maximum number of clusters distinguishable by the network.
    :param img_shape: The shape of the image inputs.
    :return: The Keras model of the clustering network with the loaded weights.
    """
    model, _, _, _ = line_net_model_4(line_num_attr, max_line_count, max_clusters, img_shape)
    # Somehow layers need to be set untrainable to save and load weights. This is due to a Keras bug.
    for layer in model.layers:
        layer.trainable = False
    model.load_weights(path, by_name=True)
    for layer in model.layers:
        layer.trainable = True
    # Freeze the image encoding network.
    model.get_layer("image_features").trainable = False

    return model


def cluster_embedding_model(line_num_attr, num_lines, embedding_dim, img_shape):
    """
    Creates the neural network for cluster description. It uses an residual multi head attention modules and
    global attention modules to condense the geometric and visual information of the lines into a cluster descriptor
    embedding.
    :param line_num_attr: The dimensionality of the geometric input data.
    :param num_lines: The maximum number of lines in a cluster.
    :param embedding_dim: The dimensionality of the descriptor embedding.
    :param img_shape: The shape of the image inputs.
    :return: A Keras model of the cluster description network.
    """
    img_input_shape = (num_lines, img_shape[0], img_shape[1], img_shape[2])
    line_inputs = kl.Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    valid_input = kl.Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    img_inputs = kl.Input(shape=img_input_shape, dtype='float32', name='images')

    # The one input is a complete hack and should be done differently. It is basically a one that is processed by
    # a linear layer that does not use bias so that it simulates trained variables.
    one_input = kl.Input(shape=(1, 1), dtype='float32', name='ones_model')

    img_feature_size = 128
    geometric_size = 384
    head_units = 512
    hidden_units = head_units * 4

    num_multis = 4
    num_adds = 4
    key_size = 128

    dropout = 0.0

    # The geometric encoding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(geometric_size, kernel_constraint=kc.max_norm(3)),
                                         name='geometric_features')(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.Dropout(dropout))(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = kl.LeakyReLU()(line_embeddings)

    # The visual encoding network.
    line_img_features = kl.Masking(mask_value=0.0)(img_inputs)
    line_img_features = get_img_model(img_input_shape, img_feature_size, dropout=0.,
                                      trainable=False)(line_img_features)

    line_embeddings = kl.Concatenate()([line_img_features, line_embeddings])

    # Build 5 multi head attention layers.
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
    # Build two global attention layers at the tail to condense information of all lines.
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
    """
    Build the triplet training model, running three descriptor networks in parallel so that the embeddings can be
    compared and a triplet loss can be formulated.
    :param line_num_attr: The dimensionality of the geometric input data.
    :param num_lines: The maximum number of lines in a cluster.
    :param embedding_dim: The dimensionality of the descriptor embedding.
    :param img_shape: The shape of the image inputs.
    :param margin: The margin hyper parameter for the triplet loss.
    :return: A ready to train Keras model of a triplet of cluster descriptor networks..
    """
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
    """
    Save the clustering model weights in the specified path.
    :param model: The clustering model.
    :param path: The path where the weights should be written to.
    """
    # Somehow layers need to be set untrainable to save and load weights. This is due to a Keras bug.
    for layer in model.layers:
        layer.trainable = False
    model.save_weights(path)
    for layer in model.layers:
        layer.trainable = True
    # Freeze the image encoding network.
    model.get_layer("cluster_embedding_model").get_layer("image_features").trainable = False


def load_cluster_embedding_model(path, line_num_attr, num_lines, embedding_dim, img_shape, margin):
    """
    Loads the cluster descriptor model and the weights from the path for inference.
    :param path: The path to the file containing the weights of the model.
    :param line_num_attr: The dimensionality of the geometric input data.
    :param num_lines: The maximum number of lines in a cluster.
    :param embedding_dim: The dimensionality of the descriptor embedding.
    :param img_shape: The shape of the image inputs.
    :param margin: The margin hyper parameter for the triplet loss.
    :return: The cluster descriptor model with the weights.
    """
    triplet_model = load_cluster_triplet_model(path, line_num_attr, num_lines, embedding_dim, img_shape, margin)
    return triplet_model.get_layer('cluster_embedding_model')


def load_cluster_triplet_model(path, line_num_attr, num_lines, embedding_dim, img_shape, margin):
    """
    Load the entire triplet model for training of the cluster descriptor model. Used during training.
    :param path: The path to the file containing the weights of the model.
    :param line_num_attr: The dimensionality of the geometric input data.
    :param num_lines: The maximum number of lines in a cluster.
    :param embedding_dim: The dimensionality of the descriptor embedding.
    :param img_shape: The shape of the image inputs.
    :param margin: The margin hyper parameter for the triplet loss.
    :return: The triplet model for training of the cluster descriptor network.
    """
    model = cluster_triplet_loss_model(line_num_attr, num_lines, embedding_dim, img_shape, margin)
    # Somehow layers need to be set untrainable to save and load weights. This is due to a Keras bug.
    for layer in model.layers:
        layer.trainable = False
    model.load_weights(path, by_name=True)
    for layer in model.layers:
        layer.trainable = True
    # Freeze the image encoding network.
    model.get_layer('cluster_embedding_model').get_layer("image_features").trainable = False

    return model

