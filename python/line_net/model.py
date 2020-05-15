
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
from tensorflow.keras.optimizers import SGD, Adam

import tensorflow as tf

from losses_and_metrics import get_kl_losses_and_metrics
from losses_and_metrics import iou_metric
from losses_and_metrics import debug_metrics


def get_img_model(input_shape, output_dim, drop_out=0.3):
    input_imgs = kl.Input(input_shape)

    # First 7 layers of vgg16.
    model = km.Sequential()
    model.add(kl.TimeDistributed(
        kl.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu")
    ))
    model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")
    ))
    model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
    ))
    model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
    ))
    model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    ))
    model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    ))
    model.add(kl.TimeDistributed(kl.BatchNormalization()))
    model.add(kl.TimeDistributed(
        kl.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    ))
    model.add(kl.TimeDistributed(kl.BatchNormalization()))

    cnn_output = model(input_imgs)
    global_avg = kl.TimeDistributed(kl.GlobalAveragePooling2D())(cnn_output)
    global_max = kl.TimeDistributed(kl.GlobalMaxPool2D())(cnn_output)
    features = kl.Concatenate()([global_avg, global_max])
    features = kl.TimeDistributed(kl.Dense(output_dim * 4))(features)
    features = kl.TimeDistributed(kl.BatchNormalization())(features)
    features = kl.Activation('relu')(features)
    features = kl.TimeDistributed(kl.Dropout(drop_out))(features)
    features = kl.TimeDistributed(kl.Dense(output_dim))(features)
    features = kl.TimeDistributed(kl.BatchNormalization())(features)
    features = kl.Activation('relu')(features)

    return km.Model(inputs=input_imgs, outputs=features, name="image_features")


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

    return km.Model(inputs=model_input, outputs=output, name='multi_head_attention_{}'.format(idx))


def get_inter_attention_layer(input_number, head_units=256, hidden_units=1024,
                              idx=0, dropout=0.2, key_size=128, n_multi_heads=2, n_add_heads=2):
    model_input = kl.Input(shape=(input_number, head_units))

    layer = get_multi_head_attention_model((input_number, head_units),
                                           idx=idx,
                                           dropout=dropout,
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

    return km.Model(inputs=model_input, outputs=layer, name='inter_attention_{}'.format(idx))


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


def line_net_model_4(line_num_attr, num_lines, img_shape):
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

    head_units = 384
    hidden_units = head_units * 4

    # The virtual camera image feature cnn:
    line_img_features = get_img_model(img_input_shape, 128)

    # The geometric embedding layer:
    line_embeddings = kl.Masking(mask_value=0.0)(line_inputs)
    line_embeddings = kl.TimeDistributed(kl.Dense(256))(line_embeddings)
    # line_embeddings = kl.Dropout(0.1)(line_embeddings)
    line_embeddings = kl.TimeDistributed(kl.BatchNormalization())(line_embeddings)
    line_embeddings = kl.LeakyReLU()(line_embeddings)

    line_embeddings = kl.Concatenate()([line_img_features, line_embeddings])

    # Build 5 multi head attention layers. Hopefully this will work.
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=0)(line_embeddings)
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=1)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=2)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=3)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=4)(layer)
    layer = get_inter_attention_layer(num_lines, head_units=head_units,
                                      hidden_units=hidden_units, idx=5)(layer)

    line_ins = kl.TimeDistributed(kl.Dense(15))(layer)
    line_ins = kl.BatchNormalization()(line_ins)
    debug_layer = line_ins
    line_ins = kl.TimeDistributed(kl.Softmax(name='instance_distribution'))(line_ins)

    loss, metrics = get_kl_losses_and_metrics(line_ins, label_input, valid_input, bg_input, num_lines)
    iou = iou_metric(label_input, unique_label_input, cluster_count_input, bg_input, valid_input, 15)
    opt = SGD(lr=0.0015, momentum=0.9)
    line_model = km.Model(inputs=[line_inputs, label_input, valid_input, bg_input, fake_input, img_inputs,
                                  unique_label_input, cluster_count_input],
                          outputs=line_ins,
                          name='line_net_model')
    line_model.compile(loss=loss,
                       optimizer='adam',
                       metrics=[iou] + metrics + debug_metrics(debug_layer),
                       experimental_run_tf_function=False)
    return line_model
