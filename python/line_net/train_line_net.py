import numpy as np
np.random.seed(123)

import time

from keras.models import Sequential, Model
import keras.layers as kl
from keras.engine.input_layer import Input
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2
from keras.layers.merge import concatenate
from keras.losses import kullback_leibler_divergence

from keras import backend as K
import tensorflow as tf

from datagenerator_framewise import LineDataGenerator


def initialize_bias(shape, dtype=float, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def drop_diagonal_and_mask(shape, validity_mask):
    eye = K.expand_dims(K.eye(size=(shape[0], shape[1]), dtype='float32'), axis=-1)

    validity_expanded = K.expand_dims(K.expand_dims(validity_mask, axis=-1), axis=-1)
    h_mask = K.cast(K.repeat_elements(validity_expanded, shape[0], -2), dtype='float32')
    v_mask = K.permute_dimensions(h_mask, (0, 2, 1, 3))

    def layer(tensor):
        print(tensor.shape)
        return (1. - eye) * tensor * h_mask * v_mask

    return layer


def get_n_max_neighbors(feature_tensor, norm_tensor, n=3):
    indices = tf.argsort(norm_tensor, axis=-2)
    max_features = tf.gather_nd(feature_tensor, indices[:, n])
    return tf.concat


def get_losses_and_metrics(compare_tensor, instancing_tensor, labels_tensor, validity_mask, bg_mask, num_lines, margin):
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

    norms = K.expand_dims(K.sqrt(K.sum(K.square(compare_tensor), axis=3)), -1)

    # Losses:

    def compare_norm_loss(y_true, y_pred):
        loss_layer = K.square(K.maximum(0., 1. - norms)) * mask_equal + \
                     K.square(K.maximum(0., norms - 0.)) * mask_not_equal
        return K.identity(loss_layer, name='compare_norm_loss')

    def kl_cross_div_loss(y_true, y_pred):
        repeated_pred = K.repeat_elements(instancing_tensor, num_lines, 1)
        h_pred = K.permute_dimensions(repeated_pred, (0, 1, 2, 3))
        v_pred = K.permute_dimensions(repeated_pred, (0, 2, 1, 3))
        d = h_pred * K.log(h_pred / v_pred)
        loss_layer = mask_equal * d + mask_not_equal * K.maximum(0., margin - d)
        return K.identity(K.sum(loss_layer, axis=3) / 32., name='kl_cross_div_loss')

    def loss(y_true, y_pred):
        return compare_norm_loss(y_true, y_pred) # + kl_cross_div_loss(y_true, y_pred) * 0.

    # Metrics:

    positives = K.cast(K.greater(norms, 0.6), dtype='float32')
    negatives = K.cast(K.greater(0.4, norms), dtype='float32')

    num_equal = K.sum(mask_equal, axis=(0, 1, 2, 3))
    num_not_equal = K.sum(mask_not_equal, axis=(0, 1, 2, 3))

    num_positives = K.sum(positives, axis=(0, 1, 2, 3))
    num_negatives = K.sum(negatives, axis=(0, 1, 2, 3))

    true_positives = K.sum(mask_equal * positives, axis=(0, 1, 2, 3))
    false_positives = K.sum(mask_not_equal * positives, axis=(0, 1, 2, 3))
    false_negatives = K.sum(mask_equal * negatives, axis=(0, 1, 2, 3))
    true_negatives = K.sum(mask_not_equal * negatives, axis=(0, 1, 2, 3))

    # Ratio true positives: true_positives / num_equal
    # Ratio true negatives: true_negatives / num_not_equal
    # Ratio false positives: false_positives / num_not_equal
    # Ratio false negatives: false_negatives / num_equal

    def tp_over_pred_p(y_true, y_pred):
        return K.identity(true_positives / num_positives, name="true_positives_of_all_predicted_positives")

    def tn_over_pred_n(y_ture, y_pred):
        return K.identity(true_negatives / num_negatives, name="tn_over_pred_n")

    def tp_over_gt_p(y_true, y_pred):
        return K.identity(true_positives / num_equal, name="true_positives_of_all_ground_truth_positives")

    def tn_over_gt_n(y_true, y_pred):
        return K.identity(true_negatives / num_not_equal, name="tn_over_gt_n")

    return loss, [tp_over_pred_p, tp_over_gt_p]
    # [compare_norm_loss, kl_cross_div_loss, tp_over_pred_p, tn_over_pred_n, tp_over_gt_p, tn_over_gt_n]


def expand_lines(num_lines):
    def expand_lines_layer(line_inputs):
        expanded = K.expand_dims(line_inputs, axis=-1)
        repeated = K.repeat_elements(expanded, num_lines, -1)
        h_input = K.permute_dimensions(repeated, (0, 1, 3, 2))
        v_input = K.permute_dimensions(repeated, (0, 3, 1, 2))
        return K.concatenate([h_input, v_input], axis=-1)

    return expand_lines_layer


def normalize_embeddings(input):
    return K.l2_normalize(input, axis=3)


def multiply(factor):
    def layer(input_tensor):
        return input_tensor * factor

    return layer


def line_net_model(line_num_attr, num_lines, margin):
    """
    Model architecture
    """

    # Some attributes for quick changing:
    first_order_embedding_size = 1
    output_embedding_size = 32

    # Inputs for geometric line information.
    line_inputs = Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = Input(shape=(num_lines,), dtype='int32', name='labels')
    validity_input = Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = Input(shape=(num_lines,), dtype='bool', name='background_mask')

    expanded_input = kl.Lambda(expand_lines(num_lines), name='expand_input')(line_inputs)

    # NN for line distance metrics.
    # Output embedding size is 64.
    compare_model = Sequential(name='first_order_compare')
    compare_model.add(kl.Convolution2D(60, kernel_size=(1, 1),
                                       input_shape=(num_lines, num_lines, line_num_attr * 2), activation='relu'))
    compare_model.add(kl.Convolution2D(60, kernel_size=(1, 1), activation='relu'))
    compare_model.add(kl.Convolution2D(16, kernel_size=(1, 1), activation='sigmoid'))
    compare_model.add(kl.Convolution2D(first_order_embedding_size, kernel_size=(1, 1), activation='sigmoid'))
    # compare_model.add(kl.Lambda(multiply(4. / first_order_embedding_size)))
    compare_model.add(kl.Lambda(drop_diagonal_and_mask((num_lines, num_lines, first_order_embedding_size),
                                                       validity_input)))

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
    line_embeddings = instancing_model(compare_embeddings)

    losses, metrics = get_losses_and_metrics(compare_embeddings,
                                             line_embeddings,
                                             label_input,
                                             validity_input,
                                             bg_input,
                                             num_lines,
                                             margin)

    line_model = Model(inputs=[line_inputs, label_input, validity_input, bg_input],
                       outputs=compare_embeddings,  # line_embeddings,
                       name='line_net_model')
    # sgd = SGD(lr=0.001, momentum=0.9)
    line_model.compile(loss=losses,
                       optimizer=Adam(lr=0.004),
                       metrics=metrics)

    return line_model


def data_generator(image_data_generator, batch_size, line_num_attr):
    while True:
        geometries, labels, valid_mask, bg_mask = image_data_generator.next_batch(batch_size)

        labels = labels.reshape((1, batch_size))
        geometries = geometries.reshape((1, batch_size, line_num_attr))
        valid_mask = valid_mask.reshape((1, batch_size))
        bg_mask = bg_mask.reshape((1, batch_size))

        yield {'lines': geometries,
               'labels': labels,
               'valid_input_mask': valid_mask,
               'background_mask': bg_mask}, \
              labels


def train():
    # Paths to line files.
    train_files = "/home/felix/line_ws/data/line_tools/interiornet_lines_split/train"

    val_files = "/home/felix/line_ws/data/line_tools/interiornet_lines_split/val"

    # The length of the geometry vector of a line.
    line_num_attr = 14
    batch_size = 150
    margin = 1.0
    bg_classes = [0, 1, 2, 20, 22]

    train_data_generator = LineDataGenerator(train_files, bg_classes, shuffle=True, data_augmentation=True)
    train_set_mean = train_data_generator.get_mean()
    train_data_generator.set_mean(train_set_mean)
    print(train_set_mean)
    val_data_generator = LineDataGenerator(val_files, bg_classes, mean=train_set_mean)

    train_generator = data_generator(train_data_generator, batch_size, line_num_attr)
    val_generator = data_generator(val_data_generator, batch_size, line_num_attr)

    # Create line net Keras model.
    line_model = line_net_model(line_num_attr, batch_size, margin)
    line_model.summary()

    line_model.fit_generator(generator=train_generator,
                             verbose=1,
                             max_queue_size=1,
                             workers=0,
                             use_multiprocessing=False,
                             epochs=50,
                             steps_per_epoch=np.floor(val_data_generator.frame_count),
                             validation_data=val_generator,
                             validation_steps=np.floor(val_data_generator.frame_count))

    val_data_generator.set_pointer(0)
    for i in range(val_data_generator.frame_count):
        geometries, labels, valid_mask, bg_mask = val_data_generator.next_batch(batch_size)

        labels = labels.reshape((1, batch_size))
        geometries = geometries.reshape((1, batch_size, line_num_attr))
        valid_mask = valid_mask.reshape((1, batch_size))
        bg_mask = bg_mask.reshape((1, batch_size))

        output = line_model.predict({'lines': geometries,
                                     'labels': labels,
                                     'valid_input_mask': valid_mask,
                                     'background_mask': bg_mask})
        output = output.reshape((batch_size, batch_size))

        np.save("output/output_frame_{}".format(i), output)


if __name__ == '__main__':
    train()



