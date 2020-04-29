import numpy as np
np.random.seed(123)

import time

import keras
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
    num_valid = tf.reduce_sum(tf.cast(validity_mask, dtype='float'))

    # norms = K.expand_dims(K.sqrt(K.sum(K.square(compare_tensor), axis=3)), -1)

    # Losses:

    def compare_norm_loss(y_true, y_pred):
        #loss_layer = K.maximum(0., K.square(1. - norm_tensor) - 0.04) * mask_equal + \
        #             K.maximum(0., K.square(norm_tensor - 0.) - 0.04) * mask_not_equal
        loss_layer = tf.nn.sigmoid_cross_entropy_with_logits(K.ones_like(norm_tensor), norm_tensor) * mask_equal + \
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

    positives = K.cast(K.greater(K.sigmoid(norm_tensor), 0.8), dtype='float32')
    negatives = K.cast(K.greater(0.2, K.sigmoid(norm_tensor)), dtype='float32')

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

    return loss, [compare_norm_loss, compare_norm_loss_2, tp_over_pred_p, tp_over_gt_p]
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
        out = kl.Dense(100, activation='relu')(feature_layer)
        out = kl.Dense(1, activation=None)(out)

        return out

    return img_compare_layer


def normalize_embeddings(input):
    return K.l2_normalize(input, axis=3)


def multiply(factor):
    def layer(input_tensor):
        return input_tensor * factor

    return layer


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
    img_inputs = Input(shape=(num_lines, img_shape[0], img_shape[1], img_shape[2]), dtype='float32', name='images')
    line_inputs = Input(shape=(num_lines, line_num_attr), dtype='float32', name='lines')
    label_input = Input(shape=(num_lines,), dtype='int32', name='labels')
    validity_input = Input(shape=(num_lines,), dtype='bool', name='valid_input_mask')
    bg_input = Input(shape=(num_lines,), dtype='bool', name='background_mask')

    # Image classifier:
    # img_shape = Input(shape=(224, 224, 3), dtype='float32')
    back_bone = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                               input_shape=(img_shape[0], img_shape[1], img_shape[2]))
    img_emb = kl.Concatenate(axis=-1)([kl.GlobalMaxPool2D()(back_bone.output),
                                       kl.GlobalAveragePooling2D()(back_bone.output)])
    img_emb = kl.Dense(100, activation='relu')(img_emb)
    img_model = keras.models.Model(inputs=back_bone.input, outputs=img_emb)

    img_features = kl.TimeDistributed(img_model)(img_inputs)
    img_norms = kl.Lambda(img_compare_features(num_lines), name='img_norms')(img_features)

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
    first_order_norm_model.add(kl.Lambda(drop_diagonal_and_mask((num_lines, num_lines, first_order_embedding_size),
                                                                validity_input)))

    compare_model_2 = Sequential(name='second_order_compare')
    compare_model_2.add(kl.Conv2D(2096, kernel_size=(1, 1),
                                  input_shape=(num_lines, num_lines, first_order_size*(n_neighbors*2 + 1)),
                                  activation='relu'))
    #compare_model_2.add(kl.Conv2D(600, kernel_size=(1, 1), activation='relu'))
    #compare_model_2.add(kl.Conv2D(600, kernel_size=(1, 1), activation='relu'))
    # compare_model_2.add(kl.Conv2D(1024, kernel_size=(1, 1), activation='relu'))
    #compare_model_2.add(kl.Conv2D(32, kernel_size=(1, 1), activation='relu'))
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

    line_model = Model(inputs=[img_inputs, line_inputs, label_input, validity_input, bg_input],
                       outputs=img_norms, #second_order_norm,  # line_embeddings,
                       name='line_net_model')
    sgd = SGD(lr=0.0001, momentum=0.9)
    line_model.compile(loss=losses,
                       optimizer=sgd,
                       metrics=metrics)

    return line_model


def data_generator(image_data_generator, max_line_count, line_num_attr, batch_size=1):
    while True:
        batch_geometries = []
        batch_labels = []
        batch_valid_mask = []
        batch_bg_mask = []
        batch_images = []

        for i in range(batch_size):
            geometries, labels, valid_mask, bg_mask, images = image_data_generator.next_batch(max_line_count)

            batch_labels.append(labels.reshape((1, max_line_count)))
            batch_geometries.append(geometries.reshape((1, max_line_count, line_num_attr)))
            batch_valid_mask.append(valid_mask.reshape((1, max_line_count)))
            batch_bg_mask.append(bg_mask.reshape((1, max_line_count)))
            batch_images.append(np.expand_dims(images, axis=0))

        geometries = np.concatenate(batch_geometries, axis=0)
        labels = np.concatenate(batch_labels, axis=0)
        valid_mask = np.concatenate(batch_valid_mask, axis=0)
        bg_mask = np.concatenate(batch_bg_mask, axis=0)
        images = np.concatenate(batch_images, axis=0)

        yield {'lines': geometries,
               'labels': labels,
               'valid_input_mask': valid_mask,
               'background_mask': bg_mask,
               'images': images}, labels


def train():
    # Paths to line files.
    train_files = "/nvme/line_ws/train_data/train"

    val_files = "/nvme/line_ws/train_data/val"

    # The length of the geometry vector of a line.
    line_num_attr = 15
    img_shape = (120, 180, 3)
    max_line_count = 150
    batch_size = 1
    margin = 1.0
    bg_classes = [0, 1, 2, 20, 22]

    train_data_generator = LineDataGenerator(train_files, bg_classes,
                                             shuffle=True,
                                             data_augmentation=True,
                                             img_shape=img_shape)
    train_set_mean = train_data_generator.get_mean()
    train_data_generator.set_mean(train_set_mean)
    print(train_set_mean)
    val_data_generator = LineDataGenerator(val_files, bg_classes, mean=train_set_mean, img_shape=img_shape)

    train_generator = data_generator(train_data_generator, max_line_count, line_num_attr, batch_size)
    val_generator = data_generator(val_data_generator, max_line_count, line_num_attr, batch_size)

    # Create line net Keras model.
    line_model = line_net_model(line_num_attr, max_line_count, img_shape, margin)
    line_model.summary()

    line_model.fit_generator(generator=train_generator,
                             verbose=1,
                             max_queue_size=1,
                             workers=0,
                             use_multiprocessing=False,
                             epochs=5,
                             steps_per_epoch=np.floor(train_data_generator.frame_count / batch_size),
                             validation_data=val_generator,
                             validation_steps=np.floor(val_data_generator.frame_count / batch_size))

    val_data_generator.set_pointer(0)
    for i in range(10):
        geometries, labels, valid_mask, bg_mask = val_data_generator.next_batch(max_line_count)

        labels = labels.reshape((1, max_line_count))
        geometries = geometries.reshape((1, max_line_count, line_num_attr))
        valid_mask = valid_mask.reshape((1, max_line_count))
        bg_mask = bg_mask.reshape((1, max_line_count))

        output = line_model.predict({'lines': geometries,
                                     'labels': labels,
                                     'valid_input_mask': valid_mask,
                                     'background_mask': bg_mask})

        print(output.shape)
        print(output[0, 0, :10, :])
        output = output.reshape((max_line_count, max_line_count))

        np.save("output/output_frame_{}".format(i), output)


if __name__ == '__main__':
    train()



