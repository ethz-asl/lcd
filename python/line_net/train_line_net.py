import numpy as np
np.random.seed(123)

import time

from keras.models import Sequential, Model
import keras.layers as kl
from keras.engine.input_layer import Input
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.losses import kullback_leibler_divergence

from keras import backend as K

import sys
sys.path.append("../")
from model import datagenerator_interiornet


def initialize_bias(shape, dtype=float, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def equal_loss(p):
    p_1 = p[0]
    p_2 = p[1]
    return kullback_leibler_divergence(p_1, p_2) + kullback_leibler_divergence(p_2, p_1)


def not_equal_loss(p):
    p_1 = p[0]
    p_2 = p[1]
    margin = K.constant(0.3, shape=(1,))
    e_1 = kullback_leibler_divergence(p_1, p_2)
    e_2 = kullback_leibler_divergence(p_2, p_1)
    return K.max([K.constant(0., shape=(1,)), kl.subtract([margin, e_1])]) + \
           K.max([K.constant(0., shape=(1,)), kl.subtract([margin, e_2])])


def line_net_loss(line_embeddings, labels, num_lines, margin):
    equal_loss_layer = kl.Lambda(equal_loss)
    not_equal_loss_layer = kl.Lambda(not_equal_loss)

    # Custom KL cross divergence instancing loss.
    def loss(y_true, y_pred):
        instancing_losses = []
        # Accumulate KL losses (only one for each line pair).
        for i in range(num_lines):
            for j in range(i, num_lines):
                R_i_j = K.cast(K.equal(labels[i], labels[j]), 'float32')
                instancing_losses.append(R_i_j * equal_loss_layer([line_embeddings[i], line_embeddings[j]]) +
                                         (1. - R_i_j) * not_equal_loss_layer([line_embeddings[i], line_embeddings[j]]))
        return K.sum(instancing_losses)

    return loss


def kl_cross_div_loss(embeddings_tensor, labels_tensor, num_lines, embedding_len, margin):
    def loss(y_true, y_pred):
        print(labels_tensor.shape)
        print(y_pred.shape)

        repeated_pred = K.repeat_elements(embeddings_tensor, num_lines, 2)
        h_pred = K.permute_dimensions(repeated_pred, (0, 1, 2, 3))
        v_pred = K.permute_dimensions(repeated_pred, (0, 2, 1, 3))

        print(h_pred.shape)
        print(v_pred.shape)

        labels = K.repeat_elements(K.cast(labels_tensor, dtype='int32'), num_lines, -2)

        h_mask = labels
        h_mask = K.repeat_elements(h_mask, embedding_len, axis=-1)
        v_mask = K.permute_dimensions(labels, (0, 2, 1, 3))
        v_mask = K.repeat_elements(v_mask, embedding_len, axis=-1)

        print(h_mask.shape)
        print(v_mask.shape)

        mask_equal = K.cast(K.equal(h_mask, v_mask), dtype='float32')
        mask_not_equal = K.cast(K.not_equal(h_mask, v_mask), dtype='float32')
        #ones = K.ones((num_lines, num_lines, embedding_len), dtype='float32')
        #zeros = K.zeros((num_lines, num_lines, embedding_len), dtype='float32')
        #margins = K.constant(margin, dtype='float32', shape=(num_lines, num_lines, embedding_len))

        d_1_2 = h_pred * K.log(h_pred / v_pred)
        d_2_1 = v_pred * K.log(v_pred / h_pred)

        print(d_1_2.shape)

        loss_layer = mask_equal * (d_1_2 + d_2_1) + \
            (1. - mask_equal) * (K.maximum(0., margin - d_1_2) + K.maximum(0., margin - d_2_1))
        #loss_layer = K.maximum(0., margin - d_2_1)# mask_not_equal * (K.maximum(0., margin - d_1_2) + K.maximum(0., margin - d_2_1))

        print("Loss layer shape.")
        print(loss_layer.shape)

        return K.sum(loss_layer, axis=(0, 1, 2, 3))

    return loss


def expand_lines(num_lines):
    def expand_lines_layer(line_inputs):
        repeated = K.repeat_elements(line_inputs, num_lines, -1)
        h_input = K.permute_dimensions(repeated, (0, 1, 3, 2))
        v_input = K.permute_dimensions(repeated, (0, 3, 1, 2))
        return K.concatenate([h_input, v_input], axis=-1)

    return expand_lines_layer


def line_net_model(line_num_attr, num_lines, margin):
    """
    Model architecture
    """

    # Some attributes for quick changing:
    first_order_embedding_size = 32
    output_embedding_size = 32

    # Inputs for geometric line informations.
    line_inputs = Input(shape=(num_lines, line_num_attr, 1), dtype='float32', name='lines')
    label_input = Input(shape=(num_lines, 1, 1), dtype='int32', name='labels')

    expanded_input = kl.Lambda(expand_lines(num_lines), name='expand_input')(line_inputs)

    # NN for line distance metrics.
    # Output embedding size is 32.
    compare_model = Sequential(name='first_order_compare')
    compare_model.add(kl.Convolution2D(60, kernel_size=(1, 1),
                                       input_shape=(num_lines, num_lines, line_num_attr * 2), activation='relu'))
    compare_model.add(kl.Convolution2D(60, kernel_size=(1, 1), activation='relu'))
    compare_model.add(kl.Convolution2D(first_order_embedding_size, kernel_size=(1, 1), activation='sigmoid'))
    compare_model.add(kl.AveragePooling2D((1, num_lines)))

    # NN for instancing distributions.
    # Output embedding size (max number of instances) is 16.
    instancing_model = Sequential(name='instancing')
    instancing_model.add(kl.Convolution2D(50, kernel_size=(1, 1),
                                          input_shape=(num_lines, 1, first_order_embedding_size), activation='relu'))
    instancing_model.add(kl.Convolution2D(output_embedding_size, kernel_size=(1, 1), activation='sigmoid'))
    instancing_model.add(kl.Softmax(axis=-1))

    line_embeddings = instancing_model(compare_model(expanded_input))

    #for i in range(num_lines):
    #    line_inputs.append(Input((line_num_attr,), dtype='float32', name="line_{}".format(i)))

    # Input for labels.

    # NN for line distance metrics.
    # Output embedding size is 32.
    #model_1 = Sequential(name="first_order_compare")
    #model_1.add(Dense(50, input_shape=(line_num_attr,), activation='relu', kernel_regularizer=l2(1e-3)))
    #model_1.add(Dense(300, activation='relu', kernel_regularizer=l2(1e-3), bias_initializer=initialize_bias))
    #model_1.add(Dense(200, activation='relu', kernel_regularizer=l2(1e-3), bias_initializer=initialize_bias))
    #model_1.add(Dense(first_order_embedding_size, activation='sigmoid', kernel_regularizer=l2(1e-3)))

    # NN for instancing distributions.
    # Output embedding size (max number of instances) is 16.
    #model_2 = Sequential(name="instancing")
    #model_2.add(Dense(50, input_shape=(first_order_embedding_size,), activation='relu'))
    #model_2.add(Dense(30, activation='relu', kernel_regularizer=l2(1e-3), bias_initializer=initialize_bias))
    #model_2.add(Dense(output_embedding_size, activation='sigmoid', kernel_regularizer=l2(1e-3)))

    # The averaged embedding for each line is created in the following.
    #line_embeddings = []
    #for i in range(num_lines):
    #    # Line distance embedding is obtained from line i with all other lines.
    #    row = []
    #    for j in range(num_lines):
    #        if i != j:
    #            line_pair = concatenate([line_inputs[i], line_inputs[j]], axis=0)
    #            row.append(model_1(line_pair))

        # These distance embeddings are then averaged to obtain the averaged embedding.
        # row_tensor = Concatenate(row, axis=1)
    #    row_mean = average(row)

        # Apply fully connected layers to obtain instance embedding for each line.
        # TODO: Add hinge loss for norm.
    #    line_embeddings.append(model_2(row_mean))

    #inputs = line_inputs
    #inputs.append(label_input)
    # TODO: Define and compile model.
    line_model = Model(inputs=[line_inputs, label_input], outputs=line_embeddings, name='line_net_model')
    sgd = SGD(lr=0.001, momentum=0.9)
    line_model.compile(loss=kl_cross_div_loss(line_embeddings, label_input, num_lines, output_embedding_size, margin),#line_net_loss(line_embeddings=line_embeddings, labels=label_input,
                            #              num_lines=num_lines, margin=margin),
                       optimizer=sgd)
                       #metrics=line_net_loss(line_embeddings=line_embeddings, labels=label_input,
                       #                      num_lines=num_lines, margin=margin))

    return line_model


def data_generator(image_data_generator, batch_size):
    while True:
        images, labels, line_types, geometries = image_data_generator.next_batch(batch_size)

        labels = np.array(labels).reshape((1, batch_size, 1, 1))
        geometries = np.array([geometries]).reshape((1, batch_size, 14, 1))
        yield {'lines': geometries, 'labels': labels}, labels


def train():
    # Paths to line files.
    train_files = [
        "/home/felix/line_ws/data/line_tools/interiornet_lines_split/train_with_line_endpoints.txt"
    ]

    val_files = [
        "/home/felix/line_ws/data/line_tools/interiornet_lines_split/val_with_line_endpoints.txt"
    ]

    # The length of the geometry vector of a line.
    line_num_attr = 14
    batch_size = 128
    margin = 1.3

    # TODO: Implement mean.
    train_data_generator = datagenerator_interiornet.ImageDataGenerator(train_files,
                                                                        np.array([0, 0, 0, 0]),
                                                                        image_type='bgr-d',
                                                                        shuffle=True)  # mean
    val_data_generator = datagenerator_interiornet.ImageDataGenerator(val_files,
                                                                      np.array([0, 0, 0, 0]),
                                                                      image_type='bgr-d')  # mean

    train_generator = data_generator(train_data_generator, batch_size)
    val_generator = data_generator(val_data_generator, batch_size)

    # Create line net Keras model.
    #tic = time.perf_counter()
    line_model = line_net_model(line_num_attr, batch_size, margin)
    #toc = time.perf_counter()
    #print("Time taken to load model: {}".format(toc-tic))
    line_model.summary()

    line_model.fit_generator(generator=train_generator,
                             verbose=1,
                             max_queue_size=1,
                             workers=0,
                             use_multiprocessing=False,
                             epochs=10,
                             steps_per_epoch=np.floor(train_data_generator.data_size / batch_size),
                             validation_data=val_generator,
                             validation_steps=np.floor(val_data_generator.data_size / batch_size))

    images, labels, line_types, geometries = train_data_generator.next_batch(batch_size)
    labels = np.array(labels).reshape((1, batch_size, 1, 1))
    geometries = np.array([geometries]).reshape((1, batch_size, 14, 1))
    output = line_model.predict({'lines': geometries, 'labels': labels})
    print(output.shape)
    print(output[0, 0, 0, :])
    output.reshape((128, 32))
    print(output[0, :])
    print(output[0, :, :, :])
    print(output)


if __name__ == '__main__':
    train()



