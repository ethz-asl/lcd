import numpy as np
np.random.seed(123)

from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, average
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


def equal_loss(p_1, p_2):
    return kullback_leibler_divergence(p_1, p_2) + kullback_leibler_divergence(p_2, p_1)


def not_equal_loss(p_1, p_2, margin):
    e_1 = kullback_leibler_divergence(p_1, p_2)
    e_2 = kullback_leibler_divergence(p_2, p_1)
    return K.max(0, margin - e_1) + K.max(0, margin - e_2)


def line_net_loss(line_embeddings, labels, num_lines):
    equal_loss_layer = Lambda(equal_loss)
    not_equal_loss_layer = Lambda(not_equal_loss)

    # Custom KL cross divergence instancing loss.
    def loss(y_true, y_pred):
        instancing_losses = []
        # Accumulate KL losses (only one for each line pair).
        for i in range(num_lines):
            for j in range(i, num_lines):
                R_i_j = K.equal(labels[i], labels[j])
                instancing_losses.append(R_i_j * equal_loss_layer(line_embeddings[i], line_embeddings[j]) +
                                         (1 - R_i_j) * not_equal_loss_layer(line_embeddings[i], line_embeddings[j]))
        return K.sum(instancing_losses)

    return loss


def line_net_model(line_num_attr, num_lines):
    """
    Model architecture
    :param line_num_attr:
    :param num_lines:
    :return:
    """

    # Some attributes for quick changing:
    first_order_embedding_size = 32
    output_embedding_size = 16

    # Inputs for geometric line informations.
    line_input = Input((num_lines, line_num_attr), dtype='float32', name="line_input")

    # Input for labels.
    label_input = Input((num_lines,), dtype='int32', name="label_input")

    # NN for line distance metrics.
    # Output embedding size is 32.
    model_1 = Sequential(name="first_order_compare")
    model_1.add(Dense(200, input_shape=(line_num_attr,), activation='relu', kernel_regularizer=l2(1e-3),
                      bias_initializer=initialize_bias))
    model_1.add(Dense(300, activation='relu', kernel_regularizer=l2(1e-3), bias_initializer=initialize_bias))
    model_1.add(Dense(200, activation='relu', kernel_regularizer=l2(1e-3), bias_initializer=initialize_bias))
    model_1.add(Dense(first_order_embedding_size, activation='sigmoid', kernel_regularizer=l2(1e-3),
                      bias_initializer=initialize_bias))

    # NN for instancing distributions.
    # Output embedding size (max number of instances) is 16.
    model_2 = Sequential(name="instancing")
    model_2.add(Dense(50, input_shape=(first_order_embedding_size,), activation='relu', kernel_regularizer=l2(1e-3),
                      bias_initializer=initialize_bias))
    model_2.add(Dense(30, activation='relu', kernel_regularizer=l2(1e-3), bias_initializer=initialize_bias))
    model_2.add(Dense(output_embedding_size, activation='sigmoid', kernel_regularizer=l2(1e-3),
                      bias_initializer=initialize_bias))

    # The averaged embedding for each line is created in the following.
    line_embeddings = []
    for i in range(num_lines):
        # Line distance embedding is obtained from line i with all other lines.
        row = []
        for j in range(num_lines):
            if i != j:
                line_pair = concatenate([line_input[i, :], line_input[j, :]], axis=0)
                row.append(model_1(line_pair))

        # These distance embeddings are then averaged to obtain the averaged embedding.
        # row_tensor = Concatenate(row, axis=1)
        row_mean = average(row)

        # Apply fully connected layers to obtain instance embedding for each line.
        # TODO: Add hinge loss for norm.
        line_embeddings.append(model_2(row_mean))

    # TODO: Define and compile model.
    line_model = Model(inputs=[line_input, label_input], outputs=line_embeddings)
    sgd = SGD(lr=0.001, momentum=0.9)
    line_model.compile(loss=line_net_loss(line_embeddings=line_embeddings, labels=label_input, num_lines=num_lines),
                       optimizer=sgd,
                       metrics=line_net_loss(line_embeddings=line_embeddings, labels=label_input, num_lines=num_lines))

    return line_model


def data_generator(image_data_generator, batch_size):
    while True:
        images, labels, line_types, geometries = image_data_generator.next_batch(batch_size)

        yield geometries, labels


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
    batch_size = 50

    # TODO: Implement mean.
    train_data_generator = datagenerator_interiornet.ImageDataGenerator(train_files,
                                                                        np.array([0, 0, 0, 0]),
                                                                        image_type='bgr-d')  # mean
    val_data_generator = datagenerator_interiornet.ImageDataGenerator(val_files,
                                                                      np.array([0, 0, 0, 0]),
                                                                      image_type='bgr-d')  # mean

    train_generator = data_generator(train_data_generator, batch_size)
    val_generator = data_generator(val_data_generator, batch_size)

    # Create line net Keras model.
    line_model = line_net_model(line_num_attr, batch_size)

    line_model.fit_generator(train_data_generator,
                             verbose=1,
                             max_queue_size=1,
                             workers=1,
                             epochs=10,
                             steps_per_epoch=10,
                             validation_data=val_data_generator,
                             validation_steps=2)


if __name__ == '__main__':
    train()



