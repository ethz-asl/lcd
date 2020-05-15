import time
import datetime
import os

import tensorflow.keras as tf_keras

import numpy as np
np.random.seed(123)

from datagenerator_framewise import LineDataGenerator
from datagenerator_framewise import data_generator
from model import line_net_model_3


def train():
    # Paths to line files.
    train_files = "/nvme/line_ws/train"
    val_files = "/nvme/line_ws/val"

    # The length of the geometry vector of a line.
    line_num_attr = 15
    img_shape = (60, 90, 3)
    max_line_count = 150
    batch_size = 20
    num_epochs = 80
    bg_classes = [0, 1, 2, 20, 22]

    # Create line net Keras model.
    # line_model = line_net_model(line_num_attr, max_line_count, img_shape, margin)
    line_model = line_net_model_3(line_num_attr, max_line_count, img_shape)
    line_model.summary()

    # log_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/120520_2010"
    # line_model.load_weights("/home/felix/line_ws/src/line_tools/python/line_net/logs/130520_2315/weights.20.hdf5",
    #                         by_name=True)
    train_set_mean = np.array([-0.00246431839, 0.0953982015,  3.15564408])

    train_data_generator = LineDataGenerator(train_files, bg_classes,
                                             shuffle=True,
                                             data_augmentation=True,
                                             img_shape=img_shape,
                                             sort=True)
    # train_set_mean = train_data_generator.get_mean()
    train_data_generator.set_mean(train_set_mean)
    print("Train set mean is: {}".format(train_set_mean))
    val_data_generator = LineDataGenerator(val_files, bg_classes, mean=train_set_mean, img_shape=img_shape, sort=True)

    train_generator = data_generator(train_data_generator, max_line_count, line_num_attr, batch_size)
    val_generator = data_generator(val_data_generator, max_line_count, line_num_attr, batch_size)

    log_path = "./logs/{}".format(datetime.datetime.now().strftime("%d%m%y_%H%M"))
    save_weights_callback = tf_keras.callbacks.ModelCheckpoint(os.path.join(log_path, "weights.{epoch:02d}.hdf5"))
    tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_path)

    line_model.fit_generator(generator=train_generator,
                             verbose=1,
                             max_queue_size=1,
                             workers=1,
                             use_multiprocessing=False,
                             epochs=num_epochs,
                             steps_per_epoch=np.floor((train_data_generator.frame_count) / batch_size),
                             validation_data=val_generator,
                             validation_steps=np.floor((val_data_generator.frame_count) / batch_size),
                             callbacks=[save_weights_callback, tensorboard_callback])


if __name__ == '__main__':
    train()



