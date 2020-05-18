import time
import datetime
import os

import tensorflow.keras as tf_keras

import numpy as np
np.random.seed(123)

from datagenerator_framewise import LineDataGenerator
from datagenerator_framewise import data_generator
from model import line_net_model_4
from model import image_pretrain_model


class LayerUnfreezeCallback(tf_keras.callbacks.Callback):
    def __init__(self, loss, opt, metrics):
        super().__init__()

        self.loss = loss
        self.opt = opt
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 15:
            transfer_layers = ["block3_conv1", "block3_conv2", "block3_conv3"]
            for layer_name in transfer_layers:
                self.model.get_layer("image_features").get_layer("vgg16_features").get_layer(layer_name).trainable = \
                    True
                print("Unfreezing layer {}.".format(layer_name))

            self.model.compile(loss=self.loss,
                               optimizer=self.opt,
                               metrics=self.metrics,
                               experimental_run_tf_function=False)

        if epoch == 30:
            transfer_layers = ["block2_conv1", "block2_conv2"]
            for layer_name in transfer_layers:
                self.model.get_layer("image_features").get_layer("vgg16_features").get_layer(layer_name).trainable = \
                    True
                print("Unfreezing layer {}.".format(layer_name))

            self.model.compile(loss=self.loss,
                               optimizer=self.opt,
                               metrics=self.metrics,
                               experimental_run_tf_function=False)


def train():
    # Paths to line files.
    train_files = "/nvme/line_ws/train"
    val_files = "/nvme/line_ws/val"

    # The length of the geometry vector of a line.
    line_num_attr = 15
    img_shape = (64, 96, 3)
    max_line_count = 150
    batch_size = 2
    num_epochs = 40
    bg_classes = [0, 1, 2, 20, 22]
    load_past = False
    past_epoch = 15
    past_path = "/home/felix/line_ws/src/line_tools/python/line_net/logs/180520_0018"
    image_weight_path = "/home/felix/line_ws/src/line_tools/python/line_net/weights/image_weights.hdf5"

    pretrain_images = False

    # Create line net Keras model.
    if pretrain_images:
        line_model, loss, opt, metrics = image_pretrain_model(line_num_attr, max_line_count, img_shape)
    else:
        line_model, loss, opt, metrics = line_net_model_4(line_num_attr, max_line_count, img_shape)
        print("Nice")
        line_model.get_layer("image_features").summary()
        line_model.get_layer("image_features").load_weights(image_weight_path)
    line_model.summary()

    if load_past:
        log_path = past_path
        line_model.load_weights(os.path.join(log_path, "weights.{}.hdf5".format(past_epoch)),
                                by_name=True)
    else:
        log_path = "./logs/{}".format(datetime.datetime.now().strftime("%d%m%y_%H%M"))

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
    train_frame_count = train_data_generator.frame_count - 8536
    val_frame_count = val_data_generator.frame_count - 1052
    val_generator = data_generator(val_data_generator, max_line_count, line_num_attr, batch_size)

    save_weights_callback = tf_keras.callbacks.ModelCheckpoint(os.path.join(log_path, "weights.{epoch:02d}.hdf5"))
    tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_path)
    callbacks = [save_weights_callback, tensorboard_callback]
    if pretrain_images:
        unfreeze_callback = LayerUnfreezeCallback(loss, opt, metrics)
        callbacks += [unfreeze_callback]

    initial_epoch = 0
    if load_past:
        initial_epoch = past_epoch
    line_model.fit_generator(generator=train_generator,
                             verbose=1,
                             max_queue_size=1,
                             workers=1,
                             use_multiprocessing=False,
                             initial_epoch=initial_epoch,
                             epochs=num_epochs,
                             steps_per_epoch=np.floor(train_frame_count / batch_size),
                             validation_data=val_generator,
                             validation_steps=np.floor(val_frame_count / batch_size),
                             callbacks=callbacks)


if __name__ == '__main__':
    train()



