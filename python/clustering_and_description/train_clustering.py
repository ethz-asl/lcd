import datetime
import os
import argparse

import tensorflow.keras as tf_keras
import tensorflow as tf

import numpy as np
np.random.seed(123)

import model
import datagenerator_framewise
import callback_utils


def train_clustering(pretrain_images=False, past_path=None, past_epoch=None):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Paths to processed train scene lines. The train validation split can be performed manually by creating a
    # validation folder and moving a fraction of the scene folders to this directory. Be sure to check that no
    # scenes are present in both the training and validation directory.
    train_files = "/nvme/line_ws/train"
    # Path to the validation scenes.
    val_files = "/nvme/line_ws/val"
    # The path to the test files for inference. If inference is not desired, remove the inference callback from the
    # list of callbacks. The inference results can be viewed with visualize_clusters.py
    test_files = "/nvme/line_ws/test"

    if not os.path.exists(train_files):
        print("ERROR: The path to the train set does not exist. Please set it in train_clustering.py.")
        print("Path to train set: " + train_files)
        exit(0)

    # Hyper-parameters of the neural network:
    # The dimensionality of the geometry vector of a line.
    line_num_attr = 15
    # The desired shape of the image inputs for the neural network.
    img_shape = (64, 96, 3)
    # The minimum number of lines during training. Can be set differently for inference.
    min_line_count = 30
    # The maximum number of lines during training. Can be set differently for inference.
    max_line_count = 160
    # The batch size.
    batch_size = 2
    # The number of epochs desired for training.
    num_epochs = 50
    # The maximum number of cluster that can be distinguished.
    # Do not forget to delete pickle files when this config is changed.
    max_clusters = 15
    # The semantic labels that should be classified as background.
    # Careful: in InteriorNet, 0 is not background, but some random class. In theory, class 0 should not exist.
    bg_classes = [0, 1, 2, 20, 22]
    # Check if past training weights should be loaded. For example if the training got interrupted.
    load_past = past_path is not None
    # The path to the pretrained weights of the image encoding layer.
    image_weight_path = "./weights/image_weights.hdf5"

    # Create line net Keras model.
    if pretrain_images:
        # If the image encoding network is to be pretrained. Use the image pretrain model, which does not contain
        # the geometric encoding layer.
        line_model, loss, opt, metrics = model.image_pretrain_model(line_num_attr, max_line_count, img_shape)
    else:
        # To train the full clustering network, it is initialized with the pretrained image encoding weights.
        line_model, loss, opt, metrics = model.line_net_model_4(line_num_attr, max_line_count, max_clusters, img_shape)
        line_model.get_layer("image_features").summary()
        line_model.get_layer("image_features").load_weights(image_weight_path)
    line_model.summary()

    # Load a past training checkpoint.
    if load_past:
        log_path = past_path
        for layer in line_model.layers:
            layer.trainable = False
        line_model.load_weights(os.path.join(log_path, "weights.{:02d}.hdf5".format(past_epoch)),
                                by_name=True)
        for layer in line_model.layers:
            layer.trainable = True
        line_model.get_layer("image_features").trainable = False
    else:
        if pretrain_images:
            log_path = "./logs/pretrain_{}".format(datetime.datetime.now().strftime("%d%m%y_%H%M"))
        else:
            log_path = "./logs/cluster_{}".format(datetime.datetime.now().strftime("%d%m%y_%H%M"))

    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    # Do not apply data augmentation if pretraining the image encoding network.
    data_augmentation = True
    if pretrain_images:
        data_augmentation = False

    # Create the data generators for the train, validation and test datasets.
    train_data_generator = datagenerator_framewise.LineDataSequence(train_files,
                                                                    batch_size,
                                                                    bg_classes,
                                                                    shuffle=True,
                                                                    img_shape=img_shape,
                                                                    min_line_count=min_line_count,
                                                                    max_line_count=max_line_count,
                                                                    data_augmentation=data_augmentation,
                                                                    max_cluster_count=max_clusters)

    val_data_generator = datagenerator_framewise.LineDataSequence(val_files,
                                                                  batch_size,
                                                                  bg_classes,
                                                                  img_shape=img_shape,
                                                                  min_line_count=min_line_count,
                                                                  max_line_count=max_line_count,
                                                                  max_cluster_count=max_clusters)

    test_data_generator = datagenerator_framewise.LineDataSequence(test_files,
                                                                   1,
                                                                   bg_classes,
                                                                   shuffle=False,
                                                                   img_shape=img_shape,
                                                                   min_line_count=0,
                                                                   max_line_count=max_line_count,
                                                                   data_augmentation=False,
                                                                   training_mode=False,
                                                                   max_cluster_count=max_clusters)

    # Callback to save the model after every epoch (This doesn't work because Keras cannot load custom models).
    # save_model_callback = tf_keras.callbacks.ModelCheckpoint(os.path.join(log_path, "weights.{epoch:02d}.hdf5"))
    # Callback to visualize the training progress on tensorboard.
    tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_path, write_graph=False, write_images=True)
    # Callback to save the
    save_callback = callback_utils.SaveCallback(os.path.join(log_path, "weights.{:02d}.hdf5"))
    # Callback to perform inference on the test set. These results can be visualized to check progress.
    inference_callback = callback_utils.InferenceCallback(test_data_generator, log_path)
    # Add a callback for learning rate decay.
    if pretrain_images:
        learning_rate_callback = callback_utils.LearningRateCallback({8: 0.00025, 14: 0.0001, 20: 0.00005})
    else:
        learning_rate_callback = callback_utils.LearningRateCallback({10: 0.000025, 15: 0.00001, 20: 0.000005})
    callbacks = [tensorboard_callback, save_callback, learning_rate_callback,
                 inference_callback]
    if pretrain_images:
        callbacks += [callback_utils.SaveImageWeightsCallback(os.path.join(log_path, "image_weights.{:02d}.hdf5"))]
        unfreeze_callback = callback_utils.LayerUnfreezeCallback(loss, opt, metrics)
        callbacks += [unfreeze_callback, inference_callback]

    # If a model checkpoint is to be loaded, the first epoch needs to be changed accordingly.
    initial_epoch = 0
    if load_past:
        initial_epoch = past_epoch

    # Perform training.
    line_model.fit(x=train_data_generator,
                   verbose=1,
                   max_queue_size=16,
                   workers=4,
                   epochs=num_epochs,
                   initial_epoch=initial_epoch,
                   use_multiprocessing=True,
                   validation_data=val_data_generator,
                   callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the clustering network or pretrain the image encoding network.')
    parser.add_argument(
        "--pretrain",
        dest='pretrain',
        action='store_true',
        help="If set, pretrains and saves the image encoding weights.")
    parser.add_argument(
        "--model_checkpoint_dir",
        default=None,
        help="If specified, the model checkpoint from past training is loaded. Epoch needs to be specified as well.")
    parser.add_argument(
        "--epoch",
        default=None,
        help="The number of the epoch from past training.")
    args = parser.parse_args()

    pretrain = args.pretrain
    model_checkpoint = args.model_checkpoint_dir
    epoch = int(args.epoch)

    if pretrain:
        if model_checkpoint is not None or epoch is not None:
            print("WARNING: Loading past model is not supported while pretraining.")
        train_clustering(pretrain_images=True)

    else:
        if model_checkpoint is not None and epoch is None:
            print("ERROR: Epoch needs to be specified if a model checkpoint is to be loaded.")
        elif model_checkpoint is not None:
            train_clustering(pretrain_images=False, past_path=model_checkpoint, past_epoch=epoch)
        else:
            train_clustering()




