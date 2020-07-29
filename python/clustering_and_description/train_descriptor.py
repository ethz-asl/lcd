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


def train_descriptor(past_path=None, past_epoch=None):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Paths to processed train scene lines. The train validation split can be performed manually by creating a
    # validation folder and moving a fraction of the scene folders to this directory. Be sure to check that no
    # scenes are present in both the training and validation directory. Also, check that no scenes on the same floor
    # are present in both the training and validation directory. For example, 3FO4IDDWIQX7_Dining_room and
    # 3FO4IDDWIQX7_Kitchen are on the same floor and contain views into the other room.
    train_files = "/nvme/line_ws/train"
    # Path to the validation scenes.
    val_files = "/nvme/line_ws/val"

    # Hyper-parameters of the neural network:
    # The dimensionality of the geometry vector of a line.
    line_num_attr = 15
    # The desired shape of the image inputs for the neural network.
    img_shape = (64, 96, 3)
    # The minimum line count per cluster for training. Can be set differently for inference.
    min_line_count = 4
    # The maximum line count per cluster for training. Can be set differently for inference.
    max_line_count = 50
    # The batch size.
    batch_size = 5
    # The number of training epochs.
    num_epochs = 40
    # The margin hyper-parameter for the triplet loss.
    margin = 0.6
    # The dimensionality of the descriptor embedding.
    embedding_dim = 128
    # The semantic labels that should be classified as background.
    # Careful: in InteriorNet, 0 is not background, but some random class. In theory, class 0 should not exist.
    bg_classes = [0, 1, 2, 20, 22]
    # The path to the pretrained weights of the image encoding layer.
    image_weight_path = "/clustering_and_description/weights/image_weights.hdf5"

    load_past = past_path is not None
    log_path = past_epoch

    # Load the model from scratch or load a past model checkpoint.
    if not load_past:
        cluster_model = model.cluster_triplet_loss_model(line_num_attr, max_line_count, embedding_dim,
                                                         img_shape, margin)
        cluster_model.get_layer("cluster_embedding_model").get_layer("image_features").load_weights(image_weight_path)
    else:
        load_path = os.path.join(log_path, "weights_only.{:02d}.hdf5".format(past_epoch))
        cluster_model = model.load_cluster_triplet_model(load_path, line_num_attr, max_line_count, embedding_dim,
                                                         img_shape, margin)
    cluster_model.summary()

    # Create the data generators for the train and validation datasets.
    train_data_generator = datagenerator_framewise.ClusterDataSequence(train_files,
                                                                       batch_size,
                                                                       bg_classes,
                                                                       shuffle=True,
                                                                       data_augmentation=True,
                                                                       img_shape=img_shape,
                                                                       min_line_count=min_line_count,
                                                                       max_line_count=max_line_count)

    val_data_generator = datagenerator_framewise.ClusterDataSequence(val_files,
                                                                     batch_size,
                                                                     bg_classes,
                                                                     shuffle=False,
                                                                     data_augmentation=False,
                                                                     img_shape=img_shape,
                                                                     min_line_count=min_line_count,
                                                                     max_line_count=max_line_count)

    if not load_past:
        log_path = "./logs/description_{}".format(datetime.datetime.now().strftime("%d%m%y_%H%M"))
    tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_path)
    save_callback = callback_utils.SaveCallback(os.path.join(log_path, "weights_only.{:02d}.hdf5"), cluster=True)
    learning_rate_callback = callback_utils.LearningRateCallback({5: 0.00005, 10: 0.000025, 20: 0.00001})
    callbacks = [tensorboard_callback, save_callback, learning_rate_callback]

    initial_epoch = 0
    if load_past:
        initial_epoch = past_epoch
    cluster_model.fit(x=train_data_generator,
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
        description='Train the cluster description network.')
    parser.add_argument(
        "--model_checkpoint_dir",
        default=None,
        help="If specified, the model checkpoint from past training is loaded. Epoch needs to be specified as well.")
    parser.add_argument(
        "--epoch",
        default=None,
        help="Path where to write the txt files with the splitting.")
    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint_dir
    epoch = args.epoch

    if model_checkpoint is not None and epoch is None:
        print("ERROR: Epoch needs to be specified if a model checkpoint is to be loaded.")
    elif model_checkpoint is not None:
        train_descriptor(past_path=model_checkpoint, past_epoch=epoch)
    else:
        train_descriptor()
