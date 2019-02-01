import tensorflow as tf
import numpy as np
import os
import argparse

from datetime import datetime

from model.datagenerator import ImageDataGenerator
from model.alexnet import AlexNet
from model.triplet_loss import batch_all_triplet_loss, \
                               batch_hardest_triplet_loss, \
                               batch_all_wohlhart_lepetit_loss
from tools.train_utils import get_train_set_mean, \
                              print_batch_triplets_statistics
from tools.lines_utils import get_label_with_line_center, get_geometric_info
from tools import pathconfig


# Set read_as_pickle to True to interpret the train/val files below as pickle
# files, False to interpret them as regular text files with the format outputted
# by split_dataset_with_labels_world.py.
def train(read_as_pickle=True):
    # Set a seed for numpy.
    np.random.seed(1)

    # Configuration settings:
    if (read_as_pickle):
        pickleandsplit_path = pathconfig.obtain_paths_and_variables(
            "PICKLEANDSPLIT_PATH")
        # * Pickle-files version: path of the pickle files to use for training
        #       and validation. Note: more than one pickle file at a time can be
        #       used for both training and validation. Therefore, train_files
        #       and val_files should both be lists.
        train_files = [
            os.path.join(pickleandsplit_path,
                         'train_0/traj_1/pickled_train.pkl')
        ]
        val_files = [
            os.path.join(pickleandsplit_path, 'train_0/traj_1/pickled_val.pkl')
        ]
    else:
        linesandimagesfolder_path = pathconfig.obtain_paths_and_variables(
            "LINESANDIMAGESFOLDER_PATH")
        # * Textfile version: path to the textfiles for the trainings and
        #       validation set.
        # TODO: fix to use this non-pickle version. The paths in train_files
        # and val_files below should work, but the current version does not
        # allow to train on several sets at a time with textfiles. Also,
        # textfiles do not contain the endpoints of the lines, but only their
        # center point, making it therefore not possible to use the line
        # direction/orthonormal representation required by the last version of
        # the network.
        train_files = [
            os.path.join(linesandimagesfolder_path, 'train_0/traj_1/train.txt')
        ]
        val_files = [
            os.path.join(linesandimagesfolder_path, 'train_0/traj_1/val.txt')
        ]
        print("Error: the current version of the network does not support text "
              "files in the old format. Please use pickle files. Exiting.")
        return

    # Either 'bgr' or 'bgr-d': type of the image fed to the network.
    image_type = 'bgr-d'
    # Type of line parametrization can be:
    # * 'direction_and_centerpoint':
    #      Each line segment is parametrized by its center point and by its unit
    #      direction vector.  To obtain invariance on the orientation of the
    #      line (i.e., given the two endpoints we do NOT want to consider one of
    #      them as the start and the other one as the end of the line segment),
    #      we enforce that the first entry should be non-negative. => 6
    #      parameters per line.
    # * 'orthonormal':
    #      A line segment is parametrized with a minimum-DOF parametrization
    #      (4 degrees of freedom) of the infinite line that it belongs to. The
    #      representation is called orthonormal. => 4 parameters per line.
    line_parametrization = 'direction_and_centerpoint'

    log_files_folder = "./logs/"

    # Learning parameters.
    learning_rate = 0.0001
    num_epochs = 30
    batch_size = 128
    # Margin of the triplet loss.
    margin = 0.2
    # Regularization hyperparameter required when using the loss based on
    # 'batch_all'/'batch_all_wohlhart_lepetit' triplet selection strategy.
    lambda_regularization = 0.1

    # Either "batch_all", "batch_hard" or "batch_all_wohlhart_lepetit". Strategy
    # for triplets selection.
    triplet_strategy = "batch_all_wohlhart_lepetit"
    # Only considered if triplet selection strategy is "batch_all".
    really_all = False

    # Network parameters.
    dropout_rate = 0.5

    # How often we want to write the tf.summary data to disk.
    display_step = 1

    # Path for tf.summary.FileWriter and to store model checkpoints.
    filewriter_path = os.path.join(log_files_folder, job_name,
                                   "triplet_loss_{}".format(triplet_strategy))
    checkpoint_path = os.path.join(
        log_files_folder, job_name,
        "triplet_loss_{}_ckpt".format(triplet_strategy))

    # Create parent path if it does not exist.
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Check if checkpoints already exist.
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

    # Create model.

    # Input placeholder.
    if image_type == 'bgr':
        input_img = tf.placeholder(
            tf.float32, [None, 227, 227, 3], name="input_img")
    elif image_type == 'bgr-d':
        input_img = tf.placeholder(
            tf.float32, [None, 227, 227, 4], name="input_img")

    # For each line, labels is in the format
    #   [line_center (3x)] [instance label (1x)]
    labels = tf.placeholder(tf.float32, [None, 4], name="labels")
    # Dropout probability.
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    # Line types.
    line_types = tf.placeholder(tf.float32, [None, 1], name="line_types")
    # Geometric information.
    if line_parametrization == 'direction_and_centerpoint':
        geometric_info = tf.placeholder(
            tf.float32, [None, 6], name="geometric_info")
    elif line_parametrization == 'orthonormal':
        geometric_info = tf.placeholder(
            tf.float32, [None, 4], name="geometric_info")
    else:
        raise ValueError("Line parametrization should be "
                         "'direction_and_centerpoint' or 'orthonormal'.")

    # Layers for which weights should not be trained.
    no_train_layers = ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2']
    # Layers for which ImageNet weights should not be loaded.
    skip_layers = ['fc8', 'fc9']

    # Initialize model.
    model = AlexNet(
        x=input_img,
        line_types=line_types,
        geometric_info=geometric_info,
        keep_prob=keep_prob,
        skip_layer=skip_layers,
        input_images=image_type)

    # Retrieve embeddings (cluster descriptors) from model output.
    embeddings = tf.nn.l2_normalize(model.fc9, axis=1)

    # Get mean of training set if the training is just starting (i.e., if no
    # previous checkpoints are found).
    if latest_checkpoint is None:
        train_set_mean = get_train_set_mean(
            train_files, image_type, read_as_pickle=read_as_pickle)
        train_set_mean_tensor = tf.convert_to_tensor(
            train_set_mean, dtype=np.float64)
        train_set_mean_variable = tf.Variable(
            initial_value=train_set_mean_tensor,
            trainable=False,
            name="train_set_mean")
    else:
        if image_type == 'bgr':
            train_set_mean_shape = (3,)
        elif image_type == 'bgr-d':
            train_set_mean_shape = (4,)
        # The value will be restored from the checkpoint.
        train_set_mean_variable = tf.get_variable(
            name="train_set_mean",
            shape=train_set_mean_shape,
            dtype=tf.float64,
            trainable=False)

    # List of trainable variables of the layers we want to train.
    var_list = [
        v for v in tf.trainable_variables()
        if v.name.split('/')[0] not in no_train_layers
    ]
    total_parameters = 0
    print("**** List of variables used for training ****")
    for var in var_list:
        shape = var.get_shape()
        var_parameters = 1
        for dim in shape:
            var_parameters *= dim.value
        print("{0} --- {1} parameters".format(var.name, var_parameters))
        total_parameters += var_parameters
    print("Total number of parameters is {}".format(total_parameters))

    # Define loss.
    with tf.name_scope("triplet_loss"):
        if triplet_strategy == "batch_all":
            # TODO: for the purpose of the printing of the statistics, when
            # using "batch_all" with really_all=True, the tensors
            # sum_valid_positive_triplets_anchor_positive_dist and
            # num_anchor_positive_pairs_with_valid_positive_triplets are
            # substituted respectively with
            # sum_valid_triplets_anchor_positive_dist and
            # num_anchor_positive_pairs_with_valid_triplets. The statistics
            # should therefore be modified accordingly.
            (loss, fraction, valid_positive_triplets, pairwise_dist,
             sum_valid_positive_triplets_anchor_positive_dist,
             num_anchor_positive_pairs_with_valid_positive_triplets,
             regularization_term) = batch_all_triplet_loss(
                 labels,
                 embeddings,
                 margin=margin,
                 lambda_regularization=lambda_regularization,
                 really_all=really_all,
                 squared=False)
        elif triplet_strategy == "batch_hard":
            (loss, mask_anchor_positive, mask_anchor_negative,
             hardest_positive_dist, hardest_negative_dist,
             hardest_positive_element, hardest_negative_element,
             pairwise_dist) = batch_hardest_triplet_loss(
                 labels, embeddings, margin=margin, squared=False)
        elif triplet_strategy == "batch_all_wohlhart_lepetit":
            (loss, fraction, valid_positive_triplets, pairwise_dist,
             sum_valid_positive_triplets_anchor_positive_dist,
             num_anchor_positive_pairs_with_valid_positive_triplets,
             regularization_term) = batch_all_wohlhart_lepetit_loss(
                 labels,
                 embeddings,
                 margin=margin,
                 lambda_regularization=lambda_regularization,
                 squared=False)
        else:
            raise ValueError(
                "Triplet strategy not recognized: {}".format(triplet_strategy))
        # The following only to assign a name to the tensor.
        loss = tf.identity(loss, name="train_loss")
    # Train operation.
    with tf.name_scope("train"):
        # Get gradients of all trainable variables.
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable
        # variables.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary.
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary.
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add embedding_mean_norm (should always be 1) to summary.
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if (triplet_strategy == "batch_all" or
            triplet_strategy == "batch_all_wohlhart_lepetit"):
        tf.summary.scalar('fraction_positive_triplets', fraction)
        tf.summary.scalar('regularization_term', regularization_term)

    # NOTE: the merged summary does not include the losses, since we want to
    # output both the training and validation loss, but the two losses should
    # obviously be computed on different data (this practically means that we
    # have to manually use add_summary rather than using merge_all).
    merged_summary = tf.summary.merge_all()

    # Add the loss to summary.
    training_loss_summary = tf.summary.scalar('training_loss', loss)
    validation_loss_summary = tf.summary.scalar('validation_loss', loss)

    # Initialize the FileWriter.
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver to store model checkpoints.
    saver = tf.train.Saver(max_to_keep=3)

    # Run model.
    with tf.Session() as sess:
        if latest_checkpoint is None:
            # Initialize all variables.
            sess.run(tf.global_variables_initializer())
            # Load the pretrained weights into the layers which are not in
            # skip_layers.
            model.load_initial_weights(sess)
            # Set first epoch to use for training as 0.
            starting_epoch = 0
        else:
            print("Found checkpoint {}".format(latest_checkpoint))
            # Load values of variables from checkpoint.
            saver.restore(sess, latest_checkpoint)
            # Set first epoch to use for training as the number in the last
            # checkpoint (note that the number saved in this filename
            # corresponds to the number of the last epoch + 1, cf. lines where
            # the checkpoints are saved).
            start_char = latest_checkpoint.find("epoch")
            if start_char == -1:
                print(
                    "File name of checkpoint is in unexpected format: did not "
                    "find ''epoch''. Exiting.")
                exit()
            else:
                start_char += 5  # Length of the string 'epoch'
                end_char = latest_checkpoint.find(".ckpt", start_char)
                if end_char == -1:
                    print(
                        "File name of checkpoint is in unexpected format: did "
                        "not find ''.ckpt''. Exiting.")
                    exit()
                else:
                    starting_epoch = int(latest_checkpoint[start_char:end_char])

        # Obtain training set mean.
        train_set_mean = sess.run(train_set_mean_variable)
        print("Mean of train set: {}".format(train_set_mean))

        # Initialize generators for image data.
        train_generator = ImageDataGenerator(
            files_list=train_files,
            horizontal_flip=False,
            shuffle=True,
            image_type=image_type,
            mean=train_set_mean,
            read_as_pickle=read_as_pickle)
        val_generator = ImageDataGenerator(
            files_list=val_files,
            shuffle=True,
            image_type=image_type,
            mean=train_set_mean,
            read_as_pickle=read_as_pickle)

        # Get the number of training/validation steps per epoch.
        train_batches_per_epoch = np.floor(
            train_generator.data_size / batch_size).astype(np.int16)
        val_batches_per_epoch = np.floor(
            val_generator.data_size / batch_size).astype(np.int16)

        print("Number of training batches per epoch = {}".format(
            train_batches_per_epoch))
        print("Number of validation batches per epoch = {}".format(
            val_batches_per_epoch))

        # Add the model graph to TensorBoard.
        writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(
            datetime.now(), filewriter_path))

        print("Starting epoch is {0}, num_epochs is {1}".format(
            starting_epoch, num_epochs))

        # Loop over number of epochs.
        for epoch in range(starting_epoch, num_epochs):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            step = 1

            while step <= train_batches_per_epoch:
                # Get a batch of images and labels.
                (batch_input_img_train, batch_labels_train,
                 batch_line_types_train
                ) = train_generator.next_batch(batch_size)
                # Pickled files have labels in the endpoints format -> convert
                # them to center format.
                labels_for_stats = batch_labels_train
                if read_as_pickle:
                    batch_start_points_train = batch_labels_train[:, :3]
                    batch_end_points_train = batch_labels_train[:, 3:6]
                    batch_geometric_info_train = get_geometric_info(
                        start_points=batch_start_points_train,
                        end_points=batch_end_points_train,
                        line_parametrization=line_parametrization)
                    batch_labels_train = get_label_with_line_center(
                        labels_batch=batch_labels_train)

                # Display statistics about triplets (for only a few epochs and
                # batches).
                if (epoch % 10 < 2 and step < 30):
                    if (triplet_strategy == 'batch_all' or
                            triplet_strategy == 'batch_all_wohlhart_lepetit'):
                        (pairwise_dist_for_stats,
                         valid_positive_triplets_for_stats,
                         sum_valid_positive_triplets_anchor_positive_dist_for_stats,
                         num_anchor_positive_pairs_with_valid_positive_triplets_for_stats,
                         loss_for_stats, regularization_term_for_stats
                        ) = sess.run(
                            [
                                pairwise_dist, valid_positive_triplets,
                                sum_valid_positive_triplets_anchor_positive_dist,
                                num_anchor_positive_pairs_with_valid_positive_triplets,
                                loss, regularization_term
                            ],
                            feed_dict={
                                input_img: batch_input_img_train,
                                labels: batch_labels_train,
                                line_types: batch_line_types_train,
                                geometric_info: batch_geometric_info_train,
                                keep_prob: dropout_rate
                            })
                        print_batch_triplets_statistics(
                            triplet_strategy=triplet_strategy,
                            images=batch_input_img_train,
                            set_mean=train_set_mean,
                            batch_index=step,
                            epoch_index=epoch,
                            write_folder='{}_logs/'.format(job_name),
                            labels=labels_for_stats,
                            pairwise_dist=pairwise_dist_for_stats,
                            valid_positive_triplets=
                            valid_positive_triplets_for_stats,
                            sum_valid_positive_triplets_anchor_positive_dist=
                            sum_valid_positive_triplets_anchor_positive_dist_for_stats,
                            num_anchor_positive_pairs_with_valid_positive_triplets
                            =num_anchor_positive_pairs_with_valid_positive_triplets_for_stats,
                            loss=loss_for_stats,
                            lambda_regularization=lambda_regularization,
                            regularization_term=regularization_term_for_stats)
                    elif (triplet_strategy == 'batch_hard'):
                        (pairwise_dist_for_stats,
                         mask_anchor_positive_for_stats,
                         mask_anchor_negative_for_stats,
                         hardest_positive_dist_for_stats,
                         hardest_negative_dist_for_stats,
                         hardest_positive_element_for_stats,
                         hardest_negative_element_for_stats,
                         loss_for_stats) = sess.run(
                             [
                                 pairwise_dist, mask_anchor_positive,
                                 mask_anchor_negative, hardest_positive_dist,
                                 hardest_negative_dist,
                                 hardest_positive_element,
                                 hardest_negative_element, loss
                             ],
                             feed_dict={
                                 input_img: batch_input_img_train,
                                 labels: batch_labels_train,
                                 line_types: batch_line_types_train,
                                 geometric_info: batch_geometric_info_train,
                                 keep_prob: dropout_rate
                             })
                        print_batch_triplets_statistics(
                            triplet_strategy=triplet_strategy,
                            images=batch_input_img_train,
                            set_mean=train_set_mean,
                            batch_index=step,
                            epoch_index=epoch,
                            write_folder='{}_logs/'.format(job_name),
                            labels=labels_for_stats,
                            pairwise_dist=pairwise_dist_for_stats,
                            mask_anchor_positive=mask_anchor_positive_for_stats,
                            mask_anchor_negative=mask_anchor_negative_for_stats,
                            hardest_positive_dist=
                            hardest_positive_dist_for_stats,
                            hardest_negative_dist=
                            hardest_negative_dist_for_stats,
                            hardest_positive_element=
                            hardest_positive_element_for_stats,
                            hardest_negative_element=
                            hardest_negative_element_for_stats,
                            loss=loss_for_stats)
                # Run the training operation.
                sess.run(
                    train_op,
                    feed_dict={
                        input_img: batch_input_img_train,
                        labels: batch_labels_train,
                        line_types: batch_line_types_train,
                        geometric_info: batch_geometric_info_train,
                        keep_prob: dropout_rate
                    })
                # Generate summary with the current batch of data and write it
                # to file.
                if step % display_step == 0:
                    (merged_s, training_s) = sess.run(
                        [merged_summary, training_loss_summary],
                        feed_dict={
                            input_img: batch_input_img_train,
                            labels: batch_labels_train,
                            line_types: batch_line_types_train,
                            geometric_info: batch_geometric_info_train,
                            keep_prob: 1.
                        })
                    writer.add_summary(merged_s,
                                       epoch * train_batches_per_epoch + step)
                    writer.add_summary(training_s,
                                       epoch * train_batches_per_epoch + step)

                step += 1

            # Validate the model on the entire validation set.
            print("{} Start validation".format(datetime.now()))
            loss_val = 0.
            val_count = 0
            for val_step in range(val_batches_per_epoch):
                (batch_input_img_val, batch_labels_val,
                 batch_line_types_val) = val_generator.next_batch(batch_size)
                # Pickled files have labels in the endpoints format -> convert
                # them to center format.
                if read_as_pickle:
                    batch_start_points_val = batch_labels_val[:, :3]
                    batch_end_points_val = batch_labels_val[:, 3:6]
                    batch_geometric_info_val = get_geometric_info(
                        start_points=batch_start_points_val,
                        end_points=batch_end_points_val,
                        line_parametrization=line_parametrization)
                    batch_labels_val = get_label_with_line_center(
                        labels_batch=batch_labels_val)
                # Obtain validation loss.
                (loss_current, validation_s) = sess.run(
                    [loss, validation_loss_summary],
                    feed_dict={
                        input_img: batch_input_img_val,
                        labels: batch_labels_val,
                        line_types: batch_line_types_val,
                        geometric_info: batch_geometric_info_val,
                        keep_prob: 1.
                    })
                loss_val += loss_current
                val_count += 1
                # Add validation loss to summary.
                writer.add_summary(validation_s,
                                   epoch * val_batches_per_epoch + val_step)
            if val_count != 0:
                loss_val = loss_val / val_count
                print("{} Average loss for validation set = {:.4f}".format(
                    datetime.now(), loss_val))

            # Reset the file pointer of the image data generator at the end of
            # each epoch.
            val_generator.reset_pointer()
            train_generator.reset_pointer()

            print("{} Saving checkpoint of model...".format(datetime.now()))
            # Save checkpoint of the model.
            checkpoint_name = os.path.join(
                checkpoint_path,
                image_type + '_model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(
                datetime.now(), checkpoint_name))

            # The following is useful if one has no access to standard output.
            with open(
                    os.path.join(log_files_folder, job_name,
                                 "epochs_completed"), "aw") as f:
                f.write("Completed epoch {}\n".format(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural network model.')
    parser.add_argument(
        "-job_name",
        help="Name of the job, to be used for the "
        "log folders.",
        default=datetime.now().strftime("%d%m%y_%H%M"))
    args = parser.parse_args()
    if args.job_name:
        job_name = args.job_name

    train()
