import tensorflow as tf
import numpy as np
import os
import argparse

from datetime import datetime

from model.datagenerator import ImageDataGenerator
from model.alexnet import AlexNet
from model.triplet_loss import batch_all_triplet_loss, batch_hardest_triplet_loss
from tools.train_set_mean import get_train_set_mean
from tools.get_line_center import get_line_center
from tools import pathconfig


def train():
    # Set a seed for numpy
    np.random.seed(1)

    # Set this to True to interpret the train/val files below as pickle files,
    # False to interpret them as regular text files with the format outputted by
    # split_dataset_with_labels_world.py
    read_as_pickle = True
    pickleandsplit_path = pathconfig.obtain_paths_and_variables(
        "PICKLEANDSPLIT_PATH")
    linesandimagesfolder_path = pathconfig.obtain_paths_and_variables(
        "LINESANDIMAGESFOLDER_PATH")

    # Configuration settings
    # Path to the textfiles for the trainings and validation set

    # For pickle version
    train_files = [
        os.path.join(pickleandsplit_path, 'train/traj_1/pickled_train.pkl')
    ]
    val_files = [
        os.path.join(pickleandsplit_path, 'train/traj_1/pickled_val.pkl')
    ]
    # For textfile version
    #train_files = [os.path.join(pickleandsplit_path, 'train/traj_1/train.txt')]
    #val_files = [os.path.join(pickleandsplit_path, 'train/traj_1/val.txt')]

    image_type = 'bgr-d'

    log_files_folder = "./logs/"

    # Learning params
    learning_rate = 0.01
    num_epochs = 30
    batch_size = 128
    margin = 0.2
    triplet_strategy = "batch_all"
    # triplet_strategy = "batch_hard"

    # Network params
    dropout_rate = 0.5
    no_train_layers = []

    # How often we want to write the tf.summary data to disk
    display_step = 1

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = os.path.join(log_files_folder, job_name,
                                   "triplet_loss_{}".format(triplet_strategy))
    checkpoint_path = os.path.join(
        log_files_folder, job_name,
        "triplet_loss_{}_ckpt".format(triplet_strategy))

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Check if checkpoints already exist
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

    # Create model
    # Placeholder for graph input and output
    if image_type == 'bgr':
        input_img = tf.placeholder(
            tf.float32, [batch_size, 227, 227, 3], name="input_img")
    elif image_type == 'bgr-d':
        input_img = tf.placeholder(
            tf.float32, [batch_size, 227, 227, 4], name="input_img")

    labels = tf.placeholder(tf.float32, [batch_size, 4], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Initialize model
    skip_layers = ['fc8']  # Don't use weights from AlexNet
    model = AlexNet(input_img, keep_prob, skip_layers, image_type)

    # Link variable to model output
    embeddings = tf.nn.l2_normalize(model.fc8, axis=1)

    # Get train_set_mean
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
        # The value will be restored from the checkpoint
        train_set_mean_variable = tf.get_variable(
            name="train_set_mean",
            shape=train_set_mean_shape,
            dtype=tf.float64,
            trainable=False)

    # List of trainable variables of the layers we want to train
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

    with tf.name_scope("triplet_loss"):
        if triplet_strategy == "batch_all":
            loss, fraction = batch_all_triplet_loss(
                labels, embeddings, margin=margin, squared=False)
        elif triplet_strategy == "batch_hard":
            loss = batch_hardest_triplet_loss(
                labels, embeddings, margin=margin, squared=False)
        else:
            raise ValueError(
                "Triplet strategy not recognized: {}".format(triplet_strategy))
        # The following only to assign a name to the tensor
        loss = tf.identity(loss, name="train_loss")
    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    if triplet_strategy == "batch_all":
        tf.summary.scalar('triplet_loss', loss)
        tf.summary.scalar('fraction_positive_triplets', fraction)
    elif triplet_strategy == "batch_hard":
        tf.summary.scalar('triplet_loss', loss)

    # Add embedding_mean_norm(should always be 1) to summary
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if latest_checkpoint is None:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Load the pretrained weights into the  layers which are not in
            # skip_layers
            model.load_initial_weights(sess)
            # Set first epoch to use for training as 0
            starting_epoch = 0
        else:
            print("Found checkpoint {}".format(latest_checkpoint))
            # Load values of variables from checkpoint
            saver.restore(sess, latest_checkpoint)
            # Set first epoch to use for training as the number in the last
            # checkpoint (note that the number saved in this filename
            # corresponds to the number of the last epoch + 1, cf. lines where
            # the checkpoints are saved)
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

        train_set_mean = sess.run(train_set_mean_variable)
        print("Mean of train set: {}".format(train_set_mean))

        # Initialize generators for image data
        train_generator = ImageDataGenerator(
            train_files,
            horizontal_flip=False,
            shuffle=True,
            image_type=image_type,
            mean=train_set_mean,
            read_as_pickle=read_as_pickle)
        val_generator = ImageDataGenerator(
            val_files,
            shuffle=True,
            image_type=image_type,
            mean=train_set_mean,
            read_as_pickle=read_as_pickle)

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = np.floor(
            train_generator.data_size / batch_size).astype(np.int16)
        val_batches_per_epoch = np.floor(
            val_generator.data_size / batch_size).astype(np.int16)

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(
            datetime.now(), filewriter_path))

        print("Starting epoch is {0}, num_epochs is {1}".format(
            starting_epoch, num_epochs))
        # Loop over number of epochs
        for epoch in range(starting_epoch, num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            step = 1

            while step < train_batches_per_epoch:

                # Get a batch of images and labels
                batch_input_img_train, batch_labels_train = train_generator.next_batch(
                    batch_size)

                # Pickled files have labels in the endpoints format -> convert
                # them to center format
                if read_as_pickle:
                    batch_labels_train = get_line_center(batch_labels_train)

                # And run the training op
                sess.run(
                    train_op,
                    feed_dict={
                        input_img: batch_input_img_train,
                        labels: batch_labels_train,
                        keep_prob: dropout_rate
                    })

                # Generate summary with the current batch of data and write to
                # file
                if step % display_step == 0:
                    s = sess.run(
                        merged_summary,
                        feed_dict={
                            input_img: batch_input_img_train,
                            labels: batch_labels_train,
                            keep_prob: 1.
                        })
                    writer.add_summary(s,
                                       epoch * train_batches_per_epoch + step)

                step += 1

            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            loss_val = 0.
            val_count = 0
            for _ in range(val_batches_per_epoch):
                batch_input_img_val, batch_labels_val = val_generator.next_batch(
                    batch_size)

                # Pickled files have labels in the endpoints format -> convert
                # them to center format
                if read_as_pickle:
                    batch_labels_val = get_line_center(batch_labels_val)

                loss_current = sess.run(
                    loss,
                    feed_dict={
                        input_img: batch_input_img_val,
                        labels: batch_labels_val,
                        keep_prob: 1.
                    })
                loss_val += loss_current
                val_count += 1
            loss_val = loss_val / val_count
            print("{} Average loss for validation set = {:.4f}".format(
                datetime.now(), loss_val))

            # Reset the file pointer of the image data generator
            val_generator.reset_pointer()
            train_generator.reset_pointer()

            print("{} Saving checkpoint of model...".format(datetime.now()))
            # save checkpoint of the model
            checkpoint_name = os.path.join(
                checkpoint_path,
                image_type + '_model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(
                datetime.now(), checkpoint_name))

            # The following is useful if one has no access to standard output
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
        default=datetime.now().strftime("%d%m%Y_%H%M"))
    args = parser.parse_args()
    if args.job_name:
        job_name = args.job_name

    train()
