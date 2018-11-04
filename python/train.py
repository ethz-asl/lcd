import tensorflow as tf
import numpy as np
import os

from datetime import datetime

from model.datagenerator import ImageDataGenerator
from model.alexnet import AlexNet
from model.triplet_loss import batch_all_triplet_loss, batch_hardest_triplet_loss
from tools.train_set_mean import get_train_set_mean
from tools import pathconfig

# Set a seed for numpy
np.random.seed(1)

# Set this to True to interpret the train/test/val files below as pickle files,
# False to interpret them as regular text files with the format outputted by
# split_dataset_with_labels_world.py
read_as_pickle = True
pickleandsplit_path = pathconfig.obtain_paths_and_variables(
    "PICKLEANDSPLIT_PATH")

# Configuration settings
# Path to the textfiles for the trainings and validation set
train_files = os.path.join(pickleandsplit_path,
                           'train/traj_1/pickled_train.pkl')
test_files = os.path.join(pickleandsplit_path, 'train/traj_1/pickled_test.pkl')
val_files = os.path.join(pickleandsplit_path, 'train/traj_1/pickled_val.pkl')

image_type = 'bgr-d'
#train_set_mean = np.array([22.4536157, 20.11461999, 5.61416132, 605.87199598])
train_set_mean = get_train_set_mean(
    train_files, image_type, read_as_pickle=read_as_pickle)
print("Mean of train set: {}".format(train_set_mean))

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
filewriter_path = "./logs/tmp/triplet_loss_{}".format(triplet_strategy)
checkpoint_path = "./logs/tmp/triplet_loss_{}_ckpt".format(triplet_strategy)

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

# TF placeholder for graph input and output
if image_type == 'bgr':
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
elif image_type == 'bgr-d':
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 4])

labels = tf.placeholder(tf.float32, [batch_size, 4])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
skip_layers = ['fc8']  # Don't use weights from AlexNet
model = AlexNet(x, keep_prob, skip_layers, image_type)

# Link variable to model output
embeddings = tf.nn.l2_normalize(model.fc8, axis=1)

# List of trainable variables of the layers we want to train
var_list = [
    v for v in tf.trainable_variables()
    if v.name.split('/')[0] not in no_train_layers
]

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

# Generator for image data
train_generator = ImageDataGenerator(
    train_files,
    horizontal_flip=False,
    shuffle=True,
    image_type=image_type,
    mean=train_set_mean,
    read_as_pickle=read_as_pickle)
val_generator = ImageDataGenerator(
    val_files,
    shuffle=False,
    image_type=image_type,
    mean=train_set_mean,
    read_as_pickle=read_as_pickle)
test_generator = ImageDataGenerator(
    test_files,
    shuffle=False,
    image_type=image_type,
    mean=train_set_mean,
    read_as_pickle=read_as_pickle)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(
    train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(
    np.int16)

#config = tf.ConfigProto(device_count={"CPU": 24}, log_device_placement=True)

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the  layers which are not in skip_layers
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_x_train, batch_labels_train = train_generator.next_batch(
                batch_size)

            # And run the training op
            sess.run(
                train_op,
                feed_dict={
                    x: batch_x_train,
                    labels: batch_labels_train,
                    keep_prob: dropout_rate
                })

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(
                    merged_summary,
                    feed_dict={
                        x: batch_x_train,
                        labels: batch_labels_train,
                        keep_prob: 1.
                    })
                writer.add_summary(s, epoch * train_batches_per_epoch + step)

            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        loss_val = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            loss_current = sess.run(
                loss, feed_dict={
                    x: batch_tx,
                    labels: batch_ty,
                    keep_prob: 1.
                })
            loss_val += loss_current
            test_count += 1
        loss_val = loss_val / test_count
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

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
