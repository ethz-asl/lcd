"""The following module allows to retrieve descriptors one line at a time (i.e.,
   it feeds the trained network with the corresponding virtual camera image to
   obtain the embeddings for the line, one line at a time).
"""
import sys
import os
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

class EmbeddingsRetriever:
    """ Retrieves embeddings for lines, given a virtual camera image and the
        line type.
    """

    def __init__(self, log_files_folder):
        self.sess = tf.InteractiveSession()
        # Restore checkpoint and model.
        saver = tf.train.import_meta_graph(
            os.path.join(log_files_folder, 'triplet_loss_batch_all_ckpt/'
                         'bgr-d_model_epoch1.ckpt.meta'))
        saver.restore(
            self.sess,
            os.path.join(log_files_folder,
                         'triplet_loss_batch_all_ckpt/bgr-d_model_epoch1.ckpt'))
        self.graph = tf.get_default_graph()
        # Input image tensor.
        self.input_img = self.graph.get_tensor_by_name('input_img:0')
        # Label of input image tensor.
        self.labels = self.graph.get_tensor_by_name('labels:0')
        # Dropout probability tensor.
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        # Embeddings tensor.
        self.embeddings = self.graph.get_tensor_by_name('l2_normalize:0')
        # Line type tensor.
        self.line_types = self.graph.get_tensor_by_name('line_types:0')
        # Retrieve mean of training set.
        train_set_mean = self.sess.run('train_set_mean:0')
        print("Train set mean is {}".format(train_set_mean))

    def get_embeddings_from_image(self, image, line_type):
        """ Given a virtual camera image and a line type, returns the
            corresponding embeddings.
        """
        start_time = timer()
        output_embeddings = self.sess.run(
            self.embeddings,
            feed_dict={
                self.input_img: image,
                self.line_types: line_type,
                self.keep_prob: 1.
            })
        end_time = timer()

        print('Time needed to retrieve desciptors for line %d: %.3f seconds' %
              (i, (end_time - start_time)))
        return output_embeddings
