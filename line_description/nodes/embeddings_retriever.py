"""The following module allows to retrieve descriptors one line at a time (i.e.,
   it feeds the trained network with the corresponding virtual camera image to
   obtain the embeddings for the line, one line at a time).
"""
import cv2
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
from tools.lines_utils import get_geometric_info


class EmbeddingsRetriever:
    """ Retrieves embeddings for lines, given a virtual camera image and the
        line type.

    Args:
        meta_file (str): Location of the .ckpt.meta file to restore the meta
            graph for the neural network.
        checkpoint_file (str): Location of the .ckpt file to restore the
            parameters of the trained neural network.
        scale_size (Tuple): Size of the input layer of the neural network.
        image_type (String): Either 'bgr' or 'brg-d', type of the image.

    Attributes:
        sess (TensorFlow session): TensorFlow session.
        graph (TensorFlow metagraph): TensorFlow metagraph restored from the
            .meta file.
        input_img (TensorFlow tensor): Tensor for the input image.
        keep_prob (TensorFlow tensor): Tensor for the the keeping probability in
            the dropout layer.
        embeddings (TensorFlow tensor): Tensor for the descriptor embeddings of
            the lines.
        line_types (TensorFlow tensor): Tensor for the types of the lines.
        geometric_info (TensorFlow tensor): Tensor for the geometric
            information.
        geometric_info_found (Boolean): True if a tensor for the geometric
            information was found in the network.
        scale_size (Tuple): Size of the input layer of the neural network.
        image_type (String): Either 'bgr' or 'brg-d', type of the image.
    """

    def __init__(self,
                 meta_file,
                 checkpoint_file,
                 scale_size=(227, 227),
                 image_type='bgr-d'):
        self.sess = tf.InteractiveSession()
        # Restore model and checkpoint.
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(self.sess, checkpoint_file)
        self.graph = tf.get_default_graph()
        # Input image tensor.
        self.input_img = self.graph.get_tensor_by_name('input_img:0')
        # Dropout probability tensor.
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        # Embeddings tensor.
        self.embeddings = self.graph.get_tensor_by_name('l2_normalize:0')
        # Line type tensor.
        self.line_types = self.graph.get_tensor_by_name('line_types:0')
        # Geometric info tensor.
        try:
            self.geometric_info = self.graph.get_tensor_by_name(
                'geometric_info:0')
        except KeyError:
            self.geometric_info_found = False
        else:
            self.geometric_info_found = True
        # Retrieve mean of training set.
        self.train_set_mean = self.sess.run('train_set_mean:0')
        print("Train set mean is {}".format(self.train_set_mean))
        self.scale_size = scale_size
        self.image_type = image_type

    def get_embeddings_from_image(self, image_bgr, image_depth, line_type,
                                  start_3D, end_3D):
        """ Given a virtual camera image, a line type and the line endpoints,
            returns the corresponding embeddings.
        """
        start_time = timer()
        # Rescale image.
        image_bgr_preprocessed = cv2.resize(
            image_bgr, (self.scale_size[0], self.scale_size[1]))
        image_bgr_preprocessed = image_bgr_preprocessed.astype(np.float32)
        image_depth_preprocessed = cv2.resize(
            image_depth, (self.scale_size[0], self.scale_size[1]))
        image_depth_preprocessed = image_depth_preprocessed.astype(np.float32)
        if self.image_type == 'bgr':
            image = image_bgr_preprocessed
        elif self.image_type == 'bgr-d':
            image = np.dstack(
                [image_bgr_preprocessed, image_depth_preprocessed])
        # Subtract mean of training set.
        image -= self.train_set_mean
        # Reshape arrays to feed them into the network.
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        line_type = np.array(line_type).reshape(-1, 1)

        # Create dictionary of the values to feed to the tensors to run the
        # operation.
        feed_dict = {
            self.input_img: image,
            self.keep_prob: 1.,
            self.line_types: line_type
        }

        # To ensure backcompatibility, geometric information is fed into the
        # network only if the version of the network trained contains the
        # associated tensor.
        if self.geometric_info_found:
            # Retrieve geometric info depending on the type of line
            # parametrization used when training.
            if (self.geometric_info.shape[1] == 4):
                # Line parametrization: 'orthonormal'.
                geometric_info = get_geometric_info(
                    start_points=start_3D,
                    end_points=end_3D,
                    line_parametrization='orthonormal')
                geometric_info = np.array(geometric_info).reshape(-1, 4)
            elif (self.geometric_info.shape[1] == 6):
                # Line parametrization: 'direction_and_centerpoint'.
                geometric_info = get_geometric_info(
                    start_points=start_3D,
                    end_points=end_3D,
                    line_parametrization='direction_and_centerpoint')
                geometric_info = np.array(geometric_info).reshape(-1, 6)
            else:
                raise ValueError("The trained geometric_info Tensor should "
                                 "have shape[1] either equal to 4 (line "
                                 "parametrization 'orthonormal') or equal to 6 "
                                 "(line parametrization "
                                 "'direction_and_centerpoint').")
            feed_dict[self.geometric_info] = geometric_info

        output_embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        end_time = timer()

        print('Time needed to retrieve descriptors for line: %.3f seconds' %
              (end_time - start_time))
        return output_embeddings
