#!/usr/bin/env python
"""ROS node that provides the response to the ImageToEmbeddings service.
"""
import numpy as np
import os
import rospy
from embeddings_retriever import EmbeddingsRetriever
from line_description.srv import ImageToEmbeddings, ImageToEmbeddingsResponse
from cv_bridge import CvBridge, CvBridgeError


class ImageToEmbeddingsConverter:
    """ Server for the service ImageToEmbeddings. Returns embeddings given a
        virtual camera image and a line type.

    Args:
        None.

    Attributes:
        embeddings_retriever (EmbeddingsRetriever): Instance of the class of the
            EmbeddingsRetriever to retrieve the embeddings given a virtual
            camera image.
    """

    def __init__(self):
        log_files_folder = '/media/francesco/line_tools_data/logs/30122018_0852/'
        self.embeddings_retriever = EmbeddingsRetriever(
            meta_file=os.path.join(log_files_folder,
                                   'triplet_loss_batch_all_ckpt/'
                                   'bgr-d_model_epoch1.ckpt.meta'),
            checkpoint_file=os.path.join(
                log_files_folder,
                'triplet_loss_batch_all_ckpt/bgr-d_model_epoch1.ckpt'))
        self.bridge = CvBridge()

    def handle_image_to_embeddings(self, req):
        try:
            virtual_camera_image_bgr = self.bridge.imgmsg_to_cv2(
                req.virtual_camera_image_bgr, "32FC3")
            virtual_camera_image_depth = self.bridge.imgmsg_to_cv2(
                req.virtual_camera_image_depth, "32FC1")
        except CvBridgeError as e:
            print(e)
        virtual_camera_image_bgr = np.asarray(virtual_camera_image_bgr)
        virtual_camera_image_depth = np.asarray(virtual_camera_image_depth)

        embeddings = self.embeddings_retriever.get_embeddings_from_image(
            virtual_camera_image_bgr, virtual_camera_image_depth, req.line_type)
        embeddings = embeddings.reshape((-1,))

        return ImageToEmbeddingsResponse(embeddings)

    def start_server(self):
        rospy.init_node('image_to_embeddings')
        s = rospy.Service('image_to_embeddings', ImageToEmbeddings,
                          self.handle_image_to_embeddings)
        rospy.spin()


if __name__ == "__main__":
    converter = ImageToEmbeddingsConverter()
    converter.start_server()
