#!/usr/bin/env python
""" ROS node that provides the response to the ImageToEmbeddings service.
"""
import tf
import numpy as np
import os
import rospy
from embeddings_retriever import EmbeddingsRetriever
from line_description.srv import ImageToEmbeddings, ImageToEmbeddingsResponse, \
                                 EmbeddingsRetrieverReady, \
                                 EmbeddingsRetrieverReadyResponse
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
        start_3D = np.array([req.start_3D.x, req.start_3D.y, req.start_3D.z])
        end_3D = np.array([req.end_3D.x, req.end_3D.y, req.end_3D.z])
        # Transform line to world coordinates.
        q = np.array([req.camera_to_world_matrix.transform.rotation.x,
                      req.camera_to_world_matrix.transform.rotation.y,
                      req.camera_to_world_matrix.transform.rotation.z,
                      req.camera_to_world_matrix.transform.rotation.w])
        t = np.array([req.camera_to_world_matrix.transform.translation.x,
                      req.camera_to_world_matrix.transform.translation.y,
                      req.camera_to_world_matrix.transform.translation.z])
        R = tf.transformations.quaternion_matrix(q)
        T = tf.transformations.translation_matrix(t)
        camera_to_world_matrix = np.dot(R, T)
        start_3D = np.dot(camera_to_world_matrix, np.hstack([start_3D, 1]))[:3]
        end_3D = np.dot(camera_to_world_matrix, np.hstack([end_3D, 1]))[:3]
        embeddings = self.embeddings_retriever.get_embeddings_from_image(
            virtual_camera_image_bgr, virtual_camera_image_depth, req.line_type,
            start_3D, end_3D)
        embeddings = embeddings.reshape((-1,))

        return ImageToEmbeddingsResponse(embeddings)

    def start_server(self):
        # Start ROS node.
        rospy.init_node('image_to_embeddings')
        # Initialize service.
        s = rospy.Service('image_to_embeddings', ImageToEmbeddings,
                          self.handle_image_to_embeddings)
        # Inform main node that initialization is completed.
        rospy.wait_for_service('embeddings_retriever_ready')
        try:
            embeddings_retriever_ready = rospy.ServiceProxy(
                'embeddings_retriever_ready', EmbeddingsRetrieverReady)
            response = embeddings_retriever_ready(True).message_received
            if (response is not True):
                print("Warning: server for service responded with {}. Expected "
                      "True.".format(response))

        except rospy.ServiceException, e:
            print(
                "Failed to call service embeddings_retriever_ready: {}".format(
                    e))

        rospy.spin()


if __name__ == "__main__":
    converter = ImageToEmbeddingsConverter()
    converter.start_server()
