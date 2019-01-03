"""ROS node that provides the response to the ImageToEmbeddings service.
"""

import os
import rospy
from embeddings_retriever import EmbeddingsRetriever
from line_description.srv import ImageToEmbeddings


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

    def handle_image_to_embeddings(self, req):
        return ImageToEmbeddingsResponse(
            self.embeddings_retriever.get_embeddings_from_image(
                req.virtual_image, req.line_type))

    def start_server(self):
        rospy.init_node('image_to_embeddings')
        s = rospy.Service('image_to_embeddings', ImageToEmbeddings,
                          self.handle_image_to_embeddings)
        rospy.spin()


if __name__ == "__main__":
    converter = ImageToEmbeddingsConverter()
    converter.start_server()
