import rospy
from embeddings_retriever import EmbeddingsRetriever
from line_description.srv import ImageToEmbeddings


class ImageToEmbeddingsConverter:
    """ Server for the service ImageToEmbeddings. Returns embeddings given a
        virtual image and a line type.
    """

    def __init__(self):
        self.embeddings_retriever = EmbeddingsRetriever(
            log_files_folder=
            '/media/francesco/line_tools_data/logs/30122018_0852/')

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
