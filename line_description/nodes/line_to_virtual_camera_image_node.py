#!/usr/bin/env python
"""ROS node that provides the response to the LineToVirtualCameraImage service.
"""
import numpy as np
import rospy
from virtual_camera_image_retriever import VirtualCameraImageRetriever
from line_description.srv import LineToVirtualCameraImage, LineToVirtualCameraImageResponse
from cv_bridge import CvBridge, CvBridgeError


class LineToVirtualCameraImageConverter:
    """ Server for the service LineToVirtualCameraImage. Returns a virtual
        camera image given a line.

    Args:
        None.

    Attributes:
        virtual_camera_image_retriever (VirtualCameraImageRetriever): Instance
            of the class of the VirtualCameraImageRetriever to retrieve the
            virtual camera image given a line.
    """

    def __init__(self):
        self.virtual_camera_image_retriever = VirtualCameraImageRetriever(
            distance_from_line=3)
        self.bridge = CvBridge()

    def handle_line_to_virtual_camera_image(self, req):
        start3D = req.line.start3D
        start3D = np.array((start3D.x, start3D.y, start3D.z))
        end3D = req.line.end3D
        end3D = np.array((end3D.x, end3D.y, end3D.z))
        hessian_left = req.line.hessian_left
        hessian_left = np.array(hessian_left)
        hessian_right = req.line.hessian_right
        hessian_right = np.array(hessian_right)
        line_type = req.line.line_type
        image_rgb = req.image_rgb
        try:
            cv_image_rgb = self.bridge.imgmsg_to_cv2(image_rgb, "32FC3")
        except CvBridgeError as e:
            print(e)
        image_rgb = np.asarray(cv_image_rgb)
        cloud = req.cloud
        try:
            cv_cloud = self.bridge.imgmsg_to_cv2(cloud, "32FC3")
        except CvBridgeError as e:
            print(e)
        cloud = np.asarray(cv_cloud)

        virtual_camera_image_rgb, virtual_camera_image_depth = \
            self.virtual_camera_image_retriever.get_virtual_camera_image(
                start3D=start3D,
                end3D=end3D,
                hessian_left=hessian_left,
                hessian_right=hessian_right,
                line_type=line_type,
                image_rgb=image_rgb,
                cloud=cloud)
        virtual_camera_image_rgb_msg = self.bridge.cv2_to_imgmsg(
            virtual_camera_image_rgb, "8UC3")
        virtual_camera_image_depth_msg = self.bridge.cv2_to_imgmsg(
            virtual_camera_image_depth, "32FC1")

        return LineToVirtualCameraImageResponse(virtual_camera_image_rgb_msg,
                                                virtual_camera_image_depth_msg)

    def start_server(self):
        rospy.init_node('line_to_virtual_camera_image')
        s = rospy.Service('line_to_virtual_camera_image',
                          LineToVirtualCameraImage,
                          self.handle_line_to_virtual_camera_image)
        rospy.spin()


if __name__ == "__main__":
    converter = LineToVirtualCameraImageConverter()
    converter.start_server()
