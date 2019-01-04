#!/usr/bin/env python
"""ROS node that provides the response to the LineToVirtualCameraImage service.
"""
import numpy as np
import rospy
from virtual_camera_image_retriever import VirtualCameraImageRetriever
from line_description.srv import LineToVirtualCameraImage


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

    def handle_line_to_virtual_camera_image(self, req):
        start3D = req.line.start3D
        end3D = req.line.end3D
        hessian_left = req.line.hessian_left
        hessian_right = req.line.hessian_right
        line_type = req.line.line_type
        image_rgb = req.image_rgb
        cloud = req.cloud
        return LineToVirtualCameraImageResponse(
            self.virtual_camera_image_retriever.get_virtual_camera_image(
                start3D=start3D, end3D=end3D, hessian_left=hessian_left,
                hessian_right=hessian_right, line_type=line_type,
                image_rgb=req.image_rgb, cloud=req.cloud))

    def start_server(self):
        rospy.init_node('line_to_virtual_camera_image')
        s = rospy.Service('line_to_virtual_camera_image',
                          LineToVirtualCameraImage,
                          self.handle_line_to_virtual_camera_image)
        rospy.spin()


if __name__ == "__main__":
    converter = LineToVirtualCameraImageConverter()
    converter.start_server()
