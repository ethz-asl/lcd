import numpy as np
import sys


class InteriorNetCameraModel:
    """ A simplified camera model class mainly for storing camera intrinsics.
        Adapted from scenenet_utils
    """

    def __init__(self, camera_intrinsics):
        # Camera projection matrix of shape (3, 4).
        self.P = camera_intrinsics
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]

    def project3dToPixel(self, point_3d):
        """ Projects 3D point to pixel.

        Args:
            point_3d (numpy array of shape (3, )): Point to be projected, in the
                format [x, y, z].

        Returns:
            pixel (numpy array of shape (2, ), and type int): Pixel position
                corresponding to the 3D point.
        """
        point_coor_homo = np.append(point_3d, [1])
        pixel_homo = self.P.dot(point_coor_homo.T)
        pixel = np.rint(pixel_homo / pixel_homo[2]).astype(int)[:2]
        return pixel

    def rgbd_to_pcl(self, rgb_image, depth_image, visualize_cloud=False):
        """ Converts RGB-D image to pointcloud. Adapted from
            https://github.com/ethz-asl/scenenet_ros_tools/blob/master/nodes/scenenet_to_rosbag.py
        """
        center_x = self.cx
        center_y = self.cy

        constant_x = 1 / self.fx
        constant_y = 1 / self.fy

        vs = np.array([
            (v - center_x) * constant_x for v in range(0, depth_image.shape[1])
        ])
        us = np.array([
            (u - center_y) * constant_y for u in range(0, depth_image.shape[0])
        ])

        # Find z coordinate of each pixel given the depth in ray length.
        z = euclidean_ray_length_to_z_coordinate(depth_image, self)
        # Convert the z coordinate from mm to m.
        z = z / 1000.0

        x = np.multiply(z, vs)
        y = z * us[:, np.newaxis]

        stacked = np.ma.dstack((x, y, z, rgb_image))
        compressed = stacked.compressed()
        pointcloud = compressed.reshape((int(compressed.shape[0] / 6), 6))

        # Optional: visualize point cloud.
        if visualize_cloud:
            xyz = pointcloud[:, :3]
            rgb = pointcloud[:, 3:6] / 255
            import pptk
            v = pptk.viewer(xyz, rgb)

        return pointcloud


def get_camera_model(fx=600, fy=600, pixel_width=640, pixel_height=480):
    """ Camera model for SceneNetRGBD dataset. Adjusted from
        https://github.com/ethz-asl/scenenet_ros_tools/blob/master/nodes/scenenet_to_rosbag.py
    """

    camera_intrinsic_matrix = camera_intrinsic_transform(
        fx, fy, pixel_width, pixel_height)
    camera_model = InteriorNetCameraModel(camera_intrinsic_matrix)
    return camera_model


def camera_intrinsic_transform(fx=600,
                               fy=600,
                               pixel_width=640,
                               pixel_height=480):
    """ Gets camera intrinsics matrix for InteriorNet dataset.
    """
    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = fx
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = fy
    camera_intrinsics[1, 2] = pixel_height / 2.0

    return camera_intrinsics


def euclidean_ray_length_to_z_coordinate(depth_image, camera_model):
    """ From https://github.com/ethz-asl/scenenet_ros_tools/blob/master/nodes/scenenet_to_rosbag.py
    """
    center_x = camera_model.cx
    center_y = camera_model.cy

    constant_x = 1 / camera_model.fx
    constant_y = 1 / camera_model.fy

    vs = np.array(
        [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
    us = np.array(
        [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

    return (np.sqrt(
        np.square(depth_image / 1000.0) /
        (1 + np.square(vs[np.newaxis, :]) + np.square(us[:, np.newaxis]))) *
            1000.0).astype(np.uint16)
