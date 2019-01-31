import numpy as np


class SceneNNCameraModel:
    """ Camera model (pinhole) for SceneNN. Partially taken from
        https://github.com/ethz-asl/scenenn_ros_tools/blob/master/nodes/scenenn_to_rosbag.py
    """

    def __init__(self, camera_intrinsics):
        # Camera projection matrix of shape (3,4).
        self.P = camera_intrinsics
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]

    def rgbd_to_pcl(self, rgb_image, depth_image, visualize_cloud=False):
        """ Converts the RGB-D image to a pointcloud. Adapted from
            https://github.com/ethz-asl/scenenn_ros_tools/blob/master/nodes/scenenn_to_rosbag.py
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

        # NOTE: contrary to SceneNetRGBD, the depth in the depth image is given
        # directly as z coordinate, instead of ray length.
        z = depth_image
        # Convert z coordinate from mm to m.
        z = z / 1000.0

        x = np.multiply(z, vs)
        y = z * us[:, np.newaxis]

        stacked = np.ma.dstack((x, y, z, rgb_image))
        compressed = stacked.compressed()
        pointcloud = compressed.reshape((int(compressed.shape[0] / 6), 6))

        # Optional: visualize point cloud.
        if (visualize_cloud):
            xyz = pointcloud[:, :3]
            rgb = pointcloud[:, 3:6] / 255
            import pptk
            v = pptk.viewer(xyz, rgb)

        return pointcloud


def get_camera_model():
    """ Returns a (pinhole) camera model for SceneNN dataset. Adjusted from
        https://github.com/ethz-asl/scenenn_ros_tools/blob/master/nodes/scenenn_to_rosbag.py
    """
    camera_intrinsics_matrix = get_camera_intrinsic_matrix()
    camera_model = SceneNNCameraModel(camera_intrinsics_matrix)

    return camera_model


def get_camera_intrinsic_matrix():
    """ Returns the intrinsic matrix of the camera. Partially taken from
        https://github.com/ethz-asl/scenenn_ros_tools/blob/master/nodes/scenenn_to_rosbag.py
    """
    # The following data is taken from the SceneNN parameter file
    # scenenn/download/scenenn_data/intrinsic/asus.ini.
    pixel_width = 640
    pixel_height = 480
    fx = 544.47329
    fy = 544.47329

    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1.0
    camera_intrinsics[0, 0] = fx
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = fy
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics
