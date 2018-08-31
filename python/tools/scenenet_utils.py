import numpy as np
import sys

import pathconfig
import scenenet_pb2 as sn

from camera_pose_and_intrinsics_example import camera_to_world_with_pose, interpolate_poses


class SceneNetCameraModel:
    """A simplified camera model class mainly for storing camera intrinsics.
    """

    def __init__(self, camera_intrinsics):
        # camera projection matrix of shape (3,4)
        self.P = camera_intrinsics
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]

    def project3dToPixel(self, point_3d):
        """Project 3d point to pixel.
        Args:
            point_3d: numpy array of shape (3, ). [x, y, z].

        Returns:
            pixel: numpy array of shape (2, ), of type int. Pixel position corresponding to the 3d point.
        """
        point_coor_homo = np.append(point_3d, [1])
        pixel_homo = self.P.dot(point_coor_homo.T)
        pixel = np.rint(pixel_homo / pixel_homo[2]).astype(int)[:2]
        return pixel


def get_camera_model():
    """Camera model for SceneNetRGBD dataset. Adjusted from https://github.com/ethz-asl/scenenet_ros_tools/blob/master/nodes/scenenet_to_rosbag.py
    """
    def camera_intrinsic_transform(vfov=45,
                                   hfov=60,
                                   pixel_width=320,
                                   pixel_height=240):
        """Get camera intrinsics matrix for SceneNetRGBD dataset"""
        camera_intrinsics = np.zeros((3, 4))
        camera_intrinsics[2, 2] = 1
        camera_intrinsics[0, 0] = (
            pixel_width / 2.0) / np.tan(np.radians(hfov / 2.0))
        camera_intrinsics[0, 2] = pixel_width / 2.0
        camera_intrinsics[1, 1] = (
            pixel_height / 2.0) / np.tan(np.radians(vfov / 2.0))
        camera_intrinsics[1, 2] = pixel_height / 2.0
        return camera_intrinsics

    camera_intrinsic_matrix = camera_intrinsic_transform()
    camera_model = SceneNetCameraModel(camera_intrinsic_matrix)
    return camera_model


def project_pcl_to_image(pointcloud, camera_model):
    """Project pointcloud to camera image.
    Args:
        pointcloud: numpy array of shape (points_number, 6). [x, y, z, r, g, b].
        camera_model: camera model class.

    Returns:
        rbg_image: numpy array of shape (240, 320, 3), dtype=np.uint8
        depth_image: numpy array of shape (240, 320), dtype=np.float32. The depth unit is mm.
    """
    rgb_image = np.zeros((240, 320, 3), dtype=np.uint8)
    depth_image = np.zeros((240, 320), dtype=np.float32)

    pcl_inside_view = pointcloud[pointcloud[:, 2] > 0, :]
    pcl_inside_view_xyz = np.hstack(
        (pcl_inside_view[:, :3], np.ones((pcl_inside_view.shape[0], 1))))

    pcl_projected = np.array(camera_model.P).dot(pcl_inside_view_xyz.T)
    pixel = np.rint(pcl_projected / pcl_projected[2, :]).astype(int)[:2, :]

    index_bool = np.logical_and(np.logical_and(0 <= pixel[0], pixel[0] < 320),
                                np.logical_and(0 <= pixel[1], pixel[1] < 240))
    pixel = pixel[:, index_bool]

    pcl_inside_view = pcl_inside_view[index_bool, :]

    # Considering occlusion, we need to be careful with the order of assignment
    # descending order according to the z coordiate
    index_sort = pcl_inside_view[:, 2].argsort()[::-1]
    pixel = pixel[:, index_sort]
    pcl_inside_view = pcl_inside_view[index_sort, :]

    rgb_image[pixel[1], pixel[0]] = pcl_inside_view[:, 3:]
    depth_image[pixel[1], pixel[0]] = pcl_inside_view[:, 2] * 1000.0  # m to mm

    # print('not projected points:', pointcloud.shape[0] - pcl_inside_view.shape[0])
    return rgb_image, depth_image


def get_origin_virtual_camera(line, distance, debug=False):
    """Get the origin of virtual camera for the line. The origin candidates are determined by the surfaces normals of the line. The virtual camera origin is chosen as the nearest point to real camera origin.
    Args:
        line: numpy array of shape (22, ). [start point(3, ), end point(3, ), left plane hessian form(4, ), right plane hessian form(4, ), left color(3, ), right color(3, ), line's type(1, ), instance label(1, )].
        distance: distance in meter between the origin of the virtual camera and the middle point of the line.

    Returns:
        origin_virtual_camera: numpy array of shape (3, ). Origin of the virtual camera.
    """
    # x axis is taken as the direction of the line
    x = (line[3:6] - line[:3]) / np.linalg.norm(line[3:6] - line[:3])
    middle_point = (line[3:6] + line[:3]) / 2
    plane1_normal = line[6:9]  # plane normal already normalized
    plane2_normal = line[10:13]

    # Get possible virtual camera optical axis, store them in the list z_cand
    n = -2  # value of n depends on the data format for line, n indicate line's type
    if line[n] == 0:  # dicontinuty line
        if debug is True:
            print("disconti")
        # discontinuty line only has one valid surface normal, the other one is [0, 0, 0, 0]
        if np.linalg.norm(plane1_normal) <= 0.001:
            z_cand = [plane2_normal, -plane2_normal]
        if np.linalg.norm(plane2_normal) <= 0.001:
            z_cand = [plane1_normal, -plane1_normal]

    if line[n] == 1:  # plane(surface) line
        if debug is True:
            print("plane")
        # Surface line has 2 similar surface normals, thus we can always take only plane 1's normal
        z_cand = [plane1_normal, -plane1_normal]

    if line[n] == 2:  # intersection line
        if debug is True:
            print("inter")
        z1 = (plane1_normal + plane2_normal) / \
            np.linalg.norm(plane1_normal + plane2_normal)
        z2 = (plane1_normal - plane2_normal) / \
            np.linalg.norm(plane1_normal - plane2_normal)
        z3 = (-plane1_normal - plane2_normal) / \
            np.linalg.norm(-plane1_normal - plane2_normal)
        z4 = (-plane1_normal + plane2_normal) / \
            np.linalg.norm(-plane1_normal + plane2_normal)

        z_cand = [z1, z2, z3, z4]

    origin_cand = []
    for z in z_cand:
        # Project vector z onto the plane which is perpendicular to the line
        p1 = np.array([0, 0, 0])
        p2 = z

        p1 = p1 - (p1 - middle_point).dot(x) * x
        p2 = p2 - (p2 - middle_point).dot(x) * x
        z = (p2 - p1) / np.linalg.norm(p2 - p1)

        origin = middle_point - distance * z
        origin_cand.append(origin)

    min_dist = np.inf
    origin_virtual_camera = origin_cand[0]

    # choose the nearest origin of virtual camera to the real camera's origin
    for origin in origin_cand:
        if np.linalg.norm(origin) < min_dist:
            min_dist = np.linalg.norm(origin)
            origin_virtual_camera = origin

    return origin_virtual_camera


def virtual_camera_pose(line, distance):
    """Get the virtual camera's pose according to the line.
    Args:
        line: numpy array of shape (22, ). [start point(3, ), end point(3, ), left plane hessian form(4, ), right plane hessian form(4, ), left color(3, ), right color(3, ), line's type(1, ), instance label(1, )].
        distance: distance in meter between the origin of the virtual camera and the middle point of the line.

    Returns:
        T: numpy array of shape (4, 4). Transformation matrix.
        z: numpy array of shape (3, ). Optical axis of the virtual camera.
    """
    x = (line[3:6] - line[:3]) / np.linalg.norm(line[3:6] - line[:3])
    middle_point = (line[3:6] + line[:3]) / 2

    origin = get_origin_virtual_camera(line, distance)
    z = (middle_point - origin) / np.linalg.norm(middle_point - origin)

    y = np.cross(z, x) / np.linalg.norm(np.cross(z, x))

    R = np.array([x, y, z])  # rotation matrix
    # translation vector expressed in this virtual camera frame
    t0 = -R.dot(origin).reshape((3, 1))
    T = np.concatenate((R, t0), axis=1)
    # with respect to the original camera frame
    T = np.vstack((T, np.array([0, 0, 0, 1])))

    return T, z


def pcl_transform(pointcloud, T):
    """Transform pointcloud according to the transformation matrix.
    Args:
        pointcloud: numpy array of shape (points_number, 6). [x, y, z, r, g, b].
        T: numpy array of shape (4, 4). Transformation matrix.

    Returns:
        pcl_new: numpy array of shape (points_number, 6). The pointcloud expressed in new coordiate frame. [x, y, z, r, g, b].
    """
    pcl_xyz = np.hstack((pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
    pcl_new_xyz = T.dot(pcl_xyz.T).T
    pcl_new = np.hstack((pcl_new_xyz[:, :3], pointcloud[:, 3:]))
    return pcl_new


def rgbd_to_pcl(rgb_image, depth_image, camera_model):
    """Convert rgb-d image to pointcloud. Adapted from https://github.com/ethz-asl/scenenet_ros_tools/blob/master/nodes/scenenet_to_rosbag.py"""
    center_x = camera_model.cx
    center_y = camera_model.cy

    constant_x = 1 / camera_model.fx
    constant_y = 1 / camera_model.fy

    vs = np.array(
        [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
    us = np.array(
        [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

    # euclidean_ray_length_to_z_coordinate
    depth_image = euclidean_ray_length_to_z_coordinate(
        depth_image, camera_model)
    # Convert depth from cm to m.
    depth_image = depth_image / 1000.0

    x = np.multiply(depth_image, vs)
    y = depth_image * us[:, np.newaxis]

    stacked = np.ma.dstack((x, y, depth_image, rgb_image))
    compressed = stacked.compressed()
    pointcloud = compressed.reshape((int(compressed.shape[0] / 6), 6))

    return pointcloud


def convert_camera_coordinates_to_world(coor_camera, trajectory, frame):
    """Convert coordinates in camera frame to coordinates in world frame.
    Args:
        coor_camera: numpy array of shape (3, ). Coordnates in camera frame. [x, y, z].
        trajectory: int. A trajectory in SceneNetRGBD datasetself.
        frame: int. A certain frame of in the trajectory.

    Returns:
        coor_world: coordinates in world frame.
    """
    trajectories = sn.Trajectories()
    try:
        with open(pathconfig.protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(
            data_root_path))
        print('Please ensure you have copied the pb file to the data directory')

    view = trajectories.trajectories[trajectory].views[frame]
    ground_truth_pose = interpolate_poses(
        view.shutter_open, view.shutter_close, 0.5)
    camera_to_world_matrix = camera_to_world_with_pose(ground_truth_pose)

    # Homogeneous coordinates
    coor_homo_camera = np.append(coor_camera, [1])
    coor_homo_world = camera_to_world_matrix.dot(coor_homo_camera)
    coor_world = coor_homo_world[:3]

    return coor_world


def euclidean_ray_length_to_z_coordinate(depth_image, camera_model):
    """From https://github.com/ethz-asl/scenenet_ros_tools/blob/master/nodes/scenenet_to_rosbag.py"""
    center_x = camera_model.cx
    center_y = camera_model.cy

    constant_x = 1 / camera_model.fx
    constant_y = 1 / camera_model.fy

    vs = np.array(
        [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
    us = np.array(
        [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

    return (np.sqrt(np.square(depth_image / 1000.0) / (1 + np.square(vs[np.newaxis, :]) + np.square(us[:, np.newaxis]))) * 1000.0).astype(np.uint16)
