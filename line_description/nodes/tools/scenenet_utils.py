import numpy as np
import sys


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
        rbg_image (numpy array of shape (240, 320, 3), dtype=np.uint8): RGB
            image of the point cloud projected to the image plane.
        depth_image (numpy array of shape (240, 320), dtype=np.float32): Depth
            image of the point cloud projected to the image plane. The depth
            unit is mm.
    """
    rgb_image = np.zeros((240, 320, 3), dtype=np.uint8)
    depth_image = np.zeros((240, 320), dtype=np.float32)

    pcl_inside_view = pointcloud[pointcloud[:, 2] > 0, :]
    pcl_inside_view_xyz = np.hstack((pcl_inside_view[:, :3],
                                     np.ones((pcl_inside_view.shape[0], 1))))

    pcl_projected = np.array(camera_model.P).dot(pcl_inside_view_xyz.T)
    pixel = np.rint(pcl_projected / pcl_projected[2, :]).astype(int)[:2, :]

    index_bool = np.logical_and(
        np.logical_and(0 <= pixel[0], pixel[0] < 320),
        np.logical_and(0 <= pixel[1], pixel[1] < 240))
    pixel = pixel[:, index_bool]

    pcl_inside_view = pcl_inside_view[index_bool, :]

    # Considering occlusion, we need to be careful with the order of assignment
    # descending order according to the z coordinate
    index_sort = pcl_inside_view[:, 2].argsort()[::-1]
    pixel = pixel[:, index_sort]
    pcl_inside_view = pcl_inside_view[index_sort, :]

    rgb_image[pixel[1], pixel[0]] = pcl_inside_view[:, 3:]
    depth_image[pixel[1], pixel[0]] = pcl_inside_view[:, 2] * 1000.0  # m to mm

    # print('not projected points:', pointcloud.shape[0] - pcl_inside_view.shape[0])
    return rgb_image, depth_image


def get_origin_virtual_camera(start3D, end3D, hessian_left, hessian_right,
                              line_type, distance):
    """Get the origin of virtual camera for the line. The origin candidates are
       determined by the surfaces normals of the line. The virtual camera origin
       is chosen as the nearest point to real camera origin.

    Args:
        start3D/end3D (numpy array of shape (3, 1)): Endpoints of the line in
            3D.
        hessian_left/right (numpy array of shape (4, 1)): Inliers planes of the
            line in hessian normal form.
        line_type (int): Type of the line: 0 -> Discontinuity, 1 -> Planar,
            2 -> Edge, 3 -> Intersection.
        distance (float): Distance of the virtual camera from the line (in
            meters).

    Returns:
    """
    # x axis is taken as the direction of the line
    x = (end3D - start3D) / np.linalg.norm(end3D - start3D)
    middle_point = (start3D + end3D) / 2
    plane1_normal = hessian_left[:3]  # plane normal already normalized
    plane2_normal = hessian_right[:3]

    # Get possible virtual camera optical axis, store them in the list z_cand
    if line_type == 0:  # Discontinuity line.
        # Discontinuity lines have only one valid surface normal, the other one
        # is set to [0, 0, 0, 0].
        if np.linalg.norm(plane1_normal) <= 0.001:
            z_cand = [plane2_normal, -plane2_normal]
        if np.linalg.norm(plane2_normal) <= 0.001:
            z_cand = [plane1_normal, -plane1_normal]

    if line_type == 1:  # Planar (surface) line
        # Surface line has 2 similar surface normals, thus we can always take
        # only the normal of plane_1.
        z_cand = [plane1_normal, -plane1_normal]

    if line_type == 2 or line_type == 3:  # Edge or intersection line
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
        # Project vector z onto the plane which is perpendicular to the line.
        p1 = np.array([0, 0, 0])
        p2 = z

        p1 = p1 - (p1 - middle_point).dot(x) * x
        p2 = p2 - (p2 - middle_point).dot(x) * x
        z = (p2 - p1) / np.linalg.norm(p2 - p1)

        origin = middle_point - distance * z
        origin_cand.append(origin)

    min_dist = np.inf
    origin_virtual_camera = origin_cand[0]

    # Choose the nearest origin of virtual camera to the origin of the real
    # camera.
    for origin in origin_cand:
        if np.linalg.norm(origin) < min_dist:
            min_dist = np.linalg.norm(origin)
            origin_virtual_camera = origin

    return origin_virtual_camera


def virtual_camera_pose(start3D, end3D, hessian_left, hessian_right, line_type,
                        distance):
    """Get the virtual camera's pose according to the line.

    Args:
        start3D/end3D (numpy array of shape (3, 1)): Endpoints of the line in
            3D.
        hessian_left/right (numpy array of shape (4, 1)): Inliers planes of the
            line in hessian normal form.
        line_type (int): Type of the line: 0 -> Discontinuity, 1 -> Planar,
            2 -> Edge, 3 -> Intersection.
        distance (float): Distance of the virtual camera from the line (in
            meters).

    Returns:
        T (numpy array of shape (4, 4)): Transformation matrix.
        z (numpy array of shape (3, )): Optical axis of the virtual camera.
    """
    x = (end3D - start3D) / np.linalg.norm(end3D - start3D)
    middle_point = (start3D + end3D) / 2

    origin = get_origin_virtual_camera(start3D, end3D, hessian_left,
                                       hessian_right, line_type, distance)
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
        pcl_new: numpy array of shape (points_number, 6). The pointcloud
            expressed in the new coordinate frame. [x, y, z, r, g, b].
    """
    pcl_xyz = np.hstack((pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
    pcl_new_xyz = T.dot(pcl_xyz.T).T
    pcl_new = np.hstack((pcl_new_xyz[:, :3], pointcloud[:, 3:]))
    return pcl_new
