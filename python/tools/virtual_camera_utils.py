import numpy as np


def get_origin_virtual_camera(start3D, end3D, hessian_left, hessian_right,
                              line_type, distance):
    """ Gets the origin of virtual camera for the line. The origin candidates
        are determined based on the surfaces normals of the line. The origin of
        the virtual camera is chosen as the point closest to real camera origin.

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
        origin_virtual_camera (numpy array of shape(3, 1)): Origin of the
            virtual camera.
    """
    # X axis is taken as the direction of the line.
    x = (end3D - start3D) / np.linalg.norm(end3D - start3D)
    middle_point = (start3D + end3D) / 2
    # Plane normals are already normalized.
    plane1_normal = hessian_left[:3]
    plane2_normal = hessian_right[:3]

    # Get possible virtual camera optical axis, store them in the list z_cand.
    if line_type == 0:  # Discontinuity line.
        # Discontinuity lines have only one valid surface normal, the other one
        # is set to [0, 0, 0, 0].
        if np.linalg.norm(plane1_normal) <= 0.001:
            z_cand = [plane2_normal, -plane2_normal]
        if np.linalg.norm(plane2_normal) <= 0.001:
            z_cand = [plane1_normal, -plane1_normal]

    if line_type == 1:  # Planar (surface) line.
        # Surface lines have 2 similar surface normals, thus we can always take
        # only the normal of plane_1.
        z_cand = [plane1_normal, -plane1_normal]

    if line_type == 2 or line_type == 3:  # Edge or intersection line.
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


def get_origin_virtual_camera_from_line(line, distance):
    """ Gets the origin of virtual camera for a line read from a file_line.

    Args:
        line (numpy array of shape (22, )): The format of the line is the
            following:
                [start point (3, ), end point (3, ),
                 left plane hessian form (4, ), right plane hessian form(4, ),
                 left color (3, ), right color(3, ), line type (1, ),
                 instance label(1, )].
        distance (float): Distance in meters between the origin of the virtual
            camera and the middle point of the line.

    Returns:
        origin_virtual_camera (numpy array of shape (3, )): Origin of the
            virtual camera.
    """
    return get_origin_virtual_camera_from_line(
        start3D=line[:3],
        end3D=line[3:6],
        hessian_left=line[6:10],
        hessian_right=line[10:14],
        line_type=line[-2],
        distance=distance)


def virtual_camera_pose(start3D, end3D, hessian_left, hessian_right, line_type,
                        distance):
    """ Gets the pose of the virtual camera according to the line.

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


def virtual_camera_pose_from_file_line(line, distance):
    """ Gets the pose of the virtual camera according to the line read from a
        file of lines.

    Args:
        line (numpy array of shape (22, )): The format of the line is the
            following:
                [start point (3, ), end point (3, ),
                 left plane hessian form (4, ), right plane hessian form(4, ),
                 left color (3, ), right color(3, ), line type (1, ),
                 instance label(1, )].
        distance (float): Distance in meters between the origin of the virtual
            camera and the middle point of the line.

    Returns:
        T (numpy array of shape (4, 4)): Transformation matrix.
        z (numpy array of shape (3, )): Optical axis of the virtual camera.
    """
    return virtual_camera_pose(
        start3D=line[:3],
        end3D=line[3:6],
        hessian_left=line[6:10],
        hessian_right=line[10:14],
        line_type=line[-2],
        distance=distance)
