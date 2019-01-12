import numpy as np
from transforms3d.euler import mat2euler


def get_label_with_line_center(labels_batch):
    """ Converts an input batch of lines with endpoints and instance label to a
        batch of lines with only center of the line and instance label.
    Args:
        labels_batch: Tensor of shape (batch_size, 7), with each row in the
                      format
                        [start point (3x)] [end point (3x)] [instance label]

    Returns:
        Tensor of shape (batch_size, 4), with each row in the format
          [center point (3x)] [instance label]
    """
    assert labels_batch.shape[1] == 7, "{}".format(labels_batch.shape)
    # Obtain center (batch_size, 6).
    center_batch = np.hstack(
        labels_batch[:, [[i], [i + 3]]].mean(axis=1) for i in range(3))
    # Concatenate instance labels column.
    return np.hstack((center_batch, labels_batch[:, [-1]]))


def get_geometric_info(start_points, end_points, line_parametrization):
    """Given a set of lines parametrized by their two endpoints, returns the
       following geometric info according to the line parametrization type:
       * 'direction_and_centerpoint':
             Each line segment is parametrized by its center point and by the
             first two entries of the unit direction vector. The third entry is
             defined automatically by enforcing that the norm should be one, and
             therefore is not fed into the network. To obtain invariance on the
             orientation of the line (i.e., given the two endpoints we do NOT
             want to consider one of them as the start and the other one as the
             end of the line segment), we enforce that the first entry should be
             non-negative. => 5 parameters per line.
       * 'orthonormal':
             A line segment is parametrized with a minimum-DOF parametrization
             (4 degrees of freedom) of the infinite line that it belongs to. The
             representation is called orthonormal. => 4 parameters per line.
    Args:
        start_points: Array of shape (num_lines, 3) containing the start point
                      of each line.
        end_points:   Array of shape (num_lines, 3) containing the end point of
                      each line.
        line_parametrization: Type of line parametrization, either
                              'direction_and_centerpoint' or 'orthonormal'.

    Returns:
        geometric_info: Array of shape (num_lines, 5) or (num_lines, 4),
                        depending on the line parametrization, that for each
                        line contains the geometric info described above.
    """
    assert (start_points.shape[0] == end_points.shape[0])
    assert (start_points.shape[1] == end_points.shape[1] == 3)
    num_lines = start_points.shape[0]
    if line_parametrization == 'direction_and_centerpoint':
        geometric_info = np.empty([num_lines, 5])
        for idx in range(num_lines):
            start_point = start_points[idx, :]
            end_point = end_points[idx, :]
            geometric_info[idx, :] = endpoints_to_centerpoint_and_direction(
                start_point, end_point)
    elif line_parametrization == 'orthonormal':
        geometric_info = np.empty([num_lines, 4])
        for idx in range(num_lines):
            start_point = start_points[idx, :]
            end_point = end_points[idx, :]
            pluecker_coordinates = endpoints_to_pluecker_coordinates(
                start_point, end_point)
            orthonormal_representation = pluecker_to_orthonormal_representation(
                pluecker_coordinates)
            geometric_info[idx, :] = orthonormal_representation
    else:
        raise ValueError("Line parametrization should be "
                         "'direction_and_centerpoint' or 'orthonormal'.")

    return geometric_info


def endpoints_to_centerpoint_and_direction(start_point, end_point):
    """ Given the endpoints of a line segment returns an array with the
        following format:
          [center point (3x)] [first two entries of unit direction vector (2x)]
        where, furthermore, the first two entries of the unit direction vector
        are such that the first one is non-negative.
    Args:
        start_point, end_point: shape (3, ): Endpoints of the input line.

    Returns:
        Array of shape (5, ), with the first three elements representing the
        center point of the line segment and the last two elements representing
        the first two entries of the unit direction vector, with the first of
        these two entries strictly non-negative.
    """
    assert (start_point.shape == end_point.shape == (3,))
    center_point = ((start_point + end_point) / 2.).reshape(3, 1)
    direction = end_point - start_point
    direction = direction / np.linalg.norm(direction)
    # Take first two entries.
    direction = direction[:2].reshape(2, 1)
    if direction[0] < 0:
        direction = -direction

    return np.vstack((center_point, direction)).reshape(5,)


def endpoints_to_pluecker_coordinates(start_point, end_point):
    """ Returns the Pluecker coordinates for a line given its endpoints (in
        regular inhomogeneous coordinates).
    Args:
        start_point, end_point: shape (3, ): Endpoints of the input line.

    Returns:
        Array of shape (6, ), with the first three elements representing n,
        the normal vector of the plane determined by the line and the origin,
        and the last three elements representing v, the direction vector of the
        line.
    """
    assert (start_point.shape == end_point.shape == (3,))
    # Normal vector of the plane determined by the line and the origin.
    n = np.cross(start_point, end_point)
    # Direction vector of the line.
    v = (end_point - start_point)

    return np.vstack((n, v)).reshape(6,)


def pluecker_to_orthonormal_representation(pluecker_coordinates):
    """ Returns the minimum-DOF (4 degrees of freedom) orthonormal
        representation proposed by [1], given Pluecker coordinates as input
        (also cf. [2]).

        The orthonormal representation (U, W) in SO(3) x SO(2) can be
        parametrized the parameter vector
            theta = (theta_1, theta_2, theta_3, theta_4),
        where theta_1, theta_2, theta_3 are Euler angles associated to the
        3-D rotation matrix U (e.g. in the order of rotation x, y, z, i.e.,
        U = R_x(theta_1) * R_y(theta_2) * R_z(theta_3)) and theta_4 is the
        rotation angle associated to the 2-D rotation matrix W.

        [1] Bartoli, Strum - "Structure-from-motion using lines: Representation,
            triangulation, and bundle adjustment".
        [2] Zuo et al. - "Robust Visual SLAM with Point and Line Features".

    Args:
        pluecker_coordinates: Array of shape (6, ), with the first three
                              elements representing n, the normal vector of the
                              plane determined by the line and the origin, and
                              the last three elements representing v, the
                              direction vector of the line.

    Returns:
        theta: Parameter vector defined above.
    """
    assert (pluecker_coordinates.shape == (6,))
    # Obtain n and v from the Pluecker coordinates.
    n = pluecker_coordinates[:3].reshape(3, 1)
    v = pluecker_coordinates[3:6].reshape(3, 1)
    n_cross_v = np.cross(n.reshape(3,), v.reshape(3,)).reshape(3, 1)
    # Extract the U matrix, w1 and w2 (c.f. eqn. (2)-(3) in [2]).
    U = np.hstack([
        n / np.linalg.norm(n), v / np.linalg.norm(v),
        (n_cross_v / np.linalg.norm(n_cross_v))
    ])
    w1 = np.linalg.norm(n)
    w2 = np.linalg.norm(v)
    # Note: W is in SO(2) => Normalize it.
    W = np.array([[w1, -w2], [w2, w1]]) / np.linalg.norm([w1, w2])
    # Check that U is a rotation matrix: U'*U should be I, where U' is U
    # transpose and I is the 3x3 identity matrix.
    if (np.linalg.norm(np.eye(3) - np.dot(U, np.transpose(U))) > 1e-4):
        raise ValueError('Matrix U is not a rotation matrix.')
    # Find the Euler angles associated to the two rotation matrices.
    theta_1, theta_2, theta_3 = mat2euler(U, axes='sxyz')
    theta_4 = np.arccos(W[0, 0])
    # Assign the angles to the parameter vector.
    theta = np.array([theta_1, theta_2, theta_3, theta_4])

    return theta
