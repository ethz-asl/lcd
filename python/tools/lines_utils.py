import numpy as np
from transforms3d.euler import mat2euler


def get_line_center(labels_batch):
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
                                  elements representing n, the normal vector of
                                  the plane determined by the line and the
                                  origin, and the last three elements
                                  representing v, the direction vector of the
                                  line.
        Returns:
            theta: Parameter vector defined above.
    """
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
    if (np.linalg.norm(np.eye(3) - np.dot(U, np.transpose(U))) > 1e-6):
        raise ValueError('Matrix U is not a rotation matrix.')
    # Find the Euler angles associated to the two rotation matrices.
    theta_1, theta_2, theta_3 = mat2euler(U, axes='sxyz')
    theta_4 = np.arccos(W[0, 0])
    # Assign the angles to the parameter vector.
    theta = np.array([theta_1, theta_2, theta_3, theta_4])

    return theta
