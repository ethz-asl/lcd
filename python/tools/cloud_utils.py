import numpy as np


def pcl_transform(pointcloud, T):
    """ Transforms a pointcloud according to the transformation matrix.

    Args:
        pointcloud (numpy array of shape (points_number, 6)): pointcloud[i, :]
            contains the information about the i-th point in the point cloud, in
            the following format:
                [x, y, z, r, g, b].
        T (numpy array of shape (4, 4)): Transformation matrix.

    Returns:
        pcl_new (numpy array of shape (points_number, 6)): The pointcloud
            expressed in the new coordinate frame, in the same format as the
            input point cloud ([x, y, z, r, g, b]).
    """
    pcl_xyz = np.hstack((pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
    pcl_new_xyz = T.dot(pcl_xyz.T).T
    pcl_new = np.hstack((pcl_new_xyz[:, :3], pointcloud[:, 3:]))
    return pcl_new


def project_pcl_to_image(pointcloud,
                         camera_model,
                         image_width=320,
                         image_height=240):
    """ Projects a pointcloud to camera image.

    Args:
        pointcloud (numpy array of shape (num_points, 6)):
            Point cloud in the format [x, y, z, r, g, b] for each point.
        camera_model (Instance of a camera model class): Camera model to use to
            reproject the point cloud on the virtual-camera image plane.
        image_width (int): Image width.
        image_height (int): Image height.

    Returns:
        rbg_image (numpy array of shape (image_height, image_width, 3),
            dtype=np.uint8): RGB image.
        depth_image (numpy array of shape (image_height, image_width),
            dtype=np.float32): Depth image, with depth in mm.
        num_nonempty_pixels (int): Number of pixels that are not empty in the
            image obtained by reprojecting the point cloud.
    """
    rgb_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    depth_image = np.zeros((image_height, image_width), dtype=np.float32)
    pixel_is_empty = np.full((image_height, image_width), fill_value=True)

    num_nonempty_pixels = 0

    pcl_inside_view = pointcloud[pointcloud[:, 2] > 0, :]
    pcl_inside_view_xyz = np.hstack((pcl_inside_view[:, :3],
                                     np.ones((pcl_inside_view.shape[0], 1))))

    pcl_projected = np.array(camera_model.P).dot(pcl_inside_view_xyz.T)
    pixel = np.rint(pcl_projected / pcl_projected[2, :]).astype(int)[:2, :]

    index_bool = np.logical_and(
        np.logical_and(0 <= pixel[0], pixel[0] < image_width),
        np.logical_and(0 <= pixel[1], pixel[1] < image_height))
    pixel = pixel[:, index_bool]

    pcl_inside_view = pcl_inside_view[index_bool, :]

    # Considering occlusion, we need to be careful with the order of assignment
    # descending order according to the z coordinate.
    index_sort = pcl_inside_view[:, 2].argsort()[::-1]
    pixel = pixel[:, index_sort]
    pcl_inside_view = pcl_inside_view[index_sort, :]

    rgb_image[pixel[1], pixel[0]] = pcl_inside_view[:, 3:]
    depth_image[pixel[1], pixel[0]] = pcl_inside_view[:, 2] * 1000.0  # m to mm

    for idx in range(len(pixel[0])):
        if pixel_is_empty[pixel[1, idx], pixel[0, idx]] == True:
            pixel_is_empty[pixel[1, idx], pixel[0, idx]] = False
            num_nonempty_pixels += 1

    return rgb_image, depth_image, num_nonempty_pixels


def project_pcl_to_image_orthogonal(pointcloud,
                                    camera_width,
                                    camera_height,
                                    camera_depth,
                                    image_width=320,
                                    image_height=240):

    """ Projects a pointcloud to a orthogonal camera image.

        Args:
            pointcloud (numpy array of shape (num_points, 6)):
                Point cloud in the format [x, y, z, r, g, b] for each point.
            camera_width (float): Camera width in meters.
            camera_height (float): Camera height in meters.
            image_width (int): Image width.
            image_height (int): Image height.

        Returns:
            rbg_image (numpy array of shape (image_height, image_width, 3),
                dtype=np.uint8): RGB image.
            depth_image (numpy array of shape (image_height, image_width),
                dtype=np.float32): Depth image, with depth in mm.
            num_nonempty_pixels (int): Number of pixels that are not empty in the
                image obtained by reprojecting the point cloud.
        """

    rgb_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    depth_image = np.zeros((image_height, image_width), dtype=np.float32)
    pixel_is_empty = np.full((image_height, image_width), fill_value=True)

    from timeit import default_timer as timer
    tic = timer()

    num_nonempty_pixels = 0
    # Z is optical axis, remove all points that lie behind camera plane.
    min_x = -camera_width / 2
    min_y = -camera_height / 2
    min_z = 0.
    max_x = camera_width / 2
    max_y = camera_height / 2
    max_z = camera_depth
    bound_x = np.logical_and(pointcloud[:, 0] >= min_x, pointcloud[:, 0] < max_x)
    bound_y = np.logical_and(pointcloud[:, 1] >= min_y, pointcloud[:, 1] < max_y)
    bound_z = np.logical_and(pointcloud[:, 2] >= min_z, pointcloud[:, 2] < max_z)

    pcl_inside_view = pointcloud[np.logical_and(np.logical_and(bound_x, bound_y), bound_z), :]
    #pcl_inside_view = pointcloud[pointcloud[:, 2] >= 0 and pointcloud[:, 1] >= 0, :]

    # Move pixels into sensor space and convert to ints.
    #pixel_floats = np.vstack((pcl_inside_view[:, 0] / camera_width * image_width + image_width / 2,
    #                   pcl_inside_view[:, 1] / camera_height * image_height + image_height / 2))
    pixel_floats = (pcl_inside_view[:, :2] * np.array([image_width / camera_width, image_height / camera_height]) +
                    np.array([image_width / 2, image_height / 2])).transpose()
    pixel = pixel = pixel_floats.astype(int) # np.rint(pixel_floats).astype(int)

    #print(pixel_floats.shape)
    #print("Transform took {} seconds.".format(timer() - tic))

    # Remove pixels outside of view frame.
    index_bool = np.logical_and(
        np.logical_and(0 <= pixel[0], pixel[0] < image_width),
        np.logical_and(0 <= pixel[1], pixel[1] < image_height))
    pixel = pixel[:, index_bool]

    pcl_inside_view = pcl_inside_view[index_bool, :]

    # Considering occlusion, we need to be careful with the order of assignment
    # descending order according to the z coordinate.
    index_sort = pcl_inside_view[:, 2].argsort()[::-1]
    pixel = pixel[:, index_sort]
    pcl_inside_view = pcl_inside_view[index_sort, :]

    rgb_image[pixel[1], pixel[0]] = pcl_inside_view[:, 3:]
    depth_image[pixel[1], pixel[0]] = pcl_inside_view[:, 2] * 1000.0  # m to mm

    return rgb_image, depth_image.astype(np.uint8)
