"""
Get virtual-camera image for each line in the given frame.
"""
import numpy as np
import cv2

from tools import cloud_utils
from tools import scenenet_utils
from tools import virtual_camera_utils


class VirtualCameraImageRetriever:
    """ Retrieves a virtual-camera image from a given set of lines.

    Args:
        distance_from_line (float): Distance of the virtual camera from the
            line (in meters).
        impainting (bool): Set to true to impaint the empty pixels of the
            virtual images. Beware that this is computationally intensive.

    Attributes:
        distance_from_line (float): Distance of the virtual camera from the
            line.
        impainting_mode_on (bool): Set to true to impaint the empty pixels of
            the virtual-camera images. Beware that this is computationally
            intensive.
    """

    def __init__(self, distance_from_line, impainting=False):
        self.distance_from_line = distance_from_line
        self.impainting_mode_on = impainting

    def get_virtual_camera_image(self, start3D, end3D, hessian_left,
                                 hessian_right, line_type, image_rgb, cloud):
        """ Returns a virtual-camera image for the given line.

        Args:
            start3D/end3D (numpy array of shape (3, 1)): Endpoints of the line
                in 3D.
            hessian_left/right (numpy array of shape (4, 1)): Inliers planes of
                the line in hessian normal form.
            line_type (int): Type of the line: 0 -> Discontinuity, 1 -> Planar,
                2 -> Edge, 3 -> Intersection.
            image_rgb (numpy array of shape(height, width, 3)): RGB image.
            cloud (numpy array of shape(height, width, 3)): Cloud image.
                cloud[i, j, :] contains the (x, y, z) coordinates of the point
                shown at pixel (i, j).

        Returns:
            RGB and depth virtual-camera images (tuple).
        """
        # Virtual camera is of the SceneNetRGBD (pinhole) camera model.
        virtual_camera = scenenet_utils.get_camera_model()

        # Obtain the pose of the virtual camera for each line.
        T, _ = virtual_camera_utils.virtual_camera_pose(
            start3D=start3D,
            end3D=end3D,
            hessian_left=hessian_left,
            hessian_right=hessian_right,
            line_type=line_type,
            distance=self.distance_from_line)

        # Draw the line in the virtual camera image (in red).
        line_3D = np.hstack([start3D, [0, 0, 255]])

        num_points_in_line = 1000
        for idx in range(num_points_in_line):
            line_3D = np.vstack([
                line_3D,
                np.hstack([(start3D + idx / float(num_points_in_line) *
                            (end3D - start3D)), [0, 0, 255]])
            ])
        # Construct the coloured point cloud.
        image_rgb = image_rgb.reshape(-1, 3)
        cloud = cloud.reshape(-1, 3)
        coloured_cloud = np.hstack([cloud, image_rgb])

        # Transform the point cloud so as to make it appear as seen from
        # the virtual camera pose.
        pcl_from_line_view = cloud_utils.pcl_transform(
            np.vstack([coloured_cloud, line_3D]), T)
        # Obtain the RGB and depth virtual camera images by reprojecting the
        # point cloud on the image plane, under the view of the virtual
        # camera.
        rgb_image_from_line_view, depth_image_from_line_view = \
           cloud_utils.project_pcl_to_image(pcl_from_line_view, virtual_camera)

        if (self.impainting_mode_on):
            # Inpaint the virtual camera image.
            reds = rgb_image_from_line_view[:, :, 2]
            greens = rgb_image_from_line_view[:, :, 1]
            blues = rgb_image_from_line_view[:, :, 0]

            mask = ((greens != 0) | (reds != 0) | (blues != 0)) * 1
            mask = np.array(mask, dtype=np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1) - mask

            rgb_image_from_line_view = cv2.inpaint(
                rgb_image_from_line_view, dilated_mask, 10, cv2.INPAINT_TELEA)
            depth_image_from_line_view = cv2.inpaint(
                depth_image_from_line_view, dilated_mask, 10, cv2.INPAINT_TELEA)

        # Return virtual image.
        return rgb_image_from_line_view, depth_image_from_line_view
