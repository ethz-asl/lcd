""" Get virtual camera image for each line in trajectory with index given as
    input.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
import argparse
from timeit import default_timer as timer

from tools import cloud_utils
from tools import virtual_camera_utils
from tools import interiornet_utils

import concurrent.futures


def world_to_cam_transform(view_pose):
    """
    Obtain the viewpoint transformation matrix from the view pose line of the InteriorNet dataset cam0.render files.
    :param view_pose: A numpy vector containing the camera pose, lookat pose and the up vector with shape (9, 1)
    :return: The 4x4 transformation matrix of the camera.
    """
    camera_pose = view_pose[:3]
    lookat_pose = view_pose[3:6]
    up = view_pose[6:]
    R = np.diag(np.ones(4))

    def normalize(vec):
        return vec / np.linalg.norm(vec)

    R[2, :3] = normalize(lookat_pose - camera_pose)
    R[0, :3] = normalize(np.cross(R[2, :3], (up - camera_pose)))
    R[1, :3] = -normalize(np.cross(R[0, :3], R[2, :3]))
    T = np.diag(np.ones(4))
    T[:3, 3] = -camera_pose
    return R.dot(T)


def parse_frames(scene_path, scene_type, traj):
    """
    Parse the view pose file of the InteriorNet scene.
    :param scene_path: The path to the scene directory.
    :param scene_type: The type of the scene, either HD1-6 or HD7.
    :param traj: The trajectory number (1, 3 or 7) if a HD1-6 scene is used.
    :return: times: The times where the camera views are taken.
             view_poses: The view poses vector with shape (N, 9) containing the camera pose, lookat pose and up vector.
    """
    if scene_type == 7:
        path = os.path.join(scene_path, "cam0.render")
    else:
        path = os.path.join(scene_path, "velocity_angular_{}_{}/cam0.render".format(traj, traj))

    # Read cam0.render file for camera pose.
    try:
        # Skip first 3 rows and every second row.
        lines = pd.read_csv(path, sep=" ", header=None, skiprows=3).iloc[::2, :]
    except IOError:
        print('cam0.render not found at location: {0}'.format(scene_path))
        print(path)
        sys.exit('Please ensure you have unzipped the files to the data directory.')

    # First three lines are comments.
    # Two lines per frame (shutter speed is 0).
    # Poses are written in the file as eye, look-at, up (see cam0.render)
    data = lines.to_numpy()

    view_poses = data[:, 1:]
    times = data[:, 0]

    # Prevent time being zero for HD7 scenes.
    if scene_type == 7:
        # Convert to nanoseconds.
        times = (times + 1.) * 1e9
    elif scene_type < 7:
        times = times

    return times, view_poses


def get_virtual_camera_images_interiornet(scene_path, scene_type, trajectory, light_type, linesfiles_path, output_path,
                                          moving_window_length=1):
    """
    Compute and save the virtual camera images of all valid lines in each frame of a InteriorNet scene.
    :param scene_path: Path to the scene.
    :param scene_type: The type of the scene, either HD1-6 or HD7.
    :param trajectory: The trajectory number (1, 3 or 7) if a HD1-6 scene is used.
    :param light_type: The light type of the scene. Original or random.
    :param linesfiles_path: The path to the directory containing the line files (output of the ROS node).
    :param output_path: The path to the output directory.
    :param moving_window_length: If sequential data is used, a moving window length can be specified to fuse point
                                 during render.
    """
    # Obtain the view poses from the scene files.
    times, view_poses = parse_frames(scene_path, scene_type, trajectory)
    # The number of views can be determined from that file.
    num_views = np.shape(view_poses)[0]

    # Photo paths are different for each scene and light types.
    if scene_type == 7 and light_type == "original":
        path_to_photos = os.path.join(scene_path, "cam0/data")
    elif scene_type == 7 and light_type == "random":
        path_to_photos = os.path.join(scene_path, "random_lighting_cam0/data")
    elif scene_type < 7:
        path_to_photos = os.path.join(scene_path, "{}_{}_{}".format(light_type, trajectory, trajectory),
                                  "cam0/data")
    else:
        print("ERROR: Wrong scene_type chosen. Please choose a valid one.")
        exit(1)

    # Depth paths as well.
    if scene_type == 7:
        path_to_depth = os.path.join(scene_path, "depth0/data")
    elif scene_type < 7:
        path_to_depth = os.path.join(scene_path, "{}_{}_{}".format(light_type, trajectory, trajectory),
                                     "depth0/data")
    else:
        print("ERROR: Wrong scene_type chosen. Please choose a valid one.")
        exit(1)

    print('Path to photos is {}'.format(path_to_photos))

    path_to_lines_root = linesfiles_path
    print('Path to lines_root is {0}'.format(path_to_lines_root))

    # Camera from which the images were taken, SceneNetRGBD (pinhole) camera
    # model.
    real_camera = interiornet_utils.get_camera_model()

    # Sliding window accumulator for point cloud, stored in world frame.
    pcl_world = np.zeros((0, 6))

    # Iterate over all frames.
    for frame_id in range(num_views):
        start_time = timer()
        photo_id = frame_id

        if scene_type < 7:
            photo_id = "{:019d}".format(int(times[photo_id]))

        # Load RGB and depth image.
        rgb_image = cv2.imread(
            os.path.join(path_to_photos, '{0}.png'.format(photo_id)),
            cv2.IMREAD_COLOR)

        depth_image = cv2.imread(
            os.path.join(path_to_depth, '{0}.png'.format(photo_id)),
            cv2.IMREAD_UNCHANGED)

        # Retrieve camera to world transformation.
        cam_to_world_t = np.linalg.inv(world_to_cam_transform(view_poses[frame_id]))

        # Obtain colored point cloud from RGB-D image.
        pcl_new = real_camera.rgbd_to_pcl(
            rgb_image=rgb_image, depth_image=depth_image, visualize_cloud=False)

        # If using a sliding window, point clouds are accumulated sequentially to obtain denser virtual camera
        # images.
        if moving_window_length > 1:
            # Add this point cloud (transformed to world frame) to the sliding window.
            pcl_world = np.vstack((cloud_utils.pcl_transform(
                pcl_new, cam_to_world_t), pcl_world))
            if np.shape(pcl_world)[0] > (np.shape(pcl_new)[0] * moving_window_length):
                pcl_world = pcl_world[:np.shape(pcl_new)[0] * moving_window_length, :]

            # Transform world point cloud back to camera frame.
            # And use this denser point cloud for virtual camera images.
            pcl = cloud_utils.pcl_transform(pcl_world, np.linalg.inv(cam_to_world_t))
        else:
            pcl = pcl_new

        path_to_lines = os.path.join(
            path_to_lines_root, 'lines_with_labels_{0}.txt'.format(frame_id))

        # Read the lines from the line file.
        try:
            data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
        except (IOError, pd.io.common.EmptyDataError):
            print("No line detected for frame {}".format(frame_id))
            continue

        data_lines = data_lines.values
        lines_count = data_lines.shape[0]

        average_time_per_line = 0

        # Multi-threaded rendering of the virtual camera images to increase computation speed.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(get_process_line_function(output_path, data_lines, frame_id, real_camera, pcl),
                         range(lines_count))

        end_time = timer()

        print('Generated virtual camera images for frame {0}'.format(frame_id))
        print('Time elapsed: {:.3f} seconds'.format(end_time - start_time))
        print('Average time per line: {:.3f} seconds'.format(
            average_time_per_line))


def get_process_line_function(output_path, data_lines, frame_id, real_camera, pcl):
    """
    This function makes thread pool execution easier for faster computation of the virtual camera images.
    It returns a function that computes the virtual camera images for one line and saves it in the corresponding
    directory.
    :param output_path: Path to the output of the frame.
    :param data_lines: The line data for the frame as output by the ROS node.
    :param frame_id: The id of the frame.
    :param real_camera: The InteriorNetCameraModel object for the camera intrinsics.
    :param pcl: The point cloud of the frame.
    """
    # Settings:
    # If the virtual camera images should be inpainted.
    inpainting = True
    # If a red line should be rendered where the line would be in the virtual camera images. Only for debugging.
    show_line = False
    # Distance between virtual camera origin and center of the line.
    cam_distance = 0.25
    # Min number of pixel per unit meter of the line. The more the line points
    # toward or away from the camera, the less pixels per meter.
    min_pixel_per_meter = 50.
    # Min and max x resolution of the virtual camera image.
    min_image_width = 40
    max_image_width = 120

    def process_line(i):
        T, _ = virtual_camera_utils.virtual_camera_pose_from_file_line(
            data_lines[i, :], cam_distance)
        start_point = data_lines[i, :3]
        end_point = data_lines[i, 3:6]
        line_length = np.linalg.norm(end_point - start_point)

        # Check the 2d to 3d length ratio to determine what the camera resolution should be to prevent aliasing.
        start_point_2d = real_camera.project3dToPixel(start_point)
        end_point_2d = real_camera.project3dToPixel(end_point)
        line_length_pixels = np.linalg.norm(end_point_2d - start_point_2d)
        # If the number of pixels per meter is too low, remove that line.
        if line_length_pixels * 1.5 > min_image_width and \
                line_length_pixels / line_length > min_pixel_per_meter:

            # Render the line in the form of a pcl if desired.
            if show_line:
                line_3d = np.hstack([start_point, [0, 0, 255]])

                num_points_in_line = 200
                for idx in range(num_points_in_line):
                    line_3d = np.vstack([
                        line_3d,
                        np.hstack([(start_point + idx / float(num_points_in_line) *
                                    (end_point - start_point)), [0, 0, 255]])
                    ])
                # Transform the point cloud so as to make it appear as seen from
                # the virtual camera pose.
                pcl_from_line_view = cloud_utils.pcl_transform(
                    np.vstack([pcl, line_3d]), T)
                # Move line points directly to the image plane, so that it is never occluded.
                pcl_from_line_view[-1000:, 2] = 0
            else:
                pcl_from_line_view = cloud_utils.pcl_transform(pcl, T)
            # Set the resolution according to line pixel density to prevent aliasing.
            img_width = min(max_image_width, int(line_length_pixels * 1.5))
            img_height = int(img_width / 1.5)
            # Obtain the RGB and depth virtual camera images by reprojecting the
            # point cloud on the image plane, under the view of the virtual
            # camera.
            (rgb_image_from_line_view, depth_image_from_line_view
             ) = cloud_utils.project_pcl_to_image_orthogonal(
                pcl_from_line_view,
                line_length * 1.5,
                line_length * 1, 0.5,
                img_width,
                img_height)

            if inpainting:
                reds = rgb_image_from_line_view[:, :, 2]
                greens = rgb_image_from_line_view[:, :, 1]
                blues = rgb_image_from_line_view[:, :, 0]

                # Create a mask that contains the neighboring black pixels of the non black pixels.
                mask = ((greens != 0) | (reds != 0) | (blues != 0)) * 1
                mask = np.array(mask, dtype=np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                kernel[0, 0] = 0
                kernel[0, -1] = 0
                kernel[-1, 0] = 0
                kernel[-1, -1] = 0
                erosion_kernel = np.ones((1, 3), np.uint8)
                # Opening to remove salt and pepper pixels.
                eroded_mask = cv2.erode(mask, erosion_kernel, iterations=1)
                opened_mask = cv2.dilate(eroded_mask, erosion_kernel, iterations=1)
                dilated_mask = cv2.dilate(opened_mask, kernel, iterations=1) - opened_mask

                rgb_image_from_line_view = cv2.bitwise_and(rgb_image_from_line_view,
                                                           rgb_image_from_line_view, mask=opened_mask)
                depth_image_from_line_view = cv2.bitwise_and(depth_image_from_line_view,
                                                             depth_image_from_line_view, mask=opened_mask)

                # Inpaint the mask using the Navier-Stokes algorithm.
                rgb_image_from_line_view = cv2.inpaint(
                    rgb_image_from_line_view, dilated_mask, 10,
                    cv2.INPAINT_NS)
                depth_image_from_line_view = cv2.inpaint(
                    depth_image_from_line_view, dilated_mask, 10,
                    cv2.INPAINT_NS)

            # Print images to file.
            cv2.imwrite(
                os.path.join(
                    output_path, 'frame_{0}/rgb/{1}.png'.format(
                        frame_id, i)), rgb_image_from_line_view)
            cv2.imwrite(
                os.path.join(output_path,
                             'frame_{0}/depth/{1}.png'.format(
                                 frame_id, i)),
                depth_image_from_line_view.astype(np.uint16))

    return process_line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get virtual camera image for each line in the input '
        'trajectory.')
    parser.add_argument("--trajectory", type=str, help="Trajectory number.")
    parser.add_argument(
        "-frame_step",
        type=int,
        help="Number of frames in one "
        "step of the rosbag used to detect lines, i.e., "
        "(frame_step - 1) frames were originally skipped after "
        "each frame inserted in the rosbag.")
    parser.add_argument(
        "-end_frame",
        type=int,
        help="Index of the last frame "
        "in the trajectory.")
    parser.add_argument(
        "-dataset_path",
        help="Path to folder containing the InteriorNet scene")
    parser.add_argument(
        "-dataset_name",
        help="If the data comes from"
        "InteriorNet, 'interiornet'.")
    parser.add_argument(
        "-linesandimagesfolder_path",
        help="Path to folder (e.g. 'data') containing text lines files (e.g. "
        "under 'data/train_0_lines') and that should store the output "
        "virtual-camera images (e.g. under 'data/train_0').")
    parser.add_argument(
        "-output_path",
        help="Data folder where to store the virtual-camera images (e.g. "
        "'data/train_0').")
    parser.add_argument(
        "-interiornet_scene_path",
        help="Path to the scene folder in InteriorNet.",
        type=str)
    parser.add_argument(
        "-interiornet_light_type",
        help="Type of lighting for InteriorNet, either original or random.",
        default='original',
        choices=['original', 'random'])
    parser.add_argument(
        "-interiornet_trajectory",
        help="Trajectory number for HD1-6 scenes in InteriorNet.",
        default=1,
        type=int,
        choices=[1, 3, 7],
        metavar='NUM')

    args = parser.parse_args()
    if (args.trajectory and args.scenenetscripts_path and args.dataset_name and
            args.dataset_path and args.linesandimagesfolder_path):
        # All strictly necessary arguments passed.
        trajectory = args.trajectory
        scenenetscripts_path = args.scenenetscripts_path
        dataset_name = args.dataset_name
        dataset_path = args.dataset_path
        linesandimagesfolder_path = args.linesandimagesfolder_path
    elif args.interiornet_scene_path and args.interiornet_light_type and args.interiornet_trajectory:
        interiornet_scene_path = args.interiornet_scene_path
        interiornet_light_type = args.interiornet_light_type
        interiornet_trajectory = args.interiornet_trajectory
        linesandimagesfolder_path = args.linesandimagesfolder_path
        dataset_name = args.dataset_name
        if not dataset_name == "interiornet":
            print("WARNING: Arguments passed for interiornet, but dataset_name is not interiornet.")
    else:
        print("ERROR: Please specify a correct dataset name and scene type. ")
        exit(0)

    if args.frame_step:
        frame_step = args.frame_step
    if args.end_frame:
        end_frame = args.end_frame

    linesfiles_path = os.path.join(linesandimagesfolder_path,
                                   '{}_lines'.format(dataset_name))
    output_path = os.path.join(linesandimagesfolder_path, dataset_name)

    if dataset_name == "interiornet":
        # Dataset from InteriorNet.
        scene_path_split = interiornet_scene_path.rsplit('/')
        scene_type = int(scene_path_split[-2][2])
        get_virtual_camera_images_interiornet(interiornet_scene_path,
                                              scene_type,
                                              interiornet_trajectory,
                                              interiornet_light_type,
                                              linesfiles_path,
                                              output_path)
    else:
        raise ValueError("Invalid dataset name. Valid names are 'val', "
                         "'train_NUM', where NUM is a number between 0 and 16, "
                         "and 'scenenn'.")
