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

from tools import camera_utils
from tools import cloud_utils
from tools import scenenn_utils
#from tools import scenenet_utils
from tools import virtual_camera_utils
from tools import pathconfig
from tools import get_protobuf_paths
from tools import interiornet_utils


def get_virtual_camera_images_scenenet_rgbd(trajectory, dataset_path, dataset_name, scenenetscripts_path):
    impainting = True
    show_line = False
    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        sys.exit('get_virtual_camera_images.py: Scenenet protobuf data not '
                 'found at location:{0}. '.format(protobuf_path) + 'Please '
                 'ensure you have copied the pb file to the data directory.')
        return

    trajectory = int(trajectory)
    traj_renderpath = trajectories.trajectories[trajectory].render_path

    path_to_photos = os.path.join(dataset_path, traj_renderpath)

    print('Path to photos is {}'.format(path_to_photos))

    path_to_lines_root = os.path.join(linesfiles_path,
                                      'traj_{0}/'.format(trajectory))

    print('Path_to_lines_root is {0}'.format(path_to_lines_root))

    # Virtual camera is of the SceneNetRGBD (pinhole) camera model.
    virtual_camera = scenenet_utils.get_camera_model()
    # Camera from which the images were taken, SceneNetRGBD (pinhole) camera
    # model.
    real_camera = scenenet_utils.get_camera_model()

    # Distance between virtual camera origin and center of the line.
    distance = 0.5
    # Min fraction of nonempty pixels in the associated virtual-camera image for
    # a line to be considered as valid.
    min_fraction_nonempty_pixels = 0.3

    # Write in a text file (that will store the histogram of percentages of
    # nonempty pixels) the distance of the virtual camera from the lines and the
    # threshold of nonempty pixels to discard lines.
    with open('hist_percentages.txt', 'w') as f:
        f.write("distance=%.3f\n" % distance)
        f.write("fraction_nonempty_pixels_threshold=%.3f\n" %
                min_fraction_nonempty_pixels)


    # Camera to world matrix retriever.
    cam_to_world = camera_utils.SceneNetCameraToWorldMatrixRetriever(trajectory,
                                                                     dataset_name,
                                                                     scenenetscripts_path)

    # Sliding window accumulator for point cloud, stored in world frame.
    pcl_world = np.zeros((0, 6))

    for frame_id in range(300):
        start_time = timer()
        photo_id = frame_id * 25

        rgb_image = cv2.imread(
            os.path.join(path_to_photos, 'photo/{0}.jpg'.format(photo_id)),
            cv2.IMREAD_COLOR)

        depth_image = cv2.imread(
            os.path.join(path_to_photos, 'depth/{0}.png'.format(photo_id)),
            cv2.IMREAD_UNCHANGED)

        # Retrieve camera to world transformation.
        cam_to_world_t = cam_to_world.get_camera_to_world_matrix(frame_id)

        # Obtain coloured point cloud from RGB-D image.
        pcl_new = real_camera.rgbd_to_pcl(
            rgb_image=rgb_image, depth_image=depth_image, visualize_cloud=False)
        # Add this point cloud (transformed to world frame) to the sliding window.
        pcl_world = np.vstack((cloud_utils.pcl_transform(pcl_new, cam_to_world_t), pcl_world))
        if np.shape(pcl_world)[0] > (np.shape(pcl_new)[0] * 10):
            pcl_world = pcl_world[:np.shape(pcl_new)[0]*10, :]

        # Transform world point cloud back to camera frame. And use this denser point cloud for virtual camera images.
        pcl = cloud_utils.pcl_transform(pcl_world, np.linalg.inv(cam_to_world_t))
        print(np.shape(pcl))

        path_to_lines = os.path.join(
            path_to_lines_root, 'lines_with_labels_{0}.txt'.format(frame_id))

        try:
            data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
        except (IOError, pd.io.common.EmptyDataError):
            print("No line detected for frame {}".format(frame_id))
            continue

        data_lines = data_lines.values
        lines_count = data_lines.shape[0]

        average_time_per_line = 0
        # A line is valid if the corresponding virtual-camera image has enough
        # nonempty pixels.
        num_valid_lines = 0
        # Initialize the histogram of the percentages of nonempty pixels.
        hist_percentage_nonempty_pixels = []
        for i in range(lines_count):
            start_time_line = timer()
            # Obtain the pose of the virtual camera for each line.
            T, _ = virtual_camera_utils.virtual_camera_pose_from_file_line(
                data_lines[i, :], distance)
            # Draw the line in the virtual camera image.
            start_point = data_lines[i, :3]
            end_point = data_lines[i, 3:6]
            line_length = np.linalg.norm(end_point - start_point)

            if show_line:
                line_3D = np.hstack([start_point, [0, 0, 255]])

                num_points_in_line = 200;
                for idx in range(num_points_in_line):
                    line_3D = np.vstack([
                        line_3D,
                        np.hstack([(start_point + idx / float(num_points_in_line) *
                                    (end_point - start_point)), [0, 0, 255]])
                    ])
                # Transform the point cloud so as to make it appear as seen from
                # the virtual camera pose.
                pcl_from_line_view = cloud_utils.pcl_transform(
                    np.vstack([pcl, line_3D]), T)
                # Move line points directly to the image plane, so that it is never occluded.
                pcl_from_line_view[-1000:, 2] = 0;
            else:
                pcl_from_line_view = cloud_utils.pcl_transform(pcl, T)
            # Obtain the RGB and depth virtual camera images by reprojecting the
            # point cloud on the image plane, under the view of the virtual
            # camera. Also obtain the number of nonempty pixels.
            (rgb_image_from_line_view, depth_image_from_line_view,
             num_nonempty_pixels) = cloud_utils.project_pcl_to_image_orthogonal(
                pcl_from_line_view, line_length * 2, line_length * 6/4, 40, 30)
            # cloud_utils.project_pcl_to_image(pcl_from_line_view, virtual_camera)
            # Discard lines that have less than the specified fraction of
            # nonempty pixels.
            fraction_nonempty_pixels = num_nonempty_pixels / float(
                rgb_image_from_line_view.shape[0] * rgb_image_from_line_view.
                shape[1])
            # Add the percentage of nonempty pixels to the histogram.
            percentage_nonempty_pixels = 100. * fraction_nonempty_pixels
            hist_percentage_nonempty_pixels.append(percentage_nonempty_pixels)
            if (fraction_nonempty_pixels > min_fraction_nonempty_pixels):
                num_valid_lines += 1
                if (impainting):
                    # Inpaint the virtual camera image.
                    reds = rgb_image_from_line_view[:, :, 2]
                    greens = rgb_image_from_line_view[:, :, 1]
                    blues = rgb_image_from_line_view[:, :, 0]

                    mask = ((greens != 0) | (reds != 0) | (blues != 0)) * 1
                    mask = np.array(mask, dtype=np.uint8)
                    kernel = np.ones((5, 5), np.uint8)
                    dilated_mask = cv2.dilate(mask, kernel, iterations=1) - mask

                    rgb_image_from_line_view = cv2.inpaint(
                        rgb_image_from_line_view, dilated_mask, 10,
                        cv2.INPAINT_NS)
                    depth_image_from_line_view = cv2.inpaint(
                        depth_image_from_line_view, dilated_mask, 10,
                        cv2.INPAINT_NS)
                end_time_line = timer()
                average_time_per_line += (end_time_line - start_time_line)

                # Upscaling to 320 x 240 resolution.
                #dim = (320, 240)
                #rgb_image_from_line_view = cv2.resize(
                #    rgb_image_from_line_view, dim, interpolation=cv2.INTER_AREA)
                #depth_image_from_line_view = cv2.resize(
                #    depth_image_from_line_view, dim, interpolation=cv2.INTER_AREA)

                # Print images to file.
                cv2.imwrite(
                    os.path.join(
                        output_path, 'traj_{0}/frame_{1}/rgb/{2}.png'.format(
                            trajectory, frame_id, i)), rgb_image_from_line_view)
                cv2.imwrite(
                    os.path.join(output_path,
                                 'traj_{0}/frame_{1}/depth/{2}.png'.format(
                                     trajectory, frame_id, i)),
                    depth_image_from_line_view.astype(np.uint16))

        end_time = timer()
        if (num_valid_lines > 0):
            average_time_per_line /= num_valid_lines

        # Append the histogram for the frame to the output text file.
        with open('hist_percentages.txt', 'aw') as f:
            for item in hist_percentage_nonempty_pixels:
                f.write("%s\n" % item)

        print('Generated virtual camera images for frame {0}'.format(frame_id))
        print('Time elapsed: {:.3f} seconds'.format(end_time - start_time))
        print('Average time per line: {:.3f} seconds'.format(
            average_time_per_line))
        print('Number of lines discarded (minimum fraction of required ' +
              'nonempty pixels = {0:.3f}) is {1}'.format(
                  min_fraction_nonempty_pixels, lines_count - num_valid_lines))


def get_virtual_camera_images_scenenn(trajectory, dataset_path):
    impainting = False
    path_to_photos = os.path.join(dataset_path, trajectory)

    print('Path to photos is {}'.format(path_to_photos))

    path_to_lines_root = os.path.join(linesfiles_path,
                                      'traj_{0}/'.format(trajectory))

    print('Path_to_lines_root is {0}'.format(path_to_lines_root))

    # Virtual camera is of the SceneNetRGBD (pinhole) camera model.
    virtual_camera = scenenet_utils.get_camera_model()
    # Camera from which the images were taken, SceneNN (pinhole) camera model.
    real_camera = scenenn_utils.get_camera_model()

    # Distance between virtual camera origin and center of the line.
    distance = 3
    # Min fraction of nonempty pixels in the associated virtual-camera image for
    # a line to be considered as valid.
    min_fraction_nonempty_pixels = 0.1

    # Write in a text file (that will store the histogram of percentages of
    # nonempty pixels) the distance of the virtual camera from the lines and the
    # threshold of nonempty pixels to discard lines.
    with open('hist_percentages.txt', 'w') as f:
        f.write("distance=%.3f\n" % distance)
        f.write("fraction_nonempty_pixels_threshold=%.3f\n" %
                min_fraction_nonempty_pixels)

    for frame_id in range(2, end_frame + 1, frame_step):
        start_time = timer()
        photo_id = frame_id

        rgb_image = cv2.imread(
            os.path.join(path_to_photos,
                         'image/image{:05d}.png'.format(photo_id)),
            cv2.IMREAD_COLOR)

        depth_image = cv2.imread(
            os.path.join(path_to_photos,
                         'depth/depth{:05d}.png'.format(photo_id)),
            cv2.IMREAD_UNCHANGED)

        # Obtain coloured point cloud from RGB-D image.
        pcl = real_camera.rgbd_to_pcl(
            rgb_image=rgb_image, depth_image=depth_image, visualize_cloud=False)
        path_to_lines = os.path.join(
            path_to_lines_root, 'lines_with_labels_{0}.txt'.format(frame_id))

        try:
            data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
        except (IOError, pd.io.common.EmptyDataError):
            print("No line detected for frame {}".format(frame_id))
            continue

        data_lines = data_lines.values
        lines_count = data_lines.shape[0]

        average_time_per_line = 0
        # A line is valid if the corresponding virtual-camera image has enough
        # nonempty pixels.
        num_valid_lines = 0
        # Initialize the histogram of the percentages of nonempty pixels.
        hist_percentage_nonempty_pixels = []
        for i in range(lines_count):
            start_time_line = timer()
            # Obtain the pose of the virtual camera for each line.
            T, _ = virtual_camera_utils.virtual_camera_pose_from_file_line(
                data_lines[i, :], distance)
            # Draw the line in the virtual camera image.
            start_point = data_lines[i, :3]
            end_point = data_lines[i, 3:6]
            line_3D = np.hstack([start_point, [0, 0, 255]])

            num_points_in_line = 1000
            for idx in range(num_points_in_line):
                line_3D = np.vstack([
                    line_3D,
                    np.hstack([(start_point + idx / float(num_points_in_line) *
                                (end_point - start_point)), [0, 0, 255]])
                ])
            # Transform the point cloud so as to make it appear as seen from
            # the virtual camera pose.
            pcl_from_line_view = cloud_utils.pcl_transform(
                np.vstack([pcl, line_3D]), T)
            # Obtain the RGB and depth virtual camera images by reprojecting the
            # point cloud on the image plane, under the view of the virtual
            # camera. Also obtain the number of nonempty pixels.
            (rgb_image_from_line_view, depth_image_from_line_view,
             num_nonempty_pixels) = cloud_utils.project_pcl_to_image(
                 pcl_from_line_view, virtual_camera)
            # Discard lines that have less than the specified fraction of
            # nonempty pixels.
            fraction_nonempty_pixels = num_nonempty_pixels / float(
                rgb_image_from_line_view.shape[0] * rgb_image_from_line_view.
                shape[1])
            # Add the percentage of nonempty pixels to the histogram.
            percentage_nonempty_pixels = 100. * fraction_nonempty_pixels
            hist_percentage_nonempty_pixels.append(percentage_nonempty_pixels)
            if (fraction_nonempty_pixels > min_fraction_nonempty_pixels):
                num_valid_lines += 1
                if (impainting):
                    # Inpaint the virtual camera image.
                    reds = rgb_image_from_line_view[:, :, 2]
                    greens = rgb_image_from_line_view[:, :, 1]
                    blues = rgb_image_from_line_view[:, :, 0]

                    mask = ((greens != 0) | (reds != 0) | (blues != 0)) * 1
                    mask = np.array(mask, dtype=np.uint8)
                    kernel = np.ones((5, 5), np.uint8)
                    dilated_mask = cv2.dilate(mask, kernel, iterations=1) - mask

                    rgb_image_from_line_view = cv2.inpaint(
                        rgb_image_from_line_view, dilated_mask, 10,
                        cv2.INPAINT_TELEA)
                    depth_image_from_line_view = cv2.inpaint(
                        depth_image_from_line_view, dilated_mask, 10,
                        cv2.INPAINT_TELEA)
                end_time_line = timer()
                average_time_per_line += (end_time_line - start_time_line)

                # Print images to file.
                cv2.imwrite(
                    os.path.join(
                        output_path, 'traj_{0}/frame_{1}/rgb/{2}.png'.format(
                            trajectory, frame_id, i)), rgb_image_from_line_view)
                cv2.imwrite(
                    os.path.join(output_path,
                                 'traj_{0}/frame_{1}/depth/{2}.png'.format(
                                     trajectory, frame_id, i)),
                    depth_image_from_line_view.astype(np.uint16))

        end_time = timer()
        if (num_valid_lines > 0):
            average_time_per_line /= num_valid_lines

        # Append the histogram for the frame to the output text file.
        with open('hist_percentages.txt', 'aw') as f:
            for item in hist_percentage_nonempty_pixels:
                f.write("%s\n" % item)

        print('Generated virtual camera images for frame {0}'.format(frame_id))
        print('Time elapsed: {:.3f} seconds'.format(end_time - start_time))
        print('Average time per line: {:.3f} seconds'.format(
            average_time_per_line))
        print('Number of lines discarded (minimum fraction of required ' +
              'nonempty pixels = {0:.3f}) is {1}'.format(
                  min_fraction_nonempty_pixels, lines_count - num_valid_lines))


def world_to_cam_transform(view_pose):
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


# InteriorNet
def get_virtual_camera_images_interiornet(scene_path, scene_type, trajectory, light_type, linesfiles_path, output_path):
    impainting = True
    show_line = False

    # TODO: Frame step.
    times, view_poses = parse_frames(scene_path, scene_type, trajectory)
    num_views = np.shape(view_poses)[0]

    # Photo paths are different for each dataset.
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

    # TODO: this has to be fixed in the future. Each scene has a different name.
    #path_to_lines_root = os.path.join(linesfiles_path,
    #                                  'traj_1/'.format(trajectory))
    path_to_lines_root = linesfiles_path

    print('Path to lines_root is {0}'.format(path_to_lines_root))

    # Camera from which the images were taken, SceneNetRGBD (pinhole) camera
    # model.
    real_camera = interiornet_utils.get_camera_model()

    # Distance between virtual camera origin and center of the line.
    distance = 0.25
    # Min fraction of nonempty pixels in the associated virtual-camera image for
    # a line to be considered as valid.
    min_fraction_nonempty_pixels = 0.0
    # Min number of pixel per unit meter of the line. The more the line points
    # toward or away from the camera, the less pixels per meter.
    min_pixel_per_meter = 50.
    # Min x resolution of the virtual camera image.
    min_image_width = 40

    # Write in a text file (that will store the histogram of percentages of
    # nonempty pixels) the distance of the virtual camera from the lines and the
    # threshold of nonempty pixels to discard lines.
    with open('hist_percentages.txt', 'w') as f:
        f.write("distance=%.3f\n" % distance)
        f.write("fraction_nonempty_pixels_threshold=%.3f\n" %
                min_fraction_nonempty_pixels)

    # Sliding window accumulator for point cloud, stored in world frame.
    pcl_world = np.zeros((0, 6))

    for frame_id in range(num_views):
        start_time = timer()
        photo_id = frame_id

        if scene_type < 7:
            photo_id = "{:019d}".format(int(times[photo_id]))

        rgb_image = cv2.imread(
            os.path.join(path_to_photos, '{0}.png'.format(photo_id)),
            cv2.IMREAD_COLOR)

        depth_image = cv2.imread(
            os.path.join(path_to_depth, '{0}.png'.format(photo_id)),
            cv2.IMREAD_UNCHANGED)

        # Retrieve camera to world transformation.
        cam_to_world_t = np.linalg.inv(world_to_cam_transform(view_poses[frame_id]))

        # Obtain coloured point cloud from RGB-D image.
        pcl_new = real_camera.rgbd_to_pcl(
            rgb_image=rgb_image, depth_image=depth_image, visualize_cloud=False)

        # Sliding window:
        sliding_window_length = 1
        if sliding_window_length > 1:
            # Add this point cloud (transformed to world frame) to the sliding window.
            pcl_world = np.vstack((cloud_utils.pcl_transform(
                pcl_new, cam_to_world_t), pcl_world))
            if np.shape(pcl_world)[0] > (np.shape(pcl_new)[0] * sliding_window_length):
                pcl_world = pcl_world[:np.shape(pcl_new)[0]*sliding_window_length, :]

            # Transform world point cloud back to camera frame.
            # And use this denser point cloud for virtual camera images.
            pcl = cloud_utils.pcl_transform(pcl_world, np.linalg.inv(cam_to_world_t))
            print(np.shape(pcl))
        else:
            pcl = pcl_new

        path_to_lines = os.path.join(
            path_to_lines_root, 'lines_with_labels_{0}.txt'.format(frame_id))

        try:
            data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
        except (IOError, pd.io.common.EmptyDataError):
            print("No line detected for frame {}".format(frame_id))
            continue

        data_lines = data_lines.values
        lines_count = data_lines.shape[0]

        average_time_per_line = 0
        # A line is valid if the corresponding virtual-camera image has enough
        # nonempty pixels.
        num_valid_lines = 0
        # Initialize the histogram of the percentages of nonempty pixels.
        hist_percentage_nonempty_pixels = []
        for i in range(lines_count):
            start_time_line = timer()
            # Obtain the pose of the virtual camera for each line.
            T, _ = virtual_camera_utils.virtual_camera_pose_from_file_line(
                data_lines[i, :], distance)
            # Draw the line in the virtual camera image.
            start_point = data_lines[i, :3]
            end_point = data_lines[i, 3:6]
            line_length = np.linalg.norm(end_point - start_point)

            start_point_2d = real_camera.project3dToPixel(start_point)
            end_point_2d = real_camera.project3dToPixel(end_point)
            line_length_pixels = np.linalg.norm(end_point_2d - start_point_2d)

            if show_line:
                line_3D = np.hstack([start_point, [0, 0, 255]])

                num_points_in_line = 200
                for idx in range(num_points_in_line):
                    line_3D = np.vstack([
                        line_3D,
                        np.hstack([(start_point + idx / float(num_points_in_line) *
                                    (end_point - start_point)), [0, 0, 255]])
                    ])
                # Transform the point cloud so as to make it appear as seen from
                # the virtual camera pose.
                pcl_from_line_view = cloud_utils.pcl_transform(
                    np.vstack([pcl, line_3D]), T)
                # Move line points directly to the image plane, so that it is never occluded.
                pcl_from_line_view[-1000:, 2] = 0
            else:
                pcl_from_line_view = cloud_utils.pcl_transform(pcl, T)
            # Obtain the RGB and depth virtual camera images by reprojecting the
            # point cloud on the image plane, under the view of the virtual
            # camera. Also obtain the number of nonempty pixels.
            (rgb_image_from_line_view, depth_image_from_line_view,
             num_nonempty_pixels) = cloud_utils.project_pcl_to_image_orthogonal(
                pcl_from_line_view,
                line_length * 1.5,
                line_length * 1, 0.5,
                int(line_length_pixels * 1.5),
                int(line_length_pixels))
            # cloud_utils.project_pcl_to_image(pcl_from_line_view, virtual_camera)
            # Discard lines that have less than the specified fraction of
            # nonempty pixels.
            fraction_nonempty_pixels = num_nonempty_pixels / float(
                rgb_image_from_line_view.shape[0] * rgb_image_from_line_view.
                shape[1])
            # Add the percentage of nonempty pixels to the histogram.
            percentage_nonempty_pixels = 100. * fraction_nonempty_pixels
            hist_percentage_nonempty_pixels.append(percentage_nonempty_pixels)
            if fraction_nonempty_pixels > min_fraction_nonempty_pixels and \
                    line_length_pixels * 1.5 > min_image_width and \
                    line_length_pixels / line_length > min_pixel_per_meter:
                num_valid_lines += 1
                if impainting:
                    # Inpaint the virtual camera image.
                    reds = rgb_image_from_line_view[:, :, 2]
                    greens = rgb_image_from_line_view[:, :, 1]
                    blues = rgb_image_from_line_view[:, :, 0]

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

                    rgb_image_from_line_view = cv2.inpaint(
                        rgb_image_from_line_view, dilated_mask, 10,
                        cv2.INPAINT_NS)
                    depth_image_from_line_view = cv2.inpaint(
                        depth_image_from_line_view, dilated_mask, 10,
                        cv2.INPAINT_NS)
                end_time_line = timer()
                average_time_per_line += (end_time_line - start_time_line)

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

        end_time = timer()
        if num_valid_lines > 0:
            average_time_per_line /= num_valid_lines

        # Append the histogram for the frame to the output text file.
        with open('hist_percentages.txt', 'aw') as f:
            for item in hist_percentage_nonempty_pixels:
                f.write("%s\n" % item)

        print('Generated virtual camera images for frame {0}'.format(frame_id))
        print('Number of valid lines is {}'.format(num_valid_lines))
        print('Time elapsed: {:.3f} seconds'.format(end_time - start_time))
        print('Average time per line total: {:.4f}'.format((end_time - start_time) / num_valid_lines))
        print('Average time per line: {:.3f} seconds'.format(
            average_time_per_line))
        print('Minimum fraction of required nonempty pixels is {}'.format(min_fraction_nonempty_pixels))
        print('Minimum number of pixels per meter is {}'.format(min_pixel_per_meter))
        print('Minimum virtual image width is {}'.format(min_image_width))
        print('Number of lines discarded is {}'.format(lines_count - num_valid_lines))


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
        "-scenenetscripts_path",
        help="Path to folder containing the scripts from pySceneNetRGBD, in "
        "particular scenenet_pb2.py (e.g. scenenetscripts_path="
        "'pySceneNetRGBD/'). Needed to extract the model of the virtual "
        "camera.")
    parser.add_argument(
        "-dataset_path",
        help="Path to folder containing the different image files from the "
        "dataset. For SceneNetRGBD datasets, the path should be such that "
        "concatenating the render path to it gives a folder with folders "
        "'depth', 'instances' and 'photo' inside (e.g. dataset_path="
        "'pySceneNetRGBD/data/train/'). For SceneNN datasets, the path should "
        "contain a subfolder XYZ for each scene (where XYZ is a three-digit ID "
        "associated to the scene, e.g. 005) and a subfolder 'intrinsic'.")
    parser.add_argument(
        "-dataset_name",
        help="If the data comes from the val or train_NUM dataset of "
        "SceneNetRGBD, either 'val' or 'train_NUM' (NUM is a number between 0 "
        "and 16). If the data comes from SceneNN, 'scenenn'. If the data comes from"
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
        # All stricly necessary arguments passed.
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
        print("WARNING: DEPRECATED. get_virtual_camera_images.py: Some arguments are missing. Using "
              "default ones in config_paths_and_variables.sh. In particular, "
              "please note that I am using dataset SceneNet.")
        # Obtain paths and variables.
        scenenetscripts_path = pathconfig.obtain_paths_and_variables(
            "SCENENET_SCRIPTS_PATH")
        linesandimagesfolder_path = pathconfig.obtain_paths_and_variables(
            "LINESANDIMAGESFOLDER_PATH")
        trajectory = pathconfig.obtain_paths_and_variables("TRAJ_NUM")
        dataset_name = pathconfig.obtain_paths_and_variables("DATASET_NAME")

    if (args.frame_step):
        frame_step = args.frame_step
    if (args.end_frame):
        end_frame = args.end_frame

    linesfiles_path = os.path.join(linesandimagesfolder_path,
                                   '{}_lines'.format(dataset_name))
    output_path = os.path.join(linesandimagesfolder_path, dataset_name)


    if (dataset_name in ["val"] + ["train_{}".format(i) for i in range(17)]):
        # Dataset from SceneNetRGBD.
        # Include the pySceneNetRGBD folder to the path and import its modules.
        sys.path.append(scenenetscripts_path)
        import scenenet_pb2 as sn

        # Find protobuf file associated to dataset_name.
        protobuf_path = get_protobuf_paths.get_protobuf_path(dataset_name)
        if protobuf_path is None:
            sys.exit('get_virtual_camera_images.py: Error in retrieving '
                     'protobuf_path.')
        if not args.dataset_path:
            scenenet_dataset_path = pathconfig.obtain_paths_and_variables(
                "SCENENET_DATASET_PATH")
            # Compose script arguments if necessary.
            dataset_path = os.path.join(scenenet_dataset_path, 'data/',
                                        dataset_name.split('_')[0])
        get_virtual_camera_images_scenenet_rgbd(trajectory, dataset_path,
                                                dataset_name, scenenetscripts_path)
    elif dataset_name == "scenenn":
        # Dataset from SceneNN.
        if not args.frame_step:
            sys.exit("It is required to indicate the frame_step when using "
                     "SceneNN dataset. Please use the argument -frame_step.")
        if not args.end_frame:
            sys.exit("It is required to indicate the index of the last frame "
                     "when using SceneNN dataset. Please use the argument "
                     "-end_frame.")
        if not args.dataset_path:
            scenenn_dataset_path = pathconfig.obtain_paths_and_variables(
                "SCENENN_DATASET_PATH")
            # Compose script arguments if necessary.
            dataset_path = os.path.join(scenenn_dataset_path, 'data/',
                                        dataset_name.split('_')[0])
        get_virtual_camera_images_scenenn(trajectory, dataset_path)
    elif dataset_name == "interiornet":
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
