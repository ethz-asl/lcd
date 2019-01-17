"""
Get virtual camera image for each line in trajectory with index given as input.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
import argparse
from timeit import default_timer as timer

from tools import cloud_utils
from tools import scenenn_utils
from tools import scenenet_utils
from tools import virtual_camera_utils
from tools import pathconfig
from tools import get_protobuf_paths


def get_virtual_camera_images_scenenet_rgbd(trajectory, dataset_path):
    impainting = False
    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('get_virtual_camera_images.py: Scenenet protobuf data not found '
              'at location:{0}. '.format(protobuf_path) + 'Please ensure you '
              'have copied the pb file to the data directory.')
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
    # Distance between virtual camera origin and line's center
    distance = 3
    for frame_id in range(300):
        start_time = timer()
        photo_id = frame_id * 25

        rgb_image = cv2.imread(
            os.path.join(path_to_photos, 'photo/{0}.jpg'.format(photo_id)),
            cv2.IMREAD_COLOR)

        depth_image = cv2.imread(
            os.path.join(path_to_photos, 'depth/{0}.png'.format(photo_id)),
            cv2.IMREAD_UNCHANGED)

        # Obtain coloured point cloud from RGB-D image.
        pcl = real_camera.rgbd_to_pcl(
            rgb_image=rgb_image, depth_image=depth_image, visualize_cloud=True)
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
            # camera.
            rgb_image_from_line_view, depth_image_from_line_view = \
                cloud_utils.project_pcl_to_image(pcl_from_line_view,
                                                 virtual_camera)
            if (impainting):
                # Inpaint the virtual camera image.
                reds = rgb_image_from_line_view[:, :, 2]
                greens = rgb_image_from_line_view[:, :, 1]
                blues = rgb_image_from_line_view[:, :, 0]

                mask = ((greens != 0) | (reds != 0) | (blues != 0)) * 1
                mask = np.array(mask, dtype=np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=1) - mask

                rgb_image_from_line_view = cv2.inpaint(rgb_image_from_line_view,
                                                       dilated_mask, 10,
                                                       cv2.INPAINT_TELEA)
                depth_image_from_line_view = cv2.inpaint(
                    depth_image_from_line_view, dilated_mask, 10,
                    cv2.INPAINT_TELEA)
            end_time_line = timer()
            average_time_per_line += ((
                end_time_line - start_time_line) / lines_count)

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

        print('Generated virtual camera images for frame {0}'.format(frame_id))
        print('Time elapsed: %.3f seconds' % (end_time - start_time))
        print('Average time per line: %.3f seconds' % average_time_per_line)


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

    # Distance between virtual camera origin and line's center.
    distance = 3
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
            rgb_image=rgb_image, depth_image=depth_image, visualize_cloud=True)
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
            # camera.
            rgb_image_from_line_view, depth_image_from_line_view = \
                cloud_utils.project_pcl_to_image(pcl_from_line_view,
                                                 virtual_camera)
            if (impainting):
                # Inpaint the virtual camera image.
                reds = rgb_image_from_line_view[:, :, 2]
                greens = rgb_image_from_line_view[:, :, 1]
                blues = rgb_image_from_line_view[:, :, 0]

                mask = ((greens != 0) | (reds != 0) | (blues != 0)) * 1
                mask = np.array(mask, dtype=np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=1) - mask

                rgb_image_from_line_view = cv2.inpaint(rgb_image_from_line_view,
                                                       dilated_mask, 10,
                                                       cv2.INPAINT_TELEA)
                depth_image_from_line_view = cv2.inpaint(
                    depth_image_from_line_view, dilated_mask, 10,
                    cv2.INPAINT_TELEA)
            end_time_line = timer()
            average_time_per_line += ((
                end_time_line - start_time_line) / lines_count)

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

        print('Generated virtual camera images for frame {0}'.format(frame_id))
        print('Time elapsed: %.3f seconds' % (end_time - start_time))
        print('Average time per line: %.3f seconds' % average_time_per_line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get virtual camera image for each line in the input '
        'trajectory.')
    parser.add_argument("-trajectory", type=str, help="Trajectory number.")
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
        "and 16). If the data comes from SceneNN, 'scenenn'.")
    parser.add_argument(
        "-linesandimagesfolder_path",
        help="Path to folder (e.g. 'data') containing text lines files (e.g. "
        "under 'data/train_lines') and that should store the output virtual "
        "camera images (e.g. under 'data/train_lines').")
    parser.add_argument(
        "-output_path",
        help="Data folder where to store the virtual camera images (e.g. "
        "'data/train').")

    args = parser.parse_args()
    if (args.trajectory and args.scenenetscripts_path and args.dataset_name and
            args.dataset_path and args.linesandimagesfolder_path):
        # All stricly necessary arguments passed.
        trajectory = args.trajectory
        scenenetscripts_path = args.scenenetscripts_path
        dataset_name = args.dataset_name
        dataset_path = args.dataset_path
        linesandimagesfolder_path = args.linesandimagesfolder_path
    else:
        print("get_virtual_camera_images.py: Some arguments are missing. Using "
              "default ones in config_paths_and_variables.sh. In particular, "
              "please note that I am using dataset SceneNetRGBD.")
        # Obtain paths and variables
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

    # Include the pySceneNetRGBD folder to the path and import its modules.
    sys.path.append(scenenetscripts_path)
    import scenenet_pb2 as sn

    if (dataset_name in ["val"] + ["train_{}".format(i) for i in range(17)]):
        # Dataset from SceneNetRGBD.
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
        get_virtual_camera_images_scenenet_rgbd(trajectory, dataset_path)
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
    else:
        raise ValueError("Invalid dataset name. Valid names are 'val', "
                         "'train_NUM', where NUM is a number between 0 and 16, "
                         "and 'scenenn'.")
