"""
Get virtual camera image for each line in trajectory with index given as input.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
import argparse

from tools import scenenet_utils
from tools import pathconfig
from tools import get_protobuf_paths


def get_virtual_camera_images():
    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('get_virtual_camera_images.py: Scenenet protobuf data not found '
              'at location:{0}'.format(protobuf_path))
        print('Please ensure you have copied the pb file to the data directory')
    traj_renderpath = trajectories.trajectories[trajectory].render_path

    path_to_photos = os.path.join(dataset_path, traj_renderpath)

    print('Path to photos is {}'.format(path_to_photos))

    path_to_lines_root = os.path.join(linesfiles_path,
                                      'traj_{0}/'.format(trajectory))

    print('Path_to_lines_root is {0}'.format(path_to_lines_root))

    # Virtual camera model
    camera_model = scenenet_utils.get_camera_model()
    # Distance between virtual camera origin and line's center
    distance = 1
    for frame_id in range(300):
        photo_id = frame_id * 25

        rgb_image = cv2.imread(
            os.path.join(path_to_photos, 'photo/{0}.jpg'.format(photo_id)),
            cv2.IMREAD_COLOR)

        depth_image = cv2.imread(
            os.path.join(path_to_photos, 'depth/{0}.png'.format(photo_id)),
            cv2.IMREAD_UNCHANGED)

        pcl = scenenet_utils.rgbd_to_pcl(rgb_image, depth_image, camera_model)
        path_to_lines = os.path.join(
            path_to_lines_root, 'lines_with_labels_{0}.txt'.format(frame_id))

        try:
            data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
        except (IOError, pd.io.common.EmptyDataError):
            print("No line detected for frame {}".format(frame_id))
            continue

        data_lines = data_lines.values
        lines_count = data_lines.shape[0]

        for i in range(lines_count):
            T, _ = scenenet_utils.virtual_camera_pose(data_lines[i, :],
                                                      distance)
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
            pcl_from_line_view = scenenet_utils.pcl_transform(
                np.vstack([pcl, line_3D]), T)
            rgb_image_from_line_view, depth_image_from_line_view = \
                scenenet_utils.project_pcl_to_image(pcl_from_line_view,
                                                    camera_model)
            # Inpaint the virtual camera image.
            reds = rgb_image_from_line_view[:, :, 2]
            greens = rgb_image_from_line_view[:, :, 1]
            blues = rgb_image_from_line_view[:, :, 0]

            mask = ((greens != 0) | (reds != 0) | (blues != 0)) * 1
            mask = np.array(mask, dtype=np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1) - mask

            dst_rgb = cv2.inpaint(rgb_image_from_line_view, dilated_mask, 10,
                                  cv2.INPAINT_TELEA)
            dst_depth = cv2.inpaint(depth_image_from_line_view, dilated_mask,
                                    10, cv2.INPAINT_TELEA)
            # Print images to file.
            cv2.imwrite(
                os.path.join(output_path,
                             'traj_{0}/frame_{1}/rgb/{2}.png'.format(
                                 trajectory, frame_id, i)), dst_rgb)
            cv2.imwrite(
                os.path.join(output_path,
                             'traj_{0}/frame_{1}/depth/{2}.png'.format(
                                 trajectory, frame_id, i)),
                dst_depth.astype(np.uint16))

        print('Generated virtual camera images for frame {0}'.format(frame_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get virtual camera image for each line in the input '
        'trajectory.')
    parser.add_argument("-trajectory", help="Trajectory number.")
    parser.add_argument(
        "-scenenetscripts_path",
        help="Path to folder containing the scripts from pySceneNetRGBD, in "
        "particular scenenet_pb2.py (e.g. scenenetscripts_path="
        "'pySceneNetRGBD/').")
    parser.add_argument(
        "-dataset_path",
        help="Path to folder containing the different image files from the "
        "dataset. The path should be such that concatenating the render path "
        "to it gives a folder with folders 'depth', 'instances' and 'photo' "
        "inside (e.g. dataset_path='pySceneNetRGBD/data/train/').")
    parser.add_argument(
        "-dataset_name",
        help="Either train or val, indicating whether "
        "the data being pickled comes from the train or val dataset of "
        "pySceneNetRGBD.")
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
            args.dataset_path and
            args.linesandimagesfolder_path):  # All arguments passed
        trajectory = int(args.trajectory)
        scenenetscripts_path = args.scenenetscripts_path
        dataset_name = args.dataset_name
        dataset_path = args.dataset_path
        linesandimagesfolder_path = args.linesandimagesfolder_path
    else:
        print("get_virtual_camera_images.py: Some arguments are missing. Using "
              "default ones in config_paths_and_variables.sh. In particular, "
              "please note that I am using dataset pySceneNetRGBD.")
        # Obtain paths and variables
        scenenet_dataset_path = pathconfig.obtain_paths_and_variables(
            "SCENENET_DATASET_PATH")
        scenenetscripts_path = pathconfig.obtain_paths_and_variables(
            "SCENENET_SCRIPTS_PATH")
        linesandimagesfolder_path = pathconfig.obtain_paths_and_variables(
            "LINESANDIMAGESFOLDER_PATH")
        trajectory = pathconfig.obtain_paths_and_variables("TRAJ_NUM")
        dataset_name = pathconfig.obtain_paths_and_variables("DATASET_NAME")
        # Compose script arguments if necessary
        dataset_path = os.path.join(scenenet_dataset_path, 'data/',
                                    dataset_name)

    linesfiles_path = os.path.join(linesandimagesfolder_path,
                                   '{}_lines'.format(dataset_name))
    output_path = os.path.join(linesandimagesfolder_path, dataset_name)
    # Find protobuf file associated to dataset_name
    protobuf_path = get_protobuf_paths.get_protobuf_path(dataset_name)
    if protobuf_path is None:
        sys.exit('get_virtual_camera_images.py: Error in retrieving '
                 'protobuf_path.')
    # Include the pySceneNetRGBD folder to the path and import its modules.
    sys.path.append(scenenetscripts_path)
    import scenenet_pb2 as sn

    get_virtual_camera_images()
