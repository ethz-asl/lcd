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


def get_virtual_camera_images():
    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(
            protobuf_path))
        print('Please ensure you have copied the pb file to the data directory')
    traj_renderpath = trajectories.trajectories[trajectory].render_path

    path_to_photos = os.path.join(dataset_path, traj_renderpath)

    print('Path to photos is {}'.format(path_to_photos))

    path_to_lines_root = os.path.join(linesfiles_path,
                                      'traj_{0}/'.format(trajectory))

    print('Path_to_lines_root is {0}'.format(path_to_lines_root))

    # Virtual camera model
    camera_model = scenenet_utils.get_camera_model()
    distance = 3
    # Distance between virtual camera origin and line's center
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
            pcl_from_line_view = scenenet_utils.pcl_transform(pcl, T)
            rgb_image_from_line_view, depth_image_from_line_view = \
                scenenet_utils.project_pcl_to_image(pcl_from_line_view,
                                                    camera_model)

            cv2.imwrite(
                os.path.join(
                    output_path, 'traj_{0}/frame_{1}/rgb/{2}.png'.format(
                        trajectory, frame_id, i)), rgb_image_from_line_view)
            cv2.imwrite(
                os.path.join(output_path,
                             'traj_{0}/frame_{1}/depth/{2}.png'.format(
                                 trajectory, frame_id, i)),
                depth_image_from_line_view.astype(np.uint16))

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
    parser.add_argument("-protobuf_path", help="Path to the protobuf file.")
    parser.add_argument(
        "-dataset_path",
        help="Path to folder containing the different image files from the "
        "dataset. The path should be such that concatenating the render path "
        "to it gives a folder with folders 'depth', 'instances' and 'photo' "
        "inside (e.g. dataset_path='pySceneNetRGBD/data/train/').")
    parser.add_argument(
        "-linesfiles_path",
        help="Path to folder containing text lines files (e.g. "
        "'data/train_lines').")
    parser.add_argument(
        "-output_path",
        help="Data folder where to store the virtual camera images (e.g. "
        "'data/train').")

    args = parser.parse_args()
    if (args.trajectory and args.scenenetscripts_path and args.protobuf_path and
            args.dataset_path and args.linesfiles_path and
            args.output_path):  # All arguments passed
        trajectory = int(args.trajectory)
        scenenetscripts_path = args.scenenetscripts_path
        protobuf_path = args.protobuf_path
        dataset_path = args.dataset_path
        linesfiles_path = args.linesfiles_path
        output_path = args.output_path
    else:
        print("Some arguments are missing. Using default ones in "
              "config_paths_and_variables.sh.")
        # Obtain paths and variables
        scenenet_dataset_path = pathconfig.obtain_paths_and_variables(
            "SCENENET_DATASET_PATH")
        scenenetscripts_path = pathconfig.obtain_paths_and_variables(
            "SCENENET_SCRIPTS_PATH")
        protobuf_path = pathconfig.obtain_paths_and_variables("PROTOBUF_PATH")
        outputdata_path = pathconfig.obtain_paths_and_variables(
            "OUTPUTDATA_PATH")
        trajectory = pathconfig.obtain_paths_and_variables("TRAJ_NUM")
        dataset_type = pathconfig.obtain_paths_and_variables("DATASET_TYPE")
        # Compose script arguments if necessary
        dataset_path = os.path.join(scenenet_dataset_path, 'data/',
                                    dataset_type)
        linesfiles_path = os.path.join(outputdata_path,
                                       '{}_lines'.format(dataset_type))
        output_path = os.path.join(outputdata_path, dataset_type)

    # Include the pySceneNetRGBD folder to the path and import its modules.
    sys.path.append(scenenetscripts_path)
    import scenenet_pb2 as sn

    get_virtual_camera_images()
