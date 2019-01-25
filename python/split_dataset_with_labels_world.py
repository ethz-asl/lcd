""" Splits the lines detected in the input trajectory into train, val and test
    set and labels each line with its center point in the world frame and its
    instance label.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse

from tools import pathconfig
from tools import scenenet_utils
from tools import get_protobuf_paths
from tools.camera_utils import SceneNetCameraToWorldMatrixRetriever


def split_dataset():
    train = []
    val = []
    test = []

    idx_for_split = 0
    for frame_id in range(start_frame, end_frame + 1, frame_step):
        idx_for_split += 1
        if idx_for_split % 5 == 1:
            val.append(frame_id)
            continue

        if idx_for_split % 5 == 3:
            test.append(frame_id)
            continue

        train.append(frame_id)

    dataset = {}
    dataset['train'] = train
    dataset['val'] = val
    dataset['test'] = test
    dataset['all_lines'] = [
        i for i in range(start_frame, end_frame + 1, frame_step)
    ]

    if (dataset_name in ["val"] + ["train_{}".format(i) for i in range(17)]):
        camera_to_world_matrix_retriever = SceneNetCameraToWorldMatrixRetriever(
            trajectory=trajectory,
            dataset_name=dataset_name,
            scenenetscripts_path=scenenetscripts_path)
    elif dataset_name == "scenenn":
        camera_to_world_matrix_retriever = SceneNNCameraToWorldMatrixRetriever(
            trajectory=trajectory, dataset_path=dataset_path)

    for key, frames in dataset.iteritems():
        for frame_id in frames:
            path_to_lines = os.path.join(
                path_to_linesfiles,
                'lines_with_labels_{0}.txt'.format(frame_id))
            try:
                data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
                data_lines = data_lines.values
                lines_count = data_lines.shape[0]
            except pd.errors.EmptyDataError:
                lines_count = 0
            # Retrieve camera-to-world matrix.
            camera_to_world_matrix = \
                camera_to_world_matrix_retriever.get_camera_to_world_matrix(
                    frame_id)

            for i in range(lines_count):
                path_to_write = path_to_virtualcameraimages + \
                    'frame_{0}/'.format(frame_id) + 'rgb/' + '{0}.png'.format(i)
                # If the virtual-camera image associated to the line does not
                # exist (because it does not contain enough nonempty pixels), do
                # not include the line in the dataset.
                if (not os.path.isfile(path_to_write)):
                    print("Virtual-camera image not found for line {} ".format(
                        i) + "in frame {}.".format(frame_id))
                    continue

                line = data_lines[i]
                line_start_point_camera = np.append(line[:3], [1])
                line_end_point_camera = np.append(line[3:6], [1])

                line_start_point_world = camera_to_world_matrix.dot(
                    line_start_point_camera)
                line_end_point_world = camera_to_world_matrix.dot(
                    line_end_point_camera)

                center_of_line = (
                    line_start_point_world[:3] + line_end_point_world[:3]) / 2

                # According to line_ros_utility::printToFile line_type is:
                # * 0 for discontinuity lines,
                # * 1 for planar lines,
                # * 2 for edge lines,
                # * 3 for intersection lines.
                line_type = int(line[-2])

                label = int(line[-1])

                with open(os.path.join(output_path, key + '.txt'), 'a') as f:
                    f.write(
                        os.path.abspath(path_to_write) + ' ' +
                        str(center_of_line[0]) + ' ' + str(center_of_line[1]) +
                        ' ' + str(center_of_line[2]) + ' ' + str(line_type) +
                        ' ' + str(label) + '\n')

                with open(
                        os.path.join(output_path,
                                     key + '_with_line_endpoints.txt'),
                        'a') as f:
                    f.write(
                        os.path.abspath(path_to_write) + ' ' +
                        str(line_start_point_world[0]) + ' ' + str(
                            line_start_point_world[1]) + ' ' + str(
                                line_start_point_world[2]) + ' ' + str(
                                    line_end_point_world[0]) + ' ' +
                        str(line_end_point_world[1]) + ' ' + str(
                            line_end_point_world[2]) + ' ' + str(line_type) +
                        ' ' + str(label) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split frames in the input trajectory in train, test and '
        'validation set.')
    parser.add_argument("-trajectory", help="Trajectory number.")
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
        help="Path to the scripts from pySceneNetRGBD (e.g. "
        "'../pySceneNetRGBD/').")
    parser.add_argument(
        "-dataset_name",
        help="Either train or val, indicating whether "
        "the data being pickled comes from the train or val dataset of "
        "pySceneNetRGBD.")
    parser.add_argument(
        "-linesandimagesfolder_path",
        help="Path to folder (e.g. 'data') containing text lines files (e.g. "
        "under 'data/train_lines') as well as virtual camera images (e.g. "
        "under 'data/train_lines').")
    parser.add_argument(
        "-output_path",
        help="Path where to write the txt files with the splitting.")
    parser.add_argument(
        "-dataset_path",
        help="Path to folder containing the different image files from the "
        "dataset. It is only needed for the SceneNN datasets, for which the "
        "path should contain a subfolder XYZ for each scene (where XYZ is a "
        "three-digit ID associated to the scene, e.g. 005) and a subfolder "
        "'intrinsic'.")

    args = parser.parse_args()
    if (args.trajectory and args.scenenetscripts_path and args.dataset_name and
            args.linesandimagesfolder_path and args.output_path):
        # All stricly arguments passed.
        trajectory = int(args.trajectory)
        scenenetscripts_path = args.scenenetscripts_path
        dataset_name = args.dataset_name
        linesandimagesfolder_path = args.linesandimagesfolder_path
        output_path = args.output_path
    else:
        print("split_dataset_with_labels_world.py: Some arguments are missing. "
              "Using default ones in config_paths_and_variables.sh.")
        # Obtain paths and variables
        scenenetscripts_path = pathconfig.obtain_paths_and_variables(
            "SCENENET_SCRIPTS_PATH")
        linesandimagesfolder_path = pathconfig.obtain_paths_and_variables(
            "LINESANDIMAGESFOLDER_PATH")
        trajectory = pathconfig.obtain_paths_and_variables("TRAJ_NUM")
        dataset_name = pathconfig.obtain_paths_and_variables("DATASET_NAME")
        output_path = os.path.join(linesandimagesfolder_path)

    if (dataset_name in ["val"] + ["train_{}".format(i) for i in range(17)]):
        # Dataset from SceneNetRGBD.
        start_frame = 0
        if not args.frame_step:
            frame_step = 1
        else:
            frame_step = args.frame_step
        if not args.end_frame:
            end_frame = 299
        else:
            end_frame = args.end_frame
    elif dataset_name == "scenenn":
        # Dataset from SceneNN.
        start_frame = 2
        if args.frame_step:
            frame_step = args.frame_step
        else:
            sys.exit("It is required to indicate the frame_step when using "
                     "SceneNN dataset. Please use the argument -frame_step.")
        if args.end_frame:
            end_frame = args.end_frame
        else:
            sys.exit("It is required to indicate the index of the last frame "
                     "when using SceneNN dataset. Please use the argument "
                     "-end_frame.")
        if args.dataset_path:
            dataset_path = args.dataset_path
        else:
            sys.exit("It is required to indicate the path of the dataset when "
                     "using SceneNN dataset. Please use the argument "
                     "-dataset_path.")

    # Compose auxiliary paths
    path_to_linesfiles = os.path.join(linesandimagesfolder_path,
                                      '{0}_lines/traj_{1}/'.format(
                                          dataset_name, trajectory))
    path_to_virtualcameraimages = os.path.join(linesandimagesfolder_path,
                                               '{0}/traj_{1}/'.format(
                                                   dataset_name, trajectory))

    split_dataset()
