"""
Split detected lines of the input trajectory to train, val and test set and label each line with its center in the world frame and the instance label.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse

from tools import pathconfig
from tools import scenenet_utils
from camera_pose_and_intrinsics_example import camera_to_world_with_pose, interpolate_poses


def split_dataset():
    sys.path.append(scenenetscripts_path)
    import scenenet_pb2 as sn

    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(
            protobuf_path))
        print('Please ensure you have copied the pb file to the datadirectory')

    frames_total = 300
    train = []
    val = []
    test = []

    for frame_id in range(frames_total):
        if frame_id % 5 == 1:
            val.append(frame_id)
            continue

        if frame_id % 5 == 3:
            test.append(frame_id)
            continue

        train.append(frame_id)

    dataset = {}
    dataset['train'] = train
    dataset['val'] = val
    dataset['test'] = test
    dataset['all_lines'] = [i for i in range(frames_total)]

    for key, frames in dataset.iteritems():
        for frame_id in frames:
            path_to_lines = os.path.join(
                path_to_linesfiles,
                'lines_with_labels_{0}.txt'.format(frame_id))

            #print('Path_to_lines is {0}'.format(path_to_lines))
            try:
                data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
                data_lines = data_lines.values
                lines_count = data_lines.shape[0]
            except pd.errors.EmptyDataError:
                lines_count = 0

            view = trajectories.trajectories[trajectory].views[frame_id]
            ground_truth_pose = interpolate_poses(view.shutter_open,
                                                  view.shutter_close, 0.5)
            camera_to_world_matrix = camera_to_world_with_pose(
                ground_truth_pose)

            for i in range(lines_count):
                path_to_write = path_to_virtualcameraimages + \
                    'frame_{0}/'.format(frame_id) + 'rgb/' + '{0}.png'.format(i)

                line = data_lines[i]
                line_start_point_camera = np.append(line[:3], [1])
                line_end_point_camera = np.append(line[3:6], [1])

                line_start_point_world = camera_to_world_matrix.dot(
                    line_start_point_camera)
                line_end_point_world = camera_to_world_matrix.dot(
                    line_end_point_camera)

                center_of_line = (
                    line_start_point_world[:3] + line_end_point_world[:3]) / 2

                label = int(line[-1])

                with open(os.path.join(output_path, key + '.txt'), 'a') as f:
                    f.write(
                        os.path.abspath(path_to_write) + ' ' +
                        str(center_of_line[0]) + ' ' + str(center_of_line[1]) +
                        ' ' + str(center_of_line[2]) + ' ' + str(label) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split frames in the input trajectory in train, test and '
        'validation set.')
    parser.add_argument("-trajectory", help="Trajectory number.")
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

    args = parser.parse_args()
    if (args.trajectory and args.scenenetscripts_path and args.dataset_name and
            args.linesandimagesfolder_path and args.output_path):
        # All arguments passed
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
    # Find protobuf file associated to dataset_name
    protobuf_path = get_protobuf_paths.get_protobuf_path(dataset_name)
    if protobuf_path is None:
        sys.exit('split_dataset_with_labels_world.py: Error in retrieving '
                 'protobuf_path.')
    # Compose auxiliary paths
    path_to_linesfiles = os.path.join(linesandimagesfolder_path,
                                      '{0}_lines/traj_{1}/'.format(
                                          dataset_name, trajectory))
    path_to_virtualcameraimages = os.path.join(linesandimagesfolder_path,
                                               '{0}/traj_{1}/'.format(
                                                   dataset_name, trajectory))

    split_dataset()
