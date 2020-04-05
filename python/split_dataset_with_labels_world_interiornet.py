""" Split the lines detected in the input trajectory into train, val and test
    set and labels each line with its center point in the world frame and its
    instance label.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import tf

from tools import pathconfig
from tools import line_file_utils
from tools.camera_utils import SceneNetCameraToWorldMatrixRetriever


def transform_to_world_rot(vector, camera_rotation):
    return transform_to_world_off_rot(
        vector, np.zeros((3,)), camera_rotation)[0:3]


def transform_to_world_off_rot(vector, camera_origin, camera_rotation):
    x = camera_origin[0]
    y = camera_origin[1]
    z = camera_origin[2]
    e0 = camera_rotation[3]
    e1 = camera_rotation[0]
    e2 = camera_rotation[1]
    e3 = camera_rotation[2]
    T = np.array([[e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2, 2 * e1 * e2 - 2 * e0 * e3, 2 * e0 * e2 + 2 * e1 * e3, x],
                  [2 * e0 * e3 + 2 * e1 * e2, e0 ** 2 - e1 ** 2 + e2 ** 2 - e3 ** 2, 2 * e2 * e3 - 2 * e0 * e1, y],
                  [2 * e1 * e3 - 2 * e0 * e2, 2 * e0 * e1 + 2 * e2 * e3, e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2, z],
                  [0, 0, 0, 1]])
    return T.dot(np.append(vector, [1.]))[0:3]


def read_all_lines(line_path):
    # Count number of files in directory.
    file_count = len([name for name in os.listdir(line_path)
                      if os.path.isfile(os.path.join(line_path, name))])
    # The number of frames is file_count / 3 because for
    # each frame 3 files are written.
    frame_count = int(file_count / 3)
    data_lines = pd.read_csv(os.path.join(line_path, "lines_with_labels_0.txt")
                             , sep=" ", header=None, nrows=1)
    column_count = data_lines.values.shape[1]
    # Append frame_id before line data.
    lines = np.zeros((0, column_count))
    # Frame id of each line.
    frame_idx = np.zeros((0,))
    # Line id of each line in its frame (for the virtual camera images.
    line_idx = np.zeros((0,))

    for i in range(frame_count):
        try:
            path_to_line = os.path.join(line_path,
                                        "lines_with_labels_{}.txt".format(i))
            data_lines = pd.read_csv(path_to_line, sep=" ", header=None)
            data_lines = data_lines.values
            line_count = data_lines.shape[0]
        except pd.errors.EmptyDataError:
            data_lines = np.zeros((0, 37))
            line_count = 0

        lines = np.vstack((lines, data_lines))
        frame_idx = np.append(frame_idx, i * np.ones((line_count,)))
        line_idx = np.append(line_idx, np.arange(line_count))

    return lines, frame_count, frame_idx, line_idx


def read_lines_min_instance(line_path, min_lines_per_instance):
    lines, frame_count, frame_idx, line_idx = read_all_lines(line_path)
    instances = np.unique(lines[:, line_file_utils.label_index()])


def split_dataset(line_files_path, virtual_images_path, output_path):
    train = []
    val = []
    test = []

    train_prob = 3. / 5.
    val_prob = 1. / 5.
    test_prob = 1. - train_prob - val_prob

    lines, frame_count, frame_idx, line_idx = read_all_lines(line_files_path)
    line_count = lines.shape[0]
    # Shuffle lines for randomness.
    shuffled_idx = np.arange(line_count)
    np.random.shuffle(shuffled_idx)

    # Split lines into train, val and test set.
    train_start = 0
    train_end = train_start + int(line_count * train_prob)
    val_start = train_end
    val_end = val_start + int(line_count * val_prob)
    test_start = val_end
    test_end = line_count

    train = shuffled_idx[train_start:train_end]
    val = shuffled_idx[val_start:val_end]
    test = shuffled_idx[test_start:test_end]

    dataset = {'train': train.tolist(),
               'val': val.tolist(),
               'test': test.tolist(),
               'all_lines': list(range(line_count))}

    lines_found = 0

    for key, set_line_idx in dataset.iteritems():
        for i in set_line_idx:
            frame_id = int(frame_idx[i])
            line_id_in_frame = int(line_idx[i])

            path_to_virtual_imgs = os.path.join(virtual_images_path,
                                                "frame_{}".format(frame_id))
            path_to_write = os.path.join(path_to_virtual_imgs,
                                         'rgb/' + '{}.png'.format(line_id_in_frame))
            # If the virtual-camera image associated to the line does not
            # exist (because it does not contain enough nonempty pixels), do
            # not include the line in the dataset.
            if not os.path.isfile(path_to_write):
                print("Virtual-camera image not found for line {} ".format(
                    line_id_in_frame) + "in frame {}.".format(frame_id))
                continue

            line = lines[i, :]
            line_dict = line_file_utils.read_line_detection_line(line)

            start_point_camera = line_dict["start_point"]
            end_point_camera = line_dict["end_point"]
            line_type = line_dict["type"]
            label = line_dict["label"]
            class_id = line_dict["class"]
            normal_1_camera = line_dict["normal_1"]
            normal_2_camera = line_dict["normal_2"]
            start_open = line_dict["start_open"]
            end_open = line_dict["end_open"]
            camera_origin = line_dict["camera_origin"]
            camera_rotation = line_dict["camera_rotation"]

            line_start_point_world = transform_to_world_off_rot(
                np.transpose(start_point_camera), camera_origin, camera_rotation)
            line_end_point_world = transform_to_world_off_rot(
                np.transpose(end_point_camera), camera_origin, camera_rotation)
            normal_1_world = transform_to_world_rot(
                np.transpose(normal_1_camera),
                camera_rotation)
            normal_2_world = transform_to_world_rot(
                np.transpose(normal_2_camera),
                camera_rotation)

            line_dict.update({
                "image_path": path_to_write,
                "start_point": line_start_point_world,
                "end_point": line_end_point_world,
                "normal_1": normal_1_world,
                "normal_2": normal_2_world,
                'frame_id': frame_id,
            })

            # center_of_line = (
            #   line_start_point_world[:3] + line_end_point_world[:3]) / 2

            lines_found = lines_found + 1

            with open(
                    os.path.join(output_path,
                                 key + '_with_line_endpoints.txt'),
                    'a') as f:
                line_file_utils.write_line_split_line(f, line_dict)

    print("Found {} lines.".format(lines_found / 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split frames in the input trajectory in train, test and '
                    'validation set.')
    parser.add_argument("-line_files_path",
                        help="Path to folder containing the line files.")
    parser.add_argument(
        "-virtual_images_path",
        help="Path to folder containing the virtual images.")
    parser.add_argument(
        "-output_path",
        help="Path where to write the txt files with the splitting.")

    args = parser.parse_args()
    if args.line_files_path and args.virtual_images_path and args.output_path:
        # All arguments passed.
        line_files_path = args.line_files_path
        virtual_images_path = args.virtual_images_path
        output_path = args.output_path
    else:
        print("ERROR: Some arguments are missing. Exiting.")
        exit(1)

    split_dataset(line_files_path, virtual_images_path, output_path)
