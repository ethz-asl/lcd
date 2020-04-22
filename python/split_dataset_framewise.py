""" Split the lines detected in the input trajectory into train, val and test
    set and labels each line with its center point in the world frame and its
    instance label.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse

from tools import line_file_utils


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


def read_frame(path):
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        data_lines = data_lines.values
        line_count = data_lines.shape[0]
    except pd.errors.EmptyDataError:
        data_lines = np.zeros((0, 37))
        line_count = 0

    return line_count, data_lines


def write_frame(line_count, data_lines, path_to_vcis, frame_id, path_write):
    for i in range(line_count):
        path_to_vci = os.path.join(path_to_vcis, 'rgb/' + '{}.png'.format(i))
        # If the virtual-camera image associated to the line does not
        # exist (because it does not contain enough nonempty pixels), do
        # not include the line in the dataset.
        if not os.path.isfile(path_to_vci):
            print("Virtual-camera image not found for line {} ".format(i) + "in frame {}.".format(frame_id))
            # print("Path is {}".format(path_to_vci))
            continue

        line = data_lines[i, :]
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
            "image_path": path_to_vci,
            # "start_point": line_start_point_world,
            # "end_point": line_end_point_world,
            # "normal_1": normal_1_world,
            # "normal_2": normal_2_world,
            'frame_id': frame_id,
        })

        with open(path_write, 'a') as f:
            line_file_utils.write_line_split_line(f, line_dict)


def mean_shift_split(data_lines, label_index):
    for i in range(len(data_lines)):
        # Frame id, line id in frame.
        data_lines[i] = np.hstack((data_lines[i], np.ones((data_lines[i].shape[0], 1)) * i,
                                   np.arange(data_lines[i].shape[0]).reshape(data_lines[i].shape[0], 1)))

    radius = 1.7

    data_lines = np.vstack(data_lines)
    label_indices = np.unique(data_lines[:, label_index])

    # Transform to world first.
    for i in range(data_lines.shape[0]):
        data_lines[i, 0:3] = transform_to_world_off_rot(np.transpose(data_lines[i, 0:3]),
                                                        data_lines[i, 31:34], data_lines[i, 34:38])
        data_lines[i, 3:6] = transform_to_world_off_rot(np.transpose(data_lines[i, 3:6]),
                                                        data_lines[i, 31:34], data_lines[i, 34:38])

    max_label = max(label_indices)
    if max_label == 65535:
        label_indices.sort()
        max_label = label_indices[-2]
    changes = []

    for label in label_indices:
        lines = data_lines[np.where(data_lines[:, label_index] == label)[0], :]

        # Perform mean shift iteratively until no line is left.
        iteration = 0
        while lines.shape[0] > 0:
            mean = (lines[0, 0:3] + lines[0, 3:6]) / 2.

            # Find new mean iteratively.
            while True:
                new_mean = np.zeros((3,))
                weight_sum = 0.
                in_radius = []
                for i in range(lines.shape[0]):
                    start = lines[i, 0:3]
                    end = lines[i, 3:6]
                    d_start = np.linalg.norm(start - mean)
                    d_end = np.linalg.norm(end - mean)
                    #if not np.isnan(min(d_start, d_end)):
                    #    print(min(d_start, d_end))
                    #    print(label)
                    if min(d_start, d_end) < radius:
                        length = np.linalg.norm(start - end)
                        # Weight the line start and end point with the length of the line.
                        new_mean = new_mean + (start + end) * length / 2.
                        weight_sum = weight_sum + length
                        in_radius.append(i)

                new_mean = new_mean / weight_sum
                if np.equal(new_mean, mean).all():
                    break

                mean = new_mean

            # If first iteration, no changes to labels.
            # After that, change label to the biggest available
            if iteration > 0:
                changes.append([lines[in_radius, -2:], max_label])
                print("Added one instance split at instance {}.".format(int(max_label)))
                max_label = max_label + 1

            lines = np.delete(lines, in_radius, axis=0)
            iteration = iteration + 1

    return changes


def split_scene(line_path, vi_path, output_path):
    # Count number of files in directory.
    frames = [name for name in os.listdir(line_path)
             if os.path.isfile(os.path.join(line_path, name)) and 'lines_with_labels' in name]
    frame_count = len(frames)

    train_prob = 4. / 5.

    train_indices = list(range(int(frame_count * train_prob)))
    val_indices = list(range(int(frame_count * train_prob), frame_count))

    line_counts = []
    data_lines = []
    for i in range(frame_count):
        frame_path = os.path.join(line_path, "lines_with_labels_{}.txt".format(i))
        line_count, frame_lines = read_frame(frame_path)
        line_counts.append(line_count)
        data_lines.append(frame_lines)

    # The index of the label column in the line data is 21.
    label_index = 21
    changes = mean_shift_split(data_lines, label_index)
    for change in changes:
        for i in range(change[0].shape[0]):
            frame_id = int(change[0][i, 0])
            line_id = int(change[0][i, 1])
            data_lines[frame_id][line_id, label_index] = change[1]

    for i in range(frame_count):
        if i in train_indices:
            key = 'train'
            out_index = train_indices.index(i)
        else:
            key = 'val'
            out_index = val_indices.index(i)

        path_to_vcis = os.path.join(vi_path, "frame_{}".format(i))
        path_to_output = os.path.join(output_path, key, "frame_{}.txt".format(out_index))

        write_frame(line_counts[i], data_lines[i], path_to_vcis, i, path_to_output)


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

    split_scene(line_files_path, virtual_images_path, output_path)
