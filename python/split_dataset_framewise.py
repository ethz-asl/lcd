""" Split the lines detected in the input trajectory into train, val and test
    set and labels each line with its center point in the world frame and its
    instance label.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse

from collections import Counter
# from numba import njit
from tools import line_file_utils


def get_transform_to_world(camera_origin, camera_rotation):
    """
    Returns the 4x4 transformation matrix from the camera origin and camera rotation.
    :param cam_origin: Cartesian origin of the camera. Numpy array of shape (3, 1)
    :param cam_rotation: Quaternion of the rotation of the camera. Numpy array of shape (4, 1)
    :return: The 4x4 transformation matrix of the camera.
    """
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
    return T


def transform_to_world_off_rot(vector, camera_origin, camera_rotation):
    """
    Transforms a vector in the camera frame into the world frame.
    :param vector: The vector to be transformed into the world frame. Numpy array of shape(3, 1)
    :param camera_origin: Cartesian origin of the camera. Numpy array of shape (3, 1)
    :param camera_rotation: Quaternion of the rotation of the camera. Numpy array of shape (4, 1)
    :return: The transformed vector.
    """
    T = get_transform_to_world(camera_origin, camera_rotation)
    return T.dot(np.vstack([vector.T, np.ones((vector.shape[0], 1))]))[0:3, :].T


def read_frame(path):
    """
    Reads the frame lines from the given path. How the lines are saved can be viewed in tools/line_file_utils.py
    :param path: The path with the text file containing the lines output by the ROS node.
    :return: line_count: The number of lines.
             data_lines: The line data.
    """
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        data_lines = data_lines.values
        line_count = data_lines.shape[0]
    except pd.errors.EmptyDataError:
        data_lines = np.zeros((0, 38))
        line_count = 0

    return line_count, data_lines


def get_valid_lines_and_vcis(data_lines, frame_id, path_to_vcis):
    """
    Extract the lines that have a valid virtual camera image. Lines do not have virtual camera images if they
    are invalid (pixel density to low, etc.). See generate_virtual_camera_images.py for details.
    All lines with no virtual camera image are discarded.
    :param data_lines: The line data as read from the text file (read_frame(path)) in the form of a numpy array with
                       shape (N, 38).
    :param frame_id: The frame id.
    :param path_to_vcis: Path to the virtual camera images of this frame.
    :return: data_lines: The new lines data without the invalid lines in the form of a numpy array with shape
                         (N_new, 38).
             vci_paths: The paths to the virtual camera image to each line in the form of a list with length N_new.
    """
    lines_to_delete = []
    vci_paths = []

    for i in range(data_lines.shape[0]):
        path_to_vci = os.path.join(path_to_vcis, 'rgb/' + '{}.png'.format(i))
        # If the virtual-camera image associated to the line does not
        # exist (because it does not contain enough nonempty pixels), do
        # not include the line in the dataset.
        if not os.path.isfile(path_to_vci):
            print("Virtual-camera image not found for line {} ".format(i) + "in frame {}.".format(frame_id))
            lines_to_delete.append(i)
            continue

        vci_paths.append(path_to_vci)

    return np.delete(data_lines, lines_to_delete, axis=0), vci_paths


def write_frame(data_lines, vci_paths, frame_id, path_write):
    """
    Writes the data for all lines of a frame into a line file used later for training.
    The data includes the virtual camera image path and the geometric data of the lines.
    :param data_lines: The data of each line in the form of a numpy array with shape (N, 38)
    :param vci_paths: The paths to the virtual camera image to each line in the form of a list with length N.
    :param frame_id: The id of the frame.
    :param path_write: The file path were the line data should be written to.
    """
    if os.path.exists(path_write):
        print("File already exists: {}".format(path_write))
        print("Skipping.")
        return

    print(data_lines.shape)
    for i in range(data_lines.shape[0]):
        path_to_vci = vci_paths[i]

        line = data_lines[i, :]
        line_dict = line_file_utils.read_line_detection_line(line)

        line_dict.update({
            "image_path": path_to_vci,
            'frame_id': frame_id,
        })

        with open(path_write, 'a') as f:
            line_file_utils.write_line_split_line(f, line_dict)


def mean_shift_split(data_lines, label_index):
    """
    Performs a mean shift split of instances that are too large. This is only necessary because of a bug in the
    InteriorNet dataset, where instances of the same class are often labelled as the same instance. This would
    unquestionably affect the clustering performance. Hopefully this gets resolved soon.
    :param data_lines: The geometric data of the lines in the form of a numpy array with shape (N, 38)
    :param label_index: The column index where the instance label is located in the data_lines array.
    :return: The changes to be applied to the data in the form of list of tuples (frame_id, line_id, new_instance_label)
    """
    # Add the frame id and line id to the line information.
    for i in range(len(data_lines)):
        # Frame id, line id in frame.
        data_lines[i] = np.hstack((data_lines[i], np.ones((data_lines[i].shape[0], 1)) * i,
                                   np.arange(data_lines[i].shape[0]).reshape(data_lines[i].shape[0], 1)))

    # The maximum radius of an instance. If an instance is larger, it will be split into more instances.
    radius = 1.7
    # The maximum mean shift operations until the algorithm is terminated.
    max_iterations = 20

    # Stack all lines from all frames.
    data_lines = np.vstack(data_lines)
    # Determine the unique instances present in the scene.
    label_indices = np.unique(data_lines[:, label_index])

    # Transform to world first.
    for i in range(data_lines.shape[0]):
        data_lines[i, 0:3] = transform_to_world_off_rot(data_lines[i, 0:3].reshape((1, 3)),
                                                        data_lines[i, 31:34], data_lines[i, 34:38])
        data_lines[i, 3:6] = transform_to_world_off_rot(data_lines[i, 3:6].reshape((1, 3)),
                                                        data_lines[i, 31:34], data_lines[i, 34:38])

    # Find the next available instance label.
    max_label = max(label_indices) + 1
    if max_label == 65535:
        label_indices.sort()
        max_label = label_indices[-2] + 1
    changes = []

    for label in label_indices:
        lines = data_lines[np.where(data_lines[:, label_index] == label)[0], :]

        # Perform mean shift iteratively until no line is left.
        iteration = 0
        while lines.shape[0] > 0:
            mean = (lines[0, 0:3] + lines[0, 3:6]) / 2.
            in_radius = []

            # Find new mean iteratively.
            mean_shift_iter = 0
            while mean_shift_iter < max_iterations:
                new_mean = np.zeros((3,))
                weight_sum = 0.
                for i in range(lines.shape[0]):
                    start = lines[i, 0:3]
                    end = lines[i, 3:6]
                    d_start = np.linalg.norm(start - mean)
                    d_end = np.linalg.norm(end - mean)
                    if min(d_start, d_end) < radius:
                        length = np.linalg.norm(start - end)
                        # Weight the line start and end point with the length of the line.
                        new_mean = new_mean + (start + end) * length / 2.
                        weight_sum = weight_sum + length
                        in_radius.append(i)

                if len(in_radius) is 0:
                    new_mean = (lines[0, 0:3] + lines[0, 3:6]) / 2.
                    in_radius.append(0)

                new_mean = new_mean / weight_sum
                if np.equal(new_mean, mean).all():
                    break

                mean = new_mean

                mean_shift_iter = mean_shift_iter + 1

            # If first iteration, no changes to labels.
            # After that, change label to the next one available (one more than the maximum label).
            if iteration > 0:
                changes.append([lines[in_radius, -2:], max_label])
                print("Added one instance split at instance {}.".format(int(max_label)))
                max_label = max_label + 1

            lines = np.delete(lines, in_radius, axis=0)
            iteration = iteration + 1

    return changes


def split_scene(line_path, vi_path, output_path):
    """
    Reads all lines from all frames in a scene, performs mean shift split of the instances and write the lines into
    files for each frame. The virtual camera image path is added to each line.
    :param line_path: The path where the frame-wise line files are located, as output by the ROS node.
    :param vi_path: The path where the virtual camera images of the scene are located.
    :param output_path: The path where the processed line data should be written to.
    """
    if "{" not in output_path:
        print("Error, output_path needs to contain brackets for frame id formatting.")
        exit(1)

    # Count number of files in directory.
    frames = [name for name in os.listdir(line_path)
              if os.path.isfile(os.path.join(line_path, name)) and 'lines_with_labels' in name]
    frame_count = len(frames)
    frames.sort()

    line_counts = []
    data_lines = []
    # Read all the frames as output by the ROS node.
    for i, frame_path in enumerate(frames):
        frame_path = os.path.join(line_path, "lines_with_labels_{}.txt".format(i))
        line_count, frame_lines = read_frame(frame_path)
        line_counts.append(line_count)
        data_lines.append(frame_lines)

    if len(data_lines) == 0:
        path_to_output = os.path.join(output_path.format('ERROR'))
        print("ERROR: NO DATA FOUND.")
        with open(path_to_output) as f:
            f.write("ERROR: NO LINE DATA")
        return

    # The index of the label column in the line data is 21.
    label_index = 21
    # Perform mean shift split of large instances.
    changes = mean_shift_split(data_lines, label_index)
    for change in changes:
        for i in range(change[0].shape[0]):
            frame_id = int(change[0][i, 0])
            line_id = int(change[0][i, 1])
            data_lines[frame_id][line_id, label_index] = change[1]

    # Append the vci_path to the line data.
    vci_path_lists = []
    for i in range(len(data_lines)):
        path_to_vcis = os.path.join(vi_path, "frame_{}".format(i))
        data_lines[i], vci_paths = get_valid_lines_and_vcis(data_lines[i], i, path_to_vcis)
        assert(data_lines[i].shape[0] == len(vci_paths))
        vci_path_lists.append(vci_paths)

    # Write the frames to their corresponding files for use during training.
    for i in range(frame_count):
        path_to_output = os.path.join(output_path.format(i))
        write_frame(data_lines[i], vci_path_lists[i], i, path_to_output)


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
