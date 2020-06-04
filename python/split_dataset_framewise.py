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


def group_lines(lines_1, lines_2):
    groups = [[{j}, set()] for j in range(lines_1.shape[0])] + \
        [[set(), {i}] for i in range(lines_2.shape[0])]

    for i in range(lines_2.shape[0]):
        line = lines_2[i, :]
        for j in range(lines_1.shape[0]):
            if lines_coincide(lines_1[j, :], line):
                grouped = []
                for k, pair in enumerate(groups):
                    if i in pair[1] or j in pair[0]:
                        grouped.append(k)

                new_group = [{j}, {i}]
                for g in grouped:
                    new_group[0] = new_group[0] | groups[g][0]
                    new_group[1] = new_group[1] | groups[g][1]

                for g in sorted(grouped, reverse=True):
                    del(groups[g])
                groups.append(new_group)

    return groups


# @njit
def lines_coincide(line_1, line_2):
    # tic = time.perf_counter()
    max_angle = 0.10
    max_dis = 0.01

    start_1 = line_1[0:3]
    end_1 = line_1[3:6]
    dir_1 = end_1 - start_1
    l_1 = np.linalg.norm(dir_1)
    dir_1_n = dir_1 / l_1

    start_2 = line_2[0:3]
    end_2 = line_2[3:6]
    dir_2 = end_2 - start_2
    l_2 = np.linalg.norm(dir_2)
    dir_2_n = dir_2 / l_2

    # Check if the angle of the line is not above a certain threshold.
    angle = np.abs(np.dot(dir_1_n, dir_2_n))

    # print("Calculating took {} seconds.".format(time.perf_counter() - tic))
    if angle > np.cos(max_angle):
        # Check if the orthogonal distance between the lines are lower than a certain threshold.
        dis_3 = np.linalg.norm(np.cross(dir_1_n, start_2 - start_1))
        dis_4 = np.linalg.norm(np.cross(dir_1_n, end_2 - start_1))

        if dis_3 < max_dis or dis_4 < max_dis:
            # Check if the lines overlap.
            x_3 = np.dot(dir_1_n, start_2 - start_1)
            x_4 = np.dot(dir_1_n, end_2 - start_1)
            if min(x_3, x_4) < 0. < max(x_3, x_4) or 0. < min(x_3, x_4) < l_1:
                return True

    return False


def fuse_line_group(lines):
    start_1 = lines[0][:3]
    end_1 = lines[0][3:6]
    l_1 = np.linalg.norm(end_1 - start_1)
    dir_1 = (end_1 - start_1) / l_1
    start_1_open = lines[0][12]
    end_1_open = lines[0][13]

    x = [0., l_1]
    points = [start_1, end_1]
    opens = [start_1_open, end_1_open]

    for line in lines[1:]:
        x.append(dir_1.dot(line[:3] - start_1))
        points.append(line[:3])
        opens.append(line[12])
        x.append(dir_1.dot(line[3:6] - start_1))
        points.append(line[3:6])
        opens.append(line[13])

    start_idx = int(np.argmin(x))
    end_idx = int(np.argmax(x))
    new_start = points[start_idx]
    new_end = points[end_idx]
    new_start_open = opens[start_idx]
    new_end_open = opens[end_idx]

    new_normal_1 = lines[0][6:9]
    new_normal_2 = lines[0][9:12]
    # Find a line that has two normals, and use those.
    for line in lines:
        n_1 = line[6:9]
        n_2 = line[9:12]
        if not (n_1 == 0.).all() and not (n_2 == 0.).all():
            new_normal_1 = n_1
            new_normal_2 = n_2
            break

    return np.hstack([new_start, new_end, new_normal_1, new_normal_2, new_start_open, new_end_open])


def transform_lines(lines, transform):
    vstacked = np.vstack([lines[:, :3].T, np.ones((1, lines.shape[0]))])
    transformed = transform.dot(vstacked)
    lines[:, :3] = transformed[:3, :].T
    lines[:, 3:6] = transform.dot(np.vstack([lines[:, 3:6].T, np.ones((1, lines.shape[0]))]))[:3, :].T
    lines[:, 6:9] = transform[:3, :3].dot(lines[:, 6:9].T).T
    lines[:, 9:12] = transform[:3, :3].dot(lines[:, 9:12].T).T

    return lines


def fuse_frames(new_frame, new_vci_paths, previous_frames, previous_vcis):
    assert(len(previous_frames) == len(previous_vcis))

    if new_frame.shape[0] == 0:
        return new_frame, new_vci_paths

    new_frame_geometries = np.hstack([new_frame[:, 0:3], new_frame[:, 3:6], new_frame[:, 23:26], new_frame[:, 26:29],
                                      new_frame[:, 29].reshape(new_frame.shape[0], 1),
                                      new_frame[:, 30].reshape(new_frame.shape[0], 1)])
    new_frame_labels = new_frame[:, 21]
    new_frame_classes = new_frame[:, 22]
    new_vcis = new_vci_paths
    cam_origin_new = new_frame[0, 31:34]
    cam_rotation_new = new_frame[0, 34:38]
    t_1 = get_transform_to_world(cam_origin_new, cam_rotation_new)

    for i, frame in enumerate(previous_frames):
        if frame.shape[0] == 0:
            continue

        frame_geometries = np.hstack([frame[:, 0:3], frame[:, 3:6], frame[:, 23:26], frame[:, 26:29],
                                      frame[:, 29].reshape(frame.shape[0], 1), frame[:, 30].reshape(frame.shape[0], 1)])
        frame_labels = frame[:, 21]
        frame_classes = frame[:, 22]
        frame_vcis = previous_vcis[i]
        cam_origin = frame[0, 31:34]
        cam_rotation = frame[0, 34:38]
        t_2 = get_transform_to_world(cam_origin, cam_rotation)
        frame_geometries = transform_lines(frame_geometries, np.linalg.inv(t_1).dot(t_2))

        assert(len(new_vcis) == new_frame_geometries.shape[0])
        print(len(frame_vcis))
        print(frame.shape)
        assert(len(frame_vcis) == frame.shape[0])
        new_frame_geometries, new_frame_labels, new_frame_classes, new_vcis = fuse_two_frames(new_frame_geometries,
                                                                                              new_frame_labels,
                                                                                              new_frame_classes,
                                                                                              new_vcis,
                                                                                              frame_geometries,
                                                                                              frame_labels,
                                                                                              frame_classes,
                                                                                              frame_vcis)

    output = np.zeros((new_frame_geometries.shape[0], new_frame.shape[1]))
    output[:, :6] = new_frame_geometries[:, :6]
    output[:, 23:29] = new_frame_geometries[:, 6:12]
    output[:, 29] = new_frame_geometries[:, 12]
    output[:, 30] = new_frame_geometries[:, 13]
    output[:, 31:34] = cam_origin_new
    output[:, 34:38] = cam_rotation_new
    output[:, 21] = new_frame_labels
    output[:, 22] = new_frame_classes

    return output, new_vcis


def fuse_two_frames(geom_1, labels_1, class_1, vcis_1, geom_2, labels_2, class_2, vcis_2):
    groups = group_lines(geom_1, geom_2)

    out_geometries = []
    out_labels = []
    out_classes = []
    out_vcis = []

    for group in groups:
        geometries = [geom_1[i, :] for i in group[0]] + [geom_2[j, :] for j in group[1]]
        labels = [labels_1[i] for i in group[0]] + [labels_2[j] for j in group[1]]
        classes = [class_1[i] for i in group[0]] + [class_2[j] for j in group[1]]
        assert(len(vcis_1) == geom_1.shape[0])
        assert(len(vcis_2) == geom_2.shape[0])
        vcis = [vcis_1[i] for i in group[0]] + [vcis_2[j] for j in group[1]]

        # Fuse lines by geometry
        fused_line = fuse_line_group(geometries)

        fused_label = Counter(labels).most_common(1)[0][0]
        fused_class = Counter(classes).most_common(1)[0][0]

        # resolutions = np.array([vci.shape[0] for vci in vcis])
        fused_vci = vcis[0]

        out_geometries.append(fused_line)
        out_labels.append(fused_label)
        out_classes.append(fused_class)
        out_vcis.append(fused_vci)

    return np.array(out_geometries), np.array(out_labels), np.array(out_classes), out_vcis


def transform_to_world_rot(vector, camera_rotation):
    return transform_to_world_off_rot(
        vector, np.zeros((3,)), camera_rotation)[0:3]


def get_transform_to_world(camera_origin, camera_rotation):
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
    T = get_transform_to_world(camera_origin, camera_rotation)
    return T.dot(np.vstack([vector.T, np.ones((vector.shape[0], 1))]))[0:3, :].T


def read_frame(path):
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        data_lines = data_lines.values
        line_count = data_lines.shape[0]
    except pd.errors.EmptyDataError:
        data_lines = np.zeros((0, 38))
        line_count = 0

    return line_count, data_lines


def get_valid_lines_and_vcis(data_lines, frame_id, path_to_vcis):
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
    if os.path.exists(path_write):
        print("File already exists: {}".format(path_write))
        print("Skipping.")
        return

    print(data_lines.shape)
    for i in range(data_lines.shape[0]):
        path_to_vci = vci_paths[i]

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

        # line_start_point_world = transform_to_world_off_rot(
        #     np.transpose(start_point_camera), camera_origin, camera_rotation)
        # line_end_point_world = transform_to_world_off_rot(
        #     np.transpose(end_point_camera), camera_origin, camera_rotation)
        # normal_1_world = transform_to_world_rot(
        #     np.transpose(normal_1_camera),
        #     camera_rotation)
        # normal_2_world = transform_to_world_rot(
        #     np.transpose(normal_2_camera),
        #     camera_rotation)

        line_dict.update({
            "image_path": path_to_vci,
            'frame_id': frame_id,
        })

        with open(path_write, 'a') as f:
            line_file_utils.write_line_split_line(f, line_dict)


def mean_shift_split(data_lines, label_index):
    # Transform lines into world coordinates.
    # for i in range(len(data_lines)):
    #     if data_lines[i].shape[0] > 0:
    #         data_lines[i][:, 0:3] = transform_to_world_off_rot(data_lines[i][:, 0:3], data_lines[i][0, 31:34],
    #                                                            data_lines[i][0, 34:38])
    #         data_lines[i][:, 3:6] = transform_to_world_off_rot(data_lines[i][:, 3:6], data_lines[i][0, 31:34],
    #                                                            data_lines[i][0, 34:38])
    #         # Frame id, line id in frame.
    #         data_lines[i] = np.hstack((data_lines[i], np.ones((data_lines[i].shape[0], 1)) * i,
    #                                    np.arange(data_lines[i].shape[0]).reshape(data_lines[i].shape[0], 1)))
    #         print(data_lines[i].shape)
    #     else:
    #         data_lines[i] = np.zeros((0, data_lines[i].shape[1] + 2))
    #         print("Empty frame detected.")
    for i in range(len(data_lines)):
        # Frame id, line id in frame.
        data_lines[i] = np.hstack((data_lines[i], np.ones((data_lines[i].shape[0], 1)) * i,
                                   np.arange(data_lines[i].shape[0]).reshape(data_lines[i].shape[0], 1)))

    radius = 1.7
    max_iterations = 20

    data_lines = np.vstack(data_lines)
    label_indices = np.unique(data_lines[:, label_index])

    # Transform to world first.
    for i in range(data_lines.shape[0]):
        data_lines[i, 0:3] = transform_to_world_off_rot(data_lines[i, 0:3].reshape((1, 3)),
                                                        data_lines[i, 31:34], data_lines[i, 34:38])
        data_lines[i, 3:6] = transform_to_world_off_rot(data_lines[i, 3:6].reshape((1, 3)),
                                                        data_lines[i, 31:34], data_lines[i, 34:38])

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
            # After that, change label to the biggest available (one more than the maximum label)
            if iteration > 0:
                changes.append([lines[in_radius, -2:], max_label])
                print("Added one instance split at instance {}.".format(int(max_label)))
                max_label = max_label + 1

            lines = np.delete(lines, in_radius, axis=0)
            iteration = iteration + 1

    return changes


def split_scene(line_path, vi_path, output_path, moving_window_length):
    if "{" not in output_path:
        print("Error, output_path needs to contain brackets for frame id formatting.")
        exit(1)

    # Count number of files in directory.
    frames = [name for name in os.listdir(line_path)
              if os.path.isfile(os.path.join(line_path, name)) and 'lines_with_labels' in name]
    frame_count = len(frames)
    frames.sort()

    # train_prob = 4. / 5.

    # train_indices = list(range(int(frame_count * train_prob)))
    # val_indices = list(range(int(frame_count * train_prob), frame_count))

    line_counts = []
    data_lines = []
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
    changes = mean_shift_split(data_lines, label_index)
    for change in changes:
        for i in range(change[0].shape[0]):
            frame_id = int(change[0][i, 0])
            line_id = int(change[0][i, 1])
            data_lines[frame_id][line_id, label_index] = change[1]

    vci_path_lists = []
    for i in range(len(data_lines)):
        path_to_vcis = os.path.join(vi_path, "frame_{}".format(i))
        data_lines[i], vci_paths = get_valid_lines_and_vcis(data_lines[i], i, path_to_vcis)
        assert(data_lines[i].shape[0] == len(vci_paths))
        vci_path_lists.append(vci_paths)

    if moving_window_length > 1:
        previous_frames = []
        previous_vci_paths = []
        for i, frame in enumerate(data_lines):
            print("Fusing frame {}.".format(i))
            # Fuse frames if moving window length is more than one.
            if i > 0:
                fused_frame, fused_vci_paths = fuse_frames(frame, vci_path_lists[i], previous_frames, previous_vci_paths)
            else:
                fused_frame = frame
                fused_vci_paths = vci_path_lists[i]
            assert(len(fused_vci_paths) == fused_frame.shape[0])
            previous_frames.append(frame)
            previous_vci_paths.append(vci_path_lists[i])
            data_lines[i] = fused_frame
            vci_path_lists[i] = fused_vci_paths
            if i > moving_window_length:
                del(previous_frames[0])
                del(previous_vci_paths[0])

            path_to_output = os.path.join(output_path.format(i))
            write_frame(data_lines[i], vci_path_lists[i], i, path_to_output)
    else:
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
