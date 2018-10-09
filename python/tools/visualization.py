import os
import open3d
import numpy as np
import pandas as pd
import pathconfig
import scenenet_pb2 as sn
import matplotlib.pyplot as plt

from camera_pose_and_intrinsics_example import camera_to_world_with_pose, interpolate_poses

path_to_lines_root = pathconfig.path_to_lines_root
protobuf_path = pathconfig.protobuf_path

trajectories = sn.Trajectories()
try:
    with open(protobuf_path, 'rb') as f:
        trajectories.ParseFromString(f.read())
except IOError:
    print('Scenenet protobuf data not found at location:{0}'.format(
        protobuf_path))
    print('Please ensure you have copied the pb file to the data directory')


def get_lines_world_coordinates_with_instances(trajectory, frames):
    """For some frames in a trajectory, get lines start and end points world coordianates and the instance labels assigned to lines.
    """
    lines_world = {}
    frames_count = {}
    lines_total = 0
    for frame_id in frames:
        # View data for current frame
        view = trajectories.trajectories[trajectory].views[frame_id]
        # Get ground true pose of camera
        ground_truth_pose = interpolate_poses(
            view.shutter_open, view.shutter_close, 0.5)
        # Transformation matrix from camera coordinate to world coordinate
        camera_to_world_matrix = camera_to_world_with_pose(ground_truth_pose)

        path_to_lines = os.path.join(path_to_lines_root,
                                     'lines_with_labels_{0}.txt'.format(frame_id))
        try:
            data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
        except IOError:
            print('No line detected for frame {}'.format(frame_id))
            continue
        data_lines = data_lines.values
        lines_count = data_lines.shape[0]

        lines_total += lines_count
        frames_count[frame_id] = lines_total

        lines_world_with_instances = np.zeros((lines_count, 7))
        for line_idx in range(lines_count):
            line = data_lines[line_idx]
            line_start_point_camera = np.append(line[:3], [1])
            line_end_point_camera = np.append(line[3:6], [1])

            # Get world coordinates of start and end point
            line_start_point_world = camera_to_world_matrix.dot(
                line_start_point_camera)
            line_end_point_world = camera_to_world_matrix.dot(
                line_end_point_camera)

            lines_world_with_instances[line_idx,
                                       :3] = line_start_point_world[:3]
            lines_world_with_instances[line_idx,
                                       3:6] = line_end_point_world[:3]
            lines_world_with_instances[line_idx, 6] = line[-1]

            line_idx += 1

        lines_world[frame_id] = lines_world_with_instances

    data_lines_world = np.vstack([lines_world[k] for k in frames])
    print('Total number of lines: {}'.format(data_lines_world.shape[0]))

    instances_total = np.unique(data_lines_world[:, -1]).shape[0]
    print('Number of unique instance labels for lines: {}'.format(instances_total))

    return data_lines_world


def pcl_lines_for_plot(data_lines, lines_color):
    """Get points on the lines for 3D visualization in open3d.
    """
    lines_number = data_lines.shape[0]
    pcl_lines = [[] for n in range(lines_number)]

    for i in range(lines_number):
        line = data_lines[i]
        start = line[:3]
        end = line[3:6]
        vector = end - start
        interpolate = np.linspace(0, 1, 100)

        points = np.vstack((start + n * vector for n in interpolate))
        if np.unique(lines_color).shape[0] > 3:
            np.random.seed(lines_color[i])
            rgb = np.random.randint(255, size=(1, 3)) / 255.0
        else:
            if lines_color[i] == 0:  # discontinuty line
                rgb = np.array([1, 0, 0])
            if lines_color[i] == 1:  # plane(surface) line
                rgb = np.array([[0, 1, 0]])
            if lines_color[i] == 2:  # intersection line
                rgb = np.array([[0, 0, 1]])
        rgbs = np.vstack((rgb for n in interpolate))

        pcl_lines[i] = open3d.PointCloud()
        pcl_lines[i].points = open3d.Vector3dVector(points)
        pcl_lines[i].colors = open3d.Vector3dVector(rgbs)
    return pcl_lines


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n).
    """

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # add some space between filters
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant',
                  constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape(
        (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')