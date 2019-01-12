import os
import open3d
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

import pathconfig
from get_protobuf_paths import get_protobuf_path

# Retrieve scenenetscripts_path and protobuf path from config file.
print('visualization.py: Using values in config_paths_and_variables.sh '
      'for SCENENET_SCRIPTS_PATH, LINESANDIMAGESFOLDER_PATH.')
scenenetscripts_path = pathconfig.obtain_paths_and_variables(
    "SCENENET_SCRIPTS_PATH")
linesandimagesfolder_path = pathconfig.obtain_paths_and_variables(
    "LINESANDIMAGESFOLDER_PATH")

sys.path.append(scenenetscripts_path)
import scenenet_pb2 as sn

from mpl_toolkits.mplot3d import Axes3D
from camera_pose_and_intrinsics_example import camera_to_world_with_pose, interpolate_poses


def get_lines_world_coordinates_with_instances(dataset_name, trajectory,
                                               frames):
    """For some frames in a trajectory, get lines start and end points world
       coordinates and the instance labels assigned to lines.
    """
    lines_world = {}
    frames_count = {}
    lines_total = 0

    path_to_lines_root = os.path.join(linesandimagesfolder_path,
                                      '{}_lines/'.format(dataset_name))
    # Find protobuf file associated to dataset_name.
    protobuf_path = get_protobuf_path(dataset_name)
    print(
        "visualization.py/get_lines_world_coordinates_with_instances: using {} as protobuf_path".
        format(protobuf_path))
    if protobuf_path is None:
        sys.exit('visualization.py: Error in retrieving protobuf_path.')

    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('visualization.py: Scenenet protobuf data not found at location:'
              '{0}'.format(protobuf_path))
        print('Please ensure you have copied the pb file to the data directory')

    frames_with_lines = []

    for frame_id in frames:
        # View data for current frame.
        view = trajectories.trajectories[trajectory].views[frame_id]
        # Get ground truth pose of camera
        ground_truth_pose = interpolate_poses(view.shutter_open,
                                              view.shutter_close, 0.5)
        # Transformation matrix from camera coordinate to world coordinate.
        camera_to_world_matrix = camera_to_world_with_pose(ground_truth_pose)

        path_to_lines = os.path.join(
            path_to_lines_root, 'traj_{0}/lines_with_labels_{1}.txt'.format(
                trajectory, frame_id))
        print('path_to_lines is {0}'.format(path_to_lines))
        try:
            data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
        except (IOError, pd.io.common.EmptyDataError):
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

            # Get world coordinates of start and end point.
            line_start_point_world = camera_to_world_matrix.dot(
                line_start_point_camera)
            line_end_point_world = camera_to_world_matrix.dot(
                line_end_point_camera)

            lines_world_with_instances[line_idx, :3] = line_start_point_world[:
                                                                              3]
            lines_world_with_instances[line_idx, 3:6] = line_end_point_world[:3]
            # Instance is last value of the line (cf. virtual_camera_pose in
            # scenenet_utils.py).
            lines_world_with_instances[line_idx, 6] = line[-1]

            line_idx += 1

        lines_world[frame_id] = lines_world_with_instances
        frames_with_lines.append(frame_id)

    print('Length of lines_world is ' + str(len(lines_world)))
    data_lines_world = np.vstack([lines_world[k] for k in frames_with_lines])
    print('Total number of lines: {}'.format(data_lines_world.shape[0]))

    instances_total = np.unique(data_lines_world[:, -1]).shape[0]
    print('Number of unique instance labels for lines: {}'.format(
        instances_total))

    return data_lines_world


def pcl_lines_for_plot(data_lines, lines_color, visualizer):
    """Get points on the lines for 3D visualization in open3d or matplotlib.
    """
    if visualizer != 'open3d' and visualizer != 'matplotlib':
        print(
            'Invalid visualizer. Valid options are \'open3d\', \'matplotlib\'.')
        return

    lines_number = data_lines.shape[0]
    pcl_lines = [[] for n in range(lines_number)]

    for i in range(lines_number):
        line = data_lines[i]
        start = line[:3]
        end = line[3:6]
        vector = end - start
        interpolate = np.linspace(0, 1, 100)

        points = np.vstack((start + n * vector for n in interpolate))
        if np.unique(lines_color).shape[0] > 4:
            np.random.seed(lines_color[i])
            rgb = np.random.randint(255, size=(1, 3)) / 255.0
        else:
            # This assumes that if no more than four different colours have been
            # assigned to the lines, these colours encode the line type.
            if lines_color[i] == 0:  # Discontinuity line (red).
                rgb = np.array([1, 0, 0])
            if lines_color[i] == 1:  # Planar line (green).
                rgb = np.array([[0, 1, 0]])
            if lines_color[i] == 2:  # Edge line (blue).
                rgb = np.array([[0, 0, 1]])
            if lines_color[i] == 3:  # Intersection line (yellow).
                rgb = np.array([[1, 1, 0]])
        rgbs = np.vstack((rgb for n in interpolate))

        if visualizer == 'open3d':
            pcl_lines[i] = open3d.PointCloud()
            pcl_lines[i].points = open3d.Vector3dVector(points)
            pcl_lines[i].colors = open3d.Vector3dVector(rgbs)
        elif visualizer == 'matplotlib':
            pcl_lines[i] = {}
            pcl_lines[i]['points'] = points
            pcl_lines[i]['color'] = tuple(rgb[0])
    return pcl_lines


def plot_lines_with_matplotlib(pcl_lines):
    """ Plots a set of lines (in the format outputted by pcl_lines_for_plot) in
        matplotlib.
    """
    num_lines = len(pcl_lines)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(num_lines):
        x = pcl_lines[i]['points'][:, 0]
        y = pcl_lines[i]['points'][:, 1]
        z = pcl_lines[i]['points'][:, 2]
        ax.plot(x, y, z, color=pcl_lines[i]['color'])
    plt.show()


def plot_lines_with_open3d(pcl_lines, window_name="Open3D"):
    """ Plots a set of lines (in the format outputted by pcl_lines_for_plot) in
        open3d.
    """
    vis = open3d.Visualizer()
    vis.create_window(window_name=window_name)
    for line in pcl_lines:
        vis.add_geometry(line)
    vis.run()
    vis.destroy_window()


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) element in a grid of size approx.
       sqrt(n) by sqrt(n).
    """
    # Normalize data for display.
    data = (data - data.min()) / (data.max() - data.min())

    # Force the number of filters to be square.
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # Add some space between filters.
    padding = (
        ((0, n**2 - data.shape[0]), (0, 1), (0, 1)) + ((0, 0),) *
        (data.ndim - 3))  # Do not pad the last dimension (if there is one).
    data = np.pad(
        data, padding, mode='constant',
        constant_values=1)  # Pad with ones (white).

    # Tile the filters into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose(
        (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
