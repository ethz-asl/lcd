import os
import numpy as np
import pandas as pd
import pathconfig
import scenenet_pb2 as sn

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

    data_lines_world = np.vstack([lines_world[k] for k in lines_world])
    print('Total number of lines: {}'.format(data_lines_world.shape[0]))

    instances_total = np.unique(data_lines_world[:, -1]).shape[0]
    print('Number of unique instance labels for lines: {}'.format(instances_total))

    return data_lines_world
