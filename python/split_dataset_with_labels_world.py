"""
Split detected lines of the input trajectory to train, val and test set and label each line with its center in the world frame and the instance label.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse

import tools.pathconfig
import scenenet_pb2 as sn

from tools import scenenet_utils
from camera_pose_and_intrinsics_example import camera_to_world_with_pose, interpolate_poses


def split_dataset():
  pySceneNetRGBD_root = tools.pathconfig.pySceneNetRGBD_root
  protobuf_path = tools.pathconfig.protobuf_path
  print('Path_to_lines_root is {0}'.format(path_to_lines_root))
  print('Path_to_lines_image is {0}'.format(path_to_lines_image))

  trajectories = sn.Trajectories()
  try:
      with open(protobuf_path, 'rb') as f:
          trajectories.ParseFromString(f.read())
  except IOError:
      print('Scenenet protobuf data not found at location:{0}'.format(
          protobuf_path))
      print('Please ensure you have copied the pb file to the datadirectory')

  '''try:
      os.remove('train.txt')
      os.remove('val.txt')
      os.remove('test.txt')
      os.remove('all_lines.txt')
  except OSError:
      pass'''

  frames_total = 300
  train = []
  val = []
  test = []

  for frame_id in range(1, frames_total):  # Start from 1, as frame 0 is
                                           # currently invalid. TODO(): set
                                           # start back to 0.
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
  # Start from 1, as frame 0 is currently invalid. TODO(): set start back to 0.
  dataset['all_lines'] = [i for i in range(1, frames_total)]

  for key, frames in dataset.iteritems():
      for frame_id in frames:
          path_to_lines = os.path.join(
              path_to_lines_root, 'lines_with_labels_{0}.txt'.format(frame_id))

          print('Path_to_lines is {0}'.format(path_to_lines))
          try:
              data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
              data_lines = data_lines.values
              lines_count = data_lines.shape[0]
          except pd.errors.EmptyDataError:
              lines_count = 0

          view = trajectories.trajectories[traj].views[frame_id]
          ground_truth_pose = interpolate_poses(
              view.shutter_open, view.shutter_close, 0.5)
          camera_to_world_matrix = camera_to_world_with_pose(ground_truth_pose)

          for i in range(lines_count):
              path_to_write = path_to_lines_image + \
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

              with open(key + '.txt', 'a') as f:
                  f.write(os.path.abspath(path_to_write) + ' ' + str(center_of_line[0]) + ' ' + str(
                      center_of_line[1]) + ' ' + str(center_of_line[2]) + ' ' + str(label) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split frames in the input trajectory in train, test and '
                    'validation set.')
    parser.add_argument(
        "-trajectory",
        default=1,
        help="Trajectory number.")
    parser.add_argument(
        "-path_to_lines_root",
        default=tools.pathconfig.path_to_lines_root,
        help="Path to the root folder containing lines.")
    parser.add_argument(
        "-path_to_lines_image",
        default=tools.pathconfig.path_to_lines_image,
        help="Path to the root folder containing images.")

    args = parser.parse_args()

    print('Given path_to_lines_root is {0}'.format(args.path_to_lines_root))
    print('Given path_to_lines_image is {0}'.format(args.path_to_lines_image))
    if args.trajectory:
        traj = int(args.trajectory)
    if args.path_to_lines_root:
        path_to_lines_root =  os.path.join(args.path_to_lines_root,
            'traj_{0}/'.format(traj))
    if args.path_to_lines_image:
        path_to_lines_image = os.path.join(args.path_to_lines_image,
            'traj_{0}/'.format(traj))

    split_dataset()
