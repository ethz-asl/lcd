"""
Get virtual camera image for each line in trajectory with index given as input.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
import argparse

from tools import scenenet_utils
from tools import pathconfig

def get_virtual_camera_images():
    # Get render path associated to the trajectory
    try:
        with open(renderpaths_path, 'r') as f:
            lines = f.readlines()
        trajectories = dict()
        for line in lines:
            traj_data = line.split()
            traj_num = int(traj_data[0])
            traj_renderpath = traj_data[1]
            trajectories[traj_num] = traj_renderpath
    except IOError:
        print('Render path file not found at location {0}.\nPlease generate '
        'a render_path file by running the script get_render_paths.py or '
        'specify a valid path for the render_path text file by using the '
        'argument -renderpath_paths.'.format(renderpaths_path))
        return
    try:
        traj_renderpath = trajectories[trajectory]
    except KeyError:
        print('Error: not able to find a trajectory with index {0}'.format(
            trajectory))
        return

    path_to_photos = os.path.join(
    pathconfig.pySceneNetRGBD_root, 'data/train/', traj_renderpath)

    print('Path to photos is' + path_to_photos)

    path_to_lines_root = '../data/train_lines/traj_' + str(trajectory) + '/'

    print('Path_to_lines_root is {0}'.format(path_to_lines_root))

    # Virtual camera model
    camera_model = scenenet_utils.get_camera_model()
    distance = 3
    # Distance between virtual camera origin and line's center
    for frame_id in range(300):
        photo_id = frame_id * 25

        rgb_image = cv2.imread(
            os.path.join(path_to_photos, 'photo/{0}.jpg'.format(photo_id)),
            cv2.IMREAD_COLOR)

        depth_image = cv2.imread(
            os.path.join(path_to_photos, 'depth/{0}.png'.format(photo_id)),
            cv2.IMREAD_UNCHANGED)

        pcl = scenenet_utils.rgbd_to_pcl(rgb_image, depth_image, camera_model)

        path_to_lines = os.path.join(
            path_to_lines_root, 'lines_with_labels_{0}.txt'.format(frame_id))

        try:
            data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
        except (IOError, pd.io.common.EmptyDataError):
            print("No line detected for frame {}".format(frame_id))
            continue

        data_lines = data_lines.values
        lines_count = data_lines.shape[0]

        for i in range(lines_count):
            T, _ = scenenet_utils.virtual_camera_pose(data_lines[i, :],
                                                      distance)
            pcl_from_line_view = scenenet_utils.pcl_transform(pcl, T)
            rgb_image_from_line_view, depth_image_from_line_view = \
                scenenet_utils.project_pcl_to_image(pcl_from_line_view,
                                                    camera_model)

            cv2.imwrite(
                '../data/train/traj_{0}/frame_{1}/rgb/{2}.png'.format(
                trajectory, frame_id, i), rgb_image_from_line_view)
            cv2.imwrite(
                '../data/train/traj_{0}/frame_{1}/depth/{2}.png'.format(
                trajectory, frame_id, i),
                depth_image_from_line_view.astype(np.uint16))

        print('frame {0} finished processing'.format(frame_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get virtual camera image for each line in the input '
                    'trajectory.')
    parser.add_argument(
        "-trajectory",
        default=1,
        help="Trajectory number.")
    parser.add_argument(
        "-renderpaths_path",
        default="render_paths.txt",
        help="Path to file containing correspondences between render paths and "
             "number of the trajectory.")

    args = parser.parse_args()
    if args.trajectory:
        trajectory = int(args.trajectory)
    if args.renderpaths_path:
        renderpaths_path = args.renderpaths_path

    get_virtual_camera_images()
