"""
Get virtual camera image for each line in trajectory 1.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd

from tools import scenenet_to_rosbag, scenenet_utils
from tools import pathconfig

path_to_photos = os.path.join(
    pathconfig.pySceneNetRGBD_root, 'data/train/0/784')  # train traj 1

path_to_lines_root = '../data/train_lines/traj_1'

# Virtual camera model
camera_model = scenenet_utils.get_camera_model()
# Distance between virtual camera origin and line's center
distance = 3
for frame_id in range(300):
    photo_id = frame_id * 25

    rgb_image = cv2.imread(
        os.path.join(path_to_photos, 'photo/{0}.jpg'.format(photo_id)), cv2.IMREAD_COLOR)

    depth_image = cv2.imread(
        os.path.join(path_to_photos, 'depth/{0}.png'.format(photo_id)), cv2.IMREAD_UNCHANGED)

    pcl = scenenet_utils.rgbd_to_pcl(rgb_image, depth_image, camera_model)

    path_to_lines = os.path.join(
        path_to_lines_root, 'lines_with_labels_{0}.txt'.format(frame_id))

    try:
        data_lines = pd.read_csv(path_to_lines, sep=" ", header=None)
    except IOError:
        print("No line detected for frame {}".format(frame_id))
        continue

    data_lines = data_lines.values
    lines_count = data_lines.shape[0]

    for i in range(lines_count):
        T, _ = scenenet_utils.virtual_camera_pose(data_lines[i, :], distance)
        pcl_from_line_view = scenenet_utils.pcl_transform(pcl, T)
        rgb_image_from_line_view, depth_image_from_line_view = scenenet_utils.project_pcl_to_image(
            pcl_from_line_view, camera_model)

        cv2.imwrite(
            '../data/train/traj_1/frame_{0}/rgb/{1}.png'.format(frame_id, i), rgb_image_from_line_view)
        cv2.imwrite(
            '../data/train/traj_1/frame_{0}/depth/{1}.png'.format(frame_id, i), depth_image_from_line_view)

    print('frame {0} finished processing'.format(frame_id))
