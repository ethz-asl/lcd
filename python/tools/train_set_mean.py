import numpy as np
import cv2
import argparse

from pickle_dataset import merge_pickled_dictionaries
from sklearn.externals import joblib


def get_train_set_mean(file_path, image_type, read_as_pickle):
    """Get the train set mean.
    """
    if read_as_pickle:
        if image_type != 'bgr' and image_type != 'bgr-d':
            raise ValueError("Image type should be 'bgr' or 'bgr-d'")
        blue_mean = 0
        green_mean = 0
        red_mean = 0
        depth_mean = 0
        total_images = 0
        pickled_dict = {}
        # Merge dictionaries extracted from all the pickle files in file_path
        for file_ in file_path:
            temp_dict = joblib.load(file_)
            merge_pickled_dictionaries(pickled_dict, temp_dict)
        for scenenetdataset_type in pickled_dict.keys():
            scenenetdataset_type_dict = pickled_dict[scenenetdataset_type]
            for trajectory_number in scenenetdataset_type_dict.keys():
                trajectory_number_dict = scenenetdataset_type_dict[
                    trajectory_number]
                for frame_number in trajectory_number_dict.keys():
                    frame_number_dict = trajectory_number_dict[frame_number]
                    for line_number in frame_number_dict['rgb'].keys():
                        total_images += 1
                        # BGR image
                        img_bgr = frame_number_dict['rgb'][line_number]['img']
                        blue_mean += np.mean(img_bgr[:, :, 0])
                        green_mean += np.mean(img_bgr[:, :, 1])
                        red_mean += np.mean(img_bgr[:, :, 2])
                        # Depth image
                        img_depth = frame_number_dict['depth'][line_number][
                            'img']
                        depth_mean += np.mean(img_depth)
        if image_type == 'bgr':
            image_mean = np.array([blue_mean, green_mean, red_mean
                                  ]) / total_images
        elif image_type == 'bgr-d':
            image_mean = np.array([blue_mean, green_mean, red_mean, depth_mean
                                  ]) / total_images
        else:
            raise ValueError("Image type should be 'bgr' or 'bgr-d'")
    else:
        image_paths = []
        # In non-pickle mode only consider first file in list
        with open(file_path[0]) as f:
            lines = f.readlines()
            for l in lines:
                items = l.split()
                image_paths.append(items[0])

        if image_type == 'bgr':
            blue_mean = 0
            green_mean = 0
            red_mean = 0
            for path_rgb in image_paths:
                img_bgr = cv2.imread(path_rgb, cv2.IMREAD_UNCHANGED)
                blue_mean += np.mean(img_bgr[:, :, 0])
                green_mean += np.mean(img_bgr[:, :, 1])
                red_mean += np.mean(img_bgr[:, :, 2])

            image_mean = np.array([blue_mean, green_mean, red_mean
                                  ]) / len(image_paths)
        elif image_type == 'bgr-d':
            blue_mean = 0
            green_mean = 0
            red_mean = 0
            depth_mean = 0
            for path_rgb in image_paths:
                img_bgr = cv2.imread(path_rgb, cv2.IMREAD_UNCHANGED)
                path_depth = path_rgb.replace('rgb', 'depth')
                img_depth = cv2.imread(path_depth, cv2.IMREAD_UNCHANGED)
                blue_mean += np.mean(img_bgr[:, :, 0])
                green_mean += np.mean(img_bgr[:, :, 1])
                red_mean += np.mean(img_bgr[:, :, 2])
                depth_mean += np.mean(img_depth)

            image_mean = np.array([blue_mean, green_mean, red_mean, depth_mean
                                  ]) / len(image_paths)
        else:
            raise ValueError("Image type should be 'bgr' or 'bgr-d'")
    return image_mean
