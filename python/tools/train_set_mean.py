import numpy as np
import cv2
import argparse


def get_train_set_mean(file_path, image_type):
    image_paths = []
    with open(file_path) as f:
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

        image_mean = np.array(
            [blue_mean, green_mean, red_mean]) / len(image_paths)
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

        image_mean = np.array(
            [blue_mean, green_mean, red_mean, depth_mean]) / len(image_paths)
    else:
        raise ValueError("Image type should be 'bgr' or 'bgr-d'")
    return image_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute train set mean")
    parser.add_argument('--image_type', default='bgr',
                        help="Image type bgr/bgr-d")

    args = parser.parse_args()
    image_type = args.image_type

    image_mean = get_train_set_mean('train.txt', image_type)
    print("Mean of train set: {}".format(image_mean))
