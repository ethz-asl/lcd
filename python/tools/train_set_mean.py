import numpy as np
import cv2


def get_train_set_mean(file_path):
    image_paths = []
    with open(file_path) as f:
        lines = f.readlines()
        labels = []
        for l in lines:
            items = l.split()
            image_paths.append(items[0])

    blue_mean = 0
    green_mean = 0
    red_mean = 0
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        blue_mean += np.mean(image[:, :, 0])
        green_mean += np.mean(image[:, :, 1])
        red_mean += np.mean(image[:, :, 2])

    image_mean = np.array([blue_mean, green_mean, red_mean]) / len(image_paths)
    return image_mean


if __name__ == '__main__':
    image_mean = get_train_set_mean('train.txt')
    print("Mean of train set: {}".format(image_mean))
