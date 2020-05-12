import os

import numpy as np
import matplotlib.pyplot as plt

from datagenerator_framewise import LineDataGenerator

def show_k_predictions():
    corrects = np.where(pred_arg == gt_arg)
    correct_values = pred_arg[corrects]
    print("Percentage of correct predictions: {}".format(corrects[0].shape[0]))
    print(correct_values)
    print(np.unique(correct_values))

    print("Number of correct preds if only k = 3:")
    print(np.where(gt_arg == 3)[0].shape[0])

    plt.hist(correct_values, bins=range(31))
    plt.show()

    for i in range(1):
        if np.argmax(predictions[i, :]) == np.argmax(gts[i, :]):
            fig, axs = plt.subplots(2)
            axs[0].bar(range(31), predictions[i, :])
            axs[1].bar(range(31), gts[i, :])
            plt.show()


def get_colors():
    return np.array([
        [1., 0., 0.],    # red
        [0., 1., 0.],    # green
        [0., 0., 1.],    # blue
        [0.75, 0.75, 0.],  # orange
        [0., 0.75, 0.75],  # orange
        [0.75, 0., 0.75],  # orange
        [0.25, 0.75, 0.],  # orange
        [0.75, 0.25, 0.],  # orange
        [0., 0.25, 0.75],  # orange
        [0., 0.75, 0.25],  # orange
        [0.25, 0., 0.75],  # orange
        [0.75, 0., 0.25],  # orange
        [0.5, 0.5, 0.8],  # orange
        [0.8, 0.5, 0.5],  # orange
        [0.5, 0.8, 0.5],  # orange
        [0.2, 0.8, 0.2],  # orange
    ])


if __name__ == '__main__':
    log_dir = "/home/felix/line_ws/src/line_tools/python/line_net/logs/110520_2158"
    test_dir = "/nvme/line_ws/test"
    bg_classes = [0, 1, 2, 20, 22]

    predictions = np.load(os.path.join(log_dir, "results/predictions_train.npy"))
    gts = np.load(os.path.join(log_dir, "results/ground_truths_train.npy"))

    predictions = np.squeeze(predictions)
    gts = np.squeeze(gts)
    print(predictions.shape)
    print(gts.shape)

    pred_arg = np.argmax(predictions, axis=-1)
    gt_arg = np.argmax(gts, axis=-1)

    colors = get_colors()

    test_data_generator = LineDataGenerator(test_dir, bg_classes,
                                            sort=True,
                                            min_line_count=0,
                                            max_cluster_count=1000000)
    geom, labels, valid, bg, img, k = test_data_generator.next_batch(150, load_images=False)
    unique_labels = np.unique(labels[np.where(np.logical_and(valid, np.logical_not(bg)))])
    print(unique_labels)

    size_x = 15
    size_y = 3
    # fig, axs = plt.subplots(5, 30)
    for i in range(size_x):
        for j in range(size_y):
            line_index = i * size_y + j
            plt.subplot(size_y, size_x, line_index + 1)
            plt.xticks([])
            plt.yticks([])

            if labels[line_index] not in unique_labels:
                color = [0.2, 0.2, 0.2]
                if not valid[line_index]:
                    plt.title('empty')
                else:
                    plt.title('background')
            else:
                lbl_index = np.where(unique_labels == labels[line_index])[0][0]
                plt.title('label: {}'.format(lbl_index))
                if lbl_index >= 15:
                    color = [0.0, 0.0, 0.0]
                else:
                    color = colors[lbl_index, :]

            plt.bar(range(15), predictions[0, line_index, :], color=color)

    plt.show()

