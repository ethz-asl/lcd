""" Creates a histogram with the distribution of the instance labels in a pickle
    file.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.externals import joblib


def compute_histogram(picklefile_path):
    """ Returns the histogram with the distribution of the instance labels in
        the input pickle file.

    Args:
        picklefile_path (string): Path of the pickle file.
    Returns:
        histogram (dict): histogram[<instance_label>] contains the number of
            occurrences for the instance label <instance_label>.
    """
    hist_instance_labels = []

    # Load file.
    try:
        pickled_dict = joblib.load(picklefile_path)
    except IOError:
        print("Pickle found not found. Exiting.")
        return

    for dataset_name in pickled_dict.keys():
        dataset_name_dict = pickled_dict[dataset_name]
        for trajectory_number in dataset_name_dict.keys():
            trajectory_number_dict = dataset_name_dict[trajectory_number]
            for frame_number in trajectory_number_dict.keys():
                frame_number_dict = trajectory_number_dict[frame_number]
                for image_type in frame_number_dict.keys():
                    image_type_dict = frame_number_dict[image_type]
                    for line_number in image_type_dict.keys():
                        line_number_dict = image_type_dict[line_number]
                        # Append image.
                        if image_type == 'rgb':
                            # Append label.
                            hist_instance_labels.append(
                                line_number_dict['labels'][-1])

    print("Total number of lines in pickle file is {}".format(
        len(hist_instance_labels)))

    instance_labels, occurrences = np.unique(
        hist_instance_labels, return_counts=True)
    instance_labels = instance_labels.astype(int)

    histogram = dict(zip(instance_labels, occurrences))

    return histogram


def display_histogram(picklefile_path):
    """ Displays the histogram with the distribution of the instance labels in
        the input pickle file.

    Args:
        picklefile_path (string): Path of the pickle file.
    """
    # Compute histogram.
    histogram = compute_histogram(picklefile_path)

    instance_labels = histogram.keys()
    occurrences = histogram.values()

    # Create histogram.
    plt.figure("Histogram of instance labels")
    plt.bar(x=instance_labels, height=occurrences)
    plt.gca().set_xticklabels(
        ['{}'.format(int(x)) for x in plt.gca().get_xticks()])
    plt.xlabel("Instance labels")
    plt.ylabel("Occurrences in pickle file")
    # Compute average number of occurrences.
    num_instances = len(instance_labels)
    average_num_occurrences = np.sum(occurrences) / num_instances
    plt.axhline(
        y=average_num_occurrences, color='k', linestyle='dashed', linewidth=1)
    plt.title(
        "Histogram of instance labels (mean number of occurrences = {:.2f})".
        format(average_num_occurrences))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a histogram of the distribution of the instance ' +
        'labels in a pickle file.')
    parser.add_argument(
        "-picklefile_path", help="Path of the pickle file.", required=True)
    try:
        args = parser.parse_args()
        if (args.picklefile_path):
            picklefile_path = args.picklefile_path
            display_histogram(picklefile_path=picklefile_path)
    except:
        pass
