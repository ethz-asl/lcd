""" Creates a histogram with the distribution of the instance labels in a pickle
    file.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
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


def display_histogram(picklefile_path, save_data=False, save_data_path=None):
    """ Displays the histogram with the distribution of the instance labels in
        the input pickle file. If specified, saves the data in a statistics
        file.

    Args:
        picklefile_path (string): Path of the pickle file.
        save_data (bool): If True the histogram statistics are printed to the
            file save_data_path.
        save_data_path (string): Path where to store the histogram statistics
            if save_data is True.
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
    plt.xlabel("Instance label")
    plt.ylabel("Number of occurrences in pickle file")
    # Compute average number of occurrences.
    num_instances = len(instance_labels)
    average_num_occurrences = np.mean(occurrences)
    # Compute standard deviation of the number of occurrences.
    std_num_occurrences = np.std(occurrences)
    plt.axhline(
        y=average_num_occurrences, color='k', linestyle='dashed', linewidth=1)
    plt.axhline(
        y=average_num_occurrences + std_num_occurrences,
        color='k',
        linestyle='dotted',
        linewidth=1)
    if (average_num_occurrences - std_num_occurrences > 0):
        plt.axhline(
            y=average_num_occurrences - std_num_occurrences,
            color='k',
            linestyle='dotted',
            linewidth=1)
    plt.title(
        "Histogram of instance labels (mean number of occurrences = {:.2f}, ".
        format(average_num_occurrences) +
        "standard deviation = {:.2f})".format(std_num_occurrences))

    plt.show()

    # Save histogram if required.
    if (save_data):
        if (save_data_path is None):
            print("Please specify a folder where to print the histogram "
                  "statistics.")
        else:
            if not os.path.exists(os.path.dirname(save_data_path)):
                try:
                    os.makedirs(os.path.dirname(save_data_path))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            with open(save_data_path, 'w') as f:
                # Print occurrences for each instance label.
                f.write("The pickle file '{0}' contains {1} lines ".format(
                    picklefile_path, np.sum(occurrences)) + "with the "
                        "following instance labels:\n")
                for _, (instance_label, occurrence) in enumerate(
                        histogram.items()):
                    f.write("- {0}: {1} occurrences\n".format(
                        instance_label, occurrence))
                # Print mean and standard deviation of the number of
                # occurrences.
                f.write("\nThe mean of the number of occurrences is {:.3f}.\n".
                        format(average_num_occurrences))
                f.write(
                    "The standard deviation of the number of occurrences is "
                    "{:.3f}.\n".format(std_num_occurrences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a histogram of the distribution of the instance ' +
        'labels in a pickle file.')
    parser.add_argument(
        "-picklefile_path", help="Path of the pickle file.", required=True)
    parser.add_argument(
        "-save_data_path",
        help="Path where to save the histogram statistics file.")
    try:
        args = parser.parse_args()
        if (args.picklefile_path):
            picklefile_path = args.picklefile_path
            if (args.save_data_path):
                display_histogram(
                    picklefile_path=picklefile_path,
                    save_data=True,
                    save_data_path=args.save_data_path)
            else:
                display_histogram(picklefile_path=picklefile_path)

    except:
        pass
