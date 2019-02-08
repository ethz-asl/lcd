import numpy as np
import os
import cv2
from collections import defaultdict
from sklearn.externals import joblib

from histogram_instances_in_pickle_file import compute_histogram


def repickle_file_with_blank_images(input_pickle_file, output_pickle_file):
    """ Repickles a pickle file, by making all the virtual-camera images blank
        (all zeros for the intensities, all zeros for the depth image).

    Args:
        input_pickle_file (string): Path of the input pickle file.
        output_pickle_file (string): Path of the output pickle file.
    """
    # Load input pickle file.
    try:
        input_pickled_dict = joblib.load(input_pickle_file)
    except IOError:
        print("Input pickle file not found at location {}.".format(
            input_pickle_file))
        return

    # Create output pickled dict as a copy of the input pickled dict.
    output_pickled_dict = input_pickled_dict

    for dataset_name in input_pickled_dict.keys():
        dataset_name_dict = input_pickled_dict[dataset_name]
        for trajectory_number in dataset_name_dict.keys():
            trajectory_number_dict = dataset_name_dict[trajectory_number]
            for frame_number in trajectory_number_dict.keys():
                frame_number_dict = trajectory_number_dict[frame_number]
                for image_type in frame_number_dict.keys():
                    image_type_dict = frame_number_dict[image_type]
                    for line_number in image_type_dict.keys():
                        # Substitute the images with blank images.
                        output_pickled_dict[dataset_name][trajectory_number][
                            frame_number][image_type][line_number][
                                'img'] = np.zeros(
                                    shape=input_pickled_dict[dataset_name]
                                    [trajectory_number][frame_number][
                                        image_type][line_number]['img'].shape,
                                    dtype=input_pickled_dict[dataset_name]
                                    [trajectory_number][frame_number][
                                        image_type][line_number]['img'].dtype)

    # Write output pickle file.
    joblib.dump(output_pickled_dict, output_pickle_file, compress=3)


def repickle_file_by_instances_occurrences(input_pickle_file,
                                           output_pickle_file,
                                           min_num_occurrences=50,
                                           max_num_occurrences=None):
    """ Repickles a pickle file, by only keeping the lines with an instance
        label that has a number of occurrences within a defined range in the
        file.

    Args:
        input_pickle_file (string): Path of the input pickle file.
        output_pickle_file (string): Path of the output pickle file.
        min_num_occurrences (int): Minimum number of occurrences that the
            instance label associated to a line should have in the input pickle
            file for the line to be included in the output pickle file.
        max_num_occurrences (int): Maximum number of occurrences that the
            instance label associated to a line should have in the input pickle
            file for the line to be included in the output pickle file.
    """
    # Compute histogram of the occurrences in the input pickle file.
    histogram = compute_histogram(input_pickle_file)
    instance_labels = histogram.keys()
    occurrences = histogram.values()

    # Set maximum number of occurrences to be one standard deviation above the
    # mean of the number of occurrences, if not set.
    if (max_num_occurrences is None):
        # Get mean number of occurrences.
        average_num_occurrences = np.mean(occurrences)
        # Get standard deviation of the number of occurrences.
        std_num_occurrences = np.std(occurrences)
        max_num_occurrences = average_num_occurrences + std_num_occurrences

    # Load input pickle file.
    try:
        input_pickled_dict = joblib.load(input_pickle_file)
    except IOError:
        print("Input pickle file not found at location {}.".format(
            input_pickle_file))
        return

    # Create output pickled dict.
    nested_dict = lambda: defaultdict(nested_dict)
    output_pickled_dict = nested_dict()

    for dataset_name in input_pickled_dict.keys():
        dataset_name_dict = input_pickled_dict[dataset_name]
        for trajectory_number in dataset_name_dict.keys():
            trajectory_number_dict = dataset_name_dict[trajectory_number]
            for frame_number in trajectory_number_dict.keys():
                frame_number_dict = trajectory_number_dict[frame_number]
                for image_type in frame_number_dict.keys():
                    image_type_dict = frame_number_dict[image_type]
                    for line_number in image_type_dict.keys():
                        line_number_dict = image_type_dict[line_number]
                        # Check number of occurrences of the instance label.
                        num_occurrence_instance = histogram[line_number_dict[
                            'labels'][-1]]
                        if (min_num_occurrences <= num_occurrence_instance <=
                                max_num_occurrences):
                            output_pickled_dict[dataset_name][
                                trajectory_number][frame_number][
                                    image_type][line_number].update(
                                        input_pickled_dict[dataset_name][
                                            trajectory_number][frame_number][
                                                image_type][line_number])
    # Convert defaultdict to dicts.
    def defaultdict_to_dict(dictionary):
        # Found image.
        if (not isinstance(dictionary, dict)) and (not isinstance(
                dictionary, defaultdict)):
            return dictionary
        # Higher level -> keep going down.
        if isinstance(dictionary, defaultdict):
            return dict((key, defaultdict_to_dict(value))
                        for key, value in dictionary.items())

    output_pickled_dict = defaultdict_to_dict(output_pickled_dict)

    # Write output pickle file.
    joblib.dump(output_pickled_dict, output_pickle_file, compress=3)


def pickle_images(input_text_file, output_pickle_file, dataset_name):
    """ Creates a pickle file containing a nested dictionary with images in
        correspondence with dataset_name ('val' or 'train_NUM' - where 'NUM' is
        a number between 0 and 16 - if the data comes from the SceneNetRGBD
        dataset, 'scenenn' if the data comes from the SceneNN dataset),
        trajectory number, frame number, image type (rgb or depth) and line
        number.

    Args:
        input_text_file (string): Path of the input text file that contains the
            list of frames to include in the pickle file, together with the
            associated images and lines.
        output_pickle_file (string): Path of the output pickle file.
        dataset_name (string): Dataset name.
    """
    nested_dict = lambda: defaultdict(nested_dict)
    data_dict = nested_dict()

    with open(input_text_file) as f:
        file_lines = f.readlines()
    file_lines = [line.strip() for line in file_lines]

    for line in file_lines:
        split_line = line.split()
        line_path = split_line[0]
        line_path_split = line_path.split('/')
        for word in line_path_split:
            # Get trajectory number.
            if word.find('traj_') != -1:
                trajectory_number = int(word.split('traj_')[1])
            # Get frame number.
            if word.find('frame_') != -1:
                frame_number = int(word.split('frame_')[1])
            # Get image type.
            if word.find('rgb') != -1 or word.find('depth') != -1:
                image_type = word
            # Get line number.
            if word.find('.png') != -1:
                line_number = int(word.split('.png')[0])
        # Get image.
        try:
            img = cv2.imread(line_path, cv2.IMREAD_UNCHANGED)
        except cv2.error as e:
            print('Trying to open image {0} resulted in error {1}'.format(
                line_path, e))
        else:
            if img is None:
                print('Image {0} returns None'.format(line_path))
            else:
                # Create entry for image in dictionary.
                data_dict[dataset_name][trajectory_number][frame_number][
                    image_type][line_number]['img'] = img
                # Create entry for labels in dictionary.
                data_dict[dataset_name][trajectory_number][
                    frame_number][image_type][line_number]['labels'] = \
                    [float(i) for i in split_line[1:-2]] + \
                    [float(split_line[-1])]
                # Create entry for line type in dictionary.
                data_dict[dataset_name][trajectory_number][
                    frame_number][image_type][line_number]['line_type'] = \
                    float(split_line[-2])

        # Current version of lines files only include paths for RGB images. The
        # following is to also load depth images.
        if image_type == 'rgb':
            line_path_depth = line_path.replace('rgb', 'depth')
            # Get image.
            try:
                img = cv2.imread(line_path_depth, cv2.IMREAD_UNCHANGED)
            except cv2.error as e:
                print('Trying to open image {0} resulted in error {1}'.format(
                    line_path_depth, e))
            else:
                if img is None:
                    print('Image {0} returns None'.format(line_path_depth))
                else:
                    # Create entry for image in dictionary.
                    data_dict[dataset_name][trajectory_number][frame_number][
                        'depth'][line_number]['img'] = img
                    # Create entry for labels in dictionary.
                    data_dict[dataset_name][trajectory_number][
                        frame_number]['depth'][line_number]['labels'] = \
                        [float(i) for i in split_line[1:-2]] + \
                        [float(split_line[-1])]
                    # Create entry for line type in dictionary.
                    data_dict[dataset_name][trajectory_number][
                        frame_number]['depth'][line_number]['line_type'] = \
                        float(split_line[-2])
    # Convert defaultdict to dicts.
    def defaultdict_to_dict(dictionary):
        # Found image.
        if (not isinstance(dictionary, dict)) and (not isinstance(
                dictionary, defaultdict)):
            return dictionary
        # Higher level -> keep going down.
        if isinstance(dictionary, defaultdict):
            return dict((key, defaultdict_to_dict(value))
                        for key, value in dictionary.items())

    data_dict = defaultdict_to_dict(data_dict)

    # Write to file.
    joblib.dump(data_dict, output_pickle_file, compress=3)


def merge_pickled_dictionaries(dict_to_update, dict_to_add):
    """ Merges pickled dictionary dict_to_add in pickled dictionary
        dict_to_update, therefore modifying dict_to_update. Values from
        dict_to_update are overwritten only if dict_to_add contains an entry
        which  has same dataset_name, trajectory_number and frame_number, i.e.,
        if both dictionaries contain the images and labels associated to frame
        frame_number in trajectory trajectory_number from the SceneNetRGBD
        dataset dataset_name. In all other cases, dictionaries are simply
        merged.
    """
    for dataset_name_dict_to_add in dict_to_add.keys():
        if dataset_name_dict_to_add in dict_to_update.keys():
            # Dataset name already in dict_to_update, check trajectory number.
            for traj_num_dict_to_add in dict_to_add[
                    dataset_name_dict_to_add].keys():
                if traj_num_dict_to_add in dict_to_update[
                        dataset_name_dict_to_add].keys():
                    # Same dataset name and trajectory_number already in
                    # dict_to_update, merge all frames.
                    dict_to_update[dataset_name_dict_to_add][
                        traj_num_dict_to_add].update(dict_to_add[
                            dataset_name_dict_to_add][traj_num_dict_to_add])
                else:
                    # Trajectory number not in dict_to_update, simply add it.
                    dict_to_update[dataset_name_dict_to_add][
                        traj_num_dict_to_add] = dict_to_add[
                            dataset_name_dict_to_add][traj_num_dict_to_add]
        else:
            # Dataset name not in dict_to_update, simply add it.
            dict_to_update[dataset_name_dict_to_add] = dict_to_add[
                dataset_name_dict_to_add]
