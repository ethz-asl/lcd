import os
import cv2
from collections import defaultdict
from sklearn.externals import joblib


def pickle_images(input_text_file, output_pickle_file, dataset_name):
    """ Creates a pickle file containing a nested dictionary with images in
    correspondence with dataset_name (i.e., whether the data being
    pickled comes from the train or val dataset of pySceneNetRGBD), trajectory
    number, frame number, image type (rgb or depth) and line number """
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
            # Get trajectory number
            if word.find('traj_') != -1:
                trajectory_number = int(word.split('traj_')[1])
            # Get frame number
            if word.find('frame_') != -1:
                frame_number = int(word.split('frame_')[1])
            # Get image type
            if word.find('rgb') != -1 or word.find('depth') != -1:
                image_type = word
            # Get line number
            if word.find('.png') != -1:
                line_number = int(word.split('.png')[0])
        # Get image
        try:
            img = cv2.imread(line_path, cv2.IMREAD_UNCHANGED)
        except cv2.error as e:
            print('Trying to open image {0} resulted in error {1}'.format(
                line_path, e))
        else:
            if img is None:
                print('Image {0} returns None'.format(line_path))
            else:
                # Create entry for image in dictionary
                data_dict[dataset_name][trajectory_number][
                    frame_number][image_type][line_number]['img'] = img
                # Create entry for labels in dictionary
                data_dict[dataset_name][trajectory_number][
                    frame_number][image_type][line_number]['labels'] = \
                    [float(i) for i in split_line[1:]]

        # Current version of lines files only include paths for rbg images. The
        # following is to also load depth images
        if image_type == 'rgb':
            line_path_depth = line_path.replace('rgb', 'depth')
            # Get image
            try:
                img = cv2.imread(line_path_depth, cv2.IMREAD_UNCHANGED)
            except cv2.error as e:
                print('Trying to open image {0} resulted in error {1}'.format(
                    line_path_depth, e))
            else:
                if img is None:
                    print('Image {0} returns None'.format(line_path_depth))
                else:
                    # Create entry for image in dictionary
                    data_dict[dataset_name][trajectory_number][
                        frame_number]['depth'][line_number]['img'] = img
                    # Create entry for labels in dictionary
                    data_dict[dataset_name][trajectory_number][
                        frame_number]['depth'][line_number]['labels'] = \
                        [float(i) for i in split_line[1:]]
    # Convert defualtdict to dicts
    def defaultdict_to_dict(dictionary):
        # Found image
        if (not isinstance(dictionary, dict)) and (not isinstance(
                dictionary, defaultdict)):
            return dictionary
        # Higher level -> keep going down
        if isinstance(dictionary, defaultdict):
            return dict((key, defaultdict_to_dict(value))
                        for key, value in dictionary.items())

    data_dict = defaultdict_to_dict(data_dict)

    # Write to file
    joblib.dump(data_dict, output_pickle_file, compress=3)


def merge_pickled_dictionaries(dict_to_update, dict_to_add):
    """ Merge pickled dictionary dict_to_add in pickled dictionary
    dict_to_update, therefore modifying dict_to_update. Values from
    dict_to_update are overwritten only if dict_to_add contains an entry which
    has same dataset_name, trajectory_number and frame_number, i.e., if both
    dictionaries contain the images and labels associated to frame frame_number
    in trajectory trajectory_number from the pySceneNetRGBD dataset
    dataset_name. In all other cases, dictionaries are simply merged.
    """
    for dataset_name_dict_to_add in dict_to_add.keys():
        if dataset_name_dict_to_add in dict_to_update.keys():
            # Dataset type already in dict_to_update, check trajectory number.
            for traj_num_dict_to_add in dict_to_add[
                    dataset_name_dict_to_add].keys():
                if traj_num_dict_to_add in dict_to_update[
                        dataset_name_dict_to_add].keys():
                    # Same dataset type and trajectory_number already in
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
            # Dataset tupe not in dict_to_update, simply add it.
            dict_to_update[dataset_name_dict_to_add] = dict_to_add[
                dataset_name_dict_to_add]
