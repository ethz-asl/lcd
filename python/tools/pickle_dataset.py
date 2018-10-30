import os
import cv2
from collections import defaultdict
from sklearn.externals import joblib

def pickle_entire_dataset(train_file, test_file, val_file, output_folder='..'):
    """ Creates pickle files for all sets (train, test, val) in the dataset """
    pickle_images(train_file, os.path.join(output_folder,'/train.pkl'))
    pickle_images(test_file, os.path.join(output_folder,'/test.pkl'))
    pickle_images(val_file, os.path.join(output_folder,'/val.pkl'))

def pickle_images(input_text_file, output_pickle_file):
    """ Creates a pickle file containing a nested dictionary with images in
    correspondence with trajectory number, frame number, image type (rgb or
    depth) and line number """
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
            if word.find('traj_')!=-1:
                trajectory_number = int(word.split('traj_')[1])
            # Get frame number
            if word.find('frame_')!=-1:
                frame_number = int(word.split('frame_')[1])
            # Get image type
            if word.find('rgb')!=-1 or word.find('depth')!=-1:
                image_type = word
            # Get line number
            if word.find('.png')!=-1:
                line_number = int(word.split('.png')[0])
        # Get image
        try:
            img = cv2.imread(line_path, cv2.IMREAD_UNCHANGED)
        except cv2.error as e:
            print('Trying to open image {0} resulted in error {1}'.format(line_path, e))

        if img is None:
            print('Image {0} returns None'.format(line_path))

        # Create entry for image in dictionary
        data_dict[trajectory_number][frame_number][image_type][line_number]['img'] = img
        # Create entry for labels in dictionary
        data_dict[trajectory_number][frame_number][image_type][line_number]['labels'] = \
            [float(i) for i in split_line[1:]]


    # Convert defualtdict to dicts
    def defaultdict_to_dict(dictionary):
        # Found image
        if (not isinstance(dictionary, dict)) and (not isinstance(dictionary, defaultdict)):
            return dictionary
        # Higher level -> keep going down
        if isinstance(dictionary, defaultdict):
            return dict((key, defaultdict_to_dict(value)) for key, value in dictionary.items())
    data_dict = defaultdict_to_dict(data_dict)

    # Write to file
    joblib.dump(data_dict, output_pickle_file)
