from sklearn.externals import joblib
import os
import argparse

from tools import pickle_dataset
from tools import pathconfig


def pickle_images():
    for word in ['train', 'test', 'val']:
        pickle_dataset.pickle_images(
            os.path.join(splittingfiles_path, '{}.txt'.format(word)),
            os.path.join(output_path, 'pickled_{}.pkl'.format(word)),
            dataset_name)
        print('Pickled {}'.format(word))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pickle files based on splitting.')
    parser.add_argument(
        "-splittingfiles_path",
        help="Path to files indicating the splitting (i.e. \{train, test, val\}"
        ".txt).")
    parser.add_argument(
        "-output_path", help="Path where to store pickle files.")
    parser.add_argument(
        "-dataset_name",
        help="Either train or val, indicating whether "
        "the data being pickled comes from the train or val dataset of "
        "pySceneNetRGBD.")

    args = parser.parse_args()
    if (args.splittingfiles_path and args.output_path and
            args.dataset_name):
        splittingfiles_path = args.splittingfiles_path
        output_path = args.output_path
        dataset_name = args.dataset_name
    else:
        print("Some arguments are missing. Using default ones in "
              "config_paths_and_variables.sh.")
        # Obtain paths and variables
        outputdata_path = pathconfig.obtain_paths_and_variables(
            "OUTPUTDATA_PATH")
        pickleandsplit_path = pathconfig.obtain_paths_and_variables(
            "PICKLEANDSPLIT_PATH")
        trajectory = pathconfig.obtain_paths_and_variables("TRAJ_NUM")
        dataset_name = pathconfig.obtain_paths_and_variables(
            "DATASET_NAME")
        # Compose script arguments if necessary
        splittingfiles_path = outputdata_path
        output_path = os.path.join(pickleandsplit_path, dataset_name,
                                   'traj_{}'.format(trajectory))

    pickle_images()
