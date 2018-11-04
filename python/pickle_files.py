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
            scenenetdataset_type)
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
        "-scenenetdataset_type",
        help="Either train or val, indicating whether "
        "the data being pickled comes from the train or val dataset of "
        "pySceneNetRGBD.")

    args = parser.parse_args()
    if (args.splittingfiles_path and args.output_path and
            args.scenenetdataset_type):
        splittingfiles_path = args.splittingfiles_path
        output_path = args.output_path
        scenenetdataset_type = args.scenenetdataset_type
    else:
        print("Some arguments are missing. Using default ones in "
              "config_paths_and_variables.sh.")
        # Obtain paths and variables
        outputdata_path = pathconfig.obtain_paths_and_variables(
            "OUTPUTDATA_PATH")
        pickleandsplit_path = pathconfig.obtain_paths_and_variables(
            "PICKLEANDSPLIT_PATH")
        trajectory = pathconfig.obtain_paths_and_variables("TRAJ_NUM")
        scenenetdataset_type = pathconfig.obtain_paths_and_variables(
            "DATASET_TYPE")
        # Compose script arguments if necessary
        splittingfiles_path = outputdata_path
        output_path = os.path.join(pickleandsplit_path, scenenetdataset_type,
                                   'traj_{}'.format(trajectory))

    pickle_images()
