""" The following script reads the 'database' containing the correspondences
    between a DATASET_NAME and a protobuf file. If called directly as a Python
    script it prints the result of the query if a match is found and nothing if
    no dataset name is given as input, if the protobuf name corresponding to the
    dataset name is not found or if the the configuration file is not found.
"""
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def get_protobuf_path(dataset_name):
    try:
        with open(os.path.join(parent_dir, 'config_protobuf_paths')) as f:
            lines = f.readlines()
            for line in lines:
                # Exclude comments.
                if line[0] != '#':
                    line_split = line.split(':')
                    name = line_split[0]
                    # Remove leading whitespaces in path if any.
                    path = line_split[1].lstrip()
                    # Remove terminating new line characters if any.
                    path = path.rstrip("\n\r")
                    if dataset_name == name:
                        return path
        return None
    except IOError:
        sys.stderr.write("Configuration file containing protobuf paths not "
                         "found. Please make sure that this script is in "
                         "python/tools and that the python folder contains the "
                         "configuration file config_protobuf_paths.")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get protobuf path from a dataset name.')
    parser.add_argument("-dataset_name", help="Dataset name.")
    args = parser.parse_args()
    if (args.dataset_name):
        path = get_protobuf_path(args.dataset_name)
        if path is not None:
            print path
