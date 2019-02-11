""" To configure path variables.
"""
import sys
import os
import subprocess

file_dir = os.path.dirname(os.path.abspath(__file__))


def obtain_paths_and_variables(name, verbose=False):
    """ Reads the current paths and variables from the package and returns the
        value of the path/variable 'name' if present.
    """
    # Execute script that produces the paths_and_variables.txt file.
    subprocess.call(
        os.path.abspath(
            os.path.join(file_dir,
                         '../../print_paths_and_variables_to_file.sh')))
    # Read file
    with open(
            os.path.abspath(
                os.path.join(file_dir, '../../paths_and_variables.txt'))) as f:
        file_lines = f.readlines()
    for line in file_lines:
        line_split = line.split(" ", 1)
        variable_name = line_split[0]
        variable_value = line_split[1].split('\n')[0]
        if verbose:
            print('Found variable {0} with value {1}.'.format(
                variable_name, variable_value))
        if variable_name == name:
            if name == "TRAJ_NUM":
                return int(variable_value)
            else:
                return variable_value
    print("Variable {} not found.".format(name) + " Please check list of valid "
          "variables in '../../config_paths_and_variables.sh'")
