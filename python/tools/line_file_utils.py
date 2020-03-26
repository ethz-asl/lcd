import numpy as np


def read_start_point(line):
    """ Reads a line of a file line (in the format of an array) and returns its
        start point.

    Args:
        line (numpy array of shape (22, )): The format of the line is the
            following:
                [start point (3, ), end point (3, ),
                 left plane hessian form (4, ), right plane hessian form(4, ),
                 left color (3, ), right color(3, ), line type (1, ),
                 instance label(1, ), normal 1 (3, ), normal 2 (3, ), start open (1, ),
                 end open (1, ), camera origin (3, ), camera rotation (4, )].

    Returns:
        start_point
    """
    return line[0:3]


def read_end_point(line):
    """ Reads a line of a file line (in the format of an array) and returns its
        end point.
    Returns:
        end_point
    """
    return line[3:6]


def read_type(line):
    """ Reads a line of a file line (in the format of an array) and returns its
        type.
    Returns:
        type
    """
    return int(line[20])


def read_label(line):
    """ Reads a line of a file line (in the format of an array) and returns its
        label.
    Returns:
        label
    """
    return int(line[21])


def read_normal_1(line):
    """ Reads a line of a file line (in the format of an array) and returns its
        first normal.
    Returns:
        left normal
    """
    return line[22:25]


def read_normal_2(line):
    """ Reads a line of a file line (in the format of an array) and returns its
        second normal.

    Returns:
        right normal
    """
    return line[25:28]


def read_start_open(line):
    """ Reads a line of a file line (in the format of an array) and returns
        if the start point is open.

    Returns:
        start_open
    """
    return bool(line[28])


def read_end_open(line):
    """ Reads a line of a file line (in the format of an array) and returns
        if the end point is open.

    Returns:
        end_open
    """
    return bool(line[29])


def read_camera_origin(line):
    """ Reads a line of a file line (in the format of an array) and returns
        the origin of the camera.

    Returns:
        camera_origin
    """
    return line[30:33]


def read_camera_rotation(line):
    """ Reads a line of a file line (in the format of an array) and returns
        the origin of the camera.

    Returns:
        camera_rotation
    """
    return line[33:37]
