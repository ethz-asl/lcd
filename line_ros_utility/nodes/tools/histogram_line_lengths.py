""" Draws a histogram of the length of all the lines in the dataset.
"""
import matplotlib.pyplot as plt
import numpy as np


class LineLengthHistogram:
    """ Creates a histograms of the lengths of all the lines in the dataset and
        displays it.
    Args:
        num_frames (int): Number of frames expected in the dataset.
    Attributes:
        num_frames (int): Number of frames expected in the dataset.
        frames_received (int): Number of frames already received.
        line_lengths (dict of list of float): line_lengths[i][j] contains the
            length of the j-th line in the i-th frame of the dataset.
    """

    def __init__(self, num_frames):
        assert (num_frames >= 0)
        self.num_frames = num_frames
        # Initialize the dictionary with the length of the lines.
        self.line_lengths = dict()
        for frame in range(num_frames):
            self.line_lengths[frame] = []
        # Set the number of frames already received to be 0.
        self.frames_received = 0

    def add_frame_to_histogram(self, line_lengths):
        """ Adds the lines in a newly received frame to the histogram.
        Args:
            line_lengths (list of float): line_lengths[i] contains the length of
                the i-th line in the newly received frame.
        """
        if (self.frames_received >= self.num_frames):
            print("Error: trying to add a frame when the expected {} were ".
                  format(self.num_frames) + "already received.")
            return
        else:
            print("Added frame {} to the histogram.".format(
                self.frames_received))
        # Insert the lines in the histogram.
        self.line_lengths[self.frames_received] = line_lengths
        # Increment the index of the current frame.
        self.frames_received += 1

    def display_histogram(self, bin_size=0.02):
        """ Displays the histogram of the received lines.
        Args:
            bin_size (float): Size of the bin in meters (i.e., step for sampling
                between the length of the shortest line and the length of the
                longest line).
        """
        if (self.frames_received < self.num_frames):
            print("Error: trying to generate the histogram, but {} more ".
                  format(self.num_frames - self.frames_received) +
                  "frames were expected.")
            return

        hist_line_lengths = []

        num_lines = 0

        # Add all frames to the histogram.
        for frame_idx in range(self.num_frames):
            hist_line_lengths += self.line_lengths[frame_idx]
            num_lines += len(self.line_lengths[frame_idx])
        # Find maximum and minimum length in the dataset.
        min_length = min(hist_line_lengths)
        max_length = max(hist_line_lengths)

        print("Total number of lines is {}".format(num_lines))

        plt.figure("Histogram of line lengths")
        plt.hist(
            hist_line_lengths,
            bins=np.arange(min_length - bin_size / 2.,
                           max_length + bin_size / 2., bin_size),
            density=True)
        plt.gca().set_xticklabels(
            ['{:.2f}'.format(x) for x in plt.gca().get_xticks()])
        plt.xlabel("Length of the lines [m]")
        plt.ylabel("Normalized frequency")
        plt.title("Total number of lines: {}. ".format(num_lines) +
                  "Min length: {0:.3f} m, max length: {1:.3f} m".format(
                      min_length, max_length))

        plt.show()
