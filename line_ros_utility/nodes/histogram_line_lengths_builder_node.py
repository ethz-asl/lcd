#!/usr/bin/env python
import rospy
from line_ros_utility.msg import LineLengthsArray
from line_detection.srv import ExtractLines

from tools.histogram_line_lengths import LineLengthHistogram


class LineLengthHistogramBuilder:
    """ Creates a histogram of the length of the lines extracted in a dataset by
        using the service ExtractLines.

    Args:
        num_frames (int): Number of frames expected in the dataset.
    Attributes:
        histogram_displayer (LineLengthHistogram): Instance of the histogram to
            be used to store the lines.
        num_frames (int): Number of frames expected in the dataset.
        frames_received (int): Number of frames already received.
    """

    def __init__(self, num_frames):
        self.histogram_displayer = LineLengthHistogram(num_frames=num_frames)
        self.num_frames = num_frames
        self.frames_received = 0

    def callback(self, line_lengths_msg):
        # Increment the number of frames received.
        self.frames_received += 1
        # Add frame to the histogram.
        self.histogram_displayer.add_frame_to_histogram(
            list(line_lengths_msg.line_lengths))
        if (self.frames_received == self.num_frames):
            # Display histogram if all frames have been received.
            self.histogram_displayer.display_histogram()

    def start(self):
        # Start ROS node.
        rospy.init_node('histogram_line_lengths_displayer')
        # Subscribe to line lengths topic.
        rospy.Subscriber(
            '/line_lengths', LineLengthsArray, self.callback, queue_size=300)

        rospy.spin()


if __name__ == "__main__":
    histogram_builder = LineLengthHistogramBuilder(num_frames=300)
    histogram_builder.start()
