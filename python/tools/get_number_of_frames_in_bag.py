import rosbag
import argparse


def get_number_of_frames():
    return bag.get_message_count('/camera/rgb/image_raw')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Returns the number of frame in a ROS bag.')
    parser.add_argument(
        "-rosbag_path", help="Path to the ROS bag.", required=True)

    args = parser.parse_args()

    if args.rosbag_path:
        bag = rosbag.Bag(args.rosbag_path)
        print(get_number_of_frames())
