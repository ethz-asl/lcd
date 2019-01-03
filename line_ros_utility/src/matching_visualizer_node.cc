#include "line_detect_describe_and_match/line_detect_describe_and_match.h"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "detect_describe_and_match_lines");
  line_ros_utility::LineDetectorDescriptorAndMatcher ls;
  ls.start();
  ros::spin();
  return 0;
}
