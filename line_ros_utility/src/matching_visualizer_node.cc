#include <ros/ros.h>

#include <line_matching/line_matching_visualizer.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "match_and_show_lines");
  line_matching::LineDetectorDescriptorAndMatcher ls;
  ls.start();
  ros::spin();
  return 0;
}
