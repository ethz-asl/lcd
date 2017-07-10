#include <ros/ros.h>

#include <line_ros_utility/line_ros_utility.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "detect_and_show_3D");
  line_ros_utility::ListenAndPublish ls;
  ls.start();
  ros::spin();
  return 0;
}
