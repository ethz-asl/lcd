#include <ros/ros.h>
#include <sstream>

#include <line_ros_utility/line_ros_utility.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "detect_and_show_3D");
  int traj_num = 1;
  if (argc >= 2) {  // Found a number-of-trajectory argument
    std::istringstream iss(argv[1]);
    if (iss >> traj_num)
      ROS_INFO("Asked to perform detection for trajectory no.%d", traj_num);
  }
  line_ros_utility::ListenAndPublish ls(traj_num);
  ls.start();
  ros::spin();
  return 0;
}
