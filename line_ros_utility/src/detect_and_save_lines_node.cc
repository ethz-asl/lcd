#include <sstream>

#include <ros/ros.h>

#include <line_ros_utility/line_ros_utility.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "detect_and_save_lines");
  std::string traj_num = "";
  std::string write_path = "";
  int start_frame = 0;
  int frame_step = 1;
  line_ros_utility::ListenAndPublish ls(traj_num, write_path, start_frame,
                                        frame_step);
  ls.start();
  ros::spin();
  return 0;
}
