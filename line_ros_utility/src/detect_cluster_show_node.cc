#include <sstream>

#include <ros/ros.h>

#include <line_ros_utility/line_ros_utility.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "detect_and_show_3D");
  int traj_num = 1;
  std::string write_path;
  if (argc >= 3) {  // Found number-of-trajectory and write-path arguments
    // Number of trajectory
    std::istringstream iss(argv[1]);
    if (iss >> traj_num) {
      ROS_INFO("Asked to perform detection for trajectory no.%d", traj_num);
    } else {
      ROS_ERROR("Unable to parse trajectory number argument.");
      return -1;
    }
    // Write path
    write_path.assign(argv[2]);
    ROS_INFO("Using %s as output path", write_path.c_str());
  } else {
    ROS_INFO("Not enough arguments. Using default paths and variables in "
              "python/config_paths_and_variables.sh.");
    std::string outputdata_path, dataset_type;
    if (!line_ros_utility::getDefaultPathsAndVariables("TRAJ_NUM", &traj_num)) {
      ROS_ERROR("Error in retrieving default argument TRAJ_NUM. Exiting."
                "Got TRAJ_NUM=%d", traj_num);
      return -1;
    }
    if (!line_ros_utility::getDefaultPathsAndVariables("OUTPUTDATA_PATH",
                                                  &outputdata_path)) {
      ROS_ERROR("Error in retrieving default argument OUTPUTDATA_PATH. "
                "Exiting.");
      return -1;
    }
    if (!line_ros_utility::getDefaultPathsAndVariables("DATASET_TYPE",
                                                  &dataset_type)){
      ROS_ERROR("Error in retrieving default argument DATASET_TYPE. Exiting.");
      return -1;
    }
    write_path = outputdata_path + "/" + dataset_type + "_lines/";
    ROS_INFO("traj_num is %d, write_path is %s", traj_num, write_path.c_str());
  }
  line_ros_utility::ListenAndPublish ls(traj_num, write_path);
  ls.start();
  ros::spin();
  return 0;
}
