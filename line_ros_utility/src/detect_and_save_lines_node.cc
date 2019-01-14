#include <sstream>

#include <ros/ros.h>

#include <line_ros_utility/line_ros_utility.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "detect_and_save_lines");
  int traj_num;
  std::string write_path;
  int start_frame;
  if (argc >= 4) {  // Found number-of-trajectory, write-path and start frame
                    // arguments.
    // Number of trajectory.
    std::istringstream iss(argv[1]);
    if (iss >> traj_num) {
      ROS_INFO("Asked to perform detection for trajectory no.%d.", traj_num);
    } else {
      ROS_ERROR("Unable to parse trajectory number argument.");
      return -1;
    }
    // Write path.
    write_path.assign(argv[2]);
    ROS_INFO("Using %s as output path", write_path.c_str());
    // Start frame. Since ROS bags from SceneNN start from frame 2 instead of
    // frame 0 (cf. scenenn_ros_tools/scenenn_to_rosbag.py) an argument that
    // handles the index assigned to the first frame is required.
    iss.clear();
    iss.str(argv[3]);
    if (iss >> start_frame) {
      ROS_INFO("Asked to label the first frame as %d.", start_frame);
    } else {
      ROS_ERROR("Unable to parse start frame argument.");
      return -1;
    }
  } else {
    ROS_INFO("Not enough arguments. Using default paths and variables in "
              "python/config_paths_and_variables.sh.");
    std::string linesandimagesfolder_path, dataset_name;
    if (!line_ros_utility::getDefaultPathsAndVariables("TRAJ_NUM", &traj_num)) {
      ROS_ERROR("Error in retrieving default argument TRAJ_NUM. Exiting."
                "Got TRAJ_NUM=%d", traj_num);
      return -1;
    }
    if (!line_ros_utility::getDefaultPathsAndVariables(
                                                  "LINESANDIMAGESFOLDER_PATH",
                                                  &linesandimagesfolder_path)) {
      ROS_ERROR("Error in retrieving default argument "
                "LINESANDIMAGESFOLDER_PATH. Exiting.");
      return -1;
    }
    if (!line_ros_utility::getDefaultPathsAndVariables("DATASET_NAME",
                                                  &dataset_name)){
      ROS_ERROR("Error in retrieving default argument DATASET_NAME. Exiting.");
      return -1;
    }
    write_path = linesandimagesfolder_path + "/" + dataset_name + "_lines/";
    start_frame = 0;
    ROS_INFO("NOTE: using start frame 0. This might not be coherent if using "
             "SceneNN dataset.");
    ROS_INFO("traj_num is %d, write_path is %s", traj_num, write_path.c_str());
  }
  line_ros_utility::ListenAndPublish ls(traj_num, write_path, start_frame);
  ls.start();
  ros::spin();
  return 0;
}
