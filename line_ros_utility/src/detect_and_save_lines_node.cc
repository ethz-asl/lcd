#include <sstream>

#include <ros/ros.h>

#include <line_ros_utility/line_ros_utility.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "detect_and_save_lines");
  std::string traj_num;
  std::string write_path;
  int start_frame;
  int frame_step;
  std::istringstream iss;
  if (argc >= 5) {  // Found number-of-trajectory, write-path, start frame and
                    // frame step arguments.
    // Number of trajectory.
    traj_num.assign(argv[1]);
    ROS_INFO("Asked to perform detection for trajectory no.%s.",
             traj_num.c_str());
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
    // Frame step. It indicates the step (in number of frames of the original
    // trajectory) between a frame in the ROS bag and the subsequent one. For
    // instance, with frame_step = 3 and start_frame = 1, the actual indices of
    // the frames received are 1, 4, 7, etc.
    iss.clear();
    iss.str(argv[4]);
    if (iss >> frame_step) {
      ROS_INFO("Asked to use a frame step of %d.", frame_step);
    } else {
      ROS_ERROR("Unable to parse frame step argument.");
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
    frame_step = 1;
    ROS_INFO("NOTE: using frame step 1. This might not be coherent if using "
             "SceneNN dataset.");
    ROS_INFO("traj_num is %d, write_path is %s", traj_num, write_path.c_str());
  }
  line_ros_utility::ListenAndPublish ls(traj_num, write_path, start_frame,
                                        frame_step);
  ls.start();
  ros::spin();
  return 0;
}
