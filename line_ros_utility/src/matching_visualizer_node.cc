#include "line_detect_describe_and_match/line_detect_describe_and_match.h"

#include <line_detection/line_detection.h>
#include <line_description/line_description.h>

#include <sstream>

#include <ros/ros.h>

int main(int argc, char** argv) {
  unsigned int detector_type_num, descriptor_type_num;
  line_detection::DetectorType detector_type;
  line_description::DescriptorType descriptor_type;
  std::istringstream iss;
  if (argc >= 3) {  // Found detector type and descriptor type arguments.
    // Detector type.
    iss.str(argv[1]);
    if (iss >> detector_type_num) {
      switch (detector_type_num) {
        case 0:
          detector_type = line_detection::DetectorType::LSD;
          ROS_INFO("Asked to use LSD detector.");
          break;
        case 1:
          detector_type = line_detection::DetectorType::EDL;
          ROS_INFO("Asked to use EDL detector.");
          break;
        case 2:
          detector_type = line_detection::DetectorType::FAST;
          ROS_INFO("Asked to use FAST detector.");
          break;
        case 3:
          detector_type = line_detection::DetectorType::HOUGH;
          ROS_INFO("Asked to use HOUGH detector.");
          break;
        default:
          ROS_ERROR("Invalid detector type provided. Valid values are integer "
                    "numbers from 0 to 3, that correspond respectively to LSD, "
                    "EDL, FAST and HOUGH.");
          return -1;
      }
    } else {
      ROS_ERROR("Unable to parse detector type argument.");
      return -1;
    }
    // Descriptor type.
    iss.clear();
    iss.str(argv[2]);
    if (iss >> descriptor_type_num) {
      switch (descriptor_type_num) {
        case 0:
          descriptor_type = line_description::DescriptorType::EMBEDDING_NN;
          ROS_INFO("Asked to use neural-network embeddings as descriptor.");
          break;
        case 1:
          descriptor_type = line_description::DescriptorType::BINARY;
          ROS_INFO("Asked to use binary descriptor.");
          break;
        default:
          ROS_ERROR("Invalid detector type provided. Valid values are integer "
                    "numbers from 0 to 1, that correspond respectively to "
                    "neural network embeddings and binary descriptor.");
          return -1;
      }
    } else {
      ROS_ERROR("Unable to parse descriptor type argument.");
      return -1;
    }
  }
  ros::init(argc, argv, "detect_describe_and_match_lines");
  line_ros_utility::LineDetectorDescriptorAndMatcher ls(detector_type,
                                                        descriptor_type);
  ls.start();
  ros::spin();
  return 0;
}
