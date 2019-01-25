#include "histogram_line_lengths_builder/histogram_line_lengths_builder.h"

#include <line_detection/line_detection.h>

#include <sstream>

#include <ros/ros.h>

int main(int argc, char** argv) {
  unsigned int detector_type_num;
  line_detection::DetectorType detector_type;
  std::unique_ptr<line_ros_utility::HistogramLineLengthsBuilder>
      histogram_builder;
  std::istringstream iss;
  if (argc >= 2) {  // Found detector type and descriptor type arguments.
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
    // Initialize node.
    ros::init(argc, argv, "create_line_lengths_histogram");
    histogram_builder =
        std::unique_ptr<line_ros_utility::HistogramLineLengthsBuilder>(
          new line_ros_utility::HistogramLineLengthsBuilder(detector_type));
  } else {
    // Initialize node. Use default argument for histogram builder.
    ros::init(argc, argv, "create_line_lengths_histogram");
    histogram_builder =
        std::unique_ptr<line_ros_utility::HistogramLineLengthsBuilder>(
          new line_ros_utility::HistogramLineLengthsBuilder());
  }
  histogram_builder->start();
  ros::spin();
  return 0;
}
