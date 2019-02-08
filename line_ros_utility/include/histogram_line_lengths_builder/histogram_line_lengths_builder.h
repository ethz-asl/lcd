#ifndef LINE_ROS_UTILITY_HISTOGRAM_LINE_LENGTHS_BUILDER_H_
#define LINE_ROS_UTILITY_HISTOGRAM_LINE_LENGTHS_BUILDER_H_

#include "line_detection/line_detection.h"

#include "line_detection/ExtractLines.h"
#include "line_ros_utility/LineLengthsArray.h"

#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <ros/ros.h>

namespace line_ros_utility {

typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>
      MySyncPolicy;

class HistogramLineLengthsBuilder {
 public:
   HistogramLineLengthsBuilder(
       line_detection::DetectorType detector_type=
           line_detection::DetectorType::LSD);
   ~HistogramLineLengthsBuilder();
   // Starts listening to the input messages.
   void start();
 protected:

 private:
   // Type of detector to use.
   line_detection::DetectorType detector_type_;
   // Publisher.
   ros::Publisher line_lengths_pub_;
   // Subscribers.
   message_filters::Synchronizer<MySyncPolicy>* sync_;
   message_filters::Subscriber<sensor_msgs::Image> image_sub_;
   message_filters::Subscriber<sensor_msgs::Image> cloud_sub_;
   message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
   // Node handle.
   ros::NodeHandle node_handle_;
   // Services.
   line_detection::ExtractLines service_extract_lines_;
   // Service clients.
   ros::ServiceClient client_extract_lines_;

  // Given an input image (and point cloud), detects lines in the image and
  // builds a histogram of the lengths of the lines.
  // Input: image_rgb_msg:              ROS message containing the RGB image
  //                                    from which to detect lines (it gets
  //                                    converted to grayscale by the
  //                                    line_extractor node).
  //
  //        cloud_msg:                  ROS message containing the point cloud
  //                                    image, in the format CV32FC3, in which
  //                                    each pixel has 3 channels that
  //                                    that correspond to the x, y and z
  //                                    coordinate of the 3D point shown at that
  //                                    pixel.
  //
  //        camera_info_msg:            ROS message containing the camera info.
  //
  //        camera_to_world_matrix_msg: ROS message containing the
  //                                    camera-to-world matrix.
   void buildHistogram(const sensor_msgs::ImageConstPtr& image_rgb_msg,
                       const sensor_msgs::ImageConstPtr& cloud_msg,
                       const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

   // Given an input image and point cloud, detect lines in the image and
   // returns them, together with an index to be assigned to the current frame.
   // Input: image_rgb_msg:   ROS message containing the RGB image from which
   //                         to detect lines (it gets converted to grayscale
   //                         by the line_extractor node).
   //
   //        cloud_msg:       ROS message containing the point cloud image, in
   //                         the format CV32FC3, in which each pixel has 3
   //                         channels that correspond to the x, y and z
   //                         coordinate of the 3D point shown at that pixel.
   //
   //        camera_info_msg: ROS message containing the camera info.
   //
   // Output: lines: Lines detected in the input image, both in 2D and 3D, and
   //                with their inlier planes and types.
   void detectLines(const sensor_msgs::ImageConstPtr& image_rgb_msg,
                    const sensor_msgs::ImageConstPtr& cloud_msg,
                    const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
                    std::vector<line_detection::Line2D3DWithPlanes>* lines);

   // Helper function that subscribes to the input topics.
   void subscribeToInputTopics();

   // Callback for the ROS messages that store the RGB image, the point cloud
   // and the camera info.
   void callback(
       const sensor_msgs::ImageConstPtr& rosmsg_image,
       const sensor_msgs::ImageConstPtr& rosmsg_cloud,
       const sensor_msgs::CameraInfoConstPtr& rosmsg_camera_info);
};

}  // namespace line_ros_utility

#endif  // LINE_ROS_UTILITY_HISTOGRAM_LINE_LENGTHS_BUILDER_H_
