#ifndef LINE_ROS_UTILITY_LINE_DETECT_DESCRIBE_AND_MATCH_H_
#define LINE_ROS_UTILITY_LINE_DETECT_DESCRIBE_AND_MATCH_H_

#include "line_description/line_description.h"
#include "line_detection/line_detection.h"
#include "line_matching/line_matching.h"

#include "line_detection/ExtractLines.h"
#include "line_description/ImageToEmbeddings.h"
#include "line_description/LineToVirtualCameraImage.h"

#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>

namespace line_ros_utility {

typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>
    MySyncPolicy;

class LineDetectorDescriptorAndMatcher {
 public:
   LineDetectorDescriptorAndMatcher(line_detection::Detector detector_type=LSD);
   ~LineDetectorDescriptorAndMatcher();
   // Starts listening to the input messages.
   void start();
   // Shows matches between consecutive frames in the input ROS messages.
   void showMatches();
 protected:

 private:
   // Type of detector to use.
   line_detection::Detector detector_type_;
   // Current frame.
   line_matching::FrameWithIndex current_frame_;
   // Instance of LineMatcher used to do the matching.
   line_matching::LineMatcher line_matcher_;
   // Data from the ROS messages.
   cv::Mat cv_image_;
   cv::Mat cv_cloud_;
   sensor_msgs::CameraInfoConstPtr camera_info_;
   // Subscribers.
   message_filters::Synchronizer<MySyncPolicy>* sync_;
   message_filters::Subscriber<sensor_msgs::Image> image_sub_;
   message_filters::Subscriber<sensor_msgs::Image> cloud_sub_;
   message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
   // Node handle.
   ros::NodeHandle node_handle_;
   // Service clients.
   ros::ServiceClient client_extract_lines_;
   ros::ServiceClient client_image_to_embeddings_;
   ros::ServiceClient client_line_to_virtual_camera_image_;

   // Given an input image and point cloud, detect lines in the image, retrieves
   // the embeddings and saves the result in the current frame.
   // Input: image_rgb:   RGB image from which to detect lines (it gets
   //                     converted to grayscale by the line_extractor node).
   //
   //        cloud:       Point cloud image, in the format CV32FC3, in which
   //                     each pixel has 3 channels that corresponds to the x, y
   //                     and z coordinate of the 3D point shown at that pixel.
   //
   //        camera_info: ROS message containing the camera info. It is left as
   //                     a message so as not to force a specific type of camera
   //                     in the interface.
   void saveLinesWithEmbeddings(
       const cv::Mat& image_rgb, const cv::Mat& cloud,
       const sensor_msgs::CameraInfoConstPtr& camera_info);

   // Given an input image and point cloud, detect lines in the image and
   // returns them, together with an index to be assigned to the current frame.
   // Input: image_rgb:    RGB image from which to detect lines (it gets
   //                      converted to grayscale by the line_extractor node).
   //
   //        cloud:        Point cloud image, in the format CV_32FC3, in which
   //                      each pixel has 3 channels that correspond to the x, y
   //                      and z coordinate of the 3D point shown at that pixel.
   //
   //        camera_info:  ROS message containing the camera info.
   //
   // Output: lines:       Lines detected in the input image, both in 2D and 3D,
   //                      and with their inlier planes and types.
   //
   //         frame_index: Index to be assigned to the current frame.
   void detectLines(const cv::Mat& image_rgb, const cv::Mat& cloud,
                    const sensor_msgs::CameraInfoConstPtr& camera_info,
                    std::vector<line_detection::Line2D3DWithPlanes>* lines,
                    int* frame_index);

   // Given a line (both in 2D and 3D, with the inlier planes and type) and a
   // coloured point cloud, returns the descriptor (embedding) associated to
   // that line.
   // Input: line:           Line the descriptor of which should be retrieved.
   //
   //        coloured_cloud: Coloured point cloud, in the format CV_32FC(6), in
   //                        which each pixel has 6 channels that correspond to
   //                        the x, y, z coordinate and the R, G, B color of the
   //                        3D point shown at that pixel.
   //
   // Output: embeddings:    Descriptor (embedding) for the input line.
   void getEmbeddings(const line_detection::Line2D3DWithPlanes& line,
                      const cv::Mat& coloured_cloud,
                      std::vector<float>* embedding);

   // Given a set of lines and their descriptors, as well as the RGB image from
   // which lines were extracted, saves them as a new frame with frame index
   // given as input (if no other frames with the same index were previously
   // received).
   // Input: lines:       Set of lines (both in 2D and 3D, with their inlier
   //                     planes and types) detected.
   //        embeddings:  Descriptors (embeddings) associated to the lines.
   //
   //        rgb_image:   RGB image from which the lines were extracted.
   //
   //        frame_index: Index to assign to the new frame.
   //
   // Output: return:     False if a frame with the same frame index was already
   //                     received, true otherwise.
   bool saveFrame(const std::vector<line_detection::Line2D3DWithPlanes>& lines,
                  const std::vector<line_description::Embedding>& embeddings,
                  const cv::Mat& rgb_image, int frame_index);

   // Callback for the ROS messages that store the RGB image, the point cloud
   // and the camera info.
   void callback(const sensor_msgs::ImageConstPtr& rosmsg_image,
                 const sensor_msgs::ImageConstPtr& rosmsg_cloud,
                 const sensor_msgs::CameraInfoConstPtr& camera_info);
};

}  // namespace line_ros_utility

#endif  // LINE_ROS_UTILITY_LINE_DETECT_DESCRIBE_AND_MATCH_H_
