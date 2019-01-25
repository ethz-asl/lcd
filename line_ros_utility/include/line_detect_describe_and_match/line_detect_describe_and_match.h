#ifndef LINE_ROS_UTILITY_LINE_DETECT_DESCRIBE_AND_MATCH_H_
#define LINE_ROS_UTILITY_LINE_DETECT_DESCRIBE_AND_MATCH_H_

#include "line_description/line_description.h"
#include "line_detection/line_detection.h"
#include "line_matching/line_matching.h"

#include "line_detection/ExtractLines.h"
#include "line_detection/ExtractKeyLines.h"
#include "line_description/EmbeddingsRetrieverReady.h"
#include "line_description/ImageToEmbeddings.h"
#include "line_description/LineToVirtualCameraImage.h"
#include "line_description/KeyLineToBinaryDescriptor.h"

#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <ros/ros.h>

namespace line_ros_utility {

typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,
    geometry_msgs::TransformStamped> MySyncPolicy;

class LineDetectorDescriptorAndMatcher {
 public:
   LineDetectorDescriptorAndMatcher(
       line_detection::DetectorType detector_type=
           line_detection::DetectorType::LSD,
       line_description::DescriptorType descriptor_type=
           line_description::DescriptorType::EMBEDDING_NN);
   ~LineDetectorDescriptorAndMatcher();
   // Starts listening to the input messages.
   void start();
   // Shows matches between consecutive frames in the input ROS messages.
   void showMatches();
 protected:

 private:
   // Type of detector to use.
   line_detection::DetectorType detector_type_;
   // Type of descriptor to use.
   line_description::DescriptorType descriptor_type_;
    // Instance of LineMatcher used to do the matching.
   line_matching::LineMatcher line_matcher_;
   // Subscribers.
   // * Binary descriptor only requires the 2D image => Use a normal subscriber.
   ros::Subscriber image_only_sub_;
   // * NN-embedding desciptor requires also 3D info => Use a subscriber for
   //   synchronized messages.
   message_filters::Synchronizer<MySyncPolicy>* sync_;
   message_filters::Subscriber<sensor_msgs::Image> image_sub_;
   message_filters::Subscriber<sensor_msgs::Image> cloud_sub_;
   message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
   message_filters::Subscriber<geometry_msgs::TransformStamped>
       camera_to_world_matrix_sub_;
   // Node handle.
   ros::NodeHandle node_handle_;
   // Services.
   line_detection::ExtractLines service_extract_lines_;
   line_detection::ExtractKeyLines service_extract_keylines_;
   line_description::LineToVirtualCameraImage
       service_line_to_virtual_camera_image_;
   line_description::ImageToEmbeddings service_image_to_embeddings_;
   line_description::KeyLineToBinaryDescriptor
       service_keyline_to_binary_descriptor_;
   // Service clients.
   ros::ServiceClient client_extract_lines_;
   ros::ServiceClient client_extract_keylines_;
   ros::ServiceClient client_line_to_virtual_camera_image_;
   ros::ServiceClient client_image_to_embeddings_;
   ros::ServiceClient client_keyline_to_binary_descriptor_;
   // Service server.
   ros::ServiceServer server_embeddings_retriever_ready_;

   // Flag used to detect whether the embeddings retriever is ready or not.
   bool embeddings_retriever_is_ready_;

   // Displays the matches between the current frame and the previous one.
   // Input: current_frame_index: Frame index of the current frame.
   void displayMatchesWithPreviousFrame(int current_frame_index);

   // Given an input image (and point cloud), these functions detect lines in
   // the image, retrieve the NN-embedding/binary descriptors and save the
   // result in the current frame of the line matcher.
   // * NN-embedding descriptor version:
   //   Input: image_rgb_msg:              ROS message containing the RGB image
   //                                      from which to detect lines (it gets
   //                                      converted to grayscale by the
   //                                      line_extractor node).
   //
   //          cloud_msg:                  ROS message containing the point
   //                                      cloud image, in the format CV32FC3,
   //                                      in which each pixel has 3 channels
   //                                      that correspond to the x, y and z
   //                                      coordinate of the 3D point shown at
   //                                      that pixel.
   //
   //          camera_info_msg:            ROS message containing the camera
   //                                      info.
   //
   //          camera_to_world_matrix_msg: ROS message containing the
   //                                      camera-to-world matrix.
   //
   //   Output: frame_index_out: Frame index with which the new frame is saved.
   void saveLinesWithNNEmbeddings(
       const sensor_msgs::ImageConstPtr& image_rgb_msg,
       const sensor_msgs::ImageConstPtr& cloud_msg,
       const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
       const geometry_msgs::TransformStampedConstPtr& camera_to_world_matrix_msg,
       int* frame_index_out);
   // * Binary descriptor version:
   //   Input: image_rgb_msg:   ROS message containing the RGB image from which
   //                           to detect lines (it gets converted to grayscale
   //                           by the line_extractor node).
   //
   //   Output: frame_index_out: Frame index with which the new frame is saved.
   void saveLinesWithBinaryDescriptors(
        const sensor_msgs::ImageConstPtr& image_rgb_msg, int* frame_index_out);

   // Given an input image and point cloud, detect lines in the image and
   // returns them, together with an index to be assigned to the current frame.
   // * NN-embedding descriptor version:
   //   Input: image_rgb_msg:   ROS message containing the RGB image from which
   //                           to detect lines (it gets converted to grayscale
   //                           by the line_extractor node).
   //
   //          cloud_msg:       ROS message containing the point cloud image, in
   //                           the format CV32FC3, in which each pixel has 3
   //                           channels that correspond to the x, y and z
   //                           coordinate of the 3D point shown at that pixel.
   //
   //          camera_info_msg: ROS message containing the camera info.
   //
   //   Output: lines:       Lines detected in the input image, both in 2D and 3D,
   //                        and with their inlier planes and types.
   //
   //           frame_index: Index to be assigned to the current frame.
   void detectLines(const sensor_msgs::ImageConstPtr& image_rgb_msg,
                    const sensor_msgs::ImageConstPtr& cloud_msg,
                    const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
                    std::vector<line_detection::Line2D3DWithPlanes>* lines,
                    int* frame_index);
   // * Binary descriptor version:
   //   Input: image_rgb_msg:   ROS message containing the RGB image from which
   //                           to detect lines (it gets converted to grayscale
   //                           by the line_extractor node).
   //
   //   Output: keylines_msgs: ROS message containing the EDL KeyLines detected
   //                          in the input image.
   //
   //           frame_index:   Index to be assigned to the current frame.
   void detectLines(const sensor_msgs::ImageConstPtr& image_rgb_msg,
                    std::vector<line_detection::KeyLine>* keylines_msgs,
                    int* frame_index);

   // Given a line (both in 2D and 3D, with the inlier planes and type) and RGB
   // image and a cloud image, returns the NN-embedding descriptor associated to
   // that line.
   // Input: line:          Line the descriptor of which should be retrieved.
   //
   //        image_rgb_msg: ROS message containing the RGB image from which to
   //                       detect lines (it gets converted to grayscale by the
   //                       line_extractor node).
   //
   //        cloud_msg:     ROS message containing the point cloud image, in the
   //                       format CV32FC3, in which each pixel has 3 channels
   //                       that correspond to the x, y and z coordinate of the
   //                       3D point shown at that pixel.
   // Output: embeddings:   NN-embedding descriptor for the input line.
   void getNNEmbeddings(
       const line_detection::Line2D3DWithPlanes& line,
       const sensor_msgs::ImageConstPtr& image_rgb_msg,
       const sensor_msgs::ImageConstPtr& cloud_msg,
       const geometry_msgs::TransformStampedConstPtr& camera_to_world_matrix_msg,
       line_description::Descriptor* embedding);
   // Given an EDL KeyLine and the RGB image it was extracted from, returns the
   // binary descriptor associated to it.
   // Input: keyline_msg:                ROS message containing the EDL KeyLine
   //                                    detected.
   //
   //        image_rgb_msg:              ROS message containing the RGB image
   //                                    from which to detect lines (it gets
   //                                    converted to grayscale by the
   //                                    line_extractor node).
   //
   //        camera_to_world_matrix_msg: ROS message containing the
   //                                    camera-to-world matrix.
   //
   // Output: descriptor:   Binary descriptor for the input line.
   void getBinaryDescriptor(const line_detection::KeyLine& keyline_msg,
                            const sensor_msgs::ImageConstPtr& image_rgb_msg,
                            line_description::Descriptor* descriptor);

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
                  const std::vector<line_description::Descriptor>& embeddings,
                  const cv::Mat& rgb_image, int frame_index);
   // Overload for the case when only 2D lines are detected (e.g. when using
   // binary descriptors).
   bool saveFrame(const std::vector<cv::Vec4f>& lines_2D,
                  const std::vector<line_description::Descriptor>& descriptors,
                  const cv::Mat& rgb_image, int frame_index);


   // Helper function that subscribes to the input topics.
   void subscribeToInputTopics();

   // Callback for the ROS service that informs that the embeddings retriever
   // is ready.
   bool embeddingsRetrieverCallback(
     line_description::EmbeddingsRetrieverReady::Request& req,
     line_description::EmbeddingsRetrieverReady::Response& res);

   // Callback for the ROS message that store the RGB image. Used when the
   // descriptor type is binary descriptor.
   void mainCallbackBinaryDescriptor(
       const sensor_msgs::ImageConstPtr& rosmsg_image);
   // Callback for the ROS messages that store the RGB image, the point cloud,
   // the camera info and the camera-to-world matrix. Used when the descriptor
   // type is neural-network embeddings.
   void mainCallbackNNEmbeddings(
       const sensor_msgs::ImageConstPtr& rosmsg_image,
       const sensor_msgs::ImageConstPtr& rosmsg_cloud,
       const sensor_msgs::CameraInfoConstPtr& rosmsg_camera_info,
       const geometry_msgs::TransformStampedConstPtr&
           rosmsg_camera_to_world_matrix);
};

}  // namespace line_ros_utility

#endif  // LINE_ROS_UTILITY_LINE_DETECT_DESCRIBE_AND_MATCH_H_
