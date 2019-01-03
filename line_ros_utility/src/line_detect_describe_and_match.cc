#include "line_matching/line_detect_describe_and_match.h"

namespace line_matching {
  LineDetectorDescriptorAndMatcher::LineDetectorDescriptorAndMatcher(
      line_detection::Detector detector_type) {
    // Set type of detector.
    detector_type_ = detector_type;
    // Initialize service clients.
    client_extract_lines_ =
        node_handle_.serviceClient<line_detection::ExtractLines>(
          "extract_lines");
    client_image_to_embeddings_ =
        node_handle_.serviceClient<line_description::ImageToEmbeddings>(
          "image_to_embeddings");
    client_line_to_virtual_camera_image_ =
        node_handle_.serviceClient<line_description::LineToVirtualCameraImage>(
          "line_to_virtual_camera_image");
    // Subscribe to input topics.
    image_sub_.subscribe(node_handle_, "/line_tools/image/rgb", 1);
    cloud_sub_.subscribe(node_handle_, "/line_tools/point_cloud", 1);
    info_sub_.subscribe(node_handle_, "/line_tools/camera_info", 1);
  }

  LineDetectorDescriptorAndMatcher::~LineDetectorDescriptorAndMatcher() {
    delete sync_;
  }

  void LineDetectorDescriptorAndMatcher::start() {
    // Connect callback.
    sync_ = new message_filters::Synchronizer<MySyncPolicy>(
        MySyncPolicy(10), image_sub_, cloud_sub_, info_sub_);
    sync_->registerCallback(
        boost::bind(&LineDetectorDescriptorAndMatcher::Callback, this, _1, _2,
                    _3));
    ROS_INFO("Main node for detection, description and matching is now ready "
             "to receive messages.");
  }

  void LineDetectorDescriptorAndMatcher::callback(
      const sensor_msgs::ImageConstPtr& rosmsg_image,
      const sensor_msgs::ImageConstPtr& rosmsg_cloud,
      const sensor_msgs::CameraInfoConstPtr& camera_info) {

  }

}  // namespace line_matching
