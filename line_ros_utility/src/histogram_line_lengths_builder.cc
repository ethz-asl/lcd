#include "histogram_line_lengths_builder/histogram_line_lengths_builder.h"

namespace line_ros_utility {
  HistogramLineLengthsBuilder::HistogramLineLengthsBuilder(
      line_detection::DetectorType detector_type) {
    // Set type of detector.
    detector_type_ = detector_type;
    // Initialize service clients.
    client_extract_lines_ =
        node_handle_.serviceClient<line_detection::ExtractLines>(
            "extract_lines");
    // Advertise line lengths topic.
    line_lengths_pub_ =
        node_handle_.advertise<line_ros_utility::LineLengthsArray>(
            "/line_lengths", 300);
  }

  HistogramLineLengthsBuilder::~HistogramLineLengthsBuilder() {
    delete sync_;
  }

  void HistogramLineLengthsBuilder::start() {
    subscribeToInputTopics();
  }

  void HistogramLineLengthsBuilder::subscribeToInputTopics() {
    // Subscribe to the image input topic.
    image_sub_.subscribe(node_handle_, "/line_tools/image/rgb", 300);
    cloud_sub_.subscribe(node_handle_, "/line_tools/point_cloud", 300);
    info_sub_.subscribe(node_handle_, "/line_tools/camera_info", 300);
    // Connect main callback.
    sync_ = new message_filters::Synchronizer<MySyncPolicy>(
        MySyncPolicy(10), image_sub_, cloud_sub_, info_sub_);
    sync_->registerCallback(
        boost::bind(
            &HistogramLineLengthsBuilder::callback, this, _1, _2, _3));
  }

  void HistogramLineLengthsBuilder::buildHistogram(
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      const sensor_msgs::ImageConstPtr& cloud_msg,
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
    std::vector<line_detection::Line2D3DWithPlanes> lines;
    std::vector<float> line_lengths;
    // Retrieve RGB image.
    cv::Mat image_rgb;
    cv_bridge::CvImageConstPtr cv_img_ptr =
        cv_bridge::toCvShare(image_rgb_msg, "rgb8");
    image_rgb = cv_img_ptr->image;
    // Retrieve cloud image.
    cv::Mat cloud;
    cv_bridge::CvImageConstPtr cv_cloud_ptr =
        cv_bridge::toCvShare(cloud_msg, "32FC3");
    cloud = cv_cloud_ptr->image;
    // Detect lines.
    detectLines(image_rgb_msg, cloud_msg, camera_info_msg, &lines);
    ROS_INFO("Number of lines detected: %lu.", lines.size());
    // Extract lengths from the lines.
    line_lengths.clear();
    for (auto& line : lines) {
      line_lengths.push_back(cv::norm(cv::Vec3f(line.line3D[3], line.line3D[4],
                                                line.line3D[5]) -
                                      cv::Vec3f(line.line3D[0], line.line3D[1],
                                                line.line3D[2])));
    }
    // Publish line lengths.
    line_ros_utility::LineLengthsArray line_lengths_msg;
    line_lengths_msg.line_lengths = line_lengths;
    line_lengths_pub_.publish(line_lengths_msg);
  }

  void HistogramLineLengthsBuilder::detectLines(
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      const sensor_msgs::ImageConstPtr& cloud_msg,
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
      std::vector<line_detection::Line2D3DWithPlanes>* lines) {
    CHECK_NOTNULL(lines);
    lines->clear();

    // Create request.
    service_extract_lines_.request.image = *image_rgb_msg;
    service_extract_lines_.request.cloud = *cloud_msg;
    service_extract_lines_.request.camera_info = *camera_info_msg;
    switch (detector_type_) {
      case line_detection::DetectorType::LSD:
        service_extract_lines_.request.detector = 0;
        break;
      case line_detection::DetectorType::EDL:
        service_extract_lines_.request.detector = 1;
        break;
      case line_detection::DetectorType::FAST:
        service_extract_lines_.request.detector = 2;
        break;
      case line_detection::DetectorType::HOUGH:
        service_extract_lines_.request.detector = 3;
        break;
      default:
        ROS_ERROR("Illegal detector type. Valid types are LSD, EDL, FAST and "
                  "HOUGH.");
        return;
    }
    // Call line extraction service.
    size_t num_lines;
    if (client_extract_lines_.call(service_extract_lines_)) {
      num_lines = service_extract_lines_.response.lines.size();
      lines->resize(num_lines);
      for (size_t i = 0; i < num_lines; ++i) {
        (*lines)[i].line3D = {
            static_cast<float>(
                service_extract_lines_.response.lines[i].start3D.x),
            static_cast<float>(
                service_extract_lines_.response.lines[i].start3D.y),
            static_cast<float>(
                service_extract_lines_.response.lines[i].start3D.z),
            static_cast<float>(
                service_extract_lines_.response.lines[i].end3D.x),
            static_cast<float>(
                service_extract_lines_.response.lines[i].end3D.y),
            static_cast<float>(
                service_extract_lines_.response.lines[i].end3D.z)};
      }
    } else {
      ROS_ERROR("Failed to call service extract_lines.");
    }
  }

  void HistogramLineLengthsBuilder::callback(
      const sensor_msgs::ImageConstPtr& rosmsg_image,
      const sensor_msgs::ImageConstPtr& rosmsg_cloud,
      const sensor_msgs::CameraInfoConstPtr& rosmsg_camera_info) {
    static unsigned int frame_index = 0;
    // Save lines to the line matcher.
    ROS_INFO("Detecting line for new frame...");
    buildHistogram(rosmsg_image, rosmsg_cloud, rosmsg_camera_info);
    ROS_INFO("...done with detecting lines and adding them to the histogram of "
             "line lengths for frame %u.", frame_index++);
  }

}  // namespace line_matching
