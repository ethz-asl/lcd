#include "line_detect_describe_and_match/line_detect_describe_and_match.h"

namespace line_ros_utility {
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
        boost::bind(&LineDetectorDescriptorAndMatcher::callback, this, _1, _2,
                    _3));
    ROS_INFO("Main node for detection, description and matching is now ready "
             "to receive messages.");
  }

  void LineDetectorDescriptorAndMatcher::displayMatchesWithPreviousFrame(
      int current_frame_index) {
    if (current_frame_index < 1) {
      ROS_INFO("Unable to display matches of frame %d with previous frame: not "
               "enough previous frames.", current_frame_index);
      return;
    }
    // Make line matcher display matches of current frame with previous frame.
    line_matcher_.displayMatches(current_frame_index - 1, current_frame_index,
                                 line_matching::MatchingMethod::MANHATTAN);
  }

  void LineDetectorDescriptorAndMatcher::saveLinesWithEmbeddings(
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      const sensor_msgs::ImageConstPtr& cloud_msg,
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
      int* frame_index_out) {
    CHECK_NOTNULL(frame_index_out);
    int frame_index;
    std::vector<line_detection::Line2D3DWithPlanes> lines;
    std::vector<line_description::Embedding> embeddings;
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
    detectLines(image_rgb_msg, cloud_msg, camera_info_msg, &lines,
                &frame_index);
    ROS_INFO("Number of lines detected: %lu.", lines.size());
    // Retrieve descriptor for all lines.
    embeddings.resize(lines.size());
    for (size_t idx = 0; idx < lines.size(); ++idx) {
      getEmbeddings(lines[idx], image_rgb_msg, cloud_msg, &embeddings[idx]);
    }
    // Save frame.
    saveFrame(lines, embeddings, image_rgb, frame_index);
    // Output the frame index of the new frame.
    *frame_index_out = frame_index;
  }

  void LineDetectorDescriptorAndMatcher::detectLines(
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      const sensor_msgs::ImageConstPtr& cloud_msg,
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
      std::vector<line_detection::Line2D3DWithPlanes>* lines,
      int* frame_index) {
    CHECK_NOTNULL(lines);
    CHECK_NOTNULL(frame_index);
    lines->clear();

    // Create request.
    service_extract_lines_.request.image = *image_rgb_msg;
    service_extract_lines_.request.cloud = *cloud_msg;
    service_extract_lines_.request.camera_info = *camera_info_msg;
    switch (detector_type_) {
      case line_detection::Detector::LSD:
        service_extract_lines_.request.detector = 0;
        break;
      case line_detection::Detector::EDL:
        service_extract_lines_.request.detector = 1;
        break;
      case line_detection::Detector::FAST:
        service_extract_lines_.request.detector = 2;
        break;
      case line_detection::Detector::HOUGH:
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
      *frame_index = service_extract_lines_.response.frame_index;
      num_lines = service_extract_lines_.response.lines.size();
      lines->resize(num_lines);
      for (size_t i = 0; i < num_lines; ++i) {
        (*lines)[i].line2D = {
            static_cast<float>(service_extract_lines_.response.start2D[i].x),
            static_cast<float>(service_extract_lines_.response.start2D[i].y),
            static_cast<float>(service_extract_lines_.response.end2D[i].x),
            static_cast<float>(service_extract_lines_.response.end2D[i].y)};
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
        (*lines)[i].hessians.resize(2);
        (*lines)[i].hessians[0] =
            {service_extract_lines_.response.lines[i].hessian_right[0],
             service_extract_lines_.response.lines[i].hessian_right[1],
             service_extract_lines_.response.lines[i].hessian_right[2],
             service_extract_lines_.response.lines[i].hessian_right[3]};
        (*lines)[i].hessians[1] =
            {service_extract_lines_.response.lines[i].hessian_left[0],
             service_extract_lines_.response.lines[i].hessian_left[1],
             service_extract_lines_.response.lines[i].hessian_left[2],
             service_extract_lines_.response.lines[i].hessian_left[3]};
        switch (service_extract_lines_.response.lines[i].line_type) {
          case 0:
            (*lines)[i].type = line_detection::LineType::DISCONT;
            break;
          case 1:
            (*lines)[i].type = line_detection::LineType::PLANE;
            break;
          case 2:
            (*lines)[i].type = line_detection::LineType::EDGE;
            break;
          case 3:
            (*lines)[i].type = line_detection::LineType::INTERSECT;
            break;
          default:
            ROS_ERROR("Found wrong line type in message. Valid numbers are "
                      "between 0 and 3.");
        }
      }
    } else {
      ROS_ERROR("Failed to call service extract_lines.");
    }
  }

  void LineDetectorDescriptorAndMatcher::getEmbeddings(
      const line_detection::Line2D3DWithPlanes& line,
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      const sensor_msgs::ImageConstPtr& cloud_msg,
      line_description::Embedding* embedding) {
    CHECK_NOTNULL(embedding);
    cv_bridge::CvImageConstPtr virtual_camera_image_ptr;
    unsigned int line_type;

    // Get line type.
    switch (line.type) {
      case line_detection::LineType::DISCONT:
        line_type = 0;
        break;
      case line_detection::LineType::PLANE:
        line_type = 1;
        break;
      case line_detection::LineType::EDGE:
        line_type = 2;
        break;
      case line_detection::LineType::INTERSECT:
        line_type = 3;
        break;
      default:
        ROS_ERROR("Illegal line type. Possible types are DISCONT, PLANE, EDGE "
                  "and INTERSECT");
        return;
    }

    // Create request for service line_to_virtual_camera_image.
    service_line_to_virtual_camera_image_.request.line.start3D.x =
        line.line3D[0];
    service_line_to_virtual_camera_image_.request.line.start3D.y =
        line.line3D[1];
    service_line_to_virtual_camera_image_.request.line.start3D.z =
        line.line3D[2];
    service_line_to_virtual_camera_image_.request.line.end3D.x = line.line3D[3];
    service_line_to_virtual_camera_image_.request.line.end3D.y = line.line3D[4];
    service_line_to_virtual_camera_image_.request.line.end3D.z = line.line3D[5];
    for (size_t i = 0; i < 4; ++i) {
      service_line_to_virtual_camera_image_.request.line.hessian_right[i] =
          line.hessians[0][i];
      service_line_to_virtual_camera_image_.request.line.hessian_left[i] =
          line.hessians[1][i];
    }
    service_line_to_virtual_camera_image_.request.line.line_type = line_type;
    service_line_to_virtual_camera_image_.request.image_rgb = *image_rgb_msg;
    service_line_to_virtual_camera_image_.request.cloud = *cloud_msg;
    // Call line_to_virtual_camera_image service.
    if (!client_line_to_virtual_camera_image_.call(
            service_line_to_virtual_camera_image_)) {
     ROS_ERROR("Failed to call service line_to_virtual_camera_image.");
     return;
    }

    // Create request for service image_to_embeddings.
    service_image_to_embeddings_.request.virtual_camera_image_bgr =
        service_line_to_virtual_camera_image_.response.virtual_camera_image_bgr;
    service_image_to_embeddings_.request.virtual_camera_image_depth =
        service_line_to_virtual_camera_image_.response.virtual_camera_image_depth;
    service_image_to_embeddings_.request.line_type = line_type;
    // Call image_to_embeddings service.
    if (client_image_to_embeddings_.call(service_image_to_embeddings_)) {
      *embedding = service_image_to_embeddings_.response.embeddings;
    } else {
     ROS_ERROR("Failed to call service image_to_embeddings.");
     return;
    }
  }

  bool LineDetectorDescriptorAndMatcher::saveFrame(
      const std::vector<line_detection::Line2D3DWithPlanes>& lines,
      const std::vector<line_description::Embedding>& embeddings,
      const cv::Mat& rgb_image, int frame_index) {
    line_matching::Frame current_frame;
    // Create frame.
    current_frame.lines.resize(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
      current_frame.lines[i].line2D = lines[i].line2D;
      current_frame.lines[i].line3D = lines[i].line3D;
      current_frame.lines[i].embeddings = embeddings[i];
    }
    current_frame.image = rgb_image;
    // Try to save frame.
    if (!line_matcher_.addFrame(current_frame, frame_index)) {
      ROS_INFO("Could not add frame with index %d, as one with the same "
               "was previously received.", frame_index);
      return false;
    }
    return true;
  }

  void LineDetectorDescriptorAndMatcher::callback(
      const sensor_msgs::ImageConstPtr& rosmsg_image,
      const sensor_msgs::ImageConstPtr& rosmsg_cloud,
      const sensor_msgs::CameraInfoConstPtr& camera_info) {
    int current_frame_index;
    // Save lines to the line matcher.
    ROS_INFO("Detecting, describing and saving line for new frame...");
    saveLinesWithEmbeddings(rosmsg_image, rosmsg_cloud, camera_info,
                            &current_frame_index);
    ROS_INFO("...done with detecting, describing and saving line for new "
             "frame.");
    ROS_INFO("Current frame index is %d", current_frame_index);
    if (current_frame_index == 1) {
      displayMatchesWithPreviousFrame(current_frame_index);
    }
  }

}  // namespace line_matching
