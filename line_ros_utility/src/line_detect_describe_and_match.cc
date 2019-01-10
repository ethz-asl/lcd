#include "line_detect_describe_and_match/line_detect_describe_and_match.h"

namespace line_ros_utility {
  LineDetectorDescriptorAndMatcher::LineDetectorDescriptorAndMatcher(
      line_detection::DetectorType detector_type,
      line_description::DescriptorType descriptor_type) {
    // Set type of detector.
    detector_type_ = detector_type;
    // Set type of descriptor.
    descriptor_type_ = descriptor_type;
    // Check that detector type and descriptor type are compatible.
    if (detector_type_ != line_detection::DetectorType::EDL &&
        descriptor_type_ == line_description::DescriptorType::BINARY) {
      ROS_ERROR("Invalid combination of detector type and descriptor type."
                "Binary descriptor can only be use with EDL detector.");
      return;
     }
    // Initialize service clients.
    if (descriptor_type_ == line_description::DescriptorType::BINARY) {
      client_extract_keylines_ =
          node_handle_.serviceClient<line_detection::ExtractKeyLines>(
            "extract_keylines");
      client_keyline_to_binary_descriptor_ =
          node_handle_.serviceClient<line_description::KeyLineToBinaryDescriptor>(
            "keyline_to_binary_descriptor");
    } else {
      client_extract_lines_ =
          node_handle_.serviceClient<line_detection::ExtractLines>(
            "extract_lines");
      client_image_to_embeddings_ =
          node_handle_.serviceClient<line_description::ImageToEmbeddings>(
            "image_to_embeddings");
      client_line_to_virtual_camera_image_ =
          node_handle_.serviceClient<line_description::LineToVirtualCameraImage>(
            "line_to_virtual_camera_image");
      // Wait for the embeddings retriever to be ready.
      embeddings_retriever_is_ready_ = false;
    }
  }

  LineDetectorDescriptorAndMatcher::~LineDetectorDescriptorAndMatcher() {
    delete sync_;
  }

  void LineDetectorDescriptorAndMatcher::start() {
    ROS_INFO("Initializing main node. Please wait...");
    if (descriptor_type_ == line_description::DescriptorType::BINARY) {
      // No need to wait for the descriptor retriever to initialize => Can
      // immediately subscribe to topics.
      subscribeToInputTopics();
    } else {
      // Advertise service that checks when the embeddings retriever is ready.
      server_embeddings_retriever_ready_ = node_handle_.advertiseService(
          "embeddings_retriever_ready",
          &LineDetectorDescriptorAndMatcher::embeddingsRetrieverCallback, this);
    }
  }

  void LineDetectorDescriptorAndMatcher::subscribeToInputTopics() {
    if (descriptor_type_ == line_description::DescriptorType::BINARY) {
      image_only_sub_ = node_handle_.subscribe(
          "/line_tools/image/rgb", 300,
          &LineDetectorDescriptorAndMatcher::mainCallbackBinaryDescriptor,
          this);
    } else {
      // Subscribe to the image input topic.
      image_sub_.subscribe(node_handle_, "/line_tools/image/rgb", 300);
      cloud_sub_.subscribe(node_handle_, "/line_tools/point_cloud", 300);
      info_sub_.subscribe(node_handle_, "/line_tools/camera_info", 300);
      // Connect main callback.
      sync_ = new message_filters::Synchronizer<MySyncPolicy>(
          MySyncPolicy(10), image_sub_, cloud_sub_, info_sub_);
      sync_->registerCallback(
          boost::bind(
              &LineDetectorDescriptorAndMatcher::mainCallbackNNEmbeddings, this,
              _1, _2, _3));
    }
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
    //line_matcher_.displayMatches(current_frame_index - 1, current_frame_index,
    //                             line_matching::MatchingMethod::EUCLIDEAN);
    line_matcher_.displayNBestMatchesPerLine(
        current_frame_index - 1, current_frame_index,
        line_matching::MatchingMethod::EUCLIDEAN, 5);
  }

  void LineDetectorDescriptorAndMatcher::saveLinesWithNNEmbeddings(
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      const sensor_msgs::ImageConstPtr& cloud_msg,
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
      int* frame_index_out) {
    CHECK_NOTNULL(frame_index_out);
    int frame_index;
    std::vector<line_detection::Line2D3DWithPlanes> lines;
    std::vector<line_description::Descriptor> embeddings;
    if (descriptor_type_ != line_description::DescriptorType::EMBEDDING_NN) {
      ROS_ERROR("Expected detector type EMBEDDING_NN, found a different one. "
                "Please use the correct function call.");
      return;
    }
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
      getNNEmbeddings(lines[idx], image_rgb_msg, cloud_msg, &embeddings[idx]);
    }
    // Save frame.
    saveFrame(lines, embeddings, image_rgb, frame_index);
    // Output the frame index of the new frame.
    *frame_index_out = frame_index;
  }

  void LineDetectorDescriptorAndMatcher::saveLinesWithBinaryDescriptors(
      const sensor_msgs::ImageConstPtr& image_rgb_msg, int* frame_index_out) {
    CHECK_NOTNULL(frame_index_out);
    int frame_index;
    std::vector<line_detection::KeyLine> keylines_msgs;
    std::vector<line_description::Descriptor> descriptors;
    std::vector<cv::Vec4f> lines_2D;
    if (descriptor_type_ != line_description::DescriptorType::BINARY) {
      ROS_ERROR("Expected detector type BINARY, found a different one. Please "
                "use the correct function call.");
      return;
    }
    // Retrieve RGB image.
    cv::Mat image_rgb;
    cv_bridge::CvImageConstPtr cv_img_ptr =
        cv_bridge::toCvShare(image_rgb_msg, "rgb8");
    image_rgb = cv_img_ptr->image;
    // Detect lines.
    detectLines(image_rgb_msg, &keylines_msgs, &frame_index);
    ROS_INFO("Number of lines detected: %lu.", keylines_msgs.size());
    // Retrieve descriptor for all lines.
    descriptors.resize(keylines_msgs.size());
    lines_2D.clear();
    for (size_t idx = 0; idx < keylines_msgs.size(); ++idx) {
      getBinaryDescriptor(keylines_msgs[idx], image_rgb_msg, &descriptors[idx]);
      // Retrieve 2D lines.
      lines_2D.push_back({keylines_msgs[idx].startPointX,
                          keylines_msgs[idx].startPointY,
                          keylines_msgs[idx].endPointX,
                          keylines_msgs[idx].endPointY});
    }
    // Save frame.
    saveFrame(lines_2D, descriptors, image_rgb, frame_index);
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

  void LineDetectorDescriptorAndMatcher::detectLines(
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      std::vector<line_detection::KeyLine>* keylines_msgs,
      int* frame_index) {
    CHECK_NOTNULL(keylines_msgs);
    CHECK_NOTNULL(frame_index);
    keylines_msgs->clear();
    // Create request.
    service_extract_keylines_.request.image = *image_rgb_msg;

    // Call keyline extraction service.
    size_t num_lines;
    if (client_extract_keylines_.call(service_extract_keylines_)) {
      *frame_index = service_extract_keylines_.response.frame_index;
      num_lines = service_extract_keylines_.response.keylines.size();
      keylines_msgs->resize(num_lines);
      for (size_t i = 0; i < num_lines; ++i) {
        (*keylines_msgs)[i] = service_extract_keylines_.response.keylines[i];
      }
    } else {
      ROS_ERROR("Failed to call service extract_keylines.");
    }
  }

  void LineDetectorDescriptorAndMatcher::getNNEmbeddings(
      const line_detection::Line2D3DWithPlanes& line,
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      const sensor_msgs::ImageConstPtr& cloud_msg,
      line_description::Descriptor* embedding) {
    CHECK_NOTNULL(embedding);
    cv_bridge::CvImageConstPtr virtual_camera_image_ptr;
    unsigned int line_type;
    if (descriptor_type_ != line_description::DescriptorType::EMBEDDING_NN) {
      ROS_ERROR("Expected detector type EMBEDDING_NN, found a different one.");
      return;
    }
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

  void LineDetectorDescriptorAndMatcher::getBinaryDescriptor(
      const line_detection::KeyLine& keyline_msg,
      const sensor_msgs::ImageConstPtr& image_rgb_msg,
      line_description::Descriptor* descriptor) {
    CHECK_NOTNULL(descriptor);
    size_t descriptor_length;
    if (descriptor_type_ != line_description::DescriptorType::BINARY) {
      ROS_ERROR("Expected detector type BINARY, found a different one.");
      return;
    }
    // Create request for service keyline_to_binary_descriptor.
    service_keyline_to_binary_descriptor_.request.keyline = keyline_msg;
    service_keyline_to_binary_descriptor_.request.image = *image_rgb_msg;
    // Call keyline_to_binary_descriptor service.
    if (client_keyline_to_binary_descriptor_.call(
          service_keyline_to_binary_descriptor_)) {
      // Return descriptor.
      descriptor->clear();
      descriptor_length =
          service_keyline_to_binary_descriptor_.response.descriptor.size();
      for (size_t i = 0; i < descriptor_length; ++i) {
        descriptor->push_back(
            service_keyline_to_binary_descriptor_.response.descriptor[i]);
      }
    } else {
     ROS_ERROR("Failed to call service keyline_to_binary_descriptor.");
     return;
    }
  }

  bool LineDetectorDescriptorAndMatcher::saveFrame(
      const std::vector<line_detection::Line2D3DWithPlanes>& lines,
      const std::vector<line_description::Descriptor>& embeddings,
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

  bool LineDetectorDescriptorAndMatcher::saveFrame(
      const std::vector<cv::Vec4f>& lines_2D,
      const std::vector<line_description::Descriptor>& descriptors,
      const cv::Mat& rgb_image, int frame_index) {
    line_matching::Frame current_frame;
    // Create frame.
    current_frame.lines.resize(lines_2D.size());
    for (size_t i = 0; i < lines_2D.size(); ++i) {
      current_frame.lines[i].line2D = lines_2D[i];
      // Since a line_matching::Frame also contains the 3D lines corresponding
      // to the 2D lines, create mock 3D lines with (0, 0, 0) endpoints. The
      // matching will be performed just by looking at the descriptors, so the
      // 3D line is not taken into account.
      current_frame.lines[i].line3D = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      current_frame.lines[i].embeddings = descriptors[i];
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

  bool LineDetectorDescriptorAndMatcher::embeddingsRetrieverCallback(
    line_description::EmbeddingsRetrieverReady::Request& req,
    line_description::EmbeddingsRetrieverReady::Response& res) {
    if (embeddings_retriever_is_ready_) {
      // This service has already been called before => Shut it down. (The new
      // call will fail).
      server_embeddings_retriever_ready_.shutdown();
      return false;
    }
    if (req.retriever_ready != true) {
      ROS_ERROR("Wrong message received by embeddings retriever. Expected "
                "true, received false.");
      return false;
    } else {
      // Configure node to listen to other topics.
      res.message_received = true;
      embeddings_retriever_is_ready_ = true;
      subscribeToInputTopics();
      return true;
    }
  }

  void LineDetectorDescriptorAndMatcher::mainCallbackBinaryDescriptor(
      const sensor_msgs::ImageConstPtr& rosmsg_image) {
    int current_frame_index;
    // Save lines to the line matcher.
    ROS_INFO("Detecting, describing and saving line for new frame...");
    saveLinesWithBinaryDescriptors(rosmsg_image, &current_frame_index);
    ROS_INFO("...done with detecting, describing and saving line for new "
             "frame.");
    ROS_INFO("Current frame index is %d", current_frame_index);
    if (current_frame_index > 0) {
      displayMatchesWithPreviousFrame(current_frame_index);
    }
  }

  void LineDetectorDescriptorAndMatcher::mainCallbackNNEmbeddings(
      const sensor_msgs::ImageConstPtr& rosmsg_image,
      const sensor_msgs::ImageConstPtr& rosmsg_cloud,
      const sensor_msgs::CameraInfoConstPtr& camera_info) {
    int current_frame_index;
    // Save lines to the line matcher.
    ROS_INFO("Detecting, describing and saving line for new frame...");
    saveLinesWithNNEmbeddings(rosmsg_image, rosmsg_cloud, camera_info,
                              &current_frame_index);
    ROS_INFO("...done with detecting, describing and saving line for new "
             "frame.");
    ROS_INFO("Current frame index is %d", current_frame_index);
    if (current_frame_index > 0) {
      displayMatchesWithPreviousFrame(current_frame_index);
    }
  }

}  // namespace line_matching
