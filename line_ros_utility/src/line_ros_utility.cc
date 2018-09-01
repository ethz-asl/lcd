#include "line_ros_utility/line_ros_utility.h"

namespace line_ros_utility {

const std::string frame_id = "line_tools_id";
const bool write_labeled_lines = true;
const bool clustering_with_random_forest = false;
const std::string kWritePath = "../data/train_lines/traj_1";

std::vector<int> clusterLinesAfterClassification(
    const std::vector<line_detection::LineWithPlanes>& lines) {
  std::vector<int> label;
  for (size_t i = 0u; i < lines.size(); ++i) {
    if (lines[i].type == line_detection::LineType::DISCONT) {
      label.push_back(0);
    } else if (lines[i].type == line_detection::LineType::PLANE) {
      label.push_back(1);
    } else {
      label.push_back(2);
    }
  }
  return label;
}

void storeLines3DinMarkerMsg(const std::vector<cv::Vec6f>& lines3D,
                             visualization_msgs::Marker* disp_lines,
                             cv::Vec3f color) {
  CHECK_NOTNULL(disp_lines);
  disp_lines->points.clear();
  if (color[0] > 1 || color[1] > 1 || color[2] > 1) cv::norm(color);
  disp_lines->action = visualization_msgs::Marker::ADD;
  disp_lines->type = visualization_msgs::Marker::LINE_LIST;
  disp_lines->scale.x = 0.03;
  disp_lines->scale.y = 0.03;
  disp_lines->color.a = 1;
  disp_lines->color.r = color[0];
  disp_lines->color.g = color[1];
  disp_lines->color.b = color[2];
  disp_lines->id = 1;
  // Fill in the line information. LINE_LIST is an array where the first point
  // is the start and the second is the end of the line. The third is then again
  // the start of the next line, and so on.
  geometry_msgs::Point p;
  for (size_t i = 0u; i < lines3D.size(); ++i) {
    p.x = lines3D[i][0];
    p.y = lines3D[i][1];
    p.z = lines3D[i][2];
    disp_lines->points.push_back(p);
    p.x = lines3D[i][3];
    p.y = lines3D[i][4];
    p.z = lines3D[i][5];
    disp_lines->points.push_back(p);
  }
}

bool printToFile(const std::vector<line_detection::LineWithPlanes>& lines3D,
                 const std::vector<int>& labels, const std::string& path) {
  CHECK(labels.size() == lines3D.size());
  std::ofstream file(path);
  if (file.is_open()) {
    for (size_t i = 0u; i < lines3D.size(); ++i) {
      for (int j = 0; j < 6; ++j) file << lines3D[i].line[j] << " ";
      for (int j = 0; j < 4; ++j) file << lines3D[i].hessians[0][j] << " ";
      for (int j = 0; j < 4; ++j) file << lines3D[i].hessians[1][j] << " ";
      for (int j = 0; j < 3; ++j) file << (int)lines3D[i].colors[0][j] << " ";
      for (int j = 0; j < 3; ++j) file << (int)lines3D[i].colors[1][j] << " ";

      if (lines3D[i].type == line_detection::LineType::DISCONT) {
        file << 0 << " ";
      } else if (lines3D[i].type == line_detection::LineType::PLANE) {
        file << 1 << " ";
      } else {
        file << 2 << " ";
      }
      file << labels[i] << std::endl;
    }
    file.close();
    return true;
  } else {
    LOG(WARNING) << "LineDetector::printToFile: File could not be opened.";
    file.close();
    return false;
  }
}

bool printToFile(const std::vector<cv::Vec4f>& lines2D,
                 const std::string& path) {
  std::ofstream file(path);
  if (file.is_open()) {
    for (size_t i = 0u; i < lines2D.size(); ++i) {
      for (size_t j = 0u; j < 3; ++j) {
        file << lines2D[i][j] << " ";
      }
      file << lines2D[i][3] << std::endl;
    }
    file.close();
    return true;
  } else {
    LOG(WARNING) << "LineDetector::printToFile: File could not be opened.";
    file.close();
    return false;
  }
}

ListenAndPublish::ListenAndPublish() : params_(), tree_classifier_() {
  ros::NodeHandle node_handle_;
  // The Pointcloud publisher and transformation for RVIZ.
  pcl_pub_ = node_handle_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
      "/vis_pointcloud", 2);
  transform_.setOrigin(tf::Vector3(0, 0, 0));
  tf::Quaternion quat;
  quat.setRPY(-line_detection::kPi / 2.0, 0.0, 0.0);
  transform_.setRotation(quat);
  // To publish the lines in 3D to rviz.
  display_clusters_.initPublishing(node_handle_);

  image_sub_.subscribe(node_handle_, "/line_tools/image/rgb", 1);
  depth_sub_.subscribe(node_handle_, "/line_tools/image/depth", 1);
  info_sub_.subscribe(node_handle_, "/line_tools/camera_info", 1);
  cloud_sub_.subscribe(node_handle_, "/line_tools/point_cloud", 1);
  instances_sub_.subscribe(node_handle_, "/line_tools/image/instances", 1);
  // Connect the dynamic reconfigure callback.
  dynamic_rcf_callback_ =
      boost::bind(&ListenAndPublish::reconfigureCallback, this, _1, _2);
  dynamic_rcf_server_.setCallback(dynamic_rcf_callback_);

  // Add the parameters utility to line_detection.
  line_detector_ = line_detection::LineDetector(&params_);
  iteration_ = 0;
  // Retrieve trees.
  if (clustering_with_random_forest) {
    tree_classifier_.getTrees();
  }
}
ListenAndPublish::~ListenAndPublish() { delete sync_; }

void ListenAndPublish::writeMatToPclCloud(
    const cv::Mat& cv_cloud, const cv::Mat& image,
    pcl::PointCloud<pcl::PointXYZRGB>* pcl_cloud) {
  CHECK_NOTNULL(pcl_cloud);
  CHECK_EQ(cv_cloud.type(), CV_32FC3);
  CHECK_EQ(image.type(), CV_8UC3);
  CHECK_EQ(cv_cloud.cols, image.cols);
  CHECK_EQ(cv_cloud.rows, image.rows);
  const size_t width = cv_cloud.cols;
  const size_t height = cv_cloud.rows;
  pcl_cloud->points.resize(width * height);
  pcl::PointXYZRGB point;
  for (size_t i = 0u; i < height; ++i) {
    for (size_t j = 0u; j < width; ++j) {
      point.x = cv_cloud.at<cv::Vec3f>(i, j)[0];
      point.y = cv_cloud.at<cv::Vec3f>(i, j)[1];
      point.z = cv_cloud.at<cv::Vec3f>(i, j)[2];
      point.r = image.at<cv::Vec3b>(i, j)[0];
      point.g = image.at<cv::Vec3b>(i, j)[1];
      point.b = image.at<cv::Vec3b>(i, j)[2];
      (*pcl_cloud)[i + j * height] = point;
    }
  }
}

void ListenAndPublish::start() {
  // The exact time synchronizer makes it possible to have a single callback
  // that recieves messages of all five topics above synchronized. This means
  // every call of the callback function recieves three messages that have the
  // same timestamp.
  sync_ = new message_filters::Synchronizer<MySyncPolicy>(
      MySyncPolicy(10), image_sub_, depth_sub_, instances_sub_, info_sub_,
      cloud_sub_);
  sync_->registerCallback(
      boost::bind(&ListenAndPublish::masterCallback, this, _1, _2, _3, _4, _5));
}

void ListenAndPublish::detectLines() {
  // Detect lines on image.
  lines2D_.clear();
  start_time_ = std::chrono::system_clock::now();
  line_detector_.detectLines(cv_img_gray_, detector_method_, &lines2D_);
  end_time_ = std::chrono::system_clock::now();
  elapsed_seconds_ = end_time_ - start_time_;
  ROS_INFO("Detecting lines 2D: %f", elapsed_seconds_.count());
}

void ListenAndPublish::projectTo3D() {
  lines3D_temp_wp_.clear();
  start_time_ = std::chrono::system_clock::now();
  line_detector_.project2Dto3DwithPlanes(cv_cloud_, cv_image_, lines2D_, true,
                                         &lines2D_kept_tmp_, &lines3D_temp_wp_);
  end_time_ = std::chrono::system_clock::now();
  elapsed_seconds_ = end_time_ - start_time_;
  ROS_INFO("Projecting to 3D: %f", elapsed_seconds_.count());
}

void ListenAndPublish::checkLines() {
  lines3D_with_planes_.clear();
  start_time_ = std::chrono::system_clock::now();
  line_detector_.runCheckOn3DLines(cv_cloud_, camera_P_, lines2D_kept_tmp_,
                                   lines3D_temp_wp_, &lines2D_kept_,
                                   &lines3D_with_planes_);
  end_time_ = std::chrono::system_clock::now();
  elapsed_seconds_ = end_time_ - start_time_;
  ROS_INFO("Check for valid lines: %f", elapsed_seconds_.count());
}

void ListenAndPublish::clusterKmeans() {
  kmeans_cluster_.setNumberOfClusters(number_of_clusters_);
  kmeans_cluster_.setLines(lines3D_with_planes_);
  // Start the clustering.
  start_time_ = std::chrono::system_clock::now();
  kmeans_cluster_.initClusteringWithHessians(0.5);
  kmeans_cluster_.runOnLinesAndHessians();
  end_time_ = std::chrono::system_clock::now();
  elapsed_seconds_ = end_time_ - start_time_;
  ROS_INFO("Clustering: %f", elapsed_seconds_.count());
}

void ListenAndPublish::clusterKmedoid() {
  start_time_ = std::chrono::system_clock::now();
  tree_classifier_.getLineDecisionPath(lines3D_with_planes_);
  tree_classifier_.computeDistanceMatrix();
  end_time_ = std::chrono::system_clock::now();
  elapsed_seconds_ = end_time_ - start_time_;
  ROS_INFO("Retrieving distance matrix: %f", elapsed_seconds_.count());

  kmedoids_cluster_.setDistanceMatrix(tree_classifier_.getDistanceMatrix());
  kmedoids_cluster_.setK(number_of_clusters_);
  start_time_ = std::chrono::system_clock::now();
  kmedoids_cluster_.cluster();
  end_time_ = std::chrono::system_clock::now();
  elapsed_seconds_ = end_time_ - start_time_;
  ROS_INFO("Cluster on distance matrix: %f", elapsed_seconds_.count());
  std::vector<size_t> labels = kmedoids_cluster_.getLabels();
  labels_rf_kmedoids_.clear();
  for (size_t i = 0u; i < labels.size(); ++i) {
    labels_rf_kmedoids_.push_back(labels[i]);
  }
}

void ListenAndPublish::initDisplay() {
  // Decide if the lines should be displayed after their classification or
  // after clustering.
  display_clusters_.setFrameID(frame_id);
  if (show_lines_or_clusters_ == 0) {
    display_clusters_.setClusters(lines3D_with_planes_,
                                  kmeans_cluster_.cluster_idx_);
  } else if (show_lines_or_clusters_ == 1) {
    display_clusters_.setClusters(
        lines3D_with_planes_, line_ros_utility::clusterLinesAfterClassification(
                                  lines3D_with_planes_));
  } else if (show_lines_or_clusters_ == 2) {
    display_clusters_.setClusters(lines3D_with_planes_, labels_);
  } else {
    display_clusters_.setClusters(lines3D_with_planes_, labels_rf_kmedoids_);
  }
}

void ListenAndPublish::publish() {
  pcl_cloud_.header.frame_id = frame_id;
  broad_caster_.sendTransform(
      tf::StampedTransform(transform_, ros::Time::now(), "map", frame_id));
  pcl_pub_.publish(pcl_cloud_);
  display_clusters_.publish();
}

void ListenAndPublish::printNumberOfLines() {
  ROS_INFO("Lines kept after projection: %lu/%lu", lines3D_with_planes_.size(),
           lines2D_.size());
}

void ListenAndPublish::reconfigureCallback(
    line_ros_utility::line_toolsConfig& config, uint32_t level) {
  params_.max_dist_between_planes = config.max_dist_between_planes;
  params_.rectangle_offset_pixels = config.rectangle_offset_pixels;
  params_.max_relative_rect_size = config.max_relative_rect_size;
  params_.max_absolute_rect_size = config.max_absolute_rect_size;
  params_.num_iter_ransac = config.num_iter_ransac;
  params_.max_error_inlier_ransac = config.max_error_inlier_ransac;
  params_.inlier_max_ransac = config.inlier_max_ransac;
  params_.min_inlier_ransac = config.min_inlier_ransac;
  params_.min_points_in_line = config.min_points_in_line;
  params_.max_deviation_inlier_line_check =
      config.max_deviation_inlier_line_check;
  params_.min_distance_between_points_hessian =
      config.min_distance_between_points_hessian;
  params_.max_cos_theta_hessian_computation =
      config.max_cos_theta_hessian_computation;
  params_.max_rating_valid_line = config.max_rating_valid_line;
  params_.canny_edges_threshold1 = config.canny_edges_threshold1;
  params_.canny_edges_threshold2 = config.canny_edges_threshold2;
  params_.canny_edges_aperture = config.canny_edges_aperture;
  params_.hough_detector_rho = config.hough_detector_rho;
  params_.hough_detector_theta = config.hough_detector_theta;
  params_.hough_detector_threshold = config.hough_detector_threshold;
  params_.hough_detector_minLineLength = config.hough_detector_minLineLength;
  params_.hough_detector_maxLineGap = config.hough_detector_maxLineGap;

  detector_method_ = config.detector;
  number_of_clusters_ = config.number_of_clusters;
  show_lines_or_clusters_ = config.clustering;
}

void ListenAndPublish::masterCallback(
    const sensor_msgs::ImageConstPtr& rosmsg_image,
    const sensor_msgs::ImageConstPtr& rosmsg_depth,
    const sensor_msgs::ImageConstPtr& rosmsg_instances,
    const sensor_msgs::CameraInfoConstPtr& rosmsg_info,
    const sensor_msgs::ImageConstPtr& rosmsg_cloud) {
  // Extract the point cloud from the message.
  cv_bridge::CvImageConstPtr cv_cloud_ptr =
      cv_bridge::toCvShare(rosmsg_cloud, "32FC3");
  cv_cloud_ = cv_cloud_ptr->image;
  CHECK(cv_cloud_.type() == CV_32FC3);
  // Extract image from message.
  cv_bridge::CvImageConstPtr cv_img_ptr =
      cv_bridge::toCvShare(rosmsg_image, "rgb8");
  cv_image_ = cv_img_ptr->image;
  // Extract depth from message.
  cv_bridge::CvImageConstPtr cv_depth_ptr = cv_bridge::toCvShare(rosmsg_depth);
  cv_depth_ = cv_depth_ptr->image;
  // Extract instances from message.
  cv_bridge::CvImageConstPtr cv_instances_ptr =
      cv_bridge::toCvShare(rosmsg_instances);
  cv_instances_ = cv_instances_ptr->image;
  // Store camera message.
  camera_info_ = rosmsg_info;
  // Get camera projection matrix
  image_geometry::PinholeCameraModel camera_model;
  camera_model.fromCameraInfo(camera_info_);
  camera_P_ = cv::Mat(camera_model.projectionMatrix());
  camera_P_.convertTo(camera_P_, CV_32F);
  // Convert image to grayscale. That is needed for the line detection.
  cv::cvtColor(cv_image_, cv_img_gray_, CV_RGB2GRAY);

  ROS_INFO("**** New Image**** Frame %lu****", iteration_);
  detectLines();
  projectTo3D();
  ROS_INFO("Kept lines: %lu/%lu", lines3D_temp_wp_.size(), lines2D_.size());
  checkLines();

  CHECK_EQ(static_cast<int>(lines3D_with_planes_.size()),
           static_cast<int>(lines2D_kept_.size()));

  printNumberOfLines();
  clusterKmeans();
  labelLinesWithInstances(lines3D_with_planes_, cv_instances_, camera_info_,
                          &labels_);

  if (clustering_with_random_forest) {
    clusterKmedoid();
  }

  if (write_labeled_lines) {
    std::string path = kWritePath + "/lines_with_labels_" +
                       std::to_string(iteration_) + ".txt";

    std::string path_2D_kept =
        kWritePath + "/lines_2D_kept_" + std::to_string(iteration_) + ".txt";

    std::string path_2D =
        kWritePath + "/lines_2D_" + std::to_string(iteration_) + ".txt";

    // 3D lines data
    printToFile(lines3D_with_planes_, labels_, path);

    // 2D lines kept (bijection with 3D lines above)
    printToFile(lines2D_kept_, path_2D_kept);

    // All 2D lines detected
    printToFile(lines2D_, path_2D);
  }
  initDisplay();
  writeMatToPclCloud(cv_cloud_, cv_image_, &pcl_cloud_);

  // The timestamp is set to 0 because rviz is not able to find the right
  // transformation otherwise.
  pcl_cloud_.header.stamp = 0;
  ROS_INFO("**** Started publishing ****");
  publish();
  ++iteration_;
}

void ListenAndPublish::labelLinesWithInstances(
    const std::vector<line_detection::LineWithPlanes>& lines,
    const cv::Mat& instances, sensor_msgs::CameraInfoConstPtr camera_info,
    std::vector<int>* labels) {
  CHECK_NOTNULL(labels);
  CHECK_EQ(instances.type(), CV_16UC1);
  labels->resize(lines.size());
  // This class is used to perform the backprojection.
  image_geometry::PinholeCameraModel camera_model;
  camera_model.fromCameraInfo(camera_info);
  // This is a voting vector, where all points on a line vote for one label and
  // the one with the highest votes wins.
  std::vector<int> labels_count;
  // For intermiedate storage.
  cv::Point2f point2D;
  unsigned short color;

  cv::Vec3f start, end, line, point3D;
  // num_checks + 1 points are reprojected onto the image.
  constexpr size_t num_checks = 10;
  // Iterate over all lines.
  for (size_t i = 0u; i < lines.size(); ++i) {
    start = {lines[i].line[0], lines[i].line[1], lines[i].line[2]};
    end = {lines[i].line[3], lines[i].line[4], lines[i].line[5]};
    line = end - start;
    // Set the labels size equal to the known_colors size and initialize them
    // with 0;
    labels_count = std::vector<int>(known_colors_.size(), 0);
    for (size_t k = 0u; k <= num_checks; ++k) {
      // Compute a point on a line.
      point3D = start + line * (k / (double)num_checks);
      // Compute its reprojection.
      point2D =
          camera_model.project3dToPixel({point3D[0], point3D[1], point3D[2]});
      // Check that the point2D lies within the image boundaries.
      point2D.x = line_detection::checkInBoundary(floor(point2D.x), 0,
                                                  instances.cols - 1);
      point2D.y = line_detection::checkInBoundary(floor(point2D.y), 0,
                                                  instances.rows - 1);
      // Get the color of the pixel.
      // color = instances.at<cv::Vec3b>(point2D);
      color = instances.at<unsigned short>(point2D);

      // Find the index of the color in the known_colors vector
      size_t j = 0;
      for (; j < known_colors_.size(); ++j) {
        if (known_colors_[j] == color) {
          break;
        }
      }
      // If we did not find the color in the known_colors, push it back to it.
      if (j == known_colors_.size()) {
        known_colors_.push_back(color);
        labels_count.push_back(0);
      }
      // Apply the vote.
      labels_count[j] += 1;
    }
    // Find the label with the highest vote.
    size_t best_guess = 0;
    for (size_t j = 1; j < labels_count.size(); ++j) {
      if (labels_count[j] > labels_count[best_guess]) {
        best_guess = j;
      }
    }
    labels->at(i) = known_colors_[best_guess];
  }
}

DisplayClusters::DisplayClusters() {
  colors_.push_back({1, 0, 0});
  colors_.push_back({0, 1, 0});
  colors_.push_back({0, 0, 1});
  colors_.push_back({1, 1, 0});
  colors_.push_back({1, 0, 1});
  colors_.push_back({0, 1, 1});
  colors_.push_back({1, 0.5, 0});
  colors_.push_back({1, 0, 0.5});
  colors_.push_back({0.5, 1, 0});
  colors_.push_back({0, 1, 0.5});
  colors_.push_back({0.5, 0, 1});
  colors_.push_back({0, 0.5, 1});

  frame_id_set_ = false;
  clusters_set_ = false;
  initialized_ = false;
}

void DisplayClusters::setFrameID(const std::string& frame_id) {
  frame_id_ = frame_id;
  frame_id_set_ = true;
}

void DisplayClusters::setClusters(
    const std::vector<line_detection::LineWithPlanes>& lines3D,
    const std::vector<int>& labels) {
  CHECK_EQ(lines3D.size(), labels.size());
  CHECK(frame_id_set_) << "line_clustering::DisplayClusters::setClusters: You "
                          "need to set the frame_id before setting the "
                          "clusters.";
  size_t N = 0u;
  line_clusters_.clear();
  for (size_t i = 0u; i < lines3D.size(); ++i) {
    // This if-clause sets the number of clusters. This works well as long the
    // clusters are indexed as an array (0,1,2,3). In any other case, it creates
    // to many clusters (which is not that bad, because empty clusters do not
    // need a lot of memory nor a lot of time to allocate), but if one label is
    // higher than the number of colors defined in the constructor (which
    // defines the number of labels that can be displayed), some clusters might
    // not be displayed.
    if (static_cast<size_t>(labels[i]) >= N) {
      N = 1u + labels[i];
      line_clusters_.resize(N);
    }
    CHECK(labels[i] >= 0) << "line_clustering::DisplayClusters::setClusters: "
                             "Negative lables are not allowed.";
    line_clusters_[labels[i]].push_back(lines3D[i].line);
  }
  marker_lines_.resize(line_clusters_.size());

  for (size_t i = 0u; i < line_clusters_.size(); ++i) {
    size_t n = i % colors_.size();
    storeLines3DinMarkerMsg(line_clusters_[i], &marker_lines_[i], colors_[n]);
    marker_lines_[i].header.frame_id = frame_id_;
    marker_lines_[i].lifetime = ros::Duration(1.1);
  }
  clusters_set_ = true;
}

void DisplayClusters::initPublishing(ros::NodeHandle& node_handle) {
  pub_.resize(colors_.size());
  for (size_t i = 0u; i < colors_.size(); ++i) {
    std::stringstream topic;
    topic << "/visualization_marker_" << i;
    pub_[i] =
        node_handle.advertise<visualization_msgs::Marker>(topic.str(), 1000);
  }
  initialized_ = true;
}

void DisplayClusters::publish() {
  CHECK(initialized_)
      << "You need to call initPublishing to advertise before publishing.";
  CHECK(frame_id_set_) << "You need to set the frame_id before publishing.";
  CHECK(clusters_set_) << "You need to set the clusters before publishing.";

  for (size_t i = 0u; i < marker_lines_.size(); ++i) {
    size_t n = i % pub_.size();
    pub_[n].publish(marker_lines_[i]);
  }
}

TreeClassifier::TreeClassifier() {
  ros::NodeHandle node_handle;
  tree_client_ =
      node_handle.serviceClient<line_ros_utility::TreeRequest>("req_trees");
  line_client_ =
      node_handle.serviceClient<line_ros_utility::RequestDecisionPath>(
          "req_decision_paths");
  header_.seq = 1;
  header_.stamp = ros::Time::now();
}

void TreeClassifier::getTrees() {
  line_ros_utility::TreeRequest tree_service;
  tree_service.request.ask_for_trees = 1;
  if (!tree_client_.call(tree_service)) {
    ROS_ERROR("Call was not succesfull. Check if random_forest.py is running.");
    ros::shutdown();
  }
  trees_.resize(tree_service.response.trees.size());
  for (size_t i = 0u; i < tree_service.response.trees.size(); ++i) {
    cv_bridge::CvImagePtr cv_ptr_ =
        cv_bridge::toCvCopy(tree_service.response.trees[i], "64FC1");
    trees_[i].children_left.clear();
    trees_[i].children_right.clear();
    for (size_t j = 0u; j < static_cast<size_t>(cv_ptr_->image.cols); ++j) {
      trees_[i].children_left.push_back(cv_ptr_->image.at<double>(0, j));
      trees_[i].children_right.push_back(cv_ptr_->image.at<double>(1, j));
    }
  }
}

void TreeClassifier::getLineDecisionPath(
    const std::vector<line_detection::LineWithPlanes>& lines) {
  line_ros_utility::RequestDecisionPath service;
  num_lines_ = lines.size();
  if (num_lines_ < 1) {
    return;
  }
  // Fill in the line.
  for (size_t i = 0u; i < num_lines_; ++i) {
    for (size_t j = 0u; j < 6; ++j) {
      service.request.lines.push_back(lines[i].line[j]);
    }
    for (size_t j = 0u; j < 4; ++j) {
      service.request.lines.push_back(lines[i].hessians[0][j]);
    }
    for (size_t j = 0u; j < 4; ++j) {
      service.request.lines.push_back(lines[i].hessians[1][j]);
    }
    for (size_t j = 0u; j < 3; ++j) {
      service.request.lines.push_back((float)lines[i].colors[0][j]);
    }
    for (size_t j = 0u; j < 3; ++j) {
      service.request.lines.push_back((float)lines[i].colors[1][j]);
    }
    if (lines[i].type == line_detection::LineType::DISCONT) {
      service.request.lines.push_back(0.0);
    } else if (lines[i].type == line_detection::LineType::PLANE) {
      service.request.lines.push_back(1.0);
    } else {
      service.request.lines.push_back(2.0);
    }
  }
  // Call the service.
  if (!line_client_.call(service)) {
    ROS_ERROR("Call was not succesfull. Check if random_forest.py is running.");
    ros::shutdown();
  }
  // Make sure the data recieved fits the stored trees_.
  CHECK_EQ(service.response.decision_paths.size(), trees_.size());
  decision_paths_.resize(trees_.size());
  // For every tree, fill in the decision paths
  for (size_t i = 0u; i < trees_.size(); ++i) {
    cv_bridge::CvImagePtr cv_ptr_ =
        cv_bridge::toCvCopy(service.response.decision_paths[i], "64FC1");
    CHECK_EQ(cv_ptr_->image.rows, 2);
    decision_paths_[i].release();
    int size[2] = {(int)num_lines_, 60000};
    decision_paths_[i].create(2, size, CV_8U);
    for (size_t j = 0u; j < static_cast<size_t>(cv_ptr_->image.cols); ++j) {
      decision_paths_[i].ref<unsigned char>(
          cv_ptr_->image.at<double>(0, j), cv_ptr_->image.at<double>(1, j)) = 1;
    }
  }
}

double TreeClassifier::computeDistance(const SearchTree& tree,
                                       const cv::SparseMat& path,
                                       size_t line_idx1, size_t line_idx2,
                                       size_t idx) {
  if (path.value<double>(line_idx1, idx) != 0 &&
      path.value<double>(line_idx2, idx) != 0) {
    if (tree.children_right[idx] == tree.children_left[idx]) {  // at leave
      return 0.0;
    } else {
      return computeDistance(tree, path, line_idx1, line_idx2,
                             tree.children_right[idx]) +
             computeDistance(tree, path, line_idx1, line_idx2,
                             tree.children_left[idx]);
    }
  } else if (path.value<double>(line_idx1, idx) != 0 ||
             path.value<double>(line_idx2, idx) != 0) {
    if (tree.children_right[idx] == tree.children_left[idx]) {
      return 1.0;
    } else {
      return computeDistance(tree, path, line_idx1, line_idx2,
                             tree.children_right[idx]) +
             computeDistance(tree, path, line_idx1, line_idx2,
                             tree.children_left[idx]) +
             1.0;
    }
  } else {
    return 0.0;
  }
}

void TreeClassifier::computeDistanceMatrix() {
  dist_matrix_ = cv::Mat(num_lines_, num_lines_, CV_32FC1, 0.0f);
  float dummy;
  for (size_t i = 0u; i < num_lines_; ++i) {
    for (size_t j = i + 1; j < num_lines_; ++j) {
      dummy = 0;
      for (size_t k = 0u; k < trees_.size(); ++k) {
        dummy += computeDistance(trees_[k], decision_paths_[k], i, j, 0);
      }
      dist_matrix_.at<float>(i, j) = dummy / (double)trees_.size();
    }
  }
}

cv::Mat TreeClassifier::getDistanceMatrix() { return dist_matrix_; }

EvalData::EvalData(const std::vector<line_detection::LineWithPlanes>& lines3D) {
  lines3D_.clear();
  for (size_t i = 0u; i < lines3D.size(); ++i) {
    lines3D_.push_back(lines3D[i].line);
  }
}

float EvalData::dist(const cv::Mat& dist_mat, size_t i, size_t j) {
  if (i < j) {
    return dist_mat.at<float>(i, j);
  } else {
    return dist_mat.at<float>(j, i);
  }
}

void EvalData::createHeatMap(const cv::Mat& image, const cv::Mat& dist_mat,
                             const size_t idx) {
  CHECK_EQ(dist_mat.cols, dist_mat.rows);
  CHECK_EQ(dist_mat.cols, lines2D_.size());
  CHECK_EQ(dist_mat.type(), CV_32FC1);
  CHECK_EQ(image.type(), CV_8UC3);
  size_t num_lines = lines2D_.size();
  cv::Vec3b color;
  float max_dist;
  float red, green, blue;
  max_dist = -1;
  for (size_t i = 0u; i < num_lines; ++i) {
    if (dist(dist_mat, i, idx) > max_dist) {
      max_dist = dist(dist_mat, i, idx);
    }
  }
  image.copyTo(heat_map_);
  for (size_t i = 0; i < num_lines; ++i) {
    if (i == idx) {
      color = {255, 255, 255};
    } else {
      getHeatMapColor(dist(dist_mat, idx, i) / max_dist, &red, &green, &blue);
      color[0] = static_cast<unsigned char>(255 * blue);
      color[1] = static_cast<unsigned char>(255 * green);
      color[2] = static_cast<unsigned char>(255 * red);
    }

    cv::line(
        heat_map_,
        {static_cast<int>(lines2D_[i][0]), static_cast<int>(lines2D_[i][1])},
        {static_cast<int>(lines2D_[i][2]), static_cast<int>(lines2D_[i][3])},
        color, 2);
  }
}

void EvalData::storeHeatMaps(const cv::Mat& image, const cv::Mat& dist_mat,
                             const std::string& path) {
  size_t num_lines = lines2D_.size();
  for (size_t i = 0u; i < num_lines; ++i) {
    createHeatMap(image, dist_mat, i);
    std::string store_path = path + "_" + std::to_string(i) + ".jpg";
    cv::imwrite(store_path, heat_map_);
  }
}

void EvalData::projectLinesTo2D(
    const sensor_msgs::CameraInfoConstPtr& camera_info) {
  cv::Point2f p1, p2;
  image_geometry::PinholeCameraModel camera_model;
  camera_model.fromCameraInfo(camera_info);
  lines2D_.clear();
  for (size_t i = 0u; i < lines3D_.size(); ++i) {
    p1 = camera_model.project3dToPixel(
        {lines3D_[i][0], lines3D_[i][1], lines3D_[i][2]});
    p2 = camera_model.project3dToPixel(
        {lines3D_[i][3], lines3D_[i][4], lines3D_[i][5]});
    lines2D_.push_back({p1.x, p1.y, p2.x, p2.y});
  }
}

// Copied from:
// http://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
bool EvalData::getHeatMapColor(float value, float* red, float* green,
                               float* blue) {
  const int NUM_COLORS = 4;
  static float color[NUM_COLORS][3] = {
      {0, 0, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
  // A static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b}
  // for each.

  int idx1;  // |-- Our desired color will be between these two indexes in
             // "color".
  int idx2;  // |
  float fractBetween =
      0;  // Fraction between "idx1" and "idx2" where our value is.

  if (value <= 0) {
    idx1 = idx2 = 0;
  }  // accounts for an input <=0
  else if (value >= 1) {
    idx1 = idx2 = NUM_COLORS - 1;
  }  // accounts for an input >=0
  else {
    value = value * (NUM_COLORS - 1);  // Will multiply value by 3.
    idx1 = floor(value);  // Our desired color will be after this index.
    idx2 = idx1 + 1;      // ... and before this index (inclusive).
    fractBetween =
        value - float(idx1);  // Distance between the two indexes (0-1).
  }

  *red = (color[idx2][0] - color[idx1][0]) * fractBetween + color[idx1][0];
  *green = (color[idx2][1] - color[idx1][1]) * fractBetween + color[idx1][1];
  *blue = (color[idx2][2] - color[idx1][2]) * fractBetween + color[idx1][2];
}

// Copied from:
// http://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
void EvalData::getValueBetweenTwoFixedColors(float value, int& red, int& green,
                                             int& blue) {
  int aR = 0;
  int aG = 0;
  int aB = 255;  // RGB for our 1st color (blue in this case).
  int bR = 255;
  int bG = 0;
  int bB = 0;  // RGB for our 2nd color (red in this case).

  red = (float)(bR - aR) * value + aR;    // Evaluated as -255*value + 255.
  green = (float)(bG - aG) * value + aG;  // Evaluates as 0.
  blue = (float)(bB - aB) * value + aB;   // Evaluates as 255*value + 0.
}

void EvalData::writeHeatMapColorBar(const std::string& path) {
  constexpr size_t height = 100;
  constexpr size_t width = 10;
  cv::Mat colorbar(height, width, CV_8UC3);
  float red, green, blue;
  for (size_t i = 0u; i < height; ++i) {
    getHeatMapColor(i / (double)height, &red, &green, &blue);
    for (size_t j = 0u; j < width; ++j) {
      colorbar.at<cv::Vec3b>(i, j) = {static_cast<unsigned char>(255 * blue),
                                      static_cast<unsigned char>(255 * green),
                                      static_cast<unsigned char>(255 * red)};
    }
  }
  cv::imwrite(path, colorbar);
}
}  // namespace line_ros_utility
