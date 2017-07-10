#include "line_ros_utility/line_ros_utility.h"
#include "line_ros_utility/line_ros_utility_inl.h"

namespace line_ros_utility {

const std::string frame_id = "line_tools_id";

std::vector<int> clusterLinesAfterClassification(
    const std::vector<line_detection::LineWithPlanes>& lines) {
  std::vector<int> label;
  for (size_t i = 0; i < lines.size(); ++i) {
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
  for (size_t i = 0; i < lines3D.size(); ++i) {
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

void labelLinesWithInstances(
    const std::vector<line_detection::LineWithPlanes>& lines,
    const cv::Mat& instances, sensor_msgs::CameraInfoConstPtr camera_info,
    std::vector<int>* labels) {
  CHECK_NOTNULL(labels);
  CHECK_EQ(instances.type(), CV_8UC3);
  labels->resize(lines.size());
  // This class is used to perform the backprojection.
  image_geometry::PinholeCameraModel camera_model;
  camera_model.fromCameraInfo(camera_info);
  // Store all colors that were found so far.
  std::vector<cv::Vec3b> known_colors;
  // This is a voting vector, where all points on a line vote for one label and
  // the one with the highest votes wins.
  std::vector<int> labels_count;
  // For intermiedate storage.
  cv::Point2f point2D;
  cv::Vec3b color;
  cv::Vec3f start, end, line, point3D;
  // num_checks + 1 points are reprojected onto the image.
  constexpr size_t num_checks = 10;
  // Iterate over all lines.
  for (size_t i = 0; i < lines.size(); ++i) {
    start = {lines[i].line[0], lines[i].line[1], lines[i].line[2]};
    end = {lines[i].line[3], lines[i].line[4], lines[i].line[5]};
    line = end - start;
    // Set the labels size equal to the known_colors size and initialize them
    // with 0;
    labels_count = std::vector<int>(known_colors.size(), 0);
    for (int k = 0; k <= num_checks; ++k) {
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
      color = instances.at<cv::Vec3b>(point2D);
      // Find the index of the color in the known_colors vector
      size_t j = 0;
      for (; j < known_colors.size(); ++j) {
        if (known_colors[j] == color) break;
      }
      // If we did not find the color in the known_colors, push it back to it.
      if (j == known_colors.size()) {
        known_colors.push_back(color);
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
    (*labels)[i] = best_guess;
  }
}

ListenAndPublish::ListenAndPublish() : params_() {
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
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
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
  line_detector_.project2Dto3DwithPlanes(cv_cloud_, lines2D_,
                                         &lines3D_temp_wp_);
  end_time_ = std::chrono::system_clock::now();
  elapsed_seconds_ = end_time_ - start_time_;
  ROS_INFO("Projecting to 3D: %f", elapsed_seconds_.count());
}

void ListenAndPublish::checkLines() {
  lines3D_with_planes_.clear();
  start_time_ = std::chrono::system_clock::now();
  line_detector_.runCheckOn3DLines(cv_cloud_, lines3D_temp_wp_,
                                   &lines3D_with_planes_);
  end_time_ = std::chrono::system_clock::now();
  elapsed_seconds_ = end_time_ - start_time_;
  ROS_INFO("Check for valid lines: %f", elapsed_seconds_.count());
}

void ListenAndPublish::cluster() {
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
  } else {
    display_clusters_.setClusters(lines3D_with_planes_, labels_);
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
  ROS_INFO("Lines kept after projection: %d/%d", lines3D_with_planes_.size(),
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
  CHECK(cv_cloud_.cols == 320);
  CHECK(cv_cloud_.rows == 240);
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

  // Convert image to grayscale. That is needed for the line detection.
  cvtColor(cv_image_, cv_img_gray_, CV_RGB2GRAY);

  ROS_INFO("**** New Image ******");
  detectLines();
  projectTo3D();
  checkLines();
  printNumberOfLines();
  cluster();
  labelLinesWithInstances(lines3D_with_planes_, cv_instances_, camera_info_,
                          &labels_);
  initDisplay();
  writeMatToPclCloud(cv_cloud_, cv_image_, &pcl_cloud_);

  // The timestamp is set to 0 because rviz is not able to find the right
  // transformation otherwise.
  pcl_cloud_.header.stamp = 0;
  ROS_INFO("**** Started publishing ****");
  publish();
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
  size_t N = 0;
  line_clusters_.clear();
  for (size_t i = 0; i < lines3D.size(); ++i) {
    // This if-clause sets the number of clusters. This works well as long the
    // clusters are indexed as an array (0,1,2,3). In any other case, it creates
    // to many clusters (which is not that bad, because empty clusters do not
    // need a lot of memory nor a lot of time to allocate), but if one label is
    // higher than the number of colors defined in the constructor (which
    // defines the number of labels that can be displayed), some clusters might
    // not be displayed.
    if (labels[i] >= N) {
      N = 1 + labels[i];
      line_clusters_.resize(N);
    }
    CHECK(labels[i] >= 0) << "line_clustering::DisplayClusters::setClusters: "
                             "Negative lables are not allowed.";
    line_clusters_[labels[i]].push_back(lines3D[i].line);
  }
  marker_lines_.resize(line_clusters_.size());
  size_t n;
  for (size_t i = 0; i < line_clusters_.size(); ++i) {
    n = i % colors_.size();
    storeLines3DinMarkerMsg(line_clusters_[i], &marker_lines_[i], colors_[n]);
    marker_lines_[i].header.frame_id = frame_id_;
    marker_lines_[i].lifetime = ros::Duration(1.1);
  }
  clusters_set_ = true;
}

void DisplayClusters::initPublishing(ros::NodeHandle& node_handle) {
  pub_.resize(colors_.size());
  for (size_t i = 0; i < colors_.size(); ++i) {
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
  size_t n;
  for (size_t i = 0; i < marker_lines_.size(); ++i) {
    n = i % pub_.size();
    pub_[n].publish(marker_lines_[i]);
  }
}

}  // namespace line_ros_utility
