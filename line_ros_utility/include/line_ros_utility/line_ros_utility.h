#ifndef LINE_ROS_UTILITY_lINE_ROS_UTILITY_H_
#define LINE_ROS_UTILITY_lINE_ROS_UTILITY_H_

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/publisher.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

#include <image_geometry/pinhole_camera_model.h>

#include <dynamic_reconfigure/server.h>
#include <line_ros_utility/line_toolsConfig.h>

#include <line_clustering/line_clustering.h>
#include <line_detection/line_detection.h>
#include <line_detection/line_detection_inl.h>

namespace line_ros_utility {

typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
    sensor_msgs::CameraInfo, sensor_msgs::Image>
    MySyncPolicy;

std::vector<int> clusterLinesAfterClassification(
    const std::vector<line_detection::LineWithPlanes>& lines);

void labelLinesWithInstances(
    const std::vector<line_detection::LineWithPlanes>& lines,
    sensor_msgs::CameraInfoConstPtr camera_info, std::vector<int>* labels);

void storeLines3DinMarkerMsg(const std::vector<cv::Vec6f>& lines3D,
                             visualization_msgs::Marker* disp_lines,
                             cv::Vec3f color);

// This class helps publishing several different clusters of lines in
// different colors, so that they are visualized by rviz. IMPORTANT: This
// function cannot display more clusters than there are colors
//            defined in the constructor. If more clusters are given to the
//            object, only the one with the highest labels are published.
class DisplayClusters {
 public:
  DisplayClusters();
  // Frame id of the marker message.
  void setFrameID(const std::string& frame_id);

  // Is used as input for the clusters to the class:
  // lines3D:   Vector of 3D lines.
  //
  // labels:    Vector of equal size as lines3D. Every entry labels the cluster
  //            the 3D line with the same index belongs to. The lables should be
  //            continous ((0,1,2,3 -> good), (0,4,8,16 -> bad)), because the
  //            highest label defines how many clusters are created (in the
  //            latter case of the example 17 clusters will be created, but only
  //            4 will contain information).
  void setClusters(const std::vector<line_detection::LineWithPlanes>& lines3D,
                   const std::vector<int>& labels);

  // This functions advertises the message.
  void initPublishing(ros::NodeHandle& node_handle);
  void publish();

 private:
  bool frame_id_set_, clusters_set_, initialized_;
  std::vector<visualization_msgs::Marker> marker_lines_;
  std::vector<std::vector<cv::Vec<float, 6> > > line_clusters_;
  std::vector<ros::Publisher> pub_;
  std::string frame_id_;
  std::vector<cv::Vec3f> colors_;
};

class ListenAndPublish {
 public:
  ListenAndPublish();
  ~ListenAndPublish();

  void start();

 protected:
  void writeMatToPclCloud(const cv::Mat& cv_cloud, const cv::Mat& image,
                          pcl::PointCloud<pcl::PointXYZRGB>* pcl_cloud);
  void detectLines();
  void projectTo3D();
  void checkLines();
  void printNumberOfLines();
  void cluster();
  void initDisplay();
  void publish();
  void reconfigureCallback(line_ros_utility::line_toolsConfig& config,
                           uint32_t level);

  void masterCallback(const sensor_msgs::ImageConstPtr& rosmsg_image,
                      const sensor_msgs::ImageConstPtr& rosmsg_depth,
                      const sensor_msgs::ImageConstPtr& rosmsg_instances,
                      const sensor_msgs::CameraInfoConstPtr& camera_info,
                      const sensor_msgs::ImageConstPtr& rosmsg_cloud);
  // Data storage.
  cv::Mat cv_image_;
  cv::Mat cv_img_gray_;
  cv::Mat cv_cloud_;
  cv::Mat cv_depth_;
  cv::Mat cv_instances_;
  pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud_;
  std::vector<cv::Vec4f> lines2D_;
  std::vector<cv::Vec<float, 6> > lines3D_;
  std::vector<line_detection::LineWithPlanes> lines3D_temp_wp_;
  std::vector<line_detection::LineWithPlanes> lines3D_with_planes_;
  std::vector<int> labels_;
  sensor_msgs::CameraInfoConstPtr camera_info_;
  // Publishers and Subscribers.
  tf::TransformBroadcaster broad_caster_;
  tf::Transform transform_;
  ros::Publisher pcl_pub_;
  message_filters::Synchronizer<MySyncPolicy>* sync_;
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::Image> instances_sub_;
  message_filters::Subscriber<sensor_msgs::Image> cloud_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
  // message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub_;
  // To store parameters.
  line_detection::LineDetectionParams params_;
  int detector_method_;
  size_t number_of_clusters_;
  size_t show_lines_or_clusters_;
  // To have the line_detection utility.
  line_detection::LineDetector line_detector_;
  line_clustering::KMeansCluster kmeans_cluster_;
  DisplayClusters display_clusters_;
  // To dynamically reconfigure parameters.
  dynamic_reconfigure::Server<line_ros_utility::line_toolsConfig>
      dynamic_rcf_server_;
  dynamic_reconfigure::Server<line_ros_utility::line_toolsConfig>::CallbackType
      dynamic_rcf_callback_;
  // To measure time.
  std::chrono::time_point<std::chrono::system_clock> start_time_, end_time_;
  std::chrono::duration<double> elapsed_seconds_;
};

}  // namespace line_ros_utility

#endif  // LINE_ROS_UTILITY_lINE_ROS_UTILITY_H_
