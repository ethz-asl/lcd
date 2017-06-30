#include <line_clustering/line_clustering.h>
#include <line_detection/line_detection.h>
#include "line_detection/line_detection_inl.h"

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
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

typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,
    sensor_msgs::PointCloud2>
    MySyncPolicy;

class listenAndPublish {
 public:
  listenAndPublish() {
    ros::NodeHandle node_handle_;
    // The Pointcloud publisher and transformation for RVIZ.
    pcl_pub_ = node_handle_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
        "/vis_pointcloud", 2);
    transform.setOrigin(tf::Vector3(0, 0, 0));
    tf::Quaternion quat;
    quat.setRPY(-3.1415 / 2, 0, 0);
    transform.setRotation(quat);
    // To publish the lines in 3D to rviz.
    display_clusters_.initPublishing(node_handle_);
    // Three topics published by scenenet_ros_tools (there are more but needed
    // here so far).
    image_sub_.subscribe(node_handle_, "/camera/rgb/image_raw", 1);
    depth_sub_.subscribe(node_handle_, "/camera/depth/image_raw", 1);
    info_sub_.subscribe(node_handle_, "/camera/rgb/camera_info", 1);
    pc_sub_.subscribe(node_handle_, "/scenenet_node/scene", 1);
  }

  ~listenAndPublish() { delete sync_; }

  void startListenAndPublishOnce() {
    // The exact time synchronizer makes it possible to have a single callback
    // that recieves messages of all three topics above synchronized. This means
    // every call of the callback function recieves three messages that have the
    // same timestamp.
    sync_ = new message_filters::Synchronizer<MySyncPolicy>(
        MySyncPolicy(10), image_sub_, depth_sub_, info_sub_, pc_sub_);
    sync_->registerCallback(boost::bind(
        &listenAndPublish::pcCallbackPublishOnce, this, _1, _2, _3, _4));
  }

  void pcCallbackPublishOnce(
      const sensor_msgs::ImageConstPtr& rosmsg_image,
      const sensor_msgs::ImageConstPtr& rosmsg_depth,
      const sensor_msgs::CameraInfoConstPtr& camera_info,
      const sensor_msgs::PointCloud2ConstPtr& rosmsg_cloud) {
    // Extract the point cloud from the message.
    pcl::fromROSMsg(*rosmsg_cloud, pcl_cloud_);
    // Extract the image from the message.
    cv_bridge::CvImageConstPtr cv_img_ptr =
        cv_bridge::toCvShare(rosmsg_image, "rgb8");
    cv_image_ = cv_img_ptr->image;
    cv_bridge::CvImageConstPtr cv_depth_ptr =
        cv_bridge::toCvShare(rosmsg_depth);
    cv_depth_ = cv_depth_ptr->image;
    cvtColor(cv_image_, cv_img_gray_, CV_RGB2GRAY);

    line_detection::pclFromSceneNetToMat(pcl_cloud_, 320, 240, cv_cloud_);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    ROS_INFO("**** New Image ******");
    start = std::chrono::system_clock::now();
    line_detector_.detectLines(cv_img_gray_, line_detection::Detector::LSD,
                               lines2D_);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    ROS_INFO("Detecting lines 2D: %f", elapsed_seconds.count());

    start = std::chrono::system_clock::now();
    std::vector<cv::Vec<float, 6> > lines3D_temp;
    std::vector<line_detection::LineWithPlanes> lines3D_temp_wp;
    line_detector_.project2Dto3DwithPlanes(cv_cloud_, lines2D_,
                                           lines3D_temp_wp);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    ROS_INFO("Projecting to 3D: %f", elapsed_seconds.count());

    start = std::chrono::system_clock::now();
    line_detector_.runCheckOn3DLines(cv_cloud_, lines3D_temp_wp, 0,
                                     lines3D_wp_);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    ROS_INFO("Check for valid lines: %f", elapsed_seconds.count());

    ROS_INFO("Lines kept after projection: %d/%d", lines3D_wp_.size(),
             lines2D_.size());

    kmeans_cluster_.setNumberOfClusters(5);
    kmeans_cluster_.setLines(lines3D_wp_);

    start = std::chrono::system_clock::now();
    kmeans_cluster_.computeMeansOfLines();
    kmeans_cluster_.runOnMeansOfLines();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    ROS_INFO("Clustering: %f", elapsed_seconds.count());

    display_clusters_.setFrameID(pcl_cloud_.header.frame_id);
    display_clusters_.setClusters(kmeans_cluster_.getLines(),
                                  kmeans_cluster_.cluster_idx_);
    // The timestamp is set to 0 because rviz is not able to find the right
    // transformation otherwise.
    pcl_cloud_.header.stamp = 0;
    // For testing reasons only one instance is published. Otherwise new point
    // clouds would be published at 1 Hz (which makes it hard to visually
    // inspect them).
    ROS_INFO("**** Started publishing ****");
    broad_caster.sendTransform(tf::StampedTransform(
        transform, ros::Time::now(), "map", pcl_cloud_.header.frame_id));
    pcl_pub_.publish(pcl_cloud_);
    display_clusters_.publish();
  }

 protected:
  // Data storage.
  cv::Mat cv_image_;
  cv::Mat cv_img_gray_;
  cv::Mat cv_cloud_;
  cv::Mat cv_depth_;
  pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud_;
  std::vector<cv::Vec4f> lines2D_;
  std::vector<cv::Vec<float, 6> > lines3D_;
  std::vector<line_detection::LineWithPlanes> lines3D_wp_;
  // Publishers and Subscribers.
  tf::TransformBroadcaster broad_caster;
  tf::Transform transform;
  ros::Publisher pcl_pub_;
  message_filters::Synchronizer<MySyncPolicy>* sync_;
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub_;
  // To have the line_detection utility.
  line_detection::LineDetector line_detector_;
  line_clustering::DisplayClusters display_clusters_;
  line_clustering::KMeansCluster kmeans_cluster_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "detect_and_show_3D");
  listenAndPublish ls;
  ls.startListenAndPublishOnce();
  ros::spin();
  return 0;
}
