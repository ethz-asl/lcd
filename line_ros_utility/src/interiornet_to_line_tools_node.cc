#include <ros/ros.h>
#include <ros/time.h>

#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/publisher.h>

#include <tf/transform_listener.h>

#include <line_ros_utility/line_ros_utility.h>

typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
    sensor_msgs::CameraInfo, sensor_msgs::PointCloud2>
    MySyncPolicy;

class convertInteriorNetToLineTools {
 public:
  convertInteriorNetToLineTools() {
    ros::NodeHandle node_handle_;

    image_pub_ =
        node_handle_.advertise<sensor_msgs::Image>("/line_tools/image/rgb", 2);
    depth_pub_ = node_handle_.advertise<sensor_msgs::Image>(
        "/line_tools/image/depth", 2);
    instances_pub_ = node_handle_.advertise<sensor_msgs::Image>(
        "/line_tools/image/instances", 2);
    info_pub_ = node_handle_.advertise<sensor_msgs::CameraInfo>(
        "/line_tools/camera_info", 2);
    cloud_pub_ = node_handle_.advertise<sensor_msgs::Image>(
        "/line_tools/point_cloud", 2);
    camera_to_world_matrix_pub_ =
        node_handle_.advertise<geometry_msgs::TransformStamped>(
            "/line_tools/camera_to_world_matrix", 2);

    image_sub_.subscribe(node_handle_, "/camera/rgb/image_raw", 1);
    depth_sub_.subscribe(node_handle_, "/camera/depth/image_raw", 1);
    info_sub_.subscribe(node_handle_, "/camera/rgb/camera_info", 1);
    pc_sub_.subscribe(node_handle_, "/interiornet_node/scene", 1);
    instances_sub_.subscribe(node_handle_, "/camera/instances/image_raw", 1);

    sync_ = new message_filters::Synchronizer<MySyncPolicy>(
        MySyncPolicy(10), image_sub_, depth_sub_, instances_sub_, info_sub_,
        pc_sub_);
    sync_->registerCallback(boost::bind(&convertInteriorNetToLineTools::callback,
                                        this, _1, _2, _3, _4, _5));
  }

  void pclFromInteriorNetToMat(const pcl::PointCloud<pcl::PointXYZRGB>& pcl_cloud,
                            cv::Mat* mat_cloud) {
    CHECK_NOTNULL(mat_cloud);
    const size_t width = 320;
    const size_t height = 240;
    CHECK_EQ(pcl_cloud.points.size(), width * height);
    mat_cloud->create(height, width, CV_32FC3);
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
        mat_cloud->at<cv::Vec3f>(i, j) =
            cv::Vec3f(pcl_cloud.points[j + i * width].x,
                      pcl_cloud.points[j + i * width].y,
                      pcl_cloud.points[j + i * width].z);
      }
    }
  }

  void callback(const sensor_msgs::ImageConstPtr& rosmsg_image,
                const sensor_msgs::ImageConstPtr& rosmsg_depth,
                const sensor_msgs::ImageConstPtr& rosmsg_instances,
                const sensor_msgs::CameraInfoConstPtr& camera_info,
                const sensor_msgs::PointCloud2ConstPtr& rosmsg_cloud) {
    ros::Time stamp;
    tf::StampedTransform transform;
    geometry_msgs::TransformStamped transform_msg;
    pcl::fromROSMsg(*rosmsg_cloud, pcl_cloud_);
    pclFromInteriorNetToMat(pcl_cloud_, &(cvimage_cloud_.image));
    cvimage_cloud_.header = rosmsg_cloud->header;
    cvimage_cloud_.encoding = "32FC3";

    cloud_pub_.publish(cvimage_cloud_.toImageMsg());
    image_pub_.publish(*rosmsg_image);
    depth_pub_.publish(*rosmsg_depth);
    info_pub_.publish(*camera_info);
    instances_pub_.publish(*rosmsg_instances);

    // Retrieve timestamp from any message above (they are all synchronized and
    // therefore have the same stamp).
    stamp = rosmsg_image->header.stamp;
    // Obtain TF message at the given timestamp.
    tf_listener_.waitForTransform("/interiornet_camera_frame", "/world", stamp,
                                  ros::Duration(1.0));
    tf_listener_.lookupTransform("/interiornet_camera_frame", "/world", stamp,
                                 transform);
    // Convert TF to geometry_msgs/TransformStamped and publish it.
    tf::transformStampedTFToMsg(transform, transform_msg);
    camera_to_world_matrix_pub_.publish(transform_msg);
  }

 protected:
  message_filters::Synchronizer<MySyncPolicy>* sync_;
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::Image> instances_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub_;
  ros::Publisher cloud_pub_;
  ros::Publisher image_pub_;
  ros::Publisher depth_pub_;
  ros::Publisher info_pub_;
  ros::Publisher instances_pub_;
  ros::Publisher camera_to_world_matrix_pub_;
  tf::TransformListener tf_listener_;

  pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud_;
  cv_bridge::CvImage cvimage_cloud_;
  cv::Mat cv_cloud_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "convert_interiornet_to_line_tools");
  convertInteriorNetToLineTools converter;
  ros::spin();
  return 0;
}
