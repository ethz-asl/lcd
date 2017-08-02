#include <ros/ros.h>

#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/publisher.h>

#include <line_ros_utility/line_ros_utility.h>

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>
    MySyncPolicy;

class convertFreiburgToLineTools {
 public:
  convertFreiburgToLineTools() {
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

    image_sub_.subscribe(node_handle_, "/camera/rgb/image_color", 1);
    depth_sub_.subscribe(node_handle_, "/camera/depth/image", 1);
    info_sub_.subscribe(node_handle_, "/camera/rgb/camera_info", 1);

    sync_ = new message_filters::Synchronizer<MySyncPolicy>(
        MySyncPolicy(10), image_sub_, depth_sub_, info_sub_);
    sync_->registerCallback(
        boost::bind(&convertFreiburgToLineTools::callback, this, _1, _2, _3));
  }

  void computePointCloudFreiburg(const cv::Mat& depth, cv::Mat* cloud) {
    CHECK_EQ(depth.type(), CV_32FC1);
    const size_t height = depth.rows;
    const size_t width = depth.cols;
    constexpr double focalLength = 525.0;
    constexpr double centerX = 319.5;
    constexpr double centerY = 239.5;
    constexpr double scalingFactor = 1;
    cv::Vec3f point3D;
    cloud->create(height, width, CV_32FC3);
    cv::Vec3f mean = {0, 0, 0};
    for (size_t u = 0; u < width; ++u) {
      for (size_t v = 0; v < height; ++v) {
        point3D[2] = depth.at<float>(v, u) / scalingFactor;
        point3D[0] = (u - centerX) * point3D[2] / focalLength;
        point3D[1] = (v - centerY) * point3D[2] / focalLength;
        cloud->at<cv::Vec3f>(v, u) = point3D;
        mean += point3D;
      }
    }
    mean = mean * 1.0f / ((double)width * height);
  }

  void createEmptyInstance(const cv::Mat& image, cv::Mat* instances) {
    const size_t height = image.rows;
    const size_t width = image.cols;
    instances->create(height, width, CV_8UC3);
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
        instances->at<cv::Vec3b>(i, j) = {255, 0, 0};
      }
    }
  }

  void callback(const sensor_msgs::ImageConstPtr& rosmsg_image,
                const sensor_msgs::ImageConstPtr& rosmsg_depth,
                const sensor_msgs::CameraInfoConstPtr& camera_info) {
    sensor_msgs::Image new_img_msg = *rosmsg_image;
    sensor_msgs::Image new_depth_msg = *rosmsg_depth;
    sensor_msgs::CameraInfo new_info_msg = *camera_info;
    new_depth_msg.header.stamp = rosmsg_image->header.stamp;
    new_info_msg.header.stamp = rosmsg_image->header.stamp;

    cv_bridge::CvImageConstPtr cv_img_ptr =
        cv_bridge::toCvShare(rosmsg_depth, "32FC1");
    computePointCloudFreiburg(cv_img_ptr->image, &(cvimage_cloud_.image));
    cvimage_cloud_.header = new_depth_msg.header;
    cvimage_cloud_.encoding = "32FC3";

    createEmptyInstance(cvimage_cloud_.image, &(cvimage_instances_.image));
    cvimage_instances_.header = new_img_msg.header;
    cvimage_instances_.encoding = "8UC3";

    ROS_INFO("publishing");
    cloud_pub_.publish(cvimage_cloud_.toImageMsg());
    image_pub_.publish(new_img_msg);
    depth_pub_.publish(new_depth_msg);
    info_pub_.publish(new_info_msg);
    instances_pub_.publish(cvimage_instances_.toImageMsg());
  }

 protected:
  message_filters::Synchronizer<MySyncPolicy>* sync_;
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
  ros::Publisher cloud_pub_;
  ros::Publisher image_pub_;
  ros::Publisher depth_pub_;
  ros::Publisher info_pub_;
  ros::Publisher instances_pub_;

  cv_bridge::CvImage cvimage_instances_;
  cv_bridge::CvImage cvimage_cloud_;
  cv::Mat cv_cloud_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "convert_scenenet_to_line_tools");
  convertFreiburgToLineTools converter;
  ros::spin();
  return 0;
}
