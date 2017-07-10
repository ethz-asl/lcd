// A test node that computes a point cloud form RGB-D data with the
// line_detection library

#include <ros/console.h>
#include <ros/ros.h>

#include <line_detection/line_detection.h>

#include <pcl/conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/publisher.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

#include <visualization_msgs/Marker.h>

int main(int argc, char** argv) {
  // This lets DEBUG messages display on console
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                     ros::console::levels::Debug)) {
    ros::console::notifyLoggerLevelsChanged();
  }
  ros::init(argc, argv, "test_point_cloud");
  ros::NodeHandle node_handle;

  // Load both depth and color image.
  cv::Mat depth;
  cv::Mat image;
  std::string img_path;
  std::string depth_path;
  if (argc == 3) {
    img_path = argv[1];
    depth_path = argv[2];
  } else {
    img_path = "hall.jpg";
    depth_path = "hall_depth.png";
  }
  image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
  depth = cv::imread(depth_path, CV_LOAD_IMAGE_UNCHANGED);
  if (!image.data) {
    ROS_INFO_STREAM(
        "Image could not be loaded. Please make shure to run the node in a "
        "directory that contains the image hall.jpg and the corresponding "
        "depth image hall_depth.png."
        << endl
        << "Alternatively give the path of the RGB image as the first argument "
           "and the path of the depth image as the second argument to the "
           "function.");
    return -1;
  }
  if (!depth.data) {
    ROS_INFO_STREAM(
        "Image could not be loaded. Please make sure to run the node in a "
        "directory that contains the image hall.jpg and the corresponding "
        "depth image hall_depth.png"
        << endl
        << "Alternatively give the path of the RGB image as the first argument "
           "and the path of the depth image as the second argument to the "
           "function.");
    return -1;
  }

  // Create a point cloud. The pointer is used to handle to the PCLVsualizer,
  // the reference is needed to handle to cloud to the computePointCloud
  // function.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>& cloud = *cloud_ptr;

  // Create the calibration matrix. The values are more or less arbitrary.
  cv::Mat K(3, 3, CV_32FC1);
  K.at<float>(0, 0) = 277.128f;
  K.at<float>(0, 1) = 0.0f;
  K.at<float>(0, 2) = 960.0f;
  K.at<float>(1, 0) = 0.0f;
  K.at<float>(1, 1) = 289.706f;
  K.at<float>(1, 2) = 540.0f;
  K.at<float>(2, 0) = 0.0f;
  K.at<float>(2, 1) = 0.0f;
  K.at<float>(2, 2) = 1.0f;

  // Compute the point cloud.
  line_detection::LineDetector line_detector;
  line_detector.computePointCloud(image, depth, K, &cloud);

  // Sparsify the point cloud for better visualization performance.
  pcl::PointCloud<pcl::PointXYZRGB> sparse_cloud;
  if (cloud.size() > 5e5) {
    for (int i = 0; i < cloud.size(); i += 4) {
      sparse_cloud.push_back(cloud.points[i]);
    }
  } else {
    sparse_cloud = cloud;
  }

  // Not to compute the 3D lines, we use a diffetent type of point cloud.
  cv::Mat pc_mat;
  cv::Mat img_gray;
  cvtColor(image, img_gray, CV_BGR2GRAY);
  cv::rgbd::depthTo3d(depth, K, pc_mat);
  std::vector<cv::Vec4f> lines2D;
  std::vector<cv::Vec<float, 6> > lines3D;
  line_detector.detectLines(img_gray, &lines2D);
  line_detector.projectLines2Dto3D(lines2D, pc_mat, &lines3D);
  // // -------- Visualize it with PCLVisualizer -------- //
  // pcl::visualization::PCLVisualizer viewer("3D Viewer");
  // viewer.setBackgroundColor(1, 1, 1);
  // viewer.addCoordinateSystem(1.0f, "global");
  // viewer.addPointCloud(cloud_ptr, "original point cloud");
  // viewer.spin();

  // -----  Visualize it with RVIZ ---------- //

  // Create a tf transform. The transform defines a fixed global frame and a
  // rotated frame with it. The rotation is 90 deg around x-axis (this goes from
  // a camera frame to a normal world frame).
  tf::TransformBroadcaster broad_caster;
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(0, 0, 0));
  tf::Quaternion quat;
  quat.setRPY(-3.1415 / 2, 0, 0);
  transform.setRotation(quat);
  // Create the publisher for the point cloud. The topic vis_pointcloud was set
  // manually in rviz.
  ros::Publisher pcl_pub =
      node_handle.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
          "vis_pointcloud", 2);
  sparse_cloud.header.frame_id = "my_frame";

  // Create the publisher for the lines.
  ros::Publisher marker_pub = node_handle.advertise<visualization_msgs::Marker>(
      "visualization_marker", 1000);
  visualization_msgs::Marker disp_lines;
  disp_lines.header.frame_id = "my_frame";
  line_detection::storeLines3DinMarkerMsg(lines3D, &disp_lines);
  // Publish the messages. Once every 10 seconds, the transform, the cloud
  // and all the lines are published. This is because rviz often fails to
  // read all the messages in one go.
  ros::Rate rate(0.1);
  while (ros::ok()) {
    broad_caster.sendTransform(
        tf::StampedTransform(transform, ros::Time::now(), "map", "my_frame"));
    pcl_pub.publish(sparse_cloud);
    marker_pub.publish(disp_lines);
    rate.sleep();
  }
  return 0;
}
