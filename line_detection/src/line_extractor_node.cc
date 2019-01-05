// This node advertises a service that extracts 2D and lines with planes from
// input images.
// "extract_lines":
//
// sensor_msgs/Image image
// sensor_msgs/Image cloud
// sensor_msgs/CameraInfo camera_info
// uint8 detector
// ---
// line_detection/Line3DWithHessians[] lines
// geometry_msgs/Point[] start2D
// geometry_msgs/Point[] end2D
// uint8 frame_index
#include <line_detection/line_detection.h>

#include <ros/ros.h>

#include <line_detection/ExtractLines.h>

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/highgui/highgui.hpp>

// Construct the line detector
line_detection::LineDetector line_detector;
// To store the lines
std::vector<cv::Vec4f> lines_2D;
std::vector<cv::Vec4f> lines_2D_fused;
std::vector<cv::Vec4f> lines_2D_tmp;
std::vector<line_detection::LineWithPlanes> lines_3D_tmp;
std::vector<line_detection::LineWithPlanes> lines_3D;
// To store the image
cv_bridge::CvImageConstPtr image_cv_ptr;
cv::Mat cv_image_rgb;
cv::Mat cv_image_gray;
// To store the point cloud
cv_bridge::CvImageConstPtr cv_cloud_ptr;
cv::Mat cv_cloud;
// Projection matrix
cv::Mat camera_P;
// Stores the index of the current frame
int frame_index;

bool detectLinesCallback(line_detection::ExtractLines::Request& req,
                         line_detection::ExtractLines::Response& res) {
  // Convert to cv_ptr (which has a member ->image (cv::Mat)).
  image_cv_ptr = cv_bridge::toCvCopy(req.image, "rgb8");
  cv_image_rgb = image_cv_ptr->image;
  cv::cvtColor(cv_image_rgb, cv_image_gray, CV_RGB2GRAY);

  // Obtain projection matrix.
  image_geometry::PinholeCameraModel camera_model;
  camera_model.fromCameraInfo(req.camera_info);
  camera_P = cv::Mat(camera_model.projectionMatrix());
  camera_P.convertTo(camera_P, CV_32F);

  // Obtain point cloud.
  cv_cloud_ptr = cv_bridge::toCvCopy(req.cloud, "32FC3");
  cv_cloud = cv_cloud_ptr->image;
  CHECK(cv_cloud.type() == CV_32FC3);

  // Detect 2D lines.
  lines_2D.clear();
  line_detector.detectLines(cv_image_gray, req.detector, &lines_2D);
  line_detector.fuseLines2D(lines_2D, &lines_2D_fused);

  // Project to 3D.
  line_detector.project2Dto3DwithPlanes(cv_cloud, cv_image_rgb, camera_P,
                                         lines_2D_fused, true, &lines_2D_tmp,
                                         &lines_3D_tmp);
  // Perform checks.
  line_detector.runCheckOn3DLines(cv_cloud, camera_P, lines_2D_tmp,
                                  lines_3D_tmp, &lines_2D, &lines_3D);

  // Store lines to the response.
  res.lines.resize(lines_2D.size());
  res.start2D.resize(lines_2D.size());
  res.end2D.resize(lines_2D.size());
  res.frame_index = frame_index++;

  for (size_t i = 0u; i < lines_2D.size(); ++i) {
    res.start2D[i].x = lines_2D[i][0];
    res.start2D[i].y = lines_2D[i][1];
    res.end2D[i].x = lines_2D[i][2];
    res.end2D[i].y = lines_2D[i][3];
    res.lines[i].start3D.x = lines_3D[i].line[0];
    res.lines[i].start3D.y = lines_3D[i].line[1];
    res.lines[i].start3D.z = lines_3D[i].line[2];
    res.lines[i].end3D.x = lines_3D[i].line[3];
    res.lines[i].end3D.y = lines_3D[i].line[4];
    res.lines[i].end3D.z = lines_3D[i].line[5];
    res.lines[i].hessian_right = {lines_3D[i].hessians[0][0],
                                  lines_3D[i].hessians[0][1],
                                  lines_3D[i].hessians[0][2],
                                  lines_3D[i].hessians[0][3]};
    res.lines[i].hessian_left = {lines_3D[i].hessians[1][0],
                                 lines_3D[i].hessians[1][1],
                                 lines_3D[i].hessians[1][2],
                                 lines_3D[i].hessians[1][3]};
    switch (lines_3D[i].type) {
      case line_detection::LineType::DISCONT:
        res.lines[i].line_type = 0;
        break;
      case line_detection::LineType::PLANE:
        res.lines[i].line_type = 1;
        break;
      case line_detection::LineType::EDGE:
        res.lines[i].line_type = 2;
        break;
      case line_detection::LineType::INTERSECT:
        res.lines[i].line_type = 3;
        break;
      default:
        ROS_ERROR("Illegal line type. Possible types are DISCONT, PLANE, EDGE "
                   "and INTERSECT");
        return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "line_detector");
  ros::NodeHandle node_handle;

  ros::ServiceServer server =
      node_handle.advertiseService("extract_lines", &detectLinesCallback);
  frame_index = 0;
  ros::spin();
}
