#include "line_detection/line_detection.h"
#include <assert.h>
#include <ros/ros.h>
#include <iostream>
#include <opencv2/rgbd.hpp>

namespace line_detection {

LineDetector::LineDetector() {
  // TODO: Can I assign some kind of a NULL pointer to these (normal NULL does
  // not work)? That would make it possible to only initialize the ones I need
  // (without a workaround with additional variables that store if a detector
  // already has been initialized).

  lsd_detector_ = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
  edl_detector_ =
      cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
  fast_detector_ = cv::ximgproc::createFastLineDetector();
}

void LineDetector::detectLines(const cv::Mat& image,
                               std::vector<cv::Vec4f>& lines, int detector) {
  // Check which detector is chosen by user. If an invalid number is given the
  // default (LSD) is chosen without a warning.
  if (detector == 1) {  // EDL_DETECTOR
    // The edl detector uses a different kind of vector to store the lines in.
    // The conversion is done later.
    std::vector<cv::line_descriptor::KeyLine> edl_lines;
    edl_detector_->detect(image, edl_lines);
    lines.clear();
    // Write lines to standard format
    for (int i = 0; i < edl_lines.size(); i++) {
      lines.push_back(cv::Vec4i(
          edl_lines[i].getStartPoint().x, edl_lines[i].getStartPoint().y,
          edl_lines[i].getEndPoint().x, edl_lines[i].getEndPoint().y));
    }

  } else if (detector == 2) {  // FAST_DETECTOR
    lines.clear();
    fast_detector_->detect(image, lines);
  } else if (detector == 3) {  // HOUGH_DETECTOR
    cv::Mat output;
    // Parameters of the Canny should not be changed (or better: the result is
    // very likely to get worse);
    cv::Canny(image, output, 50, 200, 3);
    // Here parameter changes might improve the result.
    cv::HoughLinesP(output, lines, 1, CV_PI / 180, 50, 30, 10);
  } else {  // LSD_DETECTOR
    lines.clear();
    lsd_detector_->detect(image, lines);
  }
}

void LineDetector::computePointCloud(
    const cv::Mat image, const cv::Mat& depth, const cv::Mat& K,
    pcl::PointCloud<pcl::PointXYZRGB>& point_cloud) {
  // The points are intermediatelly stored in here (because the function
  // rgbd::depthTo3d is used). This is not the fastest way, so if speed is
  // needed this functoin should be rewriten.
  cv::Mat points3d;
  cv::rgbd::depthTo3d(depth, K, points3d);

  // Here the point could gets filled with the values found by depthTo3d.
  int rows = depth.rows;
  int cols = depth.cols;
  point_cloud.width = cols;
  point_cloud.height = rows;
  point_cloud.is_dense = false;
  point_cloud.reserve(cols * rows);

  pcl::PointXYZRGB pcl_point;
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      // ROS_DEBUG("at: i = %d, j = %d", i, j);
      pcl_point.x = points3d.at<cv::Vec3f>(j, i)[0];
      pcl_point.y = points3d.at<cv::Vec3f>(j, i)[1];
      pcl_point.z = points3d.at<cv::Vec3f>(j, i)[2];
      pcl_point.r = image.at<cv::Vec3b>(j, i)[0];
      pcl_point.g = image.at<cv::Vec3b>(j, i)[1];
      pcl_point.b = image.at<cv::Vec3b>(j, i)[2];
      point_cloud.push_back(pcl_point);
    }
  }
}
// void LineDetector::paintLines(cv::Mat& image,
//                               const std::vector<cv::Vec4f>& lines,
//                               cv::Vec3b color) {
//   cv::Point2i p1, p2;
//
//   for (int i = 0; i < lines.size(); i++) {
//     p1.x = lines[i][0];
//     p1.y = lines[i][1];
//     p2.x = lines[i][2];
//     p2.y = lines[i][3];
//
//     cv::line(image, p1, p2, color, 2);
//   }
// }
}  // namespace line_detection
