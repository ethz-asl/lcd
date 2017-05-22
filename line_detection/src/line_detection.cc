#include "line_detection/line_detection.h"

#include <glog/logging.h>
#include <ros/ros.h>
#include <opencv2/rgbd.hpp>

#include <assert.h>
#include <cmath>
#include <iostream>

namespace line_detection {

bool areLinesEqual2D(const cv::Vec4f line1, const cv::Vec4f line2) {
  // First compute the difference in angle. For easier computation not the
  // actual difference in angle, but cos(theta)^2 is computed.
  float vx1 = line1[0] - line1[2];
  float vx2 = line2[0] - line2[2];
  float vy1 = line1[1] - line1[3];
  float vy2 = line2[1] - line2[3];

  float angle_difference = pow((vx1 * vx2 + vy1 * vy2), 2) /
                           ((vx1 * vx1 + vy1 * vy1) * (vx2 * vx2 + vy2 * vy2));
  // Then compute the distance of the two lines. All distances between both end
  // and start points are computed and the lowest is kept.
  float dist[4], min_dist;
  dist[0] = pow(line1[0] - line2[0], 2) + pow(line1[1] - line2[1], 2);
  dist[1] = pow(line1[0] - line2[2], 2) + pow(line1[1] - line2[3], 2);
  dist[2] = pow(line1[2] - line2[0], 2) + pow(line1[3] - line2[1], 2);
  dist[3] = pow(line1[2] - line2[2], 2) + pow(line1[3] - line2[3], 2);
  min_dist = dist[0];
  for (int i = 1; i < 4; ++i) {
    if (dist[i] < min_dist) min_dist = dist[i];
  }
  // Set the comparison parameters. These need to be adjusted correctly.
  // angle_diffrence > 0.997261 refers is equal to effective angle < 3 deg.
  if (angle_difference > 0.95 && min_dist < 5)
    return true;
  else
    return false;
}

LineDetector::LineDetector() {
  lsd_detector_ = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
  edl_detector_ =
      cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
  fast_detector_ = cv::ximgproc::createFastLineDetector();
}

void LineDetector::detectLines(const cv::Mat& image,
                               std::vector<cv::Vec4f>& lines, int detector) {
  lines.clear();
  // Check which detector is chosen by user. If an invalid number is given the
  // default (LSD) is chosen without a warning.
  if (detector == 1) {  // EDL_DETECTOR
    // The edl detector uses a different kind of vector to store the lines in.
    // The conversion is done later.
    std::vector<cv::line_descriptor::KeyLine> edl_lines;
    edl_detector_->detect(image, edl_lines);

    // Write lines to standard format
    for (size_t i = 0u; i < edl_lines.size(); ++i) {
      lines.push_back(cv::Vec4i(
          edl_lines[i].getStartPoint().x, edl_lines[i].getStartPoint().y,
          edl_lines[i].getEndPoint().x, edl_lines[i].getEndPoint().y));
    }

  } else if (detector == 2) {  // FAST_DETECTOR
    fast_detector_->detect(image, lines);
  } else if (detector == 3) {  // HOUGH_DETECTOR
    cv::Mat output;
    // Parameters of the Canny should not be changed (or better: the result is
    // very likely to get worse);
    cv::Canny(image, output, 50, 200, 3);
    // Here parameter changes might improve the result.
    cv::HoughLinesP(output, lines, 1, CV_PI / 180, 50, 30, 10);
  } else {  // LSD_DETECTOR
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

void LineDetector::projectLines2Dto3D(
    const std::vector<cv::Vec4f>& lines2D, const cv::Mat& point_cloud,
    std::vector<cv::Vec<float, 6> >& lines3D) {
  // First check if the point_cloud mat has the right format.
  if (point_cloud.type() != CV_32FC3) {
    ROS_INFO("The input matrix point_cloud must be of type CV_32FC3.");
    return;
  }
  int rows = point_cloud.rows;
  int cols = point_cloud.cols;
  lines3D.clear();
  cv::Point2f start, end, line;
  // cv::Point2i position;
  for (size_t i = 0; i < lines2D.size(); ++i) {
    start.x = floor(lines2D[i][0]);
    start.y = floor(lines2D[i][1]);
    end.x = floor(lines2D[i][2]);
    end.y = floor(lines2D[i][3]);
    // position = start;
    while (std::isnan(point_cloud.at<cv::Vec3f>(start)[0])) {
      line = end - start;
      start.x += floor(line.x / sqrt(line.x * line.x + line.y * line.y) + 0.5);
      start.y += floor(line.y / sqrt(line.x * line.x + line.y * line.y) + 0.5);
      // position = start;
      if (start.x == end.x && start.y == end.y) break;
    }
    if (start.x == end.x && start.y == end.y) continue;
    while (std::isnan(point_cloud.at<cv::Vec3f>(end)[0])) {
      line = start - end;
      end.x += floor(line.x / sqrt(line.x * line.x + line.y * line.y) + 0.5);
      end.y += floor(line.y / sqrt(line.x * line.x + line.y * line.y) + 0.5);
      // position = end;
      if (start.x == end.x && start.y == end.y) break;
    }
    if (start.x == end.x && start.y == end.y) continue;
    lines3D.push_back(cv::Vec<float, 6>(
        point_cloud.at<cv::Vec3f>(start)[0],
        point_cloud.at<cv::Vec3f>(start)[1],
        point_cloud.at<cv::Vec3f>(start)[2], point_cloud.at<cv::Vec3f>(end)[0],
        point_cloud.at<cv::Vec3f>(end)[1], point_cloud.at<cv::Vec3f>(end)[2]));
  }
  ROS_INFO("found vs given: %d, %d", lines3D.size(), lines2D.size());
}

void LineDetector::fuseLines2D(const std::vector<cv::Vec4f>& lines_in,
                               std::vector<cv::Vec4f>& lines_out) {
  lines_out.clear();
  // This list is used to remember which lines have already been assigned to a
  // cluster. Every time a line is assigned, the corresponding index is deleted
  // in this list.
  std::list<int> line_index;
  for (size_t i = 0; i < lines_in.size(); ++i) line_index.push_back(i);
  // This vector is used to store the line clusters until they are merged into
  // one line.
  std::vector<cv::Vec4f> line_cluster;
  // Iterate over all lines.
  for (size_t i = 0; i < lines_in.size(); ++i) {
    line_cluster.clear();
    // If this condition does not hold, the line lines_in[i] has already been
    // merged into a new one. If not, the algorithm tries to find lines that are
    // near this line.
    if (*(line_index.begin()) != i) {
      continue;
    } else {
      line_cluster.push_back(lines_in[i]);
      line_index.pop_front();
    }
    for (std::list<int>::iterator it = line_index.begin();
         it != line_index.end(); ++it) {
      // This loop checks if the line is near any line in the momentary cluster.
      // If yes, it assignes it to the cluster.
      for (cv::Vec4f& line : line_cluster) {
        if (areLinesEqual2D(line, lines_in[*it])) {
          line_cluster.push_back(lines_in[*it]);
          it = line_index.erase(it);
          break;
        }
      }
    }
    // If the cluster size is one, then no cluster was found.
    if (line_cluster.size() == 1) {
      lines_out.push_back(line_cluster[0]);
      continue;
    }
    // Here all the lines of a cluster are merged into one.
    int x_min = 1e4, x_max = 0, y_min = 1e4, y_max = 0, slope = 0;
    for (cv::Vec4f& line : line_cluster) {
      if (line[0] < x_min) x_min = line[0];
      if (line[0] > x_max) x_max = line[0];
      if (line[1] < y_min) y_min = line[1];
      if (line[1] > y_max) y_max = line[1];
      if (line[2] < x_min) x_min = line[2];
      if (line[2] > x_max) x_max = line[2];
      if (line[3] < y_min) y_min = line[3];
      if (line[3] > y_max) y_max = line[3];
      slope += (line[1] - line[3]) / (line[0] - line[3]);
    }
    if (slope > 0)
      lines_out.push_back(cv::Vec4f(x_min, y_min, x_max, y_max));
    else
      lines_out.push_back(cv::Vec4f(x_min, y_max, x_max, y_min));
  }
}

void LineDetector::paintLines(cv::Mat& image,
                              const std::vector<cv::Vec4f>& lines,
                              cv::Vec3b color) {
  cv::Point2i p1, p2;

  for (int i = 0; i < lines.size(); i++) {
    p1.x = lines[i][0];
    p1.y = lines[i][1];
    p2.x = lines[i][2];
    p2.y = lines[i][3];

    cv::line(image, p1, p2, color, 2);
  }
}
}  // namespace line_detection
