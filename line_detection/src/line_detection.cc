#include "line_detection/line_detection.h"

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

// NOT YET TESTED
void displayPointCloud(const cv::Mat& image, const cv::Mat& depth,
                       const cv::Mat& K) {
  std::vector<cv::Point3d> points3d;
  cv::rgbd::depthTo3d(depth, K, points3d);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  cloud->width = depth.cols * depth.rows;
  cloud->height = 1;
  cloud->is_dense = false;
  cloud->resize(cloud->width);

  for (int i = 0; i < depth.rows; i++) {
    for (int j = 0; j < depth.cols; j++) {
      cloud->points[i].x = points3d[i].x;
      cloud->points[i].y = points3d[i].y;
      cloud->points[i].z = points3d[i].z;
      cloud->points[i].r = image.at<cv::Vec3i>(j, i)[0];
      cloud->points[i].g = image.at<cv::Vec3i>(j, i)[1];
      cloud->points[i].b = image.at<cv::Vec3i>(j, i)[2];
    }
  }

  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud);
  while (!viewer.wasStopped()) {
  }
}

}  // namespace line_detection
