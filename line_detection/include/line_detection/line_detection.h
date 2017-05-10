#ifndef LINE_DETECTION_LINE_DETECTION_H_
#define LINE_DETECTION_LINE_DETECTION_H_

#include "line_detection/common.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/ximgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace line_detection {

const int LSD_DETECTOR = 0;
const int EDL_DETECTOR = 1;
const int FAST_DETECTOR = 2;
const int HOUGH_DETECTOR = 3;

class LineDetector {
 private:
  cv::Ptr<cv::LineSegmentDetector> lsd_detector_;
  cv::Ptr<cv::line_descriptor::BinaryDescriptor> edl_detector_;
  cv::Ptr<cv::ximgproc::FastLineDetector> fast_detector_;

 public:
  LineDetector();

  // detectLines:
  // Input: image:    The image on which the lines should be detected.
  //
  //        detector: 0-> LSD, 1->EDL, 2->FAST, 3-> HOUGH
  //                  Default is LSD. It is chosen even if an invalid number is
  //                  given
  //
  // Output: lines:   The lines are stored in the following format:
  //                  {start.x, start.y, end.x, end.y}
  void detectLines(const cv::Mat& image, std::vector<cv::Vec4f>& lines,
                   int detector = LSD_DETECTOR);

  // computePointCloud:
  // Input: image:    This is used to assign color values to the point cloud.
  //
  //        depth:    The actuall depth values. Is used to compute the point
  //                  cloud.
  //
  //        K:        Calibration matrix of the camera.
  //
  //        point_coud: The points found are stored in there. The cloud is
  //                    ordered. This means that for every pixel in depth/image
  //                    there is a point. Therefore some points will have NaN
  //                    values for coordinates (wherever the depth image
  //                    contains NaN/zeros).
  void computePointCloud(const cv::Mat image, const cv::Mat& depth,
                         const cv::Mat& K,
                         pcl::PointCloud<pcl::PointXYZRGB>& point_cloud);
  // void paintLines(cv::Mat& image, const std::vector<cv::Vec4f>& lines,
  //                cv::Vec3b color = {255, 0, 0});
};

}  // namespace line_detection

#include "line_detection/line_detection_inl.h"

#endif  // LINE_DETECTION_LINE_DETECTION_H_
