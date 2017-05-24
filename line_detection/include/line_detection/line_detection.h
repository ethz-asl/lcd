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

enum class Detector : unsigned int { LSD = 0, EDL = 1, FAST = 2, HOUGH = 3 };

// Returns true if lines are nearby and could be equal (low difference in angle
// and start or end point).
bool areLinesEqual2D(const cv::Vec4f line1, const cv::Vec4f line2);

double checkInBoundary(const double value, const double lower,
                       const double upper) {
  if (value < lower)
    return lower;
  else if (value > upper)
    return upper;
  else
    return value;
}

inline cv::Vec3f crossProduct(const cv::Vec3f a, const cv::Vec3f b) {
  return cv::Vec3f(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0]);
}

inline double normOfVector3D(const cv::Vec3f vector) {
  return sqrt(pow(vector[0], 2) + pow(vector[1], 2) + pow(vector[2], 2));
}

class LineDetector {
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
  void detectLines(const cv::Mat& image, Detector detector,
                   std::vector<cv::Vec4f>& lines);
  void detectLines(const cv::Mat& image, int detector,
                   std::vector<cv::Vec4f>& lines);
  void detectLines(const cv::Mat& image, std::vector<cv::Vec4f>& lines);

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

  // So far this function uses a naive approach which is not yet very robust.
  void projectLines2Dto3D(const std::vector<cv::Vec4f>& lines2D,
                          const cv::Mat& point_cloud,
                          std::vector<cv::Vec<float, 6> >& lines3D);

  // This function clusters nearby lines in line_in and summarizes it as one
  // line. All new lines are stored in lines_out.
  void fuseLines2D(const std::vector<cv::Vec4f>& lines_in,
                   std::vector<cv::Vec4f>& lines_out);

  // Simply paints all lines to image.
  void paintLines(cv::Mat& image, const std::vector<cv::Vec4f>& lines,
                  cv::Vec3b color = {255, 0, 0});

  double computeDistPointToLine3D(const cv::Vec3f& start, const cv::Vec3f& end,
                                  const cv::Vec3f& point);

 private:
  cv::Ptr<cv::LineSegmentDetector> lsd_detector_;
  cv::Ptr<cv::line_descriptor::BinaryDescriptor> edl_detector_;
  cv::Ptr<cv::ximgproc::FastLineDetector> fast_detector_;

  // Compute the slope depending on start and end point of a line.
  inline double computeSlopeOfLine(const cv::Vec4f line) {
    return (line[1] - line[3]) / (line[0] - line[3]);
  }

  // find3DLineStartAndEnd:
  // Input: point_cloud:    Mat of type CV_32FC3, that stores the 3D points. A
  //                        point can be acessed by
  //                        point_cloud.at<cv::Point3f>(j, i).x.
  //
  //        line2D:         The line that was extracted in 2D.
  //
  // Output: lines3D:       If a 3D line is found it is push_backed to this
  //                        vector. It is stored: (x_s, y_s, z_s, x_e, y_e,
  //                        z_e), wher _s/_e means start resp. end point.
  //
  // Working principle of the function: It starts at the starting point of the
  // 2D line and looks if the values in the point_cloud are not NaN there. If
  // they are not, this value is stored as the starting point. If they are NaN,
  // the function jumps to a neighbouring pixel in the direction of the end
  // point and repeats the check. This procedure is done until a non NaN point
  // is found or the search reaches the end point. If a starting point was
  // found, then the same procedure is redone from the end point. It returns
  // true if a line was found and false otherwise.
  bool find3DLineStartAndEnd(const cv::Mat& point_cloud,
                             const cv::Vec4f& line2D,
                             cv::Vec<float, 6>& lines3D);

  // findAndRate3DLine:
  // This function does exactly the same as the above, but in addition it tries
  // to rate the line. The idea is that a line which has a lot of 3D points near
  // it gets a higher rating (near 1) than a line which goes through empty space
  // (has less points near it). THIS DOES NOT WORK WELL YET.
  float findAndRate3DLine(const cv::Mat& point_cloud, const cv::Vec4f& line2D,
                          cv::Vec<float, 6>& line3D);
};

}  // namespace line_detection

#include "line_detection/line_detection_inl.h"

#endif  // LINE_DETECTION_LINE_DETECTION_H_
