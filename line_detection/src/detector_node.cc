// This node advertises a service that extracts lines from input images.
// "detect_lines":
//
// uint8 detector
// sensor_msgs/Image image
// ---
// float32[] start_x
// float32[] start_y
// float32[] end_x
// float32[] end_y
//
//  @param detector:  directly handeld to
//                    line_detection::LineDetector::detectLines
//  @param image:     Input for the line search.
//  @param start/end: Coordinates in pixels of start resp. end point of lines.

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <line_detection/RequestLineDetection.h>
#include <line_detection/line_detection.h>
#include <opencv2/highgui/highgui.hpp>

// Construct the line detector
line_detection::LineDetector line_detector;
// To store the lines
std::vector<cv::Vec4f> lines;
// To store the image
cv_bridge::CvImagePtr image_cv_ptr;

bool detectLineCallback(line_detection::RequestLineDetection::Request& req,
                        line_detection::RequestLineDetection::Response& res) {
  // Convert to cv_ptr (which has a member ->image (cv::Mat))
  image_cv_ptr = cv_bridge::toCvCopy(req.image, "mono8");
  // detect lines
  lines.clear();
  line_detector.detectLines(image_cv_ptr->image, req.detector, lines);

  // Store lines to the response.
  res.start_x.reserve(lines.size());
  res.start_y.reserve(lines.size());
  res.end_x.reserve(lines.size());
  res.end_y.reserve(lines.size());
  for (int i = 0; i < lines.size(); i++) {
    res.start_x.push_back(lines[i][0]);
    res.start_y.push_back(lines[i][1]);
    res.end_x.push_back(lines[i][2]);
    res.end_y.push_back(lines[i][3]);
  }
  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "detector");
  ros::NodeHandle node_handle;

  ros::ServiceServer server =
      node_handle.advertiseService("detect_lines", &detectLineCallback);
  ros::spin();
}
