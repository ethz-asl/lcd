#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>

#include "line_detection/common.h"
#include "line_detection/line_detection.h"
#include "line_detection/test/testing-entrypoint.h"

namespace line_detection {

class LineDetectionTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  LineDetectionTest() { SetUp(); }

  virtual ~LineDetectionTest() {}

  cv::Mat test_image;
  cv::Mat test_img_gray;
  cv::Mat test_depth_load;
  cv::Mat test_depth;
  line_detection::LineDetector line_detector;
  std::vector<cv::Vec4f> lines;

  // TODO: Why virtual?
  virtual void SetUp() {
    // Load the test image and compute a grayscale version of it.
    std::string testimage_path("test_data/hall.jpg");
    test_image = cv::imread(testimage_path, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(test_image, test_img_gray, CV_BGR2GRAY);
    // Load the depth data corresponding to the test image.
    std::string testdepth_path("test_data/hall_depth.png");
    test_depth_load = cv::imread(testdepth_path, CV_LOAD_IMAGE_UNCHANGED);
    if (test_depth_load.type() != CV_16UC1)
      test_depth_load.convertTo(test_depth, CV_16UC1);
    else
      test_depth = test_depth_load;
  }
};

TEST_F(LineDetectionTest, testLineDetection) {
  int n_lines;
  // Calling the detector with LSD.
  line_detector.detectLines(test_img_gray, lines, 0);
  n_lines = lines.size();
  EXPECT_EQ(n_lines, 716)
      << "LSD detection: Expected 84 lines to be found. Found " << n_lines;
  // Calling the detector with EDL.
  line_detector.detectLines(test_img_gray, lines, 1);
  n_lines = lines.size();
  EXPECT_EQ(n_lines, 172)
      << "EDL detection: Expected 18 lines to be found. Found " << n_lines;
  // Calling the detector with FAST.
  line_detector.detectLines(test_img_gray, lines, 2);
  n_lines = lines.size();
  EXPECT_EQ(n_lines, 598)
      << "Fast detection: Expected 70 lines to be found. Found " << n_lines;
  // Calling the detector with HOUGH.
  line_detector.detectLines(test_img_gray, lines, 3);
  n_lines = lines.size();
  EXPECT_EQ(n_lines, 165)
      << "HOUGH detection: Expected 16 lines to be found. Found " << n_lines;
}

TEST_F(LineDetectionTest, testComputePointCloud) {
  // Create calibration matrix and fill it (with non calibrated values!).
  cv::Mat K(3, 3, CV_32FC1);
  K.at<float>(0, 0) = 570.3f;
  K.at<float>(0, 1) = 0.0f;
  K.at<float>(0, 2) = 960.0f;
  K.at<float>(1, 0) = 0.0f;
  K.at<float>(1, 1) = 570.3f;
  K.at<float>(1, 2) = 540.0f;
  K.at<float>(2, 0) = 0.0f;
  K.at<float>(2, 1) = 0.0f;
  K.at<float>(2, 2) = 1.0f;
  // Point_cloud to be filled.
  pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
  // Fill the point cloud (this is the functioned that is tested here)
  line_detector.computePointCloud(test_image, test_depth, K, point_cloud);
  // Compue the mean of all entries of the point cloud
  double x_mean = 0;
  double y_mean = 0;
  double z_mean = 0;
  double r_mean = 0;
  double g_mean = 0;
  double b_mean = 0;
  for (int i = 0; i < point_cloud.size(); i++) {
    if (std::isnan(point_cloud.points[i].x)) continue;
    x_mean += point_cloud.points[i].x;
    y_mean += point_cloud.points[i].y;
    z_mean += point_cloud.points[i].z;
    r_mean += point_cloud.points[i].r;
    g_mean += point_cloud.points[i].g;
    b_mean += point_cloud.points[i].b;
  }
  x_mean = x_mean / point_cloud.size();
  y_mean = y_mean / point_cloud.size();
  z_mean = z_mean / point_cloud.size();
  r_mean = r_mean / point_cloud.size();
  g_mean = g_mean / point_cloud.size();
  b_mean = b_mean / point_cloud.size();

  // The function LineDetector::computePointCloud computes an ordered point
  // cloud. It does fill in points for which the depth image no information,
  // these are then just NaN values. But this means for every pixel there should
  // be a point in the cloud.
  EXPECT_EQ(point_cloud.size(), 1920 * 1080);
  // These are all values that were precomputed with the above calibration
  // matrix K. They are not the true values!
  EXPECT_NEAR(x_mean, 0.324596, 1e-5);
  EXPECT_NEAR(y_mean, -0.147148, 1e-5);
  EXPECT_NEAR(z_mean, 1.69212, 1e-5);
  EXPECT_NEAR(r_mean, 108.686, 1e-2);
  EXPECT_NEAR(g_mean, 117.155, 1e-2);
  EXPECT_NEAR(b_mean, 116.337, 1e-2);
}

TEST_F(LineDetectionTest, testAreLinesEqual2D) {
  EXPECT_TRUE(line_detection::areLinesEqual2D(cv::Vec4f(0, 0, 10, 10),
                                              cv::Vec4f(0, 0, 10, 10)));
  EXPECT_TRUE(line_detection::areLinesEqual2D(cv::Vec4f(0, 0, 10, 10),
                                              cv::Vec4f(10, 10, 30, 30)));
  EXPECT_FALSE(line_detection::areLinesEqual2D(cv::Vec4f(0, 0, 10, 10),
                                               cv::Vec4f(0, 0, 0, 10)));
}
}  // namespace line_detection

LINE_DETECTION_TESTING_ENTRYPOINT
