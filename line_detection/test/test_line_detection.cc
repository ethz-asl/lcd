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

  cv::Mat test_image_;
  cv::Mat test_img_gray_;
  cv::Mat test_depth_load_;
  cv::Mat test_depth_;
  line_detection::LineDetector line_detector_;
  std::vector<cv::Vec4f> lines_;

  virtual void SetUp() {
    // Load the test image and compute a grayscale version of it.
    std::string testimage_path("test_data/hall.jpg");
    test_image_ = cv::imread(testimage_path, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(test_image_, test_img_gray_, CV_BGR2GRAY);
    // Load the depth data corresponding to the test image.
    std::string testdepth_path("test_data/hall_depth.png");
    test_depth_load_ = cv::imread(testdepth_path, CV_LOAD_IMAGE_UNCHANGED);
    if (test_depth_load_.type() != CV_16UC1)
      test_depth_load_.convertTo(test_depth_, CV_16UC1);
    else
      test_depth_ = test_depth_load_;
  }
};

TEST_F(LineDetectionTest, testLSDLineDetection) {
  size_t n_lines;
  // Calling the detector with LSD.
  line_detector_.detectLines(test_img_gray_, line_detection::Detector::LSD,
                             lines_);
  n_lines = lines_.size();
  EXPECT_EQ(n_lines, 716)
      << "LSD detection: Expected 716 lines to be found. Found " << n_lines;
}
TEST_F(LineDetectionTest, testEDLLineDetection) {
  size_t n_lines;
  // Calling the detector with EDL.
  line_detector_.detectLines(test_img_gray_, line_detection::Detector::EDL,
                             lines_);
  n_lines = lines_.size();
  EXPECT_EQ(n_lines, 172)
      << "EDL detection: Expected 172 lines to be found. Found " << n_lines;
}
TEST_F(LineDetectionTest, testFASTLineDetection) {
  size_t n_lines;
  // Calling the detector with FAST.
  line_detector_.detectLines(test_img_gray_, line_detection::Detector::FAST,
                             lines_);
  n_lines = lines_.size();
  EXPECT_EQ(n_lines, 598)
      << "Fast detection: Expected 598 lines to be found. Found " << n_lines;
}
TEST_F(LineDetectionTest, testHoughLineDetection) {
  size_t n_lines;
  // Calling the detector with HOUGH.
  line_detector_.detectLines(test_img_gray_, line_detection::Detector::HOUGH,
                             lines_);
  n_lines = lines_.size();
  EXPECT_EQ(n_lines, 165)
      << "HOUGH detection: Expected 165 lines to be found. Found " << n_lines;
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
  line_detector_.computePointCloud(test_image_, test_depth_, K, point_cloud);
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

TEST_F(LineDetectionTest, testCheckInBoundary) {
  EXPECT_EQ(line_detection::checkInBoundary(1, 0, 3), 1);
  EXPECT_EQ(line_detection::checkInBoundary(-1, 0, 3), 0);
  EXPECT_EQ(line_detection::checkInBoundary(10, 0, 3), 3);
}

TEST_F(LineDetectionTest, testCrossProdcut) {
  EXPECT_EQ(
      line_detection::crossProduct(cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0)),
      cv::Vec3f(0, 0, 1));
}

TEST_F(LineDetectionTest, testdistPointToLine) {
  EXPECT_EQ(distPointToLine(cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0),
                            cv::Vec3f(0, 1, 0)),
            1);
}

TEST_F(LineDetectionTest, testScalarProduct) {
  EXPECT_EQ(
      line_detection::scalarProduct(cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0)), 0);
}

TEST_F(LineDetectionTest, testComputeDFromPlaneNormal) {
  cv::Vec3f normal(1, 2, 3);
  cv::Vec3f anchor(5, 6, 7);
  double d = line_detection::computeDfromPlaneNormal(normal, anchor);
  EXPECT_FLOAT_EQ(scalarProduct(normal, anchor) + d, 0);
}

TEST_F(LineDetectionTest, testErrorPointToPlane) {
  cv::Vec3f normal(1, 0, 0);
  cv::Vec3f point_on_plane(-2, 4, 32);
  cv::Vec3f point(4, 5, 2);
  cv::Vec4f hessian(1, 0, 0, 2);
  EXPECT_EQ(errorPointToPlane(hessian, point), 6);
  EXPECT_EQ(errorPointToPlane(normal, point_on_plane, point), 6);
  normal = {1, 1, 1};
  point_on_plane = {1, 1, 1};
  point = {0, 0, 0};
  hessian = {1, 1, 1, -3};
  hessian = hessian / normOfVector3D(normal);
  normalizeVec3D(normal);
  EXPECT_FLOAT_EQ(errorPointToPlane(hessian, point), sqrt(3));
  EXPECT_FLOAT_EQ(errorPointToPlane(normal, point_on_plane, point), sqrt(3));
}

TEST_F(LineDetectionTest, testHessianNormalFormOfPlane) {
  // Three points on the x-y-plane.
  std::vector<cv::Vec3f> points;
  points.push_back(cv::Vec3f(3, 5, 1));
  points.push_back(cv::Vec3f(5, 17, 1));
  points.push_back(cv::Vec3f(190, 3, 1));
  cv::Vec4f hessian_normal_form;
  hessianNormalFormOfPlane(points, hessian_normal_form);
  EXPECT_EQ(hessian_normal_form[0], 0);
  EXPECT_EQ(hessian_normal_form[1], 0);
  EXPECT_EQ(hessian_normal_form[2], -1);
  EXPECT_EQ(hessian_normal_form[3], 1);

  // More points one the x-y-plane
  points.push_back(cv::Vec3f(1.1, 2.3, 1));
  points.push_back(cv::Vec3f(5, 9.4, 1));
  points.push_back(cv::Vec3f(17.9, 15, 1));
  points.push_back(cv::Vec3f(150, 23, 1));
  points.push_back(cv::Vec3f(1, 1, 1));
  points.push_back(cv::Vec3f(510, 189, 1));
  hessianNormalFormOfPlane(points, hessian_normal_form);
  EXPECT_FLOAT_EQ(hessian_normal_form[0], 0);
  EXPECT_FLOAT_EQ(hessian_normal_form[1], 0);
  EXPECT_FLOAT_EQ(hessian_normal_form[2], 1);
  EXPECT_FLOAT_EQ(hessian_normal_form[3], -1);
}

TEST_F(LineDetectionTest, testPlaneRANSAC) {
  std::vector<cv::Vec3f> points;
  cv::Vec4f hessian_normal_form;
  // Points on the x-y-plane
  points.push_back(cv::Vec3f(3, 5, 0));
  points.push_back(cv::Vec3f(5, 17, 0));
  points.push_back(cv::Vec3f(190, 3, 0));
  points.push_back(cv::Vec3f(1.1, 2.3, 0));
  points.push_back(cv::Vec3f(5, 9.4, 0));
  points.push_back(cv::Vec3f(17.9, 15, 0));
  points.push_back(cv::Vec3f(150, 23, 0));
  points.push_back(cv::Vec3f(1, 1, 0));
  points.push_back(cv::Vec3f(510, 189, 0));
  points.push_back(cv::Vec3f(1, 5, 0));
  points.push_back(cv::Vec3f(5, 1, 0));
  points.push_back(cv::Vec3f(19, 3, 0));
  points.push_back(cv::Vec3f(1, 5.3, 0));
  points.push_back(cv::Vec3f(5, 12.4, 0));
  points.push_back(cv::Vec3f(1.9, 15, 0));
  points.push_back(cv::Vec3f(14.5, 3, 0));
  points.push_back(cv::Vec3f(1, 0, 0));
  points.push_back(cv::Vec3f(510, 19, 0));
  // And outliers:
  points.push_back(cv::Vec3f(1, 2, 23));
  points.push_back(cv::Vec3f(5, 7, 19));
  points.push_back(cv::Vec3f(510, 189, 3));

  EXPECT_TRUE(line_detector_.planeRANSAC(points, hessian_normal_form));
  EXPECT_FLOAT_EQ(hessian_normal_form[0], 0);
  EXPECT_FLOAT_EQ(hessian_normal_form[1], 0);
  EXPECT_FLOAT_EQ(fabs(hessian_normal_form[2]), 1);
  EXPECT_FLOAT_EQ(hessian_normal_form[3], 0);
}

TEST_F(LineDetectionTest, testFindXCoordOfPixelsOnVector) {
  cv::Point2f start(2.5, 0.3);
  cv::Point2f end(2.1, 3.9);
  std::vector<int> y_coords;
  findXCoordOfPixelsOnVector(start, end, true, y_coords);
  EXPECT_EQ(y_coords.size(), 4);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(y_coords[i], 2);

  end.x = 7.3;
  end.y = 0.7;
  findXCoordOfPixelsOnVector(start, end, true, y_coords);
  ASSERT_EQ(y_coords.size(), 4 + 1);
  EXPECT_EQ(y_coords[4], 2);
  findXCoordOfPixelsOnVector(start, end, false, y_coords);
  ASSERT_EQ(y_coords.size(), 4 + 1 + 1);
  EXPECT_EQ(y_coords[5], 8);
}

TEST_F(LineDetectionTest, testFindPointsInRectangle) {
  std::vector<cv::Point2f> corners;
  std::vector<cv::Point2i> points;
  corners.push_back(cv::Point2f(0.1, 1.9));
  corners.push_back(cv::Point2f(0.1, 0.1));
  corners.push_back(cv::Point2f(1.9, 1.9));
  corners.push_back(cv::Point2f(1.9, 0.1));
  findPointsInRectangle(corners, points);
  ASSERT_EQ(points.size(), 4);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      bool contains_element = false;
      cv::Point2i p(i, j);
      for (int k = 0; k < 4; ++k)
        if (points[k] == p) contains_element = true;
      EXPECT_TRUE(contains_element);
    }
  }
  corners.clear();
  corners.push_back(cv::Point2f(0.1, 5.9));
  corners.push_back(cv::Point2f(0.1, 0.1));
  corners.push_back(cv::Point2f(5.9, 5.9));
  corners.push_back(cv::Point2f(5.9, 0.1));
  findPointsInRectangle(corners, points);
  EXPECT_EQ(points.size(), 36);
}

TEST_F(LineDetectionTest, testGetPointOnPlaneIntersectionLine) {
  cv::Vec4f hessian1(1, 0, 0, 1);
  cv::Vec4f hessian2(0, 1, 0, 0);
  cv::Vec3f direction(0, 0, 1);
  cv::Vec3f x_0;
  getPointOnPlaneIntersectionLine(hessian1, hessian2, direction, x_0);
  EXPECT_EQ(x_0[0], -1);
  EXPECT_EQ(x_0[1], 0);
  EXPECT_EQ(x_0[2], 0);
}

TEST_F(LineDetectionTest, testFind3DlineOnPlanes) {
  cv::Vec<float, 6> line;
  cv::Vec<float, 6> line_guess(2, 2, 2);
  std::vector<cv::Vec3f> points1, points2;
  points1.push_back(cv::Vec3f(0, 0, 0));
  points1.push_back(cv::Vec3f(0.1, 0, 0));
  points1.push_back(cv::Vec3f(0.1, 0, 2));
  points1.push_back(cv::Vec3f(0, 0, 2));
  points2.push_back(cv::Vec3f(0, 0.1, 0));
  points2.push_back(cv::Vec3f(0, 0, 0));
  points2.push_back(cv::Vec3f(0, 0.1, 2));
  points2.push_back(cv::Vec3f(0, 0, 2));
  find3DlineOnPlanes(points1, points2, line_guess, line);
  EXPECT_NEAR(line[0], 0, 1e-6);
  EXPECT_NEAR(line[1], 0, 1e-6);
  EXPECT_NEAR(line[2], 2, 1e-6);
  EXPECT_NEAR(line[3], 0, 1e-6);
  EXPECT_NEAR(line[4], 0, 1e-6);
  EXPECT_NEAR(line[5], 0, 1e-6);
}

TEST_F(LineDetectionTest, testProject2Dto3DwithPlanes) {
  int N = 240;
  int M = 320;
  double scale = 0.01;
  cv::Mat cloud(N, M, CV_32FC3);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (j <= (M / 2)) {
        cloud.at<cv::Vec3f>(i, j) = cv::Vec3f(i * scale, j * scale, j * scale);
      } else {
        cloud.at<cv::Vec3f>(i, j) =
            cv::Vec3f(i * scale, j * scale, (M - j) * scale);
      }
    }
  }
  std::vector<cv::Vec4f> lines2D;
  cv::Vec4f vec1(160, 100, 160, 200);
  lines2D.push_back(vec1);
  std::vector<cv::Vec<float, 6> > lines3D;
  line_detector_.project2Dto3DwithPlanes(cloud, lines2D, lines3D);
  ASSERT_EQ(lines3D.size(), 1);
  EXPECT_NEAR(lines3D[0][0], 200 * scale, 1e-5);
  EXPECT_NEAR(lines3D[0][1], 160 * scale, 1e-5);
  EXPECT_NEAR(lines3D[0][2], 160 * scale, 1e-5);
  EXPECT_NEAR(lines3D[0][3], 100 * scale, 1e-5);
  EXPECT_NEAR(lines3D[0][4], 160 * scale, 1e-5);
  EXPECT_NEAR(lines3D[0][5], 160 * scale, 1e-5);
}

TEST_F(LineDetectionTest, testProjectPointOnPlane) {
  cv::Vec4f hessian(1, 0, 0, 0);
  cv::Vec3f point(456, 3, 2);
  cv::Vec3f projection = projectPointOnPlane(hessian, point);
  EXPECT_NEAR(projection[0], 0, 1e-5);
  EXPECT_NEAR(projection[1], 3, 1e-5);
  EXPECT_NEAR(projection[2], 2, 1e-5);
  hessian = {1, 1, 1, -3};
  hessian = hessian / normOfVector3D(cv::Vec3f(1, 1, 1));
  point = {33, 33, 33};
  projection = projectPointOnPlane(hessian, point);
  EXPECT_NEAR(projection[0], 1, 1e-5);
  EXPECT_NEAR(projection[1], 1, 1e-5);
  EXPECT_NEAR(projection[2], 1, 1e-5);
}

TEST_F(LineDetectionTest, testCheckIfValidLineBruteForce) {
  int N = 240;
  int M = 320;
  double scale = 0.01;
  cv::Mat cloud(N, M, CV_32FC3);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (j <= (M / 2)) {
        cloud.at<cv::Vec3f>(i, j) = cv::Vec3f(i * scale, j * scale, j * scale);
      } else {
        cloud.at<cv::Vec3f>(i, j) =
            cv::Vec3f(i * scale, j * scale, (M - j) * scale);
      }
    }
  }
  cv::Vec<float, 6> line3D(0, 0, 0, 10, 0, 0);
  EXPECT_TRUE(line_detector_.checkIfValidLineBruteForce(cloud, line3D)) << 1;
  EXPECT_NEAR(line3D[3], 2.4, 0.2);
  line3D = {0.5, 0.2, 0.2, 1, 0.7, 0.7};
  EXPECT_TRUE(line_detector_.checkIfValidLineBruteForce(cloud, line3D)) << 2;
  line3D = {0.5, 0.2, 0.5, 0.5, 3, 0.5};
  EXPECT_FALSE(line_detector_.checkIfValidLineBruteForce(cloud, line3D)) << 3;
}

TEST_F(LineDetectionTest, testCheckIfValidLineDiscont) {
  int N = 240;
  int M = 320;
  double scale = 0.01;
  cv::Mat cloud(N, M, CV_32FC3);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (j <= (M / 2)) {
        cloud.at<cv::Vec3f>(i, j) = cv::Vec3f(i * scale, j * scale, j * scale);
      } else {
        cloud.at<cv::Vec3f>(i, j) =
            cv::Vec3f(i * scale, j * scale, (M - j) * scale);
      }
    }
  }
  cv::Vec4f line2D(0, 0, 0, 10);
  EXPECT_TRUE(line_detector_.checkIfValidLineDiscont(cloud, line2D))
      << "test 1";
  line2D = {20, 50, 70, 50};
  EXPECT_TRUE(line_detector_.checkIfValidLineDiscont(cloud, line2D))
      << "test 2";
  line2D = {20, 50, 300, 50};
}

TEST_F(LineDetectionTest, testFind3DlinesRated) {
  int N = 240;
  int M = 320;
  double scale = 0.01;
  cv::Mat cloud(N, M, CV_32FC3);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (j <= (M / 2)) {
        cloud.at<cv::Vec3f>(i, j) = cv::Vec3f(i * scale, j * scale, j * scale);
      } else {
        cloud.at<cv::Vec3f>(i, j) =
            cv::Vec3f(i * scale, j * scale, (M - j) * scale);
      }
    }
  }
  std::vector<cv::Vec4f> lines2D;
  cv::Vec4f vec1(160, 100, 160, 200);
  lines2D.push_back(vec1);
  std::vector<cv::Vec<float, 6> > lines3D;
  std::vector<double> rating;
  line_detector_.find3DlinesRated(cloud, lines2D, lines3D, rating);
  ASSERT_EQ(lines3D.size(), 1);
  ASSERT_EQ(rating.size(), 1);
  EXPECT_EQ(rating[0], 0);
  EXPECT_NEAR(lines3D[0][0], 100 * scale, 1e-6);
  EXPECT_NEAR(lines3D[0][1], 160 * scale, 1e-6);
  EXPECT_NEAR(lines3D[0][2], 160 * scale, 1e-6);
  EXPECT_NEAR(lines3D[0][3], 200 * scale, 1e-6);
  EXPECT_NEAR(lines3D[0][4], 160 * scale, 1e-6);
  EXPECT_NEAR(lines3D[0][5], 160 * scale, 1e-6);
  vec1 = {50, 50, 250, 20};
  lines2D.push_back(vec1);
  line_detector_.find3DlinesRated(cloud, lines2D, lines3D, rating);
  ASSERT_EQ(lines3D.size(), 2);
  ASSERT_EQ(rating.size(), 2);
  EXPECT_EQ(rating[1], 1e9);
}

}  // namespace line_detection

LINE_DETECTION_TESTING_ENTRYPOINT
