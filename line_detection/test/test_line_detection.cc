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
  cv::Mat test_depth;
  line_detection::LineDetector line_detector;
  std::vector<cv::Vec4f> lines;

  // TODO: Why virtual?
  virtual void SetUp() {
    // Load the kitchen.png image and compute a grayscale version of it.
    // TODO: What is the path of this function call? Defining a global path
    // obviously does kind of defeat the purpose of such a test.
    std::string testimage_path(
        "/home/dominik/catkin_ws/src/3d_line_toolbox/line_detection/test/"
        "test_data/kitchen.png");
    test_image = cv::imread(testimage_path, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(test_image, test_img_gray, CV_BGR2GRAY);

    // // The function cv::imread fails to open the depth image correctly. The
    // // current workaround is only good for testing. The data was written to a
    // // .txt file with matlab and is read out with the following code snippet
    // test_depth.create(480, 640, CV_16UC1);
    // std::ifstream myfile;
    // myfile.open(
    //     "/home/dominik/catkin_ws/src/3d_line_toolbox/line_detection/test/"
    //     "test_data/depth.txt");
    // std::string line;
    // for (int i = 0; i < 640; i++) {
    //   for (int j = 0; j < 480; j++) {
    //     std::getline(myfile, line);
    //     test_depth.at<float>(j, i) = atoi(line.c_str());
    //   }
    // }
    // myfile.close();
  }
};

TEST_F(LineDetectionTest, testLineDetection) {
  int n_lines;
  // Calling the detector with LSD.
  line_detector.detectLines(test_img_gray, lines, 0);
  n_lines = lines.size();
  EXPECT_EQ(n_lines, 84)
      << "LSD detection: Expected 84 lines to be found. Found " << n_lines;
  // Calling the detector with EDL.
  line_detector.detectLines(test_img_gray, lines, 1);
  n_lines = lines.size();
  EXPECT_EQ(n_lines, 18)
      << "EDL detection: Expected 18 lines to be found. Found " << n_lines;
  // Calling the detector with FAST.
  line_detector.detectLines(test_img_gray, lines, 2);
  n_lines = lines.size();
  EXPECT_EQ(n_lines, 70)
      << "Fast detection: Expected 70 lines to be found. Found " << n_lines;
  // Calling the detector with HOUGH.
  line_detector.detectLines(test_img_gray, lines, 3);
  n_lines = lines.size();
  EXPECT_EQ(n_lines, 16)
      << "HOUGH detection: Expected 16 lines to be found. Found " << n_lines;

  // LOG(INFO) << "Not implemented yet";
  // TODO: load the data std::string testimage_path ("test_data/image.png")
  // TODO: Implement
}

// TEST_F(LineDetectionTest, testPointCloudViewer) {
//   cv::Mat K(3, 3, CV_32F);
//
//   K.at<float>(0, 0) = 570.3;
//   K.at<float>(0, 1) = 0;
//   K.at<float>(0, 2) = 320;
//   K.at<float>(1, 0) = 0;
//   K.at<float>(1, 1) = 570.3;
//   K.at<float>(1, 2) = 240;
//   K.at<float>(2, 0) = 0;
//   K.at<float>(2, 1) = 0;
//   K.at<float>(2, 2) = 1;
//
//   line_detection::displayPointCloud(test_depth, test_image, K);
// }

}  // namespace line_detection

LINE_DETECTION_TESTING_ENTRYPOINT
