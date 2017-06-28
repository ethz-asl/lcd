#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>

#include "line_clustering/common.h"
#include "line_clustering/line_clustering.h"
#include "line_clustering/test/testing-entrypoint.h"

namespace line_clustering {

class LineClusteringTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  LineClusteringTest() {}

  virtual ~LineClusteringTest() {}

  virtual void SetUp() {}
};

TEST_F(LineClusteringTest, testComputePerpendicularDistanceLines) {
  cv::Vec<float, 6> line1(1, 1, 1, 6, 1, 1);
  cv::Vec<float, 6> line2(1, 2, 1, 10, 2, 1);
  EXPECT_FLOAT_EQ(computePerpendicularDistanceLines(line1, line2), 1);
  line1 = {0, 0, 0, 1, 1, 0};
  line2 = {2, 0, 2, 0, 2, 2};
  EXPECT_FLOAT_EQ(computePerpendicularDistanceLines(line1, line2), 2);
  line1 = {0, 0, 0, 1, 1, 1};
  line2 = {2, 0, 2, 1, 1, 1};
  EXPECT_FLOAT_EQ(computePerpendicularDistanceLines(line1, line2), 0);
}

TEST_F(LineClusteringTest, testComputeSquareMeanDifferenceLines) {
  cv::Vec<float, 6> line1(0, 0, 0, 1, 0, 0);
  cv::Vec<float, 6> line2(0, 1, 0, 1, 1, 0);
  EXPECT_FLOAT_EQ(computeSquareMeanDifferenceLines(line1, line2), 1);
  line1 = {1.1, 3.7, 1.2, 6, 7.8, 90};
  line2 = {2, 14.4, 2, 17.3, 2, 2};
  EXPECT_FLOAT_EQ(computeSquareMeanDifferenceLines(line1, line2), 1944.1725);
}

TEST_F(LineClusteringTest, testComputeSquareNearestDifferenceLines) {
  cv::Vec<float, 6> line1(0, 0, 0, 1, 0, 0);
  cv::Vec<float, 6> line2(0, 1, 0, 1, 1, 0);
  EXPECT_FLOAT_EQ(computeSquareNearestDifferenceLines(line1, line2), 1);
  line1 = {1.1, 3.7, 1.2, 6, 7.8, 90};
  line2 = {2, 14.4, 2, 17.3, 2, 2};
  EXPECT_FLOAT_EQ(computeSquareNearestDifferenceLines(line1, line2), 115.94);
}

}  // namespace line_clustering

LINE_CLUSTERING_TESTING_ENTRYPOINT
