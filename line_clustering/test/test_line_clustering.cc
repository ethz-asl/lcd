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
  KMeansCluster* kmeans_cluster_;
  std::vector<cv::Vec<float, 6> > lines_;

  virtual ~LineClusteringTest() { delete kmeans_cluster_; }

  virtual void SetUp() {
    // Add edges of a unit cube to lines_
    lines_.push_back(cv::Vec<float, 6>(0, 0, 0, 0, 0, 1));
    lines_.push_back(cv::Vec<float, 6>(0, 0, 0, 0, 1, 0));
    lines_.push_back(cv::Vec<float, 6>(0, 0, 0, 1, 0, 0));
    lines_.push_back(cv::Vec<float, 6>(1, 0, 1, 0, 0, 1));
    lines_.push_back(cv::Vec<float, 6>(1, 0, 1, 1, 0, 0));
    lines_.push_back(cv::Vec<float, 6>(1, 0, 1, 1, 1, 1));
    lines_.push_back(cv::Vec<float, 6>(1, 1, 0, 1, 0, 0));
    lines_.push_back(cv::Vec<float, 6>(1, 1, 0, 0, 1, 0));
    lines_.push_back(cv::Vec<float, 6>(1, 1, 0, 1, 1, 1));
    lines_.push_back(cv::Vec<float, 6>(0, 1, 1, 1, 0, 0));
    lines_.push_back(cv::Vec<float, 6>(0, 1, 1, 0, 1, 0));
    lines_.push_back(cv::Vec<float, 6>(0, 1, 1, 0, 0, 1));
    // And now add a shifted cube
    cv::Vec<float, 6> shift(3, 0, 0, 3, 0, 0);
    size_t N = lines_.size();
    for (size_t i = 0; i < N; ++i) lines_.push_back(lines_[i] + shift);

    kmeans_cluster_ = new KMeansCluster(lines_, 2);
  }
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

TEST_F(LineClusteringTest, testRunOnMeansOfLines) {
  kmeans_cluster_->computeMeansOfLines();
  kmeans_cluster_->runOnMeansOfLines();
  size_t N = kmeans_cluster_->cluster_idx_.size();
  int first_label = kmeans_cluster_->cluster_idx_[0];
  for (size_t i = 1; i < N; ++i) {
    if (round(i / (double)N) == 0)
      CHECK_EQ(kmeans_cluster_->cluster_idx_[i], first_label);
    else
      CHECK_EQ(kmeans_cluster_->cluster_idx_[i], abs(first_label - 1));
  }
}
}  // namespace line_clustering

LINE_CLUSTERING_TESTING_ENTRYPOINT
