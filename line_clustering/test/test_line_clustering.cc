#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>

#include "line_clustering/common.h"
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

TEST_F(LineClusteringTest, testLineClustering) {
  EXPECT_EQ(0u, 1u);
  LOG(INFO) << "Not implemented yet";
  // TODO: Implement
}

}  // namespace line_clustering

LINE_CLUSTERING_TESTING_ENTRYPOINT
