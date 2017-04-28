#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>

#include "line_detection/common.h"
#include "line_detection/test/testing-entrypoint.h"

namespace line_detection {

class LineDetectionTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  LineDetectionTest() {}

  virtual ~LineDetectionTest() {}

  virtual void SetUp() {}
};

TEST_F(LineDetectionTest, testLineDetection) {
  EXPECT_EQ(0u, 1u);
  LOG(INFO) << "Not implemented yet";
  // TODO: load the data std::string testimage_path ("test_data/image.png")
  // TODO: Implement
}

}  // namespace line_detection

LINE_DETECTION_TESTING_ENTRYPOINT
