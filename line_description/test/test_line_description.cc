#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>

#include "line_description/common.h"
#include "line_description/test/testing-entrypoint.h"

namespace line_description {

class LineDescriptionTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  LineDescriptionTest() {}

  virtual ~LineDescriptionTest() {}

  virtual void SetUp() {}
};

TEST_F(LineDescriptionTest, testLineDescription) {
  EXPECT_EQ(0u, 1u);
  LOG(INFO) << "Not implemented yet";
  // TODO: load the data std::string testimage_path ("test_data/image.png")
  // TODO: Implement
}

}  // namespace line_description

LINE_DESCRIPTION_TESTING_ENTRYPOINT
