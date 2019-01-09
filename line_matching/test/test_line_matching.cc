#include <glog/logging.h>
#include <gtest/gtest.h>

#include "line_matching/common.h"
#include "line_matching/line_matching.h"
#include "line_matching/test/testing-entrypoint.h"

namespace line_matching {

class LineMatchingTest : public ::testing::Test {
 public:

 protected:
  LineMatchingTest() { SetUp(); }

  virtual ~LineMatchingTest() {}

  virtual void SetUp() {

  }
};

TEST_F(LineMatchingTest, testFixedSizePriorityQueue_1) {
  FixedSizePriorityQueue<float> pq(3, true);
  pq.push(3.0);
  CHECK_EQ(pq.front(), 3.0);
  pq.push(2.0);
  CHECK_EQ(pq.front(), 2.0);
  pq.pop();
  pq.push(2.0);
  pq.push(4.0);
  pq.push(-1.0);
  CHECK_EQ(pq.front(), -1.0);
  pq.pop();
  CHECK_EQ(pq.front(), 2.0);
  pq.pop();
  CHECK_EQ(pq.front(), 3.0);
  pq.pop();
}

TEST_F(LineMatchingTest, testFixedSizePriorityQueue_2) {
  FixedSizePriorityQueue<float> pq(3, false);
  pq.push(3.0);
  CHECK_EQ(pq.front(), 3.0);
  pq.pop();
  pq.push(2.0);
  CHECK_EQ(pq.front(), 2.0);
  pq.push(2.0);
  pq.push(4.0);
  pq.push(-1.0);
  CHECK_EQ(pq.front(), 4.0);
  pq.pop();
  CHECK_EQ(pq.front(), 2.0);
  pq.pop();
  CHECK_EQ(pq.front(), 2.0);
  pq.pop();
}

TEST_F(LineMatchingTest, testFixedSizePriorityQueue_3) {
  FixedSizePriorityQueue<float> pq(1, false);
  pq.push(3.0);
  CHECK_EQ(pq.front(), 3.0);
  pq.push(2.0);
  CHECK_EQ(pq.front(), 3.0);
  pq.push(2.0);
  pq.push(4.0);
  pq.push(-1.0);
  CHECK_EQ(pq.front(), 4.0);
  pq.pop();
}

}  // namespace line_matching

LINE_MATCHING_TESTING_ENTRYPOINT
