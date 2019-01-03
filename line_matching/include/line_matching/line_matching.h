#ifndef LINE_MATCHING_LINE_MATCHING_H_
#define LINE_MATCHING_LINE_MATCHING_H_

#include "line_matching/common.h"

#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace line_matching {

struct LineWithEmbeddings {
  cv::Vec4f line2D;
  cv::Vec6f line3D;
  std::vector<float> embeddings;
};

struct Frame {
  // Lines (2D and 3D) with embeddings.
  std::vector<LineWithEmbeddings> lines;
  // RGB image.
  cv::Mat image;
  // Point cloud image.
  cv::Mat cloud;
};

typedef pair<Frame, int> FrameWithIndex;

enum class MatchingMethod : unsigned int {
  HAMMING = 0,
  EUCLIDEAN = 1
};

class LineMatcher {
 public:
   LineMatcher();
   // Add the input frame with the given frame index to the set of frames
   // received if no other frame with that frame index was received.
   // Input: frame_to_add: Frame to add to the set of frames received.
   //
   //        frame_index:  Index to assign to the added frame.
   //
   // Output: False if a frame the given frame_index is already in the set of
   //         frames received, true otherwise.
   bool addFrame(const Frame& frame_to_add, int frame_index);
   // Display the matches between two frames with the given indices, if frames
   // with those indices were received.
   // Input: frame_index_1/2: Indices of the frame between which to display
   //                         matches.
   //
   //        matching_method: Method (distance) to be use to match lines.
   void displayMatches(int frame_index_1, int frame_index_2,
                       MatchingMethod matching_method);
 private:
   // Matches the two frames with the given indices, if frame with those indices
   // were received. Returns two vectors of indices with the same length,
   // that encode the correspondences between the lines from the two frames.
   // Input: frame_index_1/2:   Index of the frame between which to match lines.
   //
   //        matching_method:   Method (distance) to be use to match lines.
   //
   // Output: line_indices_1/2: Paired indices of lines from the two frames that
   //                           are matches. The line with index
   //                           line_indices_1[i] in the first frame matches the
   //                           line with index line_indices_2[i] in the second
   //                           frame.
   void matchFrames(int frame_index_1, int frame_index_2,
                    MatchingMethod matching_method,
                    std::vector<int>* line_indices_1,
                    std::vector<int>* line_indices_2);
   // Frames received.
   std::vector<Frame> frames_;
};
}  // namespace line_matching

#include "line_matching/line_matching_inl.h"

#endif  // LINE_MATCHING_LINE_MATCHING_H_
