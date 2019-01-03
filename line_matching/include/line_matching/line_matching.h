#ifndef LINE_MATCHING_LINE_MATCHING_H_
#define LINE_MATCHING_LINE_MATCHING_H_

#include "line_matching/common.h"

#include <map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
};

typedef std::pair<Frame, int> FrameWithIndex;

enum class MatchingMethod : unsigned int {
  HAMMING = 0,
  EUCLIDEAN = 1
};

struct LineMatchingParams {
  // Thresholds to define matches between lines.
  float max_difference_between_matches_hamming_ = 2.5;
  float max_difference_between_matches_euclidean_ = 2.5;

};

class LineMatcher {
 public:
   LineMatcher();
   LineMatcher(LineMatchingParams* params);

   ~LineMatcher();

   // Add the input frame with the given frame index to the set of frames
   // received if no other frame with that frame index was received.
   // Input: frame_to_add: Frame to add to the set of frames received.
   //
   //        frame_index:  Index to assign to the added frame.
   //
   // Output: False if a frame the given frame_index is already in the set of
   //         frames received, true otherwise.
   bool addFrame(const Frame& frame_to_add, unsigned int frame_index);

   // Display the matches between two frames with the given indices, if frames
   // with those indices were received.
   // Input: frame_index_1/2: Indices of the frame between which to display
   //                         matches.
   //
   //        matching_method: Method (distance) to be use to match lines.
   //
   // Output: return:         True if matching was possible, i.e., if frames
   //                         with both input frame indices were received; false
   //                         otherwise.
   bool displayMatches(unsigned int frame_index_1, unsigned int frame_index_2,
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
   //
   //         return:           True if matching was possible, i.e., if frames
   //                           with both input frame indices were received;
   //                           false otherwise.
   bool matchFrames(unsigned int frame_index_1, unsigned int frame_index_2,
                    MatchingMethod matching_method,
                    std::vector<int>* line_indices_1,
                    std::vector<int>* line_indices_2);
   // Same interface as above. Matching based on brute-force comparison of all
   // pairs of lines and selection of the best.
   bool matchFramesBruteForce(unsigned int frame_index_1,
                              unsigned int frame_index_2,
                              MatchingMethod matching_method,
                              std::vector<int>* line_indices_1,
                              std::vector<int>* line_indices_2);

   // Frames received: key = frame_index, value = frame.
   std::map<unsigned int, Frame> frames_;
   // Parameters for matching lines.
   LineMatchingParams* params_;
   // Whether the instance of the parameter struct was created by this instance
   // or not.
   bool params_is_mine_;

};
}  // namespace line_matching

#include "line_matching/line_matching_inl.h"

#endif  // LINE_MATCHING_LINE_MATCHING_H_
