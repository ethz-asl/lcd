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

// (Frame, index).
typedef std::pair<Frame, int> FrameWithIndex;
// (Rating, (index_line_in_first_frame, index_line_in_second_frame)).
typedef std::pair<float, std::pair<int, int>> MatchWithRating;

// Comparator function used to sort candidate matching by increasing rating
// (lower rating <=> better match).
inline bool operator<(const MatchWithRating& match_1,
                      const MatchWithRating& match_2) {
  return match_1.first < match_2.first;
}

enum class MatchingMethod : unsigned int {
  MANHATTAN = 0,  // Manhattan distance
  EUCLIDEAN = 1   // Euclidean distance
};

// Abstract and derived classes to compute a rating for a candidate match pair
// of descriptors.
class MatchRatingComputer {
 public:
   MatchRatingComputer(float max_difference_between_matches = 100.0f);

   // Computes the rating between two candidate matches (given their embedding
   // descriptors).
   // Input: embedding_1/2: Descriptors of the candidate matches.
   //
   // Output: rating_out: Rating of the candidate match.
   //
   //         return:     True if the rating of the match is above the threshold
   //                     defined by max_difference_between_matches_, false
   //                     otherwise.
   virtual bool computeMatchRating(const std::vector<float>& embedding_1,
                                   const std::vector<float>& embedding_2,
                                   float* rating_out) = 0;
 protected:
   // Threshold to define valid matches between lines.
   float max_difference_between_matches_;
};

class ManhattanRatingComputer : public MatchRatingComputer {
 public:
   ManhattanRatingComputer(float max_difference_between_matches = 64.0f);
   bool computeMatchRating(const std::vector<float>& embedding_1,
                           const std::vector<float>& embedding_2,
                           float* rating_out);
};

class EuclideanRatingComputer : public MatchRatingComputer {
 public:
   EuclideanRatingComputer(float max_difference_between_matches = 100.0f);
   bool computeMatchRating(const std::vector<float>& embedding_1,
                           const std::vector<float>& embedding_2,
                           float* rating_out);
};

// Main class: holds the frame and can be called to display matches.
class LineMatcher {
 public:
   LineMatcher();

   // Adds the input frame with the given frame index to the set of frames
   // received if no other frame with that frame index was received.
   // Input: frame_to_add: Frame to add to the set of frames received.
   //
   //        frame_index:  Index to assign to the added frame.
   //
   // Output: False if a frame the given frame_index is already in the set of
   //         frames received, true otherwise.
   bool addFrame(const Frame& frame_to_add, unsigned int frame_index);

   // Displays the matches between two frames with the given indices, if frames
   // with those indices were received.
   // Input: frame_index_1/2:      Indices of the frame between which to display
   //                              matches.
   //
   //        matching_method:      Method (distance) to use to match lines.
   //
   //        magnification_factor: Factor (default to 2) that expresses by how
   //                              many times the images should be enlarged for
   //                              visualization.
   //
   // Output: return: True if matching was possible, i.e., if frames with both
   //                 input frame indices were received; false otherwise.
   bool displayMatches(unsigned int frame_index_1, unsigned int frame_index_2,
                       MatchingMethod matching_method,
                       unsigned int magnification_factor=2);

   // Displays the best num_matches_per_line matches for each line in the frame
   // with frame index frame_index_1.
   // Input: frame_index_1/2:      Index of the frame between which to match
   //                              lines.
   //
   //        matching_method:      Method (distance) to use to match lines.
   //
   //        num_matches_per_line: (Maximum) number of matches to find per each
   //                              line in the first frame.
   //
   // Output: return: True if matching was possible, i.e., if frames with both
   //                 input frame indices were received; false otherwise.
   bool displayNBestMatchesPerLine(unsigned int frame_index_1,
                                   unsigned int frame_index_2,
                                   MatchingMethod matching_method,
                                   unsigned int num_matches_per_line,
                                   unsigned int magnification_factor=2);
  // Displays the best match for each line in the frame with frame index
  // frame_index_1, if the match is a valid. A match is considered valid if
  // the ratio between its rating and the rating of the second-best candidate
  // match is below a certain threshold.
  // Input: frame_index_1/2: Index of the frame between which to match lines.
  //
  //        matching_method: Method (distance) to use to match lines.
  //
  // Output: return: True if matching was possible, i.e., if frames with both
  //                 input frame indices were received; false otherwise.
  bool displayBestMatchPerLine(unsigned int frame_index_1,
                               unsigned int frame_index_2,
                               MatchingMethod matching_method,
                               unsigned int magnification_factor=2);


 private:
   // Matches the two frames with the given indices, if frame with those indices
   // were received. Returns two vectors of indices with the same length,
   // that encode the correspondences between the lines from the two frames.
   // Input: frame_index_1/2: Index of the frame between which to match lines.
   //
   //        matching_method: Method (distance) to use to match lines.
   //
   // Output: line_indices_1/2: Paired indices of lines from the two frames that
   //                           are matches. The line with index
   //                           line_indices_1[i] in the first frame matches the
   //                           line with index line_indices_2[i] in the second
   //                           frame.
   //
   //         matching_ratings: Ratings of the matches found.
   //
   //         return:           True if matching was possible, i.e., if frames
   //                           with both input frame indices were received;
   //                           false otherwise.
   bool matchFrames(unsigned int frame_index_1, unsigned int frame_index_2,
                    MatchingMethod matching_method,
                    std::vector<int>* line_indices_1,
                    std::vector<int>* line_indices_2,
                    std::vector<float>* matching_ratings);
   // Same interface as above. Matching based on brute-force comparison of all
   // pairs of lines and selection of the best.
   bool matchFramesBruteForce(unsigned int frame_index_1,
                              unsigned int frame_index_2,
                              MatchingMethod matching_method,
                              std::vector<int>* line_indices_1,
                              std::vector<int>* line_indices_2,
                              std::vector<float>* matching_ratings);

   // Matches EACH line in the frame with frame index frame_index_1 to the
   // num_matches_per_line lines in the other frame (with frame index
   // frame_index_2) that have a match with the best (lowest) ratings among all
   // lines in that frame. If less than num_matches_per_line valid matches are
   // found, returns as many matches as possible.
   // NOTE: It is possible that some lines in the frame with frame index
   // frame_index_2 are assigned to more than one line in the frame with frame
   // index frame_index_1, i.e., the assignment is not injective.
   // Input: frame_index_1/2:      Index of the frame between which to match
   //                              lines.
   //
   //        matching_method:      Method (distance) to use to match lines.
   //
   //        num_matches_per_line: (Maximum) number of matches to find per each
   //                              line in the first frame.
   //
   // Output: matches_with_ratings_vec: Vector of all matches found for all
   //                                   lines, with the matching rating and the
   //                                   indices of the pair of lines matched.
   //
   //         return:                   True if matching was possible, i.e., if
   //                                   frames with both input frame indices
   //                                   were received; false otherwise.
   bool matchFramesNBestMatchesPerLine(
       unsigned int frame_index_1, unsigned int frame_index_2,
       MatchingMethod matching_method, unsigned int num_matches_per_line,
       std::vector<MatchWithRating>* matches_with_ratings_vec);
   // Matches each line in the frame with frame index frame_index_1 to the
   // line that best matches it in the other frame (with frame index
   // frame_index_2), if the match is a valid. A match is considered valid if
   // the ratio between its rating and the rating of the second-best candidate
   // match is below a certain threshold. This is motivated by the fact that,
   // due to e.g. the high-dimensionality of the descriptor space, it is likely
   // that there are false matches which have descriptors at a similar distance
   // from the reference one in the descriptor space, and therefore have similar
   // ratings. Correct matches should instead have descriptors much closer to
   // reference descriptor than the descriptors of the false matches.
   // If the best candidate match found for a certain line in the first frame is
   // not a valid match, then the line will not be matched to any line in the
   // second frame.
   // NOTE: It is possible that some lines in the frame with frame index
   // frame_index_2 are assigned to more than one line in the frame with frame
   // index frame_index_1, i.e., the assignment is not injective.
   // Input: frame_index_1/2:      Index of the frame between which to match
   //                              lines.
   //
   //        matching_method:      Method (distance) to use to match lines.
   //
   //        num_matches_per_line: (Maximum) number of matches to find per each
   //                              line in the first frame.
   //
   // Output: matches_with_ratings_vec: Vector of all matches found for all
   //                                   lines, with the matching rating and the
   //                                   indices of the pair of lines matched.
   //
   //         return:                   True if matching was possible, i.e., if
   //                                   frames with both input frame indices
   //                                   were received; false otherwise.
   bool matchFramesBestMatchPerLine(
       unsigned int frame_index_1, unsigned int frame_index_2,
       MatchingMethod matching_method,
       std::vector<MatchWithRating>* matches_with_ratings_vec);

  // Frames received: key = frame_index, value = frame.
   std::map<unsigned int, Frame> frames_;
};
}  // namespace line_matching

#include "line_matching/line_matching_inl.h"

#endif  // LINE_MATCHING_LINE_MATCHING_H_
