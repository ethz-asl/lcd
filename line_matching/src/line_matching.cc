#include "line_matching/line_matching.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace line_matching {

LineMatcher::LineMatcher() {
}

bool LineMatcher::addFrame(const Frame& frame_to_add,
                           unsigned int frame_index) {
  // Check if a frame with the same frame index already exists.
  if (frames_.count(frame_index) != 0) {
    return false;
  }
  frames_[frame_index] = frame_to_add;
  return true;
}

bool LineMatcher::displayMatches(unsigned int frame_index_1,
                                 unsigned int frame_index_2,
                                 MatchingMethod matching_method) {
  std::vector<int> line_indices_1, line_indices_2;
  // Obtain matches for the given frame indices, if possible.
  if (!matchFrames(frame_index_1, frame_index_2, matching_method,
                   &line_indices_1, &line_indices_2)) {
    LOG(ERROR) << "Unable to match frames with indices " << frame_index_1
               << " and " << frame_index_2 << ".";
    return false;
  }
  CHECK(line_indices_1.size() == line_indices_2.size());
  // Display the two images side by side with the matched lines connected to
  // each other.
  size_t cols = frames_[frame_index_1].image.cols;
  size_t rows = frames_[frame_index_1].image.rows;
  cv::Mat large_image(cv::Size(2 * cols, rows),
                      frames_[frame_index_1].image.type());
  // Add first image.
  frames_[frame_index_1].image.copyTo(large_image(cv::Rect(0, 0, cols, rows)));
  // Add second image.
  frames_[frame_index_2].image.copyTo(large_image(cv::Rect(cols, 0, cols,
                                                           rows)));
  // Draw lines in the first image (in red).
  cv::Vec4f line2D;
  cv::Point2f center_line_1, center_line_2;
  for (auto& line_with_embedding : frames_[frame_index_1].lines) {
    line2D = line_with_embedding.line2D;
    cv::line(large_image, cv::Point(line2D[0], line2D[1]),
             cv::Point(line2D[2], line2D[3]), CV_RGB(255, 0, 0));
  }
  // Draw lines in the second image (in red).
  for (auto& line_with_embedding : frames_[frame_index_2].lines) {
    line2D = line_with_embedding.line2D;
    cv::line(large_image, cv::Point(line2D[0] + cols, line2D[1]),
             cv::Point(line2D[2] + cols, line2D[3]), CV_RGB(255, 0, 0));
  }
  // Draw yellow lines between the center of the matched lines.
  for (size_t i = 0; i < line_indices_1.size(); ++i) {
    center_line_1 = {(frames_[frame_index_1].lines[i].line2D[0] +
                      frames_[frame_index_1].lines[i].line2D[2]) / 2,
                     (frames_[frame_index_1].lines[i].line2D[1] +
                      frames_[frame_index_1].lines[i].line2D[3]) / 2};
    center_line_2 = {(frames_[frame_index_2].lines[i].line2D[0] +
                      frames_[frame_index_2].lines[i].line2D[2]) / 2,
                     (frames_[frame_index_2].lines[i].line2D[1] +
                      frames_[frame_index_2].lines[i].line2D[3]) / 2};
    cv::line(large_image, cv::Point(center_line_1.x, center_line_1.y),
             cv::Point(center_line_2.x + cols, center_line_2.y),
             CV_RGB(255, 255, 0));
  }
  // Display image.
  cv::imshow("Matches between frame " + std::to_string(frame_index_1) +
             " and frame " + std::to_string(frame_index_2), large_image);
  cv::waitKey();
  return true;
}

bool LineMatcher::matchFrames(unsigned int frame_index_1,
                              unsigned int frame_index_2,
                              MatchingMethod matching_method,
                              std::vector<int>* line_indices_1,
                              std::vector<int>* line_indices_2) {
  CHECK_NOTNULL(line_indices_1);
  CHECK_NOTNULL(line_indices_2);
  return matchFramesBruteForce(frame_index_1, frame_index_2, matching_method,
                               line_indices_1, line_indices_2);
}

bool LineMatcher::matchFramesBruteForce(unsigned int frame_index_1,
                                        unsigned int frame_index_2,
                                        MatchingMethod matching_method,
                                        std::vector<int>* line_indices_1,
                                        std::vector<int>* line_indices_2) {
  std::vector<MatchWithRating> candidate_matches;
  MatchRatingComputer* match_rating_computer;
  size_t num_lines_frame_1, num_lines_frame_2;
  size_t num_unmatched_lines_frame_1, num_unmatched_lines_frame_2;
  float rating;
  CHECK_NOTNULL(line_indices_1);
  CHECK_NOTNULL(line_indices_2);
  // Check that frames with the given frame indices exist.
  if (frames_.count(frame_index_1) == 0 || frames_.count(frame_index_2) == 0) {
    return false;
  }
  // Set the match rating computer depending on the matching method.
  switch (matching_method) {
    case MatchingMethod::MANHATTAN:
      match_rating_computer = new ManhattanRatingComputer;
      break;
    case MatchingMethod::EUCLIDEAN:
      match_rating_computer = new EuclideanRatingComputer;
      break;
    default:
      LOG(ERROR) << "Invalid matching method. Valid methods are MANHATTAN and "
                 << "EUCLIDEAN.";
      return false;
  }
  // Create vector of all possible matches with their rating.
  candidate_matches.clear();
  LineWithEmbeddings* line_1;
  LineWithEmbeddings* line_2;
  num_lines_frame_1 = frames_[frame_index_1].lines.size();
  num_lines_frame_2 = frames_[frame_index_2].lines.size();
  for (size_t idx1 = 0; idx1 < num_lines_frame_1; ++idx1) {
    line_1 = &(frames_[frame_index_1].lines[idx1]);
    for (size_t idx2 = 0; idx2 < num_lines_frame_2; ++idx2) {
      line_2 = &(frames_[frame_index_2].lines[idx2]);
      if (match_rating_computer->computeMatchRating(line_1->embeddings,
                                                    line_2->embeddings,
                                                    &rating)) {
        candidate_matches.push_back(std::make_pair(rating,
                                                   std::make_pair(idx1, idx2)));
      }
    }
  }
  // Sort vector of possible matches by increasing rating.
  std::sort(candidate_matches.begin(), candidate_matches.end(),
            match_comparator);
  // Set all the lines to be unmatched.
  std::vector<bool> line_in_frame_1_was_matched, line_in_frame_2_was_matched;
  line_in_frame_1_was_matched.assign(num_lines_frame_1, false);
  line_in_frame_2_was_matched.assign(num_lines_frame_2, false);
  num_unmatched_lines_frame_1 = num_lines_frame_1;
  num_unmatched_lines_frame_2 = num_lines_frame_2;
  // Examine the candidate matches.
  size_t current_match_idx = 0;
  size_t idx_frame_1, idx_frame_2;
  line_indices_1->clear();
  line_indices_2->clear();
  while (num_unmatched_lines_frame_1 > 0 && num_unmatched_lines_frame_2 > 0 &&
         current_match_idx < candidate_matches.size()) {
    idx_frame_1 = candidate_matches[current_match_idx].second.first;
    idx_frame_2 = candidate_matches[current_match_idx].second.second;
    if (!line_in_frame_1_was_matched[idx_frame_1] &&
        !line_in_frame_2_was_matched[idx_frame_2]) {
      line_in_frame_1_was_matched[idx_frame_1] = true;
      line_in_frame_2_was_matched[idx_frame_2] = true;
      num_unmatched_lines_frame_1--;
      num_unmatched_lines_frame_2--;
      // Output match.
      line_indices_1->push_back(idx_frame_1);
      line_indices_2->push_back(idx_frame_2);
    }
    current_match_idx++;
  }
  // Delete matching computer.
  delete match_rating_computer;

  return true;
}

MatchRatingComputer::MatchRatingComputer(float max_difference_between_matches) {
  max_difference_between_matches_ = max_difference_between_matches;
}

ManhattanRatingComputer::ManhattanRatingComputer(
    float max_difference_between_matches) {
  max_difference_between_matches_ = max_difference_between_matches;
}

EuclideanRatingComputer::EuclideanRatingComputer(
    float max_difference_between_matches) {
  max_difference_between_matches_ = max_difference_between_matches;
}

bool ManhattanRatingComputer::computeMatchRating(
    const std::vector<float>& embedding_1,
    const std::vector<float>& embedding_2, float *rating_out) {
  CHECK_NOTNULL(rating_out);
  CHECK(embedding_1.size() == embedding_2.size());
  float rating = 0.0f;
  for(size_t i = 0; i < embedding_1.size(); ++i) {
    rating += fabs(embedding_1[i] - embedding_2[i]);
  }
  if (rating > max_difference_between_matches_) {
    return false;
  }
  *rating_out = rating;
  return true;
}

bool EuclideanRatingComputer::computeMatchRating(
    const std::vector<float>& embedding_1,
    const std::vector<float>& embedding_2, float *rating_out) {
  CHECK_NOTNULL(rating_out);
  CHECK(embedding_1.size() == embedding_2.size());
  double rating = 0.0f;
  for(size_t i = 0; i < embedding_1.size(); ++i) {
    rating += pow((embedding_1[i] - embedding_2[i]), 2);
  }
  if (rating > max_difference_between_matches_) {
    return false;
  }
  *rating_out = static_cast<float>(sqrt(rating));
  return true;
}
}  // namespace line_matching
