#include "line_matching/line_matching.h"

#include <string>

namespace line_matching {

LineMatcher::LineMatcher() {
  params_ = new LineMatchingParams();
  params_is_mine_ = true;
}

LineMatcher::LineMatcher(LineMatchingParams* params) {
  params_ = params;
  params_is_mine_ = false;
}

LineMatcher::~LineMatcher() {
  if (params_is_mine_) {
    delete params_;
  }
}

bool LineMatcher::addFrame(const Frame& frame_to_add,
                           unsigned int frame_index) {
  // Check if a frame with the same frame index already exists.
  if (frames_.count(frame_index) == 0) {
    return false;
  }
  frames_[frame_index] = frame_to_add;
}

bool LineMatcher::displayMatches(unsigned int frame_index_1,
                                 unsigned int frame_index_2,
                                 MatchingMethod matching_method) {
  std::vector<int> line_indices_1, line_indices_2;
  // Obtain matches for the given frame indices, if possible.
  if (!matchFrames(frame_index_1, frame_index_2, matching_method,
                   &line_indices_1, &line_indices_2)) {
    return false;
  }
  CHECK(line_indices_1.size() == line_indices_2.size());
  // Display the two images side by side with the matched lines connected to
  // each other.
  size_t cols = frames_[frame_index_1].image.cols;
  size_t rows = frames_[frame_index_1].image.rows;
  cv::Mat large_image(cv::Size(2 * cols, rows), CV_32FC3);
  // Add first image.
  frames_[frame_index_1].image.copyTo(large_image(cv::Rect(0, 0, cols, rows)));
  // Add second image.
  frames_[frame_index_2].image.copyTo(large_image(cv::Rect(cols, 0, 2 * cols,
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
             cv::Point(center_line_2.x, center_line_2.y), CV_RGB(255, 255, 0));
  }
  // Display image.
  cv::imshow("Matches between frame " + std::to_string(frame_index_1) +
             "and frame " + std::to_string(frame_index_2), large_image);
  cv::waitKey();
}

bool LineMatcher::matchFrames(unsigned int frame_index_1,
                              unsigned int frame_index_2,
                              MatchingMethod matching_method,
                              std::vector<int>* line_indices_1,
                              std::vector<int>* line_indices_2) {
  CHECK_NOTNULL(line_indices_1);
  CHECK_NOTNULL(line_indices_2);
  matchFramesBruteForce(frame_index_1, frame_index_2, matching_method,
                        line_indices_1, line_indices_2);
}

bool LineMatcher::matchFramesBruteForce(unsigned int frame_index_1,
                                        unsigned int frame_index_2,
                                        MatchingMethod matching_method,
                                        std::vector<int>* line_indices_1,
                                        std::vector<int>* line_indices_2) {
  CHECK_NOTNULL(line_indices_1);
  CHECK_NOTNULL(line_indices_2);


}


}  // namespace line_matching
