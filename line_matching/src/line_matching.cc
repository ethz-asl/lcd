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
                                 MatchingMethod matching_method,
                                 unsigned int magnification_factor) {
  std::vector<int> line_indices_1, line_indices_2;
  std::vector<float> matching_ratings;
  // Obtain matches for the given frame indices, if possible.
  if (!matchFrames(frame_index_1, frame_index_2, matching_method,
                   &line_indices_1, &line_indices_2, &matching_ratings)) {
    LOG(ERROR) << "Unable to match frames with indices " << frame_index_1
               << " and " << frame_index_2 << ".";
    return false;
  }
  CHECK(line_indices_1.size() == line_indices_2.size());
  unsigned int num_matches = line_indices_1.size();
  LOG(INFO) << "Found a total of " << num_matches << " matches between frame "
            << frame_index_1 << " and " << frame_index_2 << ".";
  LOG(INFO) << "Frame " << frame_index_1 << " originally contained "
            << frames_[frame_index_1].lines.size() << " lines, while frame "
            << frame_index_2 << " originally contained "
            << frames_[frame_index_2].lines.size() << " lines.";
  // Display the two images side by side with the matched lines connected to
  // each other.
  size_t cols = frames_[frame_index_1].image.cols;
  size_t rows = frames_[frame_index_1].image.rows;
  cv::Mat large_image(cv::Size(2 * cols, rows),
                      frames_[frame_index_1].image.type());
  cv::Mat temp_image(cv::Size(2 * cols, rows),
                      frames_[frame_index_1].image.type());
  // Add first image.
  frames_[frame_index_1].image.copyTo(large_image(cv::Rect(0, 0, cols, rows)));
  // Add second image.
  frames_[frame_index_2].image.copyTo(large_image(cv::Rect(cols, 0, cols,
                                                           rows)));
  cv::Vec4f line2D_1, line2D_2;
  cv::Point2f center_line_1, center_line_2;
  std::string window_title;
  unsigned int idx_1, idx_2;
  // Display matches one at a time.
  for (size_t i = 0; i < num_matches; ++i) {
    idx_1 = line_indices_1[i];
    idx_2 = line_indices_2[i];
    large_image.copyTo(temp_image);
    // Draw lines in the first image (in red).
    line2D_1 = frames_[frame_index_1].lines[idx_1].line2D;
    cv::line(temp_image, cv::Point(line2D_1[0], line2D_1[1]),
             cv::Point(line2D_1[2], line2D_1[3]), CV_RGB(255, 0, 0));
    // Draw lines in the second image (in red).
    line2D_2 = frames_[frame_index_2].lines[idx_2].line2D;
    cv::line(temp_image, cv::Point(line2D_2[0] + cols, line2D_2[1]),
             cv::Point(line2D_2[2] + cols, line2D_2[3]), CV_RGB(255, 0, 0));
    center_line_1 = {(frames_[frame_index_1].lines[idx_1].line2D[0] +
                      frames_[frame_index_1].lines[idx_1].line2D[2]) / 2,
                     (frames_[frame_index_1].lines[idx_1].line2D[1] +
                      frames_[frame_index_1].lines[idx_1].line2D[3]) / 2};
    center_line_2 = {(frames_[frame_index_2].lines[idx_2].line2D[0] +
                      frames_[frame_index_2].lines[idx_2].line2D[2]) / 2,
                     (frames_[frame_index_2].lines[idx_2].line2D[1] +
                      frames_[frame_index_2].lines[idx_2].line2D[3]) / 2};
    // Draw yellow lines between the center of the matched lines.
    cv::line(temp_image, cv::Point(center_line_1.x, center_line_1.y),
             cv::Point(center_line_2.x + cols, center_line_2.y),
             CV_RGB(255, 255, 0));
    // Resize image.
    cv::resize(temp_image, temp_image,
               cv::Size(temp_image.size().width * magnification_factor,
                        temp_image.size().height * magnification_factor));
    // Display image.
    window_title = "Match " + std::to_string(i) + "/" +
                   std::to_string(num_matches) + " between frame " +
                   std::to_string(frame_index_1) + " and frame " +
                   std::to_string(frame_index_2) + ". Rating of current match "
                   "is " + std::to_string(matching_ratings[i]);
    cv::imshow(window_title, temp_image);
    cv::waitKey();
    try {
      cv::destroyWindow(window_title);
    }
    catch (cv::Exception& e) {
      LOG(INFO) << "Did not close window '" + window_title + "' because it was "
                << "not open.";
    }
  }
  return true;
}

bool LineMatcher::displayNBestMatchesPerLine(
    unsigned int frame_index_1, unsigned int frame_index_2,
    MatchingMethod matching_method, unsigned int num_matches_per_line,
    unsigned int magnification_factor) {
  std::vector<MatchWithRating> matches_with_ratings_vec;
  // Obtain matches for the given frame indices, if possible.
  if (!matchFramesNBestMatchesPerLine(frame_index_1, frame_index_2,
                                      matching_method, num_matches_per_line,
                                      &matches_with_ratings_vec)) {
    LOG(ERROR) << "Unable to match frames with indices " << frame_index_1
               << " and " << frame_index_2 << ".";
    return false;
  }
  LOG(INFO) << "Matched the lines: " << matches_with_ratings_vec .size()
            << " total matches.";
  // Display the two images side by side with the matched lines connected to
  // each other.
  size_t cols = frames_[frame_index_1].image.cols;
  size_t rows = frames_[frame_index_1].image.rows;
  cv::Mat large_image(cv::Size(2 * cols, rows),
                      frames_[frame_index_1].image.type());
  cv::Mat temp_image(cv::Size(2 * cols, rows),
                      frames_[frame_index_1].image.type());
  // Add first image.
  frames_[frame_index_1].image.copyTo(large_image(cv::Rect(0, 0, cols, rows)));
  // Add second image.
  frames_[frame_index_2].image.copyTo(large_image(cv::Rect(cols, 0, cols,
                                                           rows)));
  cv::Vec4f line2D_1, line2D_2;
  cv::Point2f center_line_1, center_line_2;
  std::string window_title;
  unsigned int idx_1, idx_2;
  float match_rating;
  // Display matches one at a time.
  for (size_t i = 0; i < matches_with_ratings_vec.size(); ++i) {
    match_rating = matches_with_ratings_vec[i].first;
    idx_1 = matches_with_ratings_vec[i].second.first;
    idx_2 = matches_with_ratings_vec[i].second.second;
    large_image.copyTo(temp_image);
    // Draw lines in the first image (in red).
    line2D_1 = frames_[frame_index_1].lines[idx_1].line2D;
    cv::line(temp_image, cv::Point(line2D_1[0], line2D_1[1]),
             cv::Point(line2D_1[2], line2D_1[3]), CV_RGB(255, 0, 0));
    // Draw lines in the second image (in red).
    line2D_2 = frames_[frame_index_2].lines[idx_2].line2D;
    cv::line(temp_image, cv::Point(line2D_2[0] + cols, line2D_2[1]),
             cv::Point(line2D_2[2] + cols, line2D_2[3]), CV_RGB(255, 0, 0));
    center_line_1 = {(frames_[frame_index_1].lines[idx_1].line2D[0] +
                      frames_[frame_index_1].lines[idx_1].line2D[2]) / 2,
                     (frames_[frame_index_1].lines[idx_1].line2D[1] +
                      frames_[frame_index_1].lines[idx_1].line2D[3]) / 2};
    center_line_2 = {(frames_[frame_index_2].lines[idx_2].line2D[0] +
                      frames_[frame_index_2].lines[idx_2].line2D[2]) / 2,
                     (frames_[frame_index_2].lines[idx_2].line2D[1] +
                      frames_[frame_index_2].lines[idx_2].line2D[3]) / 2};
    // Draw yellow lines between the center of the matched lines.
    cv::line(temp_image, cv::Point(center_line_1.x, center_line_1.y),
             cv::Point(center_line_2.x + cols, center_line_2.y),
             CV_RGB(255, 255, 0));
    // Resize image.
    cv::resize(temp_image, temp_image,
               cv::Size(temp_image.size().width * magnification_factor,
                        temp_image.size().height * magnification_factor));
    // Display image.
    window_title = "Match between line " + std::to_string(idx_1) + " in " +
                   "frame " + std::to_string(frame_index_1) + " and line " +
                   std::to_string(idx_2) + " in frame " +
                   std::to_string(frame_index_2) + ". Rating of current match "
                   "is " + std::to_string(match_rating);
    cv::imshow(window_title, temp_image);
    cv::waitKey();
    try {
      cv::destroyWindow(window_title);
    }
    catch (cv::Exception& e) {
      LOG(INFO) << "Did not close window '" + window_title + "' because it was "
                << "not open.";
    }
  }
  return true;
}

bool LineMatcher::displayBestMatchPerLine(unsigned int frame_index_1,
                                          unsigned int frame_index_2,
                                          MatchingMethod matching_method,
                                          unsigned int magnification_factor) {
  std::vector<MatchWithRating> matches_with_ratings_vec;
  // Obtain matches for the given frame indices, if possible.
  if (!matchFramesBestMatchPerLine(frame_index_1, frame_index_2,
                                   matching_method, &matches_with_ratings_vec)
                                   ) {
    LOG(ERROR) << "Unable to match frames with indices " << frame_index_1
               << " and " << frame_index_2 << ".";
    return false;
  }
  LOG(INFO) << "Matched the lines: " << matches_with_ratings_vec .size()
            << " total matches.";
  // Display the two images side by side with the matched lines connected to
  // each other.
  size_t cols = frames_[frame_index_1].image.cols;
  size_t rows = frames_[frame_index_1].image.rows;
  cv::Mat large_image(cv::Size(2 * cols, rows),
                      frames_[frame_index_1].image.type());
  cv::Mat temp_image(cv::Size(2 * cols, rows),
                      frames_[frame_index_1].image.type());
  // Add first image.
  frames_[frame_index_1].image.copyTo(large_image(cv::Rect(0, 0, cols, rows)));
  // Add second image.
  frames_[frame_index_2].image.copyTo(large_image(cv::Rect(cols, 0, cols,
                                                           rows)));
  cv::Vec4f line2D_1, line2D_2;
  cv::Point2f center_line_1, center_line_2;
  std::string window_title;
  unsigned int idx_1, idx_2;
  float match_rating;
  // Display matches one at a time.
  for (size_t i = 0; i < matches_with_ratings_vec.size(); ++i) {
    match_rating = matches_with_ratings_vec[i].first;
    idx_1 = matches_with_ratings_vec[i].second.first;
    idx_2 = matches_with_ratings_vec[i].second.second;
    large_image.copyTo(temp_image);
    // Draw lines in the first image (in red).
    line2D_1 = frames_[frame_index_1].lines[idx_1].line2D;
    cv::line(temp_image, cv::Point(line2D_1[0], line2D_1[1]),
             cv::Point(line2D_1[2], line2D_1[3]), CV_RGB(255, 0, 0));
    // Draw lines in the second image (in red).
    line2D_2 = frames_[frame_index_2].lines[idx_2].line2D;
    cv::line(temp_image, cv::Point(line2D_2[0] + cols, line2D_2[1]),
             cv::Point(line2D_2[2] + cols, line2D_2[3]), CV_RGB(255, 0, 0));
    center_line_1 = {(frames_[frame_index_1].lines[idx_1].line2D[0] +
                      frames_[frame_index_1].lines[idx_1].line2D[2]) / 2,
                     (frames_[frame_index_1].lines[idx_1].line2D[1] +
                      frames_[frame_index_1].lines[idx_1].line2D[3]) / 2};
    center_line_2 = {(frames_[frame_index_2].lines[idx_2].line2D[0] +
                      frames_[frame_index_2].lines[idx_2].line2D[2]) / 2,
                     (frames_[frame_index_2].lines[idx_2].line2D[1] +
                      frames_[frame_index_2].lines[idx_2].line2D[3]) / 2};
    // Draw yellow lines between the center of the matched lines.
    cv::line(temp_image, cv::Point(center_line_1.x, center_line_1.y),
             cv::Point(center_line_2.x + cols, center_line_2.y),
             CV_RGB(255, 255, 0));
    // Resize image.
    cv::resize(temp_image, temp_image,
               cv::Size(temp_image.size().width * magnification_factor,
                        temp_image.size().height * magnification_factor));
    // Display image.
    window_title = "Match between line " + std::to_string(idx_1) + " in " +
                   "frame " + std::to_string(frame_index_1) + " and line " +
                   std::to_string(idx_2) + " in frame " +
                   std::to_string(frame_index_2) + ". Rating of current match "
                   "is " + std::to_string(match_rating);
    cv::imshow(window_title, temp_image);
    cv::waitKey();
    try {
      cv::destroyWindow(window_title);
    }
    catch (cv::Exception& e) {
      LOG(INFO) << "Did not close window '" + window_title + "' because it was "
                << "not open.";
    }
  }
  return true;
}

bool LineMatcher::matchFrames(unsigned int frame_index_1,
                              unsigned int frame_index_2,
                              MatchingMethod matching_method,
                              std::vector<int>* line_indices_1,
                              std::vector<int>* line_indices_2,
                              std::vector<float>* matching_ratings) {
  CHECK_NOTNULL(line_indices_1);
  CHECK_NOTNULL(line_indices_2);
  return matchFramesBruteForce(frame_index_1, frame_index_2, matching_method,
                               line_indices_1, line_indices_2,
                               matching_ratings);
}

bool LineMatcher::matchFramesBruteForce(unsigned int frame_index_1,
                                        unsigned int frame_index_2,
                                        MatchingMethod matching_method,
                                        std::vector<int>* line_indices_1,
                                        std::vector<int>* line_indices_2,
                                        std::vector<float>* matching_ratings) {
  std::vector<MatchWithRating> candidate_matches;
  MatchRatingComputer* match_rating_computer;
  size_t num_lines_frame_1, num_lines_frame_2;
  size_t num_unmatched_lines_frame_1, num_unmatched_lines_frame_2;
  float rating;
  CHECK_NOTNULL(line_indices_1);
  CHECK_NOTNULL(line_indices_2);
  CHECK_NOTNULL(matching_ratings);
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
  std::sort(candidate_matches.begin(), candidate_matches.end());
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
  matching_ratings->clear();
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
      matching_ratings->push_back(candidate_matches[current_match_idx].first);
    }
    current_match_idx++;
  }
  // Delete matching computer.
  delete match_rating_computer;

  return true;
}

bool LineMatcher::matchFramesBestMatchPerLine(
    unsigned int frame_index_1, unsigned int frame_index_2,
    MatchingMethod matching_method,
    std::vector<MatchWithRating>* matches_with_ratings_vec) {
  std::vector<MatchWithRating> candidate_matches;
  MatchRatingComputer* match_rating_computer;
  size_t num_lines_frame_1, num_lines_frame_2;
  // Auxiliary variables used to compute the matches.
  float rating;
  MatchWithRating candidate_match_curr_line;
  // The true argument is to keep elements ordered in ascending order. The queue
  // keeps the best two elements..
  FixedSizePriorityQueue<MatchWithRating> best_matches_curr_line(2, true);
  LineWithEmbeddings* line_1;
  LineWithEmbeddings* line_2;
  // To return only the best match.
  MatchWithRating best_candidate_match, second_best_candidate_match;
  float ratio_best_two_ratings;
  // As in SIFT.
  constexpr float max_ratio_best_two_ratings = 0.8;

  CHECK_NOTNULL(matches_with_ratings_vec);

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

  num_lines_frame_1 = frames_[frame_index_1].lines.size();
  num_lines_frame_2 = frames_[frame_index_2].lines.size();

  matches_with_ratings_vec->clear();
  LOG(INFO) << "Matching lines.";
  for (size_t idx1 = 0; idx1 < num_lines_frame_1; ++idx1) {
    line_1 = &(frames_[frame_index_1].lines[idx1]);
    best_matches_curr_line.clear();
    for (size_t idx2 = 0; idx2 < num_lines_frame_2; ++idx2) {
      line_2 = &(frames_[frame_index_2].lines[idx2]);
      if (match_rating_computer->computeMatchRating(line_1->embeddings,
                                                    line_2->embeddings,
                                                    &rating)) {
        candidate_match_curr_line.first = rating;
        candidate_match_curr_line.second = std::make_pair(idx1, idx2);
        best_matches_curr_line.push(candidate_match_curr_line);
      }
    }
    // Add the candidate matches found to the output if they are valid matches.
    CHECK(best_matches_curr_line.size() >= 0 &&
          best_matches_curr_line.size() <= 2);
    if (best_matches_curr_line.size() == 2) {
      // Two candidate matches are found.
      best_candidate_match = best_matches_curr_line.front();
      best_matches_curr_line.pop();
      second_best_candidate_match = best_matches_curr_line.front();
      best_matches_curr_line.pop();
      // Compute the ratio of the ratings of the two candidate matches.
      ratio_best_two_ratings = best_candidate_match.first /
        second_best_candidate_match.first;
      LOG(INFO) << "Ratio of best two ratings is: " << ratio_best_two_ratings;
      if (ratio_best_two_ratings < max_ratio_best_two_ratings) {
        // The ratio of the best rating and the second best rating is less than
        // the threshold set above, than the best candidate match is a valid
        // match.
        matches_with_ratings_vec->push_back(best_candidate_match);
      }
    } else if (best_matches_curr_line.size() == 1) {
      // Only one candidate match found => Select it as a valid match.
      matches_with_ratings_vec->push_back(best_matches_curr_line.front());
      best_matches_curr_line.pop();
    }
  }

  // Delete matching computer.
  delete match_rating_computer;

  return true;
}

bool LineMatcher::matchFramesNBestMatchesPerLine(
    unsigned int frame_index_1, unsigned int frame_index_2,
    MatchingMethod matching_method, unsigned int num_matches_per_line,
    std::vector<MatchWithRating>* matches_with_ratings_vec) {
  std::vector<MatchWithRating> candidate_matches;
  MatchRatingComputer* match_rating_computer;
  size_t num_lines_frame_1, num_lines_frame_2;
  // Auxiliary variables used to compute the matches.
  float rating;
  MatchWithRating candidate_match_curr_line;
  // The true argument is to keep elements ordered in ascending order.
  FixedSizePriorityQueue<MatchWithRating> best_matches_curr_line(
    num_matches_per_line, true);
  LineWithEmbeddings* line_1;
  LineWithEmbeddings* line_2;

  CHECK_NOTNULL(matches_with_ratings_vec);

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

  num_lines_frame_1 = frames_[frame_index_1].lines.size();
  num_lines_frame_2 = frames_[frame_index_2].lines.size();

  matches_with_ratings_vec->clear();

  for (size_t idx1 = 0; idx1 < num_lines_frame_1; ++idx1) {
    line_1 = &(frames_[frame_index_1].lines[idx1]);
    best_matches_curr_line.clear();
    for (size_t idx2 = 0; idx2 < num_lines_frame_2; ++idx2) {
      line_2 = &(frames_[frame_index_2].lines[idx2]);
      if (match_rating_computer->computeMatchRating(line_1->embeddings,
                                                    line_2->embeddings,
                                                    &rating)) {
        candidate_match_curr_line.first = rating;
        candidate_match_curr_line.second = std::make_pair(idx1, idx2);
        best_matches_curr_line.push(candidate_match_curr_line);
      }
    }
    // Add the matches found to the output.
    while (!best_matches_curr_line.empty()) {
      matches_with_ratings_vec->push_back(best_matches_curr_line.front());
      best_matches_curr_line.pop();
    }
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
