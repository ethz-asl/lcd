#include "line_matching/line_matching.h"

namespace line_matching {

LineMatcher::LineMatcher() {

}

bool LineMatcher::addFrame(const Frame& frame_to_add, int frame_index) {


}

void LineMatcher::displayMatches(int frame_index_1, int frame_index_2,
                                 MatchingMethod matching_method) {


}

void LineMatcher::matchFrames(int frame_index_1, int frame_index_2,
                              MatchingMethod matching_method,
                              std::vector<int>* line_indices_1,
                              std::vector<int>* line_indices_2) {


}

}  // namespace line_matching
