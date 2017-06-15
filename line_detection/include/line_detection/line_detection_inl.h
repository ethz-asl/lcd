#ifndef LINE_DETECTION_LINE_DETECTION_INL_H_
#define LINE_DETECTION_LINE_DETECTION_INL_H_

#include "line_detection/line_detection.h"

namespace line_detection {

template <typename T>
void getNUniqueRandomElements(const std::vector<T>& in, int N,
                              std::vector<T>& out,
                              std::default_random_engine& generator) {
  CHECK(in.size() > N);
  out.clear();
  out.reserve(N);

  int max = in.size();
  int indeces[max];
  int swap_helper, idx;
  for (int i = 0; i < max; ++i) indeces[i] = i;

  std::uniform_int_distribution<int>* distribution;
  for (int i = max; i > max - N; --i) {
    distribution = new std::uniform_int_distribution<int>(0, i - 1);
    idx = (*distribution)(generator);
    out.push_back(in[indeces[idx]]);
    swap_helper = indeces[idx];
    indeces[idx] = indeces[i - 1];
    indeces[i - 1] = swap_helper;
  }
  delete distribution;
}

template <typename T>
void getNUniqueRandomElements(const std::vector<T>& in, int N,
                              std::vector<T>& out) {
  unsigned seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  getNUniqueRandomElements(in, N, out, generator);
}

}  // namespace line_detection

#endif  // LINE_DETECTION_LINE_DETECTION_INL_H_
