#ifndef LINE_DETECTION_LINE_DETECTION_INL_H_
#define LINE_DETECTION_LINE_DETECTION_INL_H_

#include "line_detection/line_detection.h"

namespace line_detection {

// This function samples a specific number of unique elements from an array.
// Input: in:         From this vector the elements are sampled.
//        N:          The number of samples taken.
//        generator:  A random engine that is handled to the uniform number
//                    sampler (std::uniform_int_distribution). It is used as an
//                    input here, so that the seed may be set outside of this
//                    function. This is useful if you run the function in a fast
//                    loop. Proposed engine: std::default_random_engine.
//
// Output: out:       A vector containing N unique samples of in.
template <typename T>
void getNUniqueRandomElements(const std::vector<T>& in, size_t num_samples,
                              std::default_random_engine* generator,
                              std::vector<T>* out) {
  CHECK_NOTNULL(generator);
  CHECK_NOTNULL(out);
  CHECK(in.size() > num_samples);
  out->clear();
  out->reserve(num_samples);
  // The algorithm uses the Fisher-Yates Shuffle to guarantee that no element is
  // sampled twice.
  size_t max = in.size();
  int idx;
  // From this array the indices of an element is sampled.
  int indices[max];
  for (size_t i = 0; i < max; ++i) {
    indices[i] = i;
  }
  std::uniform_int_distribution<int>* distribution;
  for (size_t i = max; i > max - num_samples; --i) {
    distribution = new std::uniform_int_distribution<int>(0, i - 1);
    idx = (*distribution)(*generator);
    out->push_back(in[indices[idx]]);
    indices[idx] = indices[i - 1];
  }
  delete distribution;
}

// An overload, that allows the use without specifyng an random engine. Be
// careful if this is used in a loop.
template <typename T>
void getNUniqueRandomElements(const std::vector<T>& in, size_t num_samples,
                              std::vector<T>* out) {
  unsigned seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  getNUniqueRandomElements(in, num_samples, &generator, out);
}

}  // namespace line_detection

#endif  // LINE_DETECTION_LINE_DETECTION_INL_H_
