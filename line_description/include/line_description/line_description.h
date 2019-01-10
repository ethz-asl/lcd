#ifndef LINE_DESCRIPTION_LINE_DESCRIPTION_H_
#define LINE_DESCRIPTION_LINE_DESCRIPTION_H_

#include "line_description/common.h"

#include <vector>

#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

namespace line_description {

typedef std::vector<float> Descriptor;

enum class DescriptorType : unsigned int {
  EMBEDDING_NN = 0,
  BINARY = 1
};

class LineDescriber {
 public:
   LineDescriber(DescriptorType descriptor_type);

   // Returns the descriptor(s) for the given input line(s) in the input image.
   // * Binary descriptor version:
   //   Input: keyline(s): KeyLine object(s) associated to input line(s).
   //
   //          image:      Image from which the input line(s) was (/were)
   //                      extracted.
   //
   //   Output: descriptor(s): Descriptor(s) for the input line(s).
   void describeLine(const cv::line_descriptor::KeyLine& keyline,
                     const cv::Mat& image, Descriptor* descriptor);
   void describeLines(const std::vector<cv::line_descriptor::KeyLine>& keylines,
                      const cv::Mat& image,
                      std::vector<Descriptor>* descriptors);
 protected:

 private:
  // Type of descriptor to use.
  DescriptorType descriptor_type_;
  // Instance of describer object for each descriptor type.
  cv::Ptr<cv::line_descriptor::BinaryDescriptor> binary_descriptor_;
};
}  // namespace line_description

#include "line_description/line_description_inl.h"

#endif  // LINE_DESCRIPTION_LINE_DESCRIPTION_H_
