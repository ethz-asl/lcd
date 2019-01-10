#include "line_description/line_description.h"

/*#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
*/
namespace line_description {
  LineDescriber::LineDescriber(DescriptorType descriptor_type) {
    switch (descriptor_type) {
      case DescriptorType::BINARY:
        descriptor_type_ = descriptor_type;
        binary_descriptor_ =
            cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        break;
      default:
        LOG(ERROR) << "Trying to initialize the line describer with an invalid "
                   << "descriptor type. Valid types are: BINARY.";
    }
  }

  void LineDescriber::describeLine(const cv::line_descriptor::KeyLine& keyline,
                                   const cv::Mat& image,
                                   Descriptor* descriptor) {
    CHECK_NOTNULL(descriptor);
    cv::Mat cv_descriptor;
    if (descriptor_type_ != DescriptorType::BINARY) {
      LOG(ERROR) << "Trying to use wrong interface to describe line. Expected "
                 << "BINARY descriptor type.";
      return;
    }
    // Create one-element vector.
    std::vector<cv::line_descriptor::KeyLine> keyline_vec;
    keyline_vec.push_back(keyline);
    binary_descriptor_->compute(image, keyline_vec, cv_descriptor);
    LOG(INFO) << "cv_descriptor.size().height is " << cv_descriptor.size().height;
    LOG(INFO) << "cv_descriptor.size().width is " << cv_descriptor.size().width;
    CHECK(cv_descriptor.size().height == 1 && cv_descriptor.size().width == 32);
    // Transform binary descriptor to a Descriptor object.
    descriptor->clear();
    for (size_t i = 0; i < 32; ++i) {
      // Descriptor is of type CV_8UC1.
      descriptor->push_back(
          static_cast<float>(cv_descriptor.at<unsigned char>(0, i)) / 255.0);
      LOG(INFO) << "Just added " << (*descriptor)[i] << " to the descriptor.";
    }
  }

  void LineDescriber::describeLines(
      const std::vector<cv::line_descriptor::KeyLine>& keylines,
      const cv::Mat& image, std::vector<Descriptor>* descriptors) {
    CHECK_NOTNULL(descriptors);
    cv::Mat cv_descriptors;
    if (descriptor_type_ != DescriptorType::BINARY) {
      LOG(ERROR) << "Trying to use wrong interface to describe lines. Expected "
                 << "BINARY descriptor type.";
      return;
    }
    // Create a copy (binding the argument of compute to const argument keyline
    // would discard qualifiers).
    std::vector<cv::line_descriptor::KeyLine> keylines_copy = keylines;
    binary_descriptor_->compute(image, keylines_copy, cv_descriptors);
    CHECK(cv_descriptors.size().height == keylines.size() &&
          cv_descriptors.size().width == 32);
    // Transform binary descriptors to an array of Descriptor objects.
    descriptors->clear();
    Descriptor temp_descriptor(32);
    for (size_t line_idx = 0; line_idx < keylines.size(); ++line_idx) {
      for (size_t i = 0; i < 32; ++i) {
        // Descriptor is of type CV_8UC1.
        temp_descriptor[i] =
            static_cast<float>(cv_descriptors.at<unsigned char>(line_idx, i)) /
                               255.0;
      }
      descriptors->push_back(temp_descriptor);
    }
  }


}  // namespace line_description
