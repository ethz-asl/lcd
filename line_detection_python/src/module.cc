#include "line_detection_python/ndarray_converter.h"
#include "pybind11/pybind11.h"

#include <line_detection/line_detection.h>

namespace line_detection {
cv::Mat detect3DLines(const cv::Mat cv_image_, const cv::Mat cv_image_depth_,
                      const cv::Mat cv_cloud_, const cv::Mat camera_P_) {
  // Detect 2D lines.
  line_detection::LineDetector line_detector_;
  cv::Mat cv_img_gray_;
  cv::cvtColor(cv_image_, cv_img_gray_, CV_RGB2GRAY);
  std::vector<cv::Vec4f> lines2D_;
  // Detect with LSD detector
  line_detector_.detectLines(cv_img_gray_, 0, &lines2D_);

  // Project 2D lines to 3D.
  std::vector<cv::Vec4f> lines2D_kept_tmp_, lines2D_kept_;
  std::vector<line_detection::LineWithPlanes> lines3D_temp_wp_,
      lines3D_with_planes_;
  line_detector_.project2Dto3DwithPlanes(cv_cloud_, cv_image_, camera_P_,
                                         lines2D_, true, &lines2D_kept_tmp_,
                                         &lines3D_temp_wp_);

  // Check if line is valid.
  line_detector_.runCheckOn3DLines(cv_cloud_, camera_P_, lines2D_kept_tmp_,
                                   lines3D_temp_wp_, &lines2D_kept_,
                                   &lines3D_with_planes_);

  // Number of 3D lines.
  size_t lines_number = lines3D_with_planes_.size();

  // 3D lines' start and end points.
  cv::Mat output(lines_number, 6, CV_32F);
  for (size_t i = 0u; i < lines_number; ++i) {
    for (size_t j = 0u; j < 6; ++j) {
      output.at<float>(i, j) = lines3D_with_planes_[i].line[j];
    }
  }
  return output;
}
}  // namespace line_detection

namespace py = pybind11;

PYBIND11_MODULE(py_line_detection, m) {
  NDArrayConverter::init_numpy();

  m.def(
      "detect3DLines",
      (cv::Mat(*)(const cv::Mat, const cv::Mat, const cv::Mat, const cv::Mat)) &
          line_detection::detect3DLines,
      "Detect 3D lines in rgb-d image.", py::arg("bgr_image"),
      py::arg("depth_image"), py::arg("pointcloud"),
      py::arg("camera_projection_matrix"));
}
