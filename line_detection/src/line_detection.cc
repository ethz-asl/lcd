#include "line_detection/line_detection.h"

#include <cstdlib>

namespace line_detection {
cv::Vec3f projectPointOnPlane(const cv::Vec4f& hessian,
                              const cv::Vec3f& point) {
  cv::Vec3f x_0, normal;
  normal = {hessian[0], hessian[1], hessian[2]};
  size_t non_zero;
  for (non_zero = 0; non_zero < 3; ++non_zero) {
    if (fabs(hessian[non_zero]) > 0.1) break;
  }
  for (size_t i = 0; i < 3; ++i) {
    if (i == non_zero)
      x_0[i] = -hessian[3] / hessian[non_zero];
    else
      x_0[i] = 0;
  }
  return point - (point - x_0).dot(normal) * normal;
}

bool findIntersectionBetweenPlaneAndLine(const cv::Vec4f& plane,
                                         const cv::Vec3f& line_direction,
                                         cv::Vec3f* intersection_point) {
  // Plane is parametrized as: a*x+b*y+c*z+d=0.
  float a, b, c, d;
  cv::Vec3f normal_vector({plane[0], plane[1], plane[2]});
  d /= cv::norm(normal_vector);
  normalizeVector3D(&normal_vector);
  a = normal_vector[0];
  b = normal_vector[1];
  c = normal_vector[2];
  // Check that the line is not parallel to the plane, i.e., that it is not
  // perpendicular to the normal vector.
  if (normal_vector.dot(line_direction) == 0.)
    return false;
  float gamma = -d / (a * line_direction[0] + b * line_direction[1] +
    c * line_direction[3]);
  for (size_t i = 0; i < 3; ++i)
    intersection_point[i] = gamma * line_direction[i];
  return true;
}

bool findPlaneThroughPointAndLine(const cv::Vec3f& point,
                                  const cv::Vec3f& line_start,
                                  const cv::Vec3f& line_end,
                                  cv::Vec4f* plane) {
  // Check that the point does not belong to the line.
  if (line_start - point == point - line_end)
    return false;
  cv::Vec3f normal_vector = (line_end - point).cross(line_start - point);
  normalizeVector3D(&normal_vector);
  float a, b, c, d;
  a = normal_vector[0];
  b = normal_vector[1];
  c = normal_vector[2];
  // Find d so that, e.g. the point belongs to the plane.
  d = -(a * point[0] + b * point[1] + c * point[2]);
  (*plane)[0] = a;
  (*plane)[1] = b;
  (*plane)[2] = c;
  (*plane)[3] = d;
  return true;
}



bool areLinesEqual2D(const cv::Vec4f line1, const cv::Vec4f line2) {
  // First compute the difference in angle. For easier computation not the
  // actual difference in angle, but cos(theta)^2 is computed, using the
  // definition of dot product.
  float vx1 = line1[0] - line1[2];
  float vx2 = line2[0] - line2[2];
  float vy1 = line1[1] - line1[3];
  float vy2 = line2[1] - line2[3];

  float cos_sq_angle_difference = pow((vx1 * vx2 + vy1 * vy2), 2) /
                                  ((vx1 * vx1 + vy1 * vy1) *
                                  (vx2 * vx2 + vy2 * vy2));
  // Then compute the distance of the two lines. All distances between both
  // end and start points are computed and the lowest is kept.
  float dist[4], min_dist;
  dist[0] = pow(line1[0] - line2[0], 2) + pow(line1[1] - line2[1], 2);
  dist[1] = pow(line1[0] - line2[2], 2) + pow(line1[1] - line2[3], 2);
  dist[2] = pow(line1[2] - line2[0], 2) + pow(line1[3] - line2[1], 2);
  dist[3] = pow(line1[2] - line2[2], 2) + pow(line1[3] - line2[3], 2);
  min_dist = dist[0];
  for (size_t i = 1; i < 4; ++i) {
    if (dist[i] < min_dist) min_dist = dist[i];
  }

  // If angle difference and minimum distance are less than the thresholds,
  // return true. Note that since we want angle_difference ~= 0 it must hold
  // that cos(angle_difference) ~= 1 => cos^2(angle_difference) ~= 1.
  constexpr double kCosSqAngleDifference = 0.95;
  constexpr double kMinDistance = 0.5;
  if (cos_sq_angle_difference > kCosSqAngleDifference &&
      min_dist < kMinDistance) {
    return true;
  } else {
    return false;
  }
}

void findXCoordOfPixelsOnVector(const cv::Point2f& start,
                                const cv::Point2f& end, bool left_side,
                                std::vector<int>* x_coord) {
  CHECK_NOTNULL(x_coord);
  int top = floor(start.y);
  int bottom = ceil(end.y);
  int height = bottom - top;
  float x_start, width;
  x_start = floor(start.x) + 0.5;
  width = floor(end.x) - floor(start.x);
  CHECK(height > 0) << "Important: the following statement must hold: start.y "
                       "< end.y. "
                    << height;
  if (height == 1) {
    if (left_side) {
      x_coord->push_back(floor(start.x));
    } else {
      x_coord->push_back(ceil(end.x));
    }
    return;
  }
  for (int i = 0; i < height; ++i) {
    x_coord->push_back(int(x_start + i * width / (height - 1)));
  }
}

void findPointsInRectangle(std::vector<cv::Point2f> corners,
                           std::vector<cv::Point2i>* points) {
  CHECK_NOTNULL(points);
  CHECK_EQ(corners.size(), 4)
      << "The rectangle must be defined by exactly 4 corner points.";
  // Find the relative positions of the points.
  // int upper, lower, left, right, store_i;
  std::vector<int> idx{0, 1, 2, 3};
  // This part finds out if two of the points have equal y values. This may
  // not be very likely for some data, but if it happens it can produce
  // unpredictable outcome. If this is the case, the rectangle is rotated by
  // 0.1 degree. This should not make a difference, because the pixels have
  // integer values anyway (so a corner point of 100.1 and 100.2 gives the
  // same result).
  bool some_points_have_equal_height = false;
  // Check all y values against all others.
  for (size_t i = 0; i < 4; ++i) {
    for(size_t j = i+1; j < 4; ++j) {
      if (corners[i].y == corners[j].y){
        some_points_have_equal_height = true;
        break;
      }
    }
  }
  // Do the rotation.
  if (some_points_have_equal_height) {
    constexpr float kRotationDeg = 0.1;
    const float rotation_rad = degToRad(kRotationDeg);
    for (size_t i = 0u; i < 4u; ++i)
      corners[i] = {
          cos(rotation_rad) * corners[i].x - sin(rotation_rad) * corners[i].y,
          cos(rotation_rad) * corners[i].x + sin(rotation_rad) * corners[i].y};
  }
  // The points are set to lowest, highest, most right and most left in this
  // order. It does work because the preprocessing done guarantees that no two
  // points have the same y coordinate.
  cv::Point2f upper, lower, left, right;
  upper = corners[0];
  for (int i = 1; i < 4; ++i) {
    if (upper.y > corners[i].y) {
      upper = corners[i];
    }
  }
  lower.y = -1e6;
  for (int i = 0; i < 4; ++i) {
    if (lower.y < corners[i].y && corners[i] != upper) {
      lower = corners[i];
    }
  }
  left.x = 1e6;
  for (int i = 0; i < 4; ++i) {
    if (left.x > corners[i].x && corners[i] != upper && corners[i] != lower) {
      left = corners[i];
    }
  }
  for (int i = 0; i < 4; ++i) {
    if (corners[i] != left && corners[i] != upper && corners[i] != lower) {
      right = corners[i];
    }
  }
  // With the ordering given, the border pixels can be found as pixels, that
  // lie on the border vectors.
  std::vector<int> left_border;
  std::vector<int> right_border;
  findXCoordOfPixelsOnVector(upper, left, true, &left_border);
  findXCoordOfPixelsOnVector(upper, right, false, &right_border);
  // Pop_back is used because otherwise the corners[left/right] pixels would
  // be counted twice.
  left_border.pop_back();
  right_border.pop_back();
  findXCoordOfPixelsOnVector(left, lower, true, &left_border);
  findXCoordOfPixelsOnVector(right, lower, false, &right_border);
  if (left_border.size() > right_border.size()) {
    left_border.pop_back();
  } else if (left_border.size() < right_border.size()) {
    right_border.pop_back();
  }
  CHECK_EQ(left_border.size(), right_border.size());
  // Iterate over all pixels in the rectangle.
  points->clear();
  int x, y;
  for (size_t i = 0; i < left_border.size(); ++i) {
    y = floor(upper.y) + i;
    // y = floor(corners[upper].y) + i;
    x = left_border[i];
    do {
      points->push_back(cv::Point2i(x, y));
      ++x;
    } while (x <= right_border[i]);
  }
}

bool getPointOnPlaneIntersectionLine(const cv::Vec4f& hessian1,
                                     const cv::Vec4f& hessian2,
                                     const cv::Vec3f& direction,
                                     cv::Vec3f* x_0) {
  CHECK_NOTNULL(x_0);
  // The problem can be solved with a under determined linear system. See
  // http://mathworld.wolfram.com/Plane-PlaneIntersection.html
  cv::Mat m(2, 2, CV_32FC1);
  cv::Mat b(2, 1, CV_32FC1);
  cv::Mat x_0_mat(2, 1, CV_32FC1);
  int non_zero, count = 0;
  // Because the system is underdetemined, we can set an element of our
  // solution to zero. We just have to check that the corresponding element in
  // the direction vector is non-zero. For numerical stability we check here
  // that the element is greater than 0.1. Given that the vector is
  // normalized, at least one element always meets this condition.
  for (non_zero = 2; non_zero >= 0; --non_zero) {
    if (fabs(direction[non_zero]) > 0.1) break;
  }
  // Fill in the matrices for m*x_0 = b and solve the system.
  for (int i = 0; i < 3; ++i) {
    if (i == non_zero) continue;
    m.at<float>(0, count) = hessian1[i];
    m.at<float>(1, count) = hessian2[i];
    ++count;
  }
  b.at<float>(0, 0) = -hessian1[3];
  b.at<float>(1, 0) = -hessian2[3];
  bool success = cv::solve(m, b, x_0_mat);
  count = 0;
  // When filling in the solution we must again take into account that we
  // assumend a certain component to be zero.
  for (int i = 0; i < 3; ++i) {
    if (i == non_zero) {
      (*x_0)[i] = 0;
      continue;
    }
    (*x_0)[i] = x_0_mat.at<float>(count, 0);
    ++count;
  }
  return success;
}

cv::Mat getImageOfLine(const cv::Vec4f& line, const cv::Mat background_image,
                       const int scale_factor) {
   // Display line with rectangles in the image
   int cols = background_image.cols;
   int rows = background_image.rows;

   cv::Mat img_for_display(rows, cols, CV_8UC3);
   cv::resize(background_image, img_for_display, img_for_display.size());

   cv::Vec2f start({line[0], line[1]});
   cv::Vec2f end({line[2], line[3]});
   // Line
   cv::line(img_for_display, cv::Point(start[0], start[1]),
            cv::Point(end[0], end[1]),
            cv::Scalar(255, 0, 0));  // Red
  // Resize image
   cv::resize(img_for_display, img_for_display,
     img_for_display.size()*scale_factor);
   return img_for_display;
}

cv::Mat getImageOfLineWithRectangles(const cv::Vec4f& line,
                                     const std::vector<cv::Point2f>& rect_left,
                                     const std::vector<cv::Point2f>& rect_right,
                                     const cv::Mat background_image,
                                     const int scale_factor) {
  // Display line with rectangles in the image
  int cols = background_image.cols;
  int rows = background_image.rows;

  cv::Mat img_for_display(rows, cols, CV_8UC3);
  cv::resize(background_image, img_for_display, img_for_display.size());

  cv::Vec2f start({line[0], line[1]});
  cv::Vec2f end({line[2], line[3]});
  // Line
  cv::line(img_for_display, cv::Point(start[0], start[1]),
           cv::Point(end[0], end[1]),
           cv::Scalar(0, 0, 0));  // Black
  // Left rectangle
  cv::line(img_for_display, rect_left[0], rect_left[1],
           cv::Scalar(255, 0, 255)); // Purple
  cv::line(img_for_display, rect_left[2], rect_left[3],
           cv::Scalar(255, 0, 255)); // Purple
  cv::line(img_for_display, rect_left[1], rect_left[3],
           cv::Scalar(255, 0, 255)); // Purple
  // Left rectangle
  cv::line(img_for_display, rect_right[0], rect_right[1],
           cv::Scalar(255, 255, 0)); // Cyan
  cv::line(img_for_display, rect_right[2], rect_right[3],
           cv::Scalar(255, 255, 0)); // Cyan
  cv::line(img_for_display, rect_right[1], rect_right[3],
           cv::Scalar(255, 255, 0)); // Cyan
  cv::resize(img_for_display, img_for_display,
    img_for_display.size()*scale_factor);
  return img_for_display;
}



void displayLineWithPointsAndPlanes(const cv::Vec3f& start,
                                    const cv::Vec3f& end,
                                    const cv::Vec3f& start_guess,
                                    const cv::Vec3f& end_guess,
                                    const std::vector<cv::Vec3f>& inliers1,
                                    const std::vector<cv::Vec3f>& inliers2,
                                    const cv::Vec4f& hessian1,
                                    const cv::Vec4f& hessian2) {
  // Write YAML file containing the information to be parsed by the Python
  // script.
  boost::filesystem::path line_tools_root_path(
    line_tools_paths::kLineToolsRootPath);
  boost::filesystem::path filename("line_with_points_and_planes.yaml");
  boost::filesystem::path full_path = line_tools_root_path / filename;
  std::ofstream out(full_path.string());
  out << "# Line" << std::endl;
  out << "start: [" << start[0] << ", " << start[1] << ", " << start[2] << "]"
      << std::endl;
  out << "end: [" << end[0] << ", " << end[1] << ", " << end[2] << "]"
      << std::endl;
  out << "# Line_guess" << std::endl;
  out << "start_guess: [" << start_guess[0] << ", " << start_guess[1] << ", "
      << start_guess[2] << "]" << std::endl;
  out << "end_guess: [" << end_guess[0] << ", " << end_guess[1] << ", "
      << end_guess[2] << "]" << std::endl;
  out << "# Hessians" << std::endl;
  out << "hessians:" << std::endl;
  out << "  0: [" << hessian1[0] << ", " << hessian1[1] << ", "
      << hessian1[2] << ", " << hessian1[3] << "]" << std::endl;
  out << "  1: [" << hessian2[0] << ", " << hessian2[1] << ", "
      << hessian2[2] << ", " << hessian2[3] << "]" << std::endl;
  out << "# Inlier points" << std::endl;
  out << "inliers:" << std::endl;
  out << "  0:" << std::endl;
  for (size_t i = 0; i < inliers1.size(); ++i) {
    out << "    - [" << inliers1[i][0] << ", " << inliers1[i][1] << ", "
        << inliers1[i][2] << "]" << std::endl;
  }
  out << "  1:" << std::endl;
  for (size_t i = 0; i < inliers2.size(); ++i) {
    out << "    - [" << inliers2[i][0] << ", " << inliers2[i][1] << ", "
        << inliers2[i][2] << "]" << std::endl;
  }
  out.close();
  // Call python script
  boost::filesystem::path python_script_rel_path(
    "python/display_line_with_points_and_planes.py");
  full_path = line_tools_root_path / python_script_rel_path;
  std::string command = "python " + full_path.string();
  system(command.c_str());
}

LineDetector::LineDetector() {
  lsd_detector_ = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
  edl_detector_ =
      cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
  fast_detector_ = cv::ximgproc::createFastLineDetector();
  params_ = new LineDetectionParams();
  params_is_mine_ = true;
}
LineDetector::LineDetector(LineDetectionParams* params) {
  lsd_detector_ = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
  edl_detector_ =
      cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
  fast_detector_ = cv::ximgproc::createFastLineDetector();
  params_ = params;
  params_is_mine_ = false;
}
LineDetector::~LineDetector() {
  if (params_is_mine_) {
    delete params_;
  }
}

void LineDetector::detectLines(const cv::Mat& image, int detector,
                               std::vector<cv::Vec4f>* lines) {
  if (detector == 0)
    detectLines(image, Detector::LSD, lines);
  else if (detector == 1)
    detectLines(image, Detector::EDL, lines);
  else if (detector == 2)
    detectLines(image, Detector::FAST, lines);
  else if (detector == 3)
    detectLines(image, Detector::HOUGH, lines);
  else {
    LOG(WARNING)
        << "LineDetector::detectLines: Detector choice not valid, LSD was "
           "chosen as default.";
    detectLines(image, Detector::LSD, lines);
  }
}
void LineDetector::detectLines(const cv::Mat& image, Detector detector,
                               std::vector<cv::Vec4f>* lines) {
  CHECK_NOTNULL(lines);
  lines->clear();
  // Check which detector is chosen by user. If an invalid number is given the
  // default (LSD) is chosen without a warning.
  if (detector == Detector::LSD) {
    lsd_detector_->detect(image, *lines);
  } else if (detector == Detector::EDL) {  // EDL_DETECTOR
    // The edl detector uses a different kind of vector to store the lines in.
    // The conversion is done later.
    std::vector<cv::line_descriptor::KeyLine> edl_lines;
    edl_detector_->detect(image, edl_lines);

    // Write lines to standard format
    for (size_t i = 0u; i < edl_lines.size(); ++i) {
      lines->push_back(cv::Vec4f(
          edl_lines[i].getStartPoint().x, edl_lines[i].getStartPoint().y,
          edl_lines[i].getEndPoint().x, edl_lines[i].getEndPoint().y));
    }

  } else if (detector == Detector::FAST) {  // FAST_DETECTOR
    fast_detector_->detect(image, *lines);
  } else if (detector == Detector::HOUGH) {  // HOUGH_DETECTOR
    cv::Mat output;
    // Parameters of the Canny should not be changed (or better: the result is
    // very likely to get worse).
    cv::Canny(image, output, params_->canny_edges_threshold1,
              params_->canny_edges_threshold2, params_->canny_edges_aperture);
    // Here parameter changes might improve the result.
    cv::HoughLinesP(output, *lines, params_->hough_detector_rho,
                    params_->hough_detector_theta,
                    params_->hough_detector_threshold,
                    params_->hough_detector_minLineLength,
                    params_->hough_detector_maxLineGap);
  }
}
void LineDetector::detectLines(const cv::Mat& image,
                               std::vector<cv::Vec4f>* lines) {
  detectLines(image, Detector::LSD, lines);
}

bool LineDetector::hessianNormalFormOfPlane(
    const std::vector<cv::Vec3f>& points, cv::Vec4f* hessian_normal_form) {
  CHECK_NOTNULL(hessian_normal_form);
  const int num_points = points.size();
  CHECK(num_points >= 3);
  if (num_points == 3) {  // In this case an exact solution can be computed.
    cv::Vec3f vec1 = points[1] - points[0];
    cv::Vec3f vec2 = points[2] - points[0];
    // This checks first if the points were too close.
    double norms = cv::norm(vec1) * cv::norm(vec2);
    if (norms < params_->min_distance_between_points_hessian) return false;
    // Then if they lie on a line. The angle between the vectors must at least
    // be 2 degrees.
    double cos_theta = fabs(vec1.dot(vec2)) / norms;
    if (cos_theta > params_->max_cos_theta_hessian_computation) return false;
    // The normal already defines the orientation of the plane (it is
    // perpendicular to both vectors, since they must lie within the plane).
    cv::Vec3f normal = vec1.cross(vec2);
    // Now bring the plane into the hessian normal form.
    *hessian_normal_form =
        cv::Vec4f(normal[0], normal[1], normal[2],
                  computeDfromPlaneNormal(normal, points[0]));
    *hessian_normal_form = (*hessian_normal_form) / cv::norm(normal);
    return true;
  } else {  // If there are more than 3 points, the solution is approximate.
    cv::Vec3f mean(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < num_points; ++i) {
      mean += points[i] / num_points;
    }
    cv::Mat A(3, num_points, CV_64FC1);
    for (size_t i = 0u; i < static_cast<size_t>(num_points); ++i) {
      A.at<double>(0, i) = points[i][0] - mean[0];
      A.at<double>(1, i) = points[i][1] - mean[1];
      A.at<double>(2, i) = points[i][2] - mean[2];
    }
    cv::Mat U, W, Vt;
    cv::SVD::compute(A, W, U, Vt);
    cv::Vec3f normal;
    if (U.type() == CV_64FC1) {
      normal =
          cv::Vec3f(U.at<double>(0, 2), U.at<double>(1, 2), U.at<double>(2, 2));
    } else if (U.type() == CV_32FC1) {
      normal =
          cv::Vec3f(U.at<float>(0, 2), U.at<float>(1, 2), U.at<float>(2, 2));
    }
    *hessian_normal_form = cv::Vec4f(normal[0], normal[1], normal[2],
                                     computeDfromPlaneNormal(normal, mean));
    return true;
  }
}

void LineDetector::projectLines2Dto3D(const std::vector<cv::Vec4f>& lines2D,
                                      const cv::Mat& point_cloud,
                                      std::vector<cv::Vec6f>* lines3D) {
  CHECK_NOTNULL(lines3D);
  // First check if the point_cloud mat has the right format.
  CHECK_EQ(point_cloud.type(), CV_32FC3)
      << "The input matrix point_cloud must be of type CV_32FC3.";
  lines3D->clear();
  cv::Point2i start, end;
  for (size_t i = 0; i < lines2D.size(); ++i) {
    start.x = floor(lines2D[i][0]);
    start.y = floor(lines2D[i][1]);
    end.x = floor(lines2D[i][2]);
    end.y = floor(lines2D[i][3]);
    if (!std::isnan(point_cloud.at<cv::Vec3f>(start)[0]) &&
        !std::isnan(point_cloud.at<cv::Vec3f>(end)[0])) {
      lines3D->push_back(cv::Vec6f(point_cloud.at<cv::Vec3f>(start)[0],
                                   point_cloud.at<cv::Vec3f>(start)[1],
                                   point_cloud.at<cv::Vec3f>(start)[2],
                                   point_cloud.at<cv::Vec3f>(end)[0],
                                   point_cloud.at<cv::Vec3f>(end)[1],
                                   point_cloud.at<cv::Vec3f>(end)[2]));
    }
  }
}

void LineDetector::fuseLines2D(const std::vector<cv::Vec4f>& lines_in,
                               std::vector<cv::Vec4f>* lines_out) {
  CHECK_NOTNULL(lines_out);
  lines_out->clear();
  // This list is used to remember which lines have already been assigned to a
  // cluster. Every time a line is assigned, the corresponding index is
  // deleted in this list.
  std::list<int> line_index;
  for (size_t i = 0; i < lines_in.size(); ++i) line_index.push_back(i);
  // This vector is used to store the line clusters until they are merged into
  // one line.
  std::vector<cv::Vec4f> line_cluster;
  // Iterate over all lines.
  for (size_t i = 0u; i < lines_in.size(); ++i) {
    line_cluster.clear();
    // If this condition does not hold, the line lines_in[i] has already been
    // merged into a new one. If not, the algorithm tries to find lines that
    // are near this line.
    if (*(line_index.begin()) != static_cast<int>(i)) {
      continue;
    } else {
      line_cluster.push_back(lines_in[i]);
      line_index.pop_front();
    }
    for (std::list<int>::iterator it = line_index.begin();
         it != line_index.end(); ++it) {
      // This loop checks if the line is near any line in the momentary
      // cluster. If yes, it assignes it to the cluster.
      for (cv::Vec4f& line : line_cluster) {
        if (areLinesEqual2D(line, lines_in[*it])) {
          line_cluster.push_back(lines_in[*it]);
          it = line_index.erase(it);
          break;
        }
      }
    }
    // If the cluster size is one, then no cluster was found.
    if (line_cluster.size() == 1) {
      lines_out->push_back(line_cluster[0]);
      continue;
    }
    // Here all the lines of a cluster are merged into one.
    int x_min = 1e4, x_max = 0, y_min = 1e4, y_max = 0, slope = 0;
    for (cv::Vec4f& line : line_cluster) {
      if (line[0] < x_min) x_min = line[0];
      if (line[0] > x_max) x_max = line[0];
      if (line[1] < y_min) y_min = line[1];
      if (line[1] > y_max) y_max = line[1];
      if (line[2] < x_min) x_min = line[2];
      if (line[2] > x_max) x_max = line[2];
      if (line[3] < y_min) y_min = line[3];
      if (line[3] > y_max) y_max = line[3];
      slope += computeSlopeOfLine(line);
    }
    if (slope > 0)
      lines_out->push_back(cv::Vec4f(x_min, y_min, x_max, y_max));
    else
      lines_out->push_back(cv::Vec4f(x_min, y_max, x_max, y_min));
  }
}

void LineDetector::paintLines(const std::vector<cv::Vec4f>& lines,
                              cv::Mat* image, cv::Vec3b color) {
  cv::Point2i p1, p2;

  for (size_t i = 0u; i < lines.size(); ++i) {
    p1.x = lines[i][0];
    p1.y = lines[i][1];
    p2.x = lines[i][2];
    p2.y = lines[i][3];

    constexpr int kThickness = 1;
    cv::line(*image, p1, p2, color, kThickness);
  }
}

bool LineDetector::find3DLineStartAndEnd(const cv::Mat& point_cloud,
                                         const cv::Vec4f& line2D,
                                         cv::Vec6f* line3D, cv::Point2f* start,
                                         cv::Point2f* end) {
  CHECK_NOTNULL(line3D);
  CHECK_NOTNULL(start);
  CHECK_NOTNULL(end);
  CHECK_EQ(point_cloud.type(), CV_32FC3)
      << "The input matrix point_cloud must be of type CV_32FC3.";
  // A floating point value that decribes a position in an image is always
  // within the pixel described through the floor operation.
  start->x = floor(line2D[0]);
  start->y = floor(line2D[1]);
  end->x = floor(line2D[2]);
  end->y = floor(line2D[3]);
  // Search for a non NaN value on the line. Effectively these two while loops
  // just make unit steps (one pixel) from start to end (first loop) and then
  // from end to start (second loop) until a non NaN point is found.
  cv::LineIterator it_start_end(point_cloud, *(start), *(end), 8);
  // Search for a non NaN value on the line.
  // From starting point
  //LOG(INFO) << "it_start_end at (" << start->x << ", " << start-> y << ") and "
  //          << "is " << (std::isnan(point_cloud.at<cv::Vec3f>(*(start))[0]) ?
  //             "" : "not ") << "NaN";
  while (std::isnan(point_cloud.at<cv::Vec3f>(*(start))[0])) {
    ++it_start_end;
    *(start) = it_start_end.pos();
    //LOG(INFO) << "it_start_end at (" << start->x << ", " << start-> y << ") "
    //          << "and is " << (std::isnan(point_cloud.at<cv::Vec3f>(*(start))[0]) ?
    //             "" : "not ") << "NaN";
    if (start->x == end->x && start->y == end->y) break;
  }
  if (start->x == end->x && start->y == end->y) return false;
  // From ending point
  cv::LineIterator it_end_start(point_cloud, *(end), *(start), 8);
  //LOG(INFO) << "it_end_start at (" << start->x << ", " << start-> y << ") and "
  //          << "is " << (std::isnan(point_cloud.at<cv::Vec3f>(*(start))[0]) ?
  //             "" : "not ") << "NaN";
  while (std::isnan(point_cloud.at<cv::Vec3f>(*(end))[0])) {
    ++it_end_start;
    *(end) = it_end_start.pos();
    //LOG(INFO) << "it_end_start at (" <<end->x << ", " << end-> y << ") "
    //          << "and is " << (std::isnan(point_cloud.at<cv::Vec3f>(*(end))[0]) ?
    //             "" : "not ") << "NaN";
    if (start->x == end->x && start->y == end->y) break;
  }
  if (start->x == end->x && start->y == end->y) return false;
  *line3D = cv::Vec6f(point_cloud.at<cv::Vec3f>(*(start))[0],
                      point_cloud.at<cv::Vec3f>(*(start))[1],
                      point_cloud.at<cv::Vec3f>(*(start))[2],
                      point_cloud.at<cv::Vec3f>(*(end))[0],
                      point_cloud.at<cv::Vec3f>(*(end))[1],
                      point_cloud.at<cv::Vec3f>(*(end))[2]);
  return true;
}

double LineDetector::findAndRate3DLine(const cv::Mat& point_cloud,
                                       const cv::Vec4f& line2D,
                                       cv::Vec6f* line3D, int* num_points) {
  CHECK_NOTNULL(line3D);
  CHECK_NOTNULL(num_points);
  CHECK_EQ(point_cloud.type(), CV_32FC3)
      << "The input matrix point_cloud must be of type CV_32FC3.";
  cv::Point2f start, end, rate_it;
  if (!find3DLineStartAndEnd(point_cloud, line2D, line3D, &start, &end)) {
    return 1e9;
  }

  // In addition to find3DLineStartAndEnd, this function also rates the
  // line. The rating is based on the average distance between 3D line and
  // 3D points considered as on the 3D line (i.e., 2D points lie on the 2D
  // line)
  double rating = 0.0;
  double rating_temp;
  *num_points = 0;
  rate_it = start;

  int num_nan_points = 0;

  cv::LineIterator it_start_end_found(point_cloud, start, end, 8);
  while (!(rate_it.x == end.x && rate_it.y == end.y)) {
    if (std::isnan(point_cloud.at<cv::Vec3f>(rate_it)[0])){
      ++num_nan_points;
      continue;
    }
    rating_temp = distPointToLine(point_cloud.at<cv::Vec3f>(start),
                                  point_cloud.at<cv::Vec3f>(end),
                                  point_cloud.at<cv::Vec3f>(rate_it));
    ++it_start_end_found;
    rate_it = it_start_end_found.pos();
    rating += rating_temp;
    ++(*num_points);
  }

  //LOG(INFO) << "Found " << num_nan_points << " NaN points for this line.";
  return rating / (*num_points);
}
double LineDetector::findAndRate3DLine(const cv::Mat& point_cloud,
                                       const cv::Vec4f& line2D,
                                       cv::Vec6f* line3D) {
  int num_points;
  return findAndRate3DLine(point_cloud, line2D, line3D, &num_points);
}

std::vector<cv::Vec4f> LineDetector::checkLinesInBounds(
    const std::vector<cv::Vec4f>& lines2D, size_t x_max, size_t y_max) {
  CHECK(x_max > 0);
  CHECK(y_max > 0);
  std::vector<cv::Vec4f> new_lines;
  new_lines.reserve(lines2D.size());
  double x_bound = static_cast<double>(x_max) - 1e-9;
  double y_bound = static_cast<double>(y_max) - 1e-9;
  for (size_t i = 0; i < lines2D.size(); ++i) {
    new_lines.push_back(
        {static_cast<float>(checkInBoundary(lines2D[i][0], 0.0, x_bound)),
         static_cast<float>(checkInBoundary(lines2D[i][1], 0.0, y_bound)),
         static_cast<float>(checkInBoundary(lines2D[i][2], 0.0, x_bound)),
         static_cast<float>(checkInBoundary(lines2D[i][3], 0.0, y_bound))});
  }
  return new_lines;
}

bool LineDetector::getRectanglesFromLine(const cv::Vec4f& line,
                                         std::vector<cv::Point2f>* rect_left,
                                         std::vector<cv::Point2f>* rect_right) {
  CHECK_NOTNULL(rect_right);
  CHECK_NOTNULL(rect_left);
  // The offset defines how far away from the line the nearest corner points
  // are.
  double offset = params_->rectangle_offset_pixels;
  double relative_rect_size = params_->max_relative_rect_size;
  // Defines the length of the side perpendicular to the line.
  // Exactly as above, but defines a numerical maximum.
  double max_rect_size = params_->max_absolute_rect_size;
  double eff_rect_size = max_rect_size;
  cv::Point2f start(line[0], line[1]);
  cv::Point2f end(line[2], line[3]);
  cv::Point2f line_dir = end - start;
  cv::Point2f go_left(-line_dir.y, line_dir.x);
  cv::Point2f go_right(line_dir.y, -line_dir.x);
  double norm = sqrt(pow(line_dir.x, 2) + pow(line_dir.y, 2));
  if (eff_rect_size > norm * relative_rect_size)
    eff_rect_size = norm * relative_rect_size;
  rect_left->resize(4);
  (*rect_left)[0] = start + offset / norm * go_left;
  (*rect_left)[1] = start + eff_rect_size / norm * go_left;
  (*rect_left)[2] = end + offset / norm * go_left;
  (*rect_left)[3] = end + eff_rect_size / norm * go_left;
  rect_right->resize(4);
  (*rect_right)[0] = start + offset / norm * go_right;
  (*rect_right)[1] = start + eff_rect_size / norm * go_right;
  (*rect_right)[2] = end + offset / norm * go_right;
  (*rect_right)[3] = end + eff_rect_size / norm * go_right;
  // TODO(ff): Check if function needs to be of non void return type.
}

void LineDetector::assignColorToLines(const cv::Mat& image,
                                      const std::vector<cv::Point2i>& points,
                                      LineWithPlanes* line3D) {
  CHECK_NOTNULL(line3D);
  CHECK_EQ(image.type(), CV_8UC3);
  long long x1 = 0, x2 = 0, x3 = 0;
  int num_points = points.size();
  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].x < 0 || points[i].x >= image.cols || points[i].y < 0 ||
        points[i].y >= image.rows) {
      continue;
    }
    x1 += image.at<cv::Vec3b>(points[i])[0];
    x2 += image.at<cv::Vec3b>(points[i])[1];
    x3 += image.at<cv::Vec3b>(points[i])[2];
  }
  line3D->colors.push_back({static_cast<unsigned char>(x1 / num_points),
                            static_cast<unsigned char>(x2 / num_points),
                            static_cast<unsigned char>(x3 / num_points)});
}

// DEPRECATED
bool LineDetector::find3DlineOnPlanes(const std::vector<cv::Vec3f>& points1,
                                      const std::vector<cv::Vec3f>& points2,
                                      const cv::Vec6f& line_guess,
                                      cv::Vec6f* line) {
  LineWithPlanes line_with_planes;
  if (find3DlineOnPlanes(points1, points2, line_guess, &line_with_planes)) {
    *line = line_with_planes.line;
    return true;
  } else {
    return false;
  }
}

// DEPRECATED
bool LineDetector::find3DlineOnPlanes(const std::vector<cv::Vec3f>& points1,
                                      const std::vector<cv::Vec3f>& points2,
                                      const cv::Vec6f& line_guess,
                                      LineWithPlanes* line) {
  CHECK_NOTNULL(line);
  size_t N1 = points1.size();
  size_t N2 = points2.size();
  if (N1 < 3 || N2 < 3) return false;
  cv::Vec3f mean1, mean2, normal1, normal2;
  line->hessians.resize(2);
  // cv::Vec4f hessian1, hessian2;
  // Fit a plane model to the two sets of points individually.
  if (!hessianNormalFormOfPlane(points1, &(line->hessians[0])))
    LOG(WARNING) << "find3DlineOnPlanes: search for hessian failed.";
  if (!hessianNormalFormOfPlane(points2, &(line->hessians[1])))
    LOG(WARNING) << "find3DlineOnPlanes: search for hessian failed.";
  // Extract the two plane normals.
  normal1 = {line->hessians[0][0], line->hessians[0][1], line->hessians[0][2]};
  normal2 = {line->hessians[1][0], line->hessians[1][1], line->hessians[1][2]};
  // This parameter defines at which point 2 lines are concerned to be near.
  // This distance is computed from the means of the two set of points. If the
  // distance is higher than this value, it is assumed that the line is not
  // the intersection of the two planes, but just lies on the one that is in
  // the foreground.
  constexpr double angle_difference = 0.995;
  mean1 = computeMean(points1);
  mean2 = computeMean(points2);
  if (cv::norm(mean1 - mean2) < params_->max_dist_between_planes) {
    // Checks if the planes are parallel.
    if (fabs(normal1.dot(normal2)) > angle_difference) {
      line->line = line_guess;
      line->type = LineType::PLANE;
      return true;
    } else {
      // The line lying on both planes must be perpendicular to both normals,
      // so it can be computed with the cross product.
      cv::Vec3f direction = normal1.cross(normal2);
      normalizeVector3D(&direction);
      // Now a point on the intersection line is searched.
      cv::Vec3f x_0;
      getPointOnPlaneIntersectionLine(line->hessians[0], line->hessians[1],
                                      direction, &x_0);
      // This part searches for start and end point, because so far we only
      // have a line from and to infinity. The procedure used here projects
      // all points in both sets onto the line and then chooses the pair of
      // points that maximizes the distance of the line.
      double dist;
      double dist_min = 1e9;
      double dist_max = -1e9;
      for (size_t i = 0; i < N1; ++i) {
        dist = direction.dot(points1[i] - x_0);
        if (dist < dist_min) dist_min = dist;
        if (dist > dist_max) dist_max = dist;
      }
      for (size_t i = 0; i < N2; ++i) {
        dist = direction.dot(points2[i] - x_0);
        if (dist < dist_min) dist_min = dist;
        if (dist > dist_max) dist_max = dist;
      }
      cv::Vec3f start, end;
      start = x_0 + direction * dist_min;
      end = x_0 + direction * dist_max;
      line->line = {start[0], start[1], start[2], end[0], end[1], end[2]};

      line->type = LineType::EDGE;
      return true;
    }
  } else {
    // If we reach this point, we have a discontinuity. We then try to fit a
    // line to the set of points that lies closer to the origin (and therefore
    // closer to the camera). This is in most cases a reasonable assumption,
    // since the line most of the time belongs to the object that obscures the
    // background (which causes the discontinuity).
    // The fitting is done in 3 steps:
    //  1.  Project the line_guess onto the plane fitted to the point set.
    //  2.  Find the line parallel to the projected one that goes through the
    //      point in the set of the points that is nearest to the line.
    //  3.  From all points in the set: Project them on to the line and choose
    //      the combination of start/end point that maximizes the line distance.
    cv::Vec3f start(line_guess[0], line_guess[1], line_guess[2]);
    cv::Vec3f end(line_guess[3], line_guess[4], line_guess[5]);
    cv::Vec3f direction = end - start;
    normalizeVector3D(&direction);

    const std::vector<cv::Vec3f>* points_new;
    int idx;
    if (cv::norm(mean1) < cv::norm(mean2)) {
      idx = 0;
      points_new = &points1;
    } else {
      idx = 1;
      points_new = &points2;
    }
    line->hessians[abs(idx - 1)] = {0.0f, 0.0f, 0.0f, 0.0f};
    start = projectPointOnPlane(line->hessians[idx], start);
    end = projectPointOnPlane(line->hessians[idx], end);
    cv::Vec3f nearest_point;
    double min_dist = 1e9;
    double dist, dist_dir, dist_dir_min, dist_dir_max;
    dist_dir_min = 1e9;
    dist_dir_max = -1e9;
    for (size_t i = 0; i < points_new->size(); ++i) {
      // dist is used to find the nearest point to the line.
      dist =
          cv::norm(start - (*points_new)[i]) + cv::norm(end - (*points_new)[i]);
      if (dist < min_dist) {
        min_dist = dist;
        nearest_point = (*points_new)[i];
      }
      // dist_dir is used to find the points that maximize the line.
      dist_dir = direction.dot((*points_new)[i] - start);
      if (dist_dir < dist_dir_min) dist_dir_min = dist_dir;
      if (dist_dir > dist_dir_max) dist_dir_max = dist_dir;
    }
    cv::Vec3f x_0 = projectPointOnLine(nearest_point, direction, start);
    start = x_0 + direction * dist_dir_min;
    end = x_0 + direction * dist_dir_max;
    line->line = {start[0], start[1], start[2], end[0], end[1], end[2]};
    line->type = LineType::DISCONT;
    return true;
  }
}

bool LineDetector::find3DlineOnPlanes(const std::vector<cv::Vec3f>& points1,
                                      const std::vector<cv::Vec3f>& points2,
                                      const cv::Vec6f& line_guess,
                                      const cv::Mat& cloud,
                                      const cv::Mat& camera_P,
                                      const bool planes_found,
                                      LineWithPlanes* line) {
  CHECK_NOTNULL(line);
  size_t N1 = points1.size();
  size_t N2 = points2.size();
  if (N1 < 3 || N2 < 3) return false;
  cv::Vec3f mean1, mean2, normal1, normal2;
  line->hessians.resize(2);
  // cv::Vec4f hessian1, hessian2;
  // Fit a plane model to the two sets of points individually.
  if (!hessianNormalFormOfPlane(points1, &(line->hessians[0])))
    LOG(WARNING) << "find3DlineOnPlanes: search for hessian failed.";
  if (!hessianNormalFormOfPlane(points2, &(line->hessians[1])))
    LOG(WARNING) << "find3DlineOnPlanes: search for hessian failed.";
  // Extract the two plane normals.
  normal1 = {line->hessians[0][0], line->hessians[0][1], line->hessians[0][2]};
  normal2 = {line->hessians[1][0], line->hessians[1][1], line->hessians[1][2]};

  // Compute mean points of the two sets.
  mean1 = computeMean(points1);
  mean2 = computeMean(points2);

  // To consider a line found as valid. It should have enough number of inliers
  // and enough inliers around line's center.
  bool enough_num_inliers, enough_inliers_around_center;

  // If the distance along the plane1/2's normal direction between the two
  // sets of points is small and both support planes for the line are found,
  // the line should be either intersection line or surface line, otherwise
  // the line is discontinuity line.
  if (fabs((mean1 - mean2).dot(normal1)) < params_->max_dist_between_planes &&
      fabs((mean1 - mean2).dot(normal2)) < params_->max_dist_between_planes &&
      planes_found) {
    // Concatenate the two set of points. For surface and intersection line,
    // points1 and points2 are different and thus no repetition of points. The
    // latter is also ensured by the fact that planes_found is True.
    std::vector<cv::Vec3f> points;
    points.reserve(points1.size() + points2.size());
    points.insert(points.end(), points1.begin(), points1.end());
    points.insert(points.end(), points2.begin(), points2.end());

    // Checks if the planes are parallel. If the angle between the two planes'
    // normal vectors is small, they are parallel and the line is surface
    // line.
    constexpr double kAngleDifference = 0.95;
    // Note: since the two normal vectors have unitary norm by definition of
    // Hessian normal form, their dot product is the cosine of the angle between
    // them.
    if (fabs(normal1.dot(normal2)) > kAngleDifference) {
      // Get the initial line guess
      cv::Vec3f start_guess(line_guess[0], line_guess[1], line_guess[2]);
      cv::Vec3f end_guess(line_guess[3], line_guess[4], line_guess[5]);

      cv::Vec3f start, end;
      enough_num_inliers =
          adjustLineUsingInliers(points, start_guess, end_guess, &start, &end);
      // Fix orientation w.r.t. reference line if needed
      adjustLineOrientationGivenReferenceLine(line_guess, &start, &end);

      line->line = {start[0], start[1], start[2], end[0], end[1], end[2]};
      line->type = LineType::PLANE;

      enough_inliers_around_center =
          checkIfValidLineUsingInliers(points, start, end);

      if (enough_num_inliers && enough_inliers_around_center) {
        num_planar_lines++;
        return true;
      } else {
        return false;
      }
    } else {
      // The line lying on both planes must be perpendicular to both normals,
      // so it can be computed with the cross product.
      cv::Vec3f direction = normal1.cross(normal2);
      normalizeVector3D(&direction);
      // Now a point on the intersection line is searched.
      cv::Vec3f x_0;
      getPointOnPlaneIntersectionLine(line->hessians[0], line->hessians[1],
                                      direction, &x_0);
      cv::Vec3f start_guess = x_0;
      cv::Vec3f end_guess = x_0 + direction;

      cv::Vec3f start, end;
      enough_num_inliers =
          adjustLineUsingInliers(points, start_guess, end_guess, &start, &end);
      // Fix orientation w.r.t. reference line if needed
      adjustLineOrientationGivenReferenceLine(line_guess, &start, &end);

      if (!enough_num_inliers)
        return false;
      enough_inliers_around_center =
          checkIfValidLineUsingInliers(points, start, end);
      if (!enough_inliers_around_center)
        return false;

      line->line = {start[0], start[1], start[2], end[0], end[1], end[2]};

      cv::Vec3f start_guess_init = {line_guess[0], line_guess[1],
                                    line_guess[2]};
      cv::Vec3f end_guess_init = {line_guess[3], line_guess[4], line_guess[5]};
      if (visualization_mode_on_) {
        displayLineWithPointsAndPlanes(start, end, start_guess_init,
                                       end_guess_init, points1, points2,
                                       line->hessians[0], line->hessians[1]);
      }
      // Project line re-adjusted through inliers in 2D and add it to the
      // background image.
      cv::Vec4f line_2D;
      project3DLineTo2D(*line, camera_P, &line_2D);


      if (visualization_mode_on_) {
       // Update background image
        background_image_ = getImageOfLine(line_2D, background_image_, 1);
      }

      // Line can now be either an edge or on an intersection line.
      if (!assignEdgeOrIntersectionLineType(cloud, camera_P, points1, points2,
                                            line)) {
        LOG(ERROR) << "Could not assign neither edge- nor intersection- line "
                   << "type to line (" << line->line[0] << ", " << line->line[1]
                   << ", " << line->line[2] << ") -- (" << line->line[3] << ", "
                   << line->line[4] << ", " << line->line[5] << ")";

        return false;
      } else {
        LOG(INFO) << "Successfully determined type "
                  << (line->type==LineType::EDGE ? "EDGE " : "INTERSECT ")
                  << "for line (" << line->line[0] << ", " << line->line[1]
                  << ", " << line->line[2] << ") -- (" << line->line[3] << ", "
                  << line->line[4] << ", " << line->line[5] << ")";
        return true;
      }
    }
  } else {
    // If we reach this point, we have a discontinuity. We then try to fit a
    // line to the set of points that lies closer to the origin (and therefore
    // closer to the camera). This is in most cases a reasonable assumption,
    // since the line most of the time belongs to the object that obscures the
    // background (which causes the discontinuity).
    // The fitting is done in 3 steps:
    //  1.  Project the line_guess onto the plane fitted to the point set.
    //  2.  Find the line parallel to the projected one that goes through the
    //      point in the set of the points that is nearest to the line.
    //  3.  From all points in the set: Project them on to the line and choose
    //      the combination of start/end point that maximizes the line
    //      distance.
    cv::Vec3f start_guess(line_guess[0], line_guess[1], line_guess[2]);
    cv::Vec3f end_guess(line_guess[3], line_guess[4], line_guess[5]);

    const std::vector<cv::Vec3f>* points;
    int idx;
    if (cv::norm(mean1) < cv::norm(mean2)) {
      idx = 0;
      points = &points1;
    } else {
      idx = 1;
      points = &points2;
    }
    line->hessians[abs(idx - 1)] = {0.0f, 0.0f, 0.0f, 0.0f};

    start_guess = projectPointOnPlane(line->hessians[idx], start_guess);
    end_guess = projectPointOnPlane(line->hessians[idx], end_guess);

    cv::Vec3f direction = end_guess - start_guess;
    normalizeVector3D(&direction);

    cv::Vec3f nearest_point;
    getNearestPointToLine(*points, start_guess, end_guess, &nearest_point);

    // Update start point guess of the line as the nearest point
    start_guess = nearest_point;
    end_guess = nearest_point + direction;

    cv::Vec3f start, end;
    enough_num_inliers =
        adjustLineUsingInliers(*points, start_guess, end_guess, &start, &end);
    // Fix orientation w.r.t. reference line if needed
    adjustLineOrientationGivenReferenceLine(line_guess, &start, &end);


    line->line = {start[0], start[1], start[2], end[0], end[1], end[2]};
    line->type = LineType::DISCONT;

    enough_inliers_around_center =
        checkIfValidLineUsingInliers(*points, start, end);
    if (enough_num_inliers && enough_inliers_around_center) {
      num_discontinuity_lines++;
      return true;
    } else {
      return false;
    }
  }
}

bool LineDetector::assignEdgeOrIntersectionLineType(const cv::Mat& cloud,
    const cv::Mat& camera_P, const std::vector<cv::Vec3f>& inliers_right,
    const std::vector<cv::Vec3f>& inliers_left, LineWithPlanes* line) {
  CHECK_NOTNULL(line);
  // First step: if the two planes around the original line form a
  // convex angle, set the line type to be EDGE, otherwise both EDGE and
  // INTERSECTION line type are possible and a further test is required.
  // Let us note that a plane admits two different orientations, i.e., given the
  // direction of its normal vector, the latteassignEdger can point either "towards" the
  // camera or in the opposite way.
  directHessianTowardsOrigin(&(line->hessians[0]));
  directHessianTowardsOrigin(&(line->hessians[1]));
  // Compute mean points (needed if using
  // determineConvexityFromViewpointGivenLineAndMeanPoints).
  cv::Vec3f mean_point_right = computeMean(inliers_right);
  cv::Vec3f mean_point_left = computeMean(inliers_left);
  bool convex_true_concave_false;
  cv::Vec3f origin({0.0f, 0.0f, 0.0f});
  if (determineConvexityFromViewpointGivenLineAndInlierPoints(*line,
    inliers_right, inliers_left, origin, &convex_true_concave_false)) {
      if (convex_true_concave_false) {
        // Convex => Edge
        line->type = LineType::EDGE;
        num_edge_lines++;
        return true;
      }
  } else {
    // This case should never be entered
    return false;
  }
  // Concave => Use the following method: prolonge the line from its endpoints,
  // and prolonge the inlier planes as well. If on any on the two sides one of
  // the two prolonged inlier planes contains points while the other is empty,
  // then the line is an intersection line (one of the two objects in contact
  // with each other reaches its boundary). If none of the two sides is of this
  // type, but for at least one side both the prolonged inlier planes contain
  // points, than the line is of type edge. If this is not the case either,
  // there is not enough information to determine the type of the line.
  LOG(INFO) << "Line with concave planes. Using method of prolonged lines to "
            << "determine edge/intersection line type.";

  // As a first step extend the 3D line from both endpoints and extract the two
  // extensions as line segments.
  cv::Vec3f start(line->line[0], line->line[1], line->line[2]);
  cv::Vec3f end(line->line[3], line->line[4], line->line[5]);
  cv::Vec3f direction = end - start;
  normalizeVector3D(&direction);
  // Line prolonged before start
  cv::Vec3f start_line_before_start =
      start - params_->extension_length_for_edge_or_intersection * direction;
  cv::Vec3f end_line_before_start = start;
  // Line prolonged after end
  cv::Vec3f start_line_after_end = end;
  cv::Vec3f end_line_after_end =
      end + params_->extension_length_for_edge_or_intersection * direction;

  // Check which of the prolonged planes contain (enough) points that are valid
  // fit to them.
  bool right_plane_enough_valid_points_before_start;
  bool left_plane_enough_valid_points_before_start;
  bool right_plane_enough_valid_points_after_end;
  bool left_plane_enough_valid_points_after_end;
  checkIfValidPointsOnPlanesGivenProlongedLine(
      cloud, camera_P, start_line_before_start, end_line_before_start,
      line->hessians, &right_plane_enough_valid_points_before_start,
      &left_plane_enough_valid_points_before_start);
  checkIfValidPointsOnPlanesGivenProlongedLine(
      cloud, camera_P, start_line_after_end, end_line_after_end, line->hessians,
      &right_plane_enough_valid_points_after_end,
      &left_plane_enough_valid_points_after_end);
  // Show image with two prolonged lines.
  cv::Mat img_for_display = background_image_;
  if (visualization_mode_on_) {
    cv::resize(background_image_, img_for_display,
               img_for_display.size() * scale_factor_for_visualization);
    cv::imshow("Line with rectangles and prolonged line/planes",
                img_for_display);
    cv::waitKey();
  }

  // Convert booleans to string.
  std::string point_planes_config;
  point_planes_config.push_back(
      int(left_plane_enough_valid_points_before_start) + '0');
  point_planes_config.push_back(
      int(right_plane_enough_valid_points_before_start) + '0');
  point_planes_config.push_back(
      int(left_plane_enough_valid_points_after_end) + '0');
  point_planes_config.push_back(
      int(right_plane_enough_valid_points_after_end) + '0');
  // Possible cases:
  // - [0][0]/[0][0] -> Edge line.
  // - All other cases -> Intersection line.
  if (point_planes_config == "0000") {
    line->type = LineType::EDGE;
    num_edge_lines++;
    occurrences_config_prolonged_plane[0][0][0][0]++;
  } else {
    if (point_planes_config == "0001" || point_planes_config == "0010" ||
        point_planes_config == "0100" || point_planes_config == "1000") {
      occurrences_config_prolonged_plane[1][0][0][0]++;
    } else if (point_planes_config == "1100" || point_planes_config == "0011") {
      occurrences_config_prolonged_plane[1][1][0][0]++;
    } else if (point_planes_config == "1010" || point_planes_config == "0101") {
      occurrences_config_prolonged_plane[1][0][1][0]++;
    } else if (point_planes_config == "1001" || point_planes_config == "0110") {
      occurrences_config_prolonged_plane[1][0][0][1]++;
    } else if (point_planes_config == "1110" || point_planes_config == "1101" ||
               point_planes_config == "1011" || point_planes_config == "0111") {
      occurrences_config_prolonged_plane[1][1][1][0]++;
    } else if (point_planes_config == "1111") {
      occurrences_config_prolonged_plane[1][1][1][1]++;
    } else {
      LOG(ERROR) << "Found a case for the configuration valid points/prolonged "
                 << "planes that should be impossible.";
      return false;
    }
    line->type = LineType::INTERSECT;
    num_intersection_lines++;
  }
  return true;
}

bool LineDetector::determineConvexityFromViewpointGivenLineAndInlierPoints(
  const LineWithPlanes& line, const std::vector<cv::Vec3f>& inliers_1,
  const std::vector<cv::Vec3f>& inliers_2, const cv::Vec3f& viewpoint,
  bool* convex_true_concave_false) {
  CHECK_NOTNULL(convex_true_concave_false);
  // Orient normal vectors towards the viewpoint (if not done before).
  cv::Vec4f hessians[2];
  hessians[0] = line.hessians[0];
  hessians[1] = line.hessians[1];
  directHessianTowardsPoint(viewpoint, &hessians[0]);
  directHessianTowardsPoint(viewpoint, &hessians[1]);
  // Let us note that each plane is divided by the other inlier plane (with
  // which it intersects in correspondence to the line) into two half-planes.
  // Only one half-plane, however, will actually be visible from the viewpoint
  // and will contain the points, whereas the other should - if the line and the
  // planes are good fit to the data - ideally not contain any point.
  // At first we find each of these half-planes.
  bool halfplane_1_is_behind_plane_2, halfplane_2_is_behind_plane_1;
  int num_inliers_1_behind_plane_2 = 0;
  int num_inliers_1_ahead_of_plane_2 = 0;
  int num_inliers_2_behind_plane_1 = 0;
  int num_inliers_2_ahead_of_plane_1 = 0;
  cv::Vec4f inlier_homo;
  // For further considerations on why the dot product works for this task, see
  // directHessianTowardsPoint.
  for (size_t i = 0; i < inliers_1.size(); ++i) {
    inlier_homo = cv::Vec4f({inliers_1[i][0], inliers_1[i][1], inliers_1[i][2],
                            1.0f});
    if (hessians[1].dot(inlier_homo) > 0)
      ++num_inliers_1_ahead_of_plane_2;
    else
      ++num_inliers_1_behind_plane_2;
  }
  if (num_inliers_1_behind_plane_2 > num_inliers_1_ahead_of_plane_2)
    halfplane_1_is_behind_plane_2 = true;
  else
    halfplane_1_is_behind_plane_2 = false;
  for (size_t i = 0; i < inliers_2.size(); ++i) {
    inlier_homo = cv::Vec4f({inliers_2[i][0], inliers_2[i][1], inliers_2[i][2],
                            1.0f});
    if (hessians[0].dot(inlier_homo) > 0)
      ++num_inliers_2_ahead_of_plane_1;
    else
      ++num_inliers_2_behind_plane_1;
  }
  if (num_inliers_2_behind_plane_1 > num_inliers_2_ahead_of_plane_1)
    halfplane_2_is_behind_plane_1 = true;
  else
    halfplane_2_is_behind_plane_1 = false;
  // Infer convexity/concavity.
  if (halfplane_1_is_behind_plane_2 && halfplane_2_is_behind_plane_1) {
      // Convex angle
      *convex_true_concave_false = true;
      return true;
  } else if (!halfplane_1_is_behind_plane_2 && !halfplane_2_is_behind_plane_1) {
      // Concave angle
      *convex_true_concave_false = false;
      return true;
  } else {
    // This case should never be entered
    LOG(ERROR) << "Error in determining the concavity/convexity of the angle "
               << "between the two planes around the line with the following "
               << "3D coordinates: (" << line.line[0] << ", " << line.line[1]
               << ", " << line.line[2] << ") -- (" << line.line[3] << ", "
               << line.line[4] << ", " << line.line[5] << "). Hessians are: ["
               << hessians[0][0] << ", " << hessians[0][1] << ", "
               << hessians[0][2] << ", " << hessians[0][3] << "] and ["
               << hessians[1][0] << ", " << hessians[1][1] << ", "
               << hessians[1][2] << ", " << hessians[1][3] << "].";
    num_lines_discarded_for_convexity_concavity++;
    return false;
  }
}

bool LineDetector::determineConvexityFromViewpointGivenLineAndMeanPoints(
  const LineWithPlanes& line, const cv::Vec3f& mean_point_1,
  const cv::Vec3f& mean_point_2, const cv::Vec3f& viewpoint,
  bool* convex_true_concave_false) {
  CHECK_NOTNULL(convex_true_concave_false);
  // Take any point on the line that connects a generic pair of points, each of
  // which belonging to one of the two planes. If this point is "in front of"
  // (i.e., in the orientation of the normal vector) the planes than we have a
  // convex angle, otherwise a concave angle.
  // As points belonging to the planes, take the projections of the two means
  // on the planes.

  // Compute projection of mean points on the planes.
  cv::Vec3f mean_point_1_proj =
      projectPointOnPlane(line.hessians[0], mean_point_1);
  cv::Vec3f mean_point_2_proj =
      projectPointOnPlane(line.hessians[1], mean_point_2);
  cv::Vec3f mean_of_mean_points = (mean_point_1_proj + mean_point_2_proj) / 2;
  cv::Vec4f mean_hom = {mean_of_mean_points[0], mean_of_mean_points[1],
                        mean_of_mean_points[2], 1.0};
  if (line.hessians[0].dot(mean_hom) > 0 &&
      line.hessians[1].dot(mean_hom) > 0) {
      // Concave angle
      *convex_true_concave_false = false;
      return true;
  } else if (line.hessians[0].dot(mean_hom) < 0 &&
      line.hessians[1].dot(mean_hom) < 0) {
      // Convex angle
      *convex_true_concave_false = true;
      return true;
  } else {
    // This case should never be entered
    LOG(ERROR) << "Error in determining the concavity/convexity of the angle "
               << "between the two planes around the line with the following "
               << "3D coordinates: (" << line.line[0] << ", " << line.line[1]
               << ", " << line.line[2] << ") -- (" << line.line[3] << ", "
               << line.line[4] << ", " << line.line[5] << "). Hessians are: ["
               << line.hessians[0][0] << ", " << line.hessians[0][1] << ", "
               << line.hessians[0][2] << ", " << line.hessians[0][3] << "] "
               << "and [" << line.hessians[1][0] << ", "
               << line.hessians[1][1] << ", " << line.hessians[1][2] << ", "
               << line.hessians[1][3] << "]. Mean point 1 is ("
               << mean_point_1_proj[0] << ", " << mean_point_1_proj[1]
               << ", " << mean_point_1_proj[2] << "). Mean point 2 is ("
               << mean_point_2_proj[0] << ", " << mean_point_2_proj[1]
               << ", " << mean_point_2_proj[2] << "). Mean of mean points is ("
               << mean_of_mean_points[0] << ", " << mean_of_mean_points[1]
               << ", " << mean_of_mean_points[2] << ").";
    return false;
  }
}

void LineDetector::checkIfValidPointsOnPlanesGivenProlongedLine(
    const cv::Mat& cloud, const cv::Mat& camera_P, const cv::Vec3f& start,
    const cv::Vec3f& end, const std::vector<cv::Vec4f>& hessians,
    bool* right_plane_enough_valid_points,
    bool* left_plane_enough_valid_points) {
  CHECK_NOTNULL(left_plane_enough_valid_points);
  CHECK_NOTNULL(right_plane_enough_valid_points);
  double max_deviation = params_->max_error_inlier_ransac;
  // Get 2D coordinates of the endpoints of the line segment.
  cv::Vec4f prolonged_line;
  project3DLineTo2D(start, end, camera_P, &prolonged_line);

  // For both the left and the right side of the line: Find a rectangle
  // defining a patch, find all points within the patch. We will later try to
  // fit a plane to these points, in such a way that the plane is parallel to
  // the inlier plane of the original line that is on the same side of the line
  // as it is.
  std::vector<cv::Point2f> rect_left, rect_right;
  std::vector<cv::Point2i> points_in_rect;
  std::vector<cv::Vec3f> points_left_plane, points_right_plane;
  getRectanglesFromLine(prolonged_line, &rect_left, &rect_right);


  if (visualization_mode_on_) {
    // Display image of prolonged line.
    background_image_ = getImageOfLineWithRectangles(prolonged_line, rect_left,
                                                     rect_right,
                                                     background_image_, 1);
  }

  // Find points for the left side.
  findPointsInRectangle(rect_left, &points_in_rect);
  points_left_plane.clear();
  for (size_t j = 0; j < points_in_rect.size(); ++j) {
    if (points_in_rect[j].x < 0 || points_in_rect[j].x >= cloud.cols ||
        points_in_rect[j].y < 0 || points_in_rect[j].y >= cloud.rows) {
      continue;
    }
    if (std::isnan(cloud.at<cv::Vec3f>(points_in_rect[j])[0])) continue;
    points_left_plane.push_back(cloud.at<cv::Vec3f>(points_in_rect[j]));
  }
  LOG(INFO) << "Left rectangle contains " << points_left_plane.size()
            << " points.";

  // Find points for the right side.
  findPointsInRectangle(rect_right, &points_in_rect);
  points_right_plane.clear();
  for (size_t j = 0; j < points_in_rect.size(); ++j) {
    if (points_in_rect[j].x < 0 || points_in_rect[j].x >= cloud.cols ||
        points_in_rect[j].y < 0 || points_in_rect[j].y >= cloud.rows) {
      continue;
    }
    if (std::isnan(cloud.at<cv::Vec3f>(points_in_rect[j])[0])) continue;
    points_right_plane.push_back(cloud.at<cv::Vec3f>(points_in_rect[j]));
  }
  LOG(INFO) << "Right rectangle contains " << points_right_plane.size()
            << " points.";

  // If the number of points around the plane is too small, either the line
  // segment is too short (but this should not be the case if
  // extension_length_for_edge_or_intersection is properly set) or the line
  // segment is near the edge of the image. Therefore, not enough points can be
  // counted to determine whether there are enough valid points on the two
  // sides.
  bool enough_left_points_to_count = true;
  bool enough_right_points_to_count = true;
  if (points_left_plane.size() < params_->min_points_in_prolonged_rect) {
    *left_plane_enough_valid_points = false;
    enough_left_points_to_count = false;
  }

  if (points_right_plane.size() < params_->min_points_in_prolonged_rect) {
    *right_plane_enough_valid_points = false;
    enough_right_points_to_count = false;
  }

  // Now check if the points around the two planes could be part of the two
  // planes around the original line, i.e., if they could belong to the same
  // object as the points on the corresponding plane around the original line.
  // To do so, count how many points in the two planes around the prolonged line
  // are consistent with the hessians of the original line.
  int valid_points_left_plane = 0, valid_points_right_plane = 0;
  cv::Vec4f hessian_left_plane, hessian_right_plane;
  std::vector<cv::Vec3f>::iterator it;
  // According to the way hessians were assigned to the lines in
  // project2Dto3DwithPlanes, the map between hessians and side is
  // hessians[0] -> right, hessians[1] -> left.
  valid_points_left_plane = 0;
  valid_points_right_plane = 0;
  hessian_left_plane = hessians[1];
  hessian_right_plane = hessians[0];

  if (enough_left_points_to_count) {
    for (it = points_left_plane.begin(); it != points_left_plane.end(); ++it) {
      if (errorPointToPlane(hessian_left_plane, *it) < max_deviation)
        ++valid_points_left_plane;
    }
    // Determine if enough valid points are found for the left plane.
    if (valid_points_left_plane < params_-> max_points_for_empty_rectangle)
      *left_plane_enough_valid_points = false;
    else
      *left_plane_enough_valid_points = true;
  }
  if (enough_right_points_to_count) {
    for (it = points_right_plane.begin(); it != points_right_plane.end();
         ++it) {
      if (errorPointToPlane(hessian_right_plane, *it) < max_deviation)
        ++valid_points_right_plane;
    }
    // Determine if enough valid points are found for the right plane.
    if (valid_points_right_plane < params_-> max_points_for_empty_rectangle)
      *right_plane_enough_valid_points = false;
    else
      *right_plane_enough_valid_points = true;
  }
  LOG(INFO) << "Found " << valid_points_left_plane << " valid points on the "
            << "left plane and " << valid_points_right_plane << " valid points "
            << "on the right plane.";
}

bool LineDetector::planeRANSAC(const std::vector<cv::Vec3f>& points,
                               cv::Vec4f* hessian_normal_form) {
  const size_t N = points.size();
  double inlier_fraction_min = params_->min_inlier_ransac;
  std::vector<cv::Vec3f> inliers;
  planeRANSAC(points, &inliers);
  // If we found not enough inlier, return false. This is important because
  // there might not be a solution (and we dont want to propose one if there
  // is none).
  if (inliers.size() <= inlier_fraction_min * N) return false;
  // Now we compute the final model parameters with all the inliers.
  return hessianNormalFormOfPlane(inliers, hessian_normal_form);
}
void LineDetector::planeRANSAC(const std::vector<cv::Vec3f>& points,
                               std::vector<cv::Vec3f>* inliers) {
  CHECK_NOTNULL(inliers);
  // Set parameters and do a sanity check.
  const int N = points.size();
  inliers->clear();
  const int max_it = params_->num_iter_ransac;
  constexpr int number_of_model_params = 3;
  double max_deviation = params_->max_error_inlier_ransac;
  double inlier_fraction_max = params_->inlier_max_ransac;
  CHECK(N > number_of_model_params) << "Not enough points to use RANSAC.";
  // Declare variables that are used for the RANSAC.
  std::vector<cv::Vec3f> random_points, inlier_candidates;
  cv::Vec3f normal;
  cv::Vec4f hessian_normal_form;
  // Set a random seed.
  unsigned seed = 1;
  std::default_random_engine generator(seed);
  // Start RANSAC.
  for (int iter = 0; iter < max_it; ++iter) {
    // Get number_of_model_params unique elements from points.
    getNUniqueRandomElements(points, number_of_model_params, &generator,
                             &random_points);
    // It might happen that the randomly chosen points lie on a line. In this
    // case, hessianNormalFormOfPlane would return false.
    if (!hessianNormalFormOfPlane(random_points, &hessian_normal_form))
      continue;
    normal = cv::Vec3f(hessian_normal_form[0], hessian_normal_form[1],
                       hessian_normal_form[2]);
    // Check which of the points are inlier with the current plane model.
    inlier_candidates.clear();
    for (int j = 0; j < N; ++j) {
      if (errorPointToPlane(hessian_normal_form, points[j]) < max_deviation) {
        inlier_candidates.push_back(points[j]);
      }
    }
    // If we found more inliers than in any previous run, we store them
    // as global inliers.
    if (inlier_candidates.size() > inliers->size()) {
      *inliers = inlier_candidates;
    }

    // Usual not part of RANSAC: stop early if we have enough inliers.
    // This feature is here because it might be that we have a very
    // high inlier percentage. In this case RANSAC finds the right
    // model within the first few iterations and all later iterations
    // are just wasted run time.
    if (inliers->size() > inlier_fraction_max * N) break;
  }
}

void LineDetector::project2Dto3DwithPlanes(
    const cv::Mat& cloud, const std::vector<cv::Vec4f>& lines2D,
    std::vector<cv::Vec6f>* lines3D) {
  CHECK_NOTNULL(lines3D);
  std::vector<LineWithPlanes> lines_with_planes;
  lines3D->clear();
  project2Dto3DwithPlanes(cloud, lines2D, &lines_with_planes);
  for (size_t i = 0; i < lines_with_planes.size(); ++i)
    lines3D->push_back(lines_with_planes[i].line);
}
void LineDetector::project2Dto3DwithPlanes(
    const cv::Mat& cloud, const std::vector<cv::Vec4f>& lines2D,
    std::vector<LineWithPlanes>* lines3D) {
  cv::Mat image;
  project2Dto3DwithPlanes(cloud, image, lines2D, false, lines3D);
}

// DEPRECATED
void LineDetector::project2Dto3DwithPlanes(
    const cv::Mat& cloud, const cv::Mat& image,
    const std::vector<cv::Vec4f>& lines2D_in, const bool set_colors,
    std::vector<LineWithPlanes>* lines3D) {
  CHECK_NOTNULL(lines3D);
  CHECK_EQ(cloud.type(), CV_32FC3);
  // Declare all variables before the main loop.
  std::vector<cv::Point2f> rect_left, rect_right;
  std::vector<cv::Point2i> points_in_rect;
  std::vector<cv::Vec3f> plane_point_cand, inliers_left, inliers_right;
  std::vector<cv::Vec6f> lines3D_cand;
  std::vector<double> rating;
  cv::Point2i start, end;
  LineWithPlanes line3D_true;
  // Parameter: Fraction of inlier that must be found for the plane model to
  // be valid.
  double min_inliers = params_->min_inlier_ransac;
  double max_rating = params_->max_rating_valid_line;
  bool right_found, left_found;
  constexpr size_t min_points_for_ransac = 3;
  // This is a first guess of the 3D lines. They are used in some cases, where
  // the lines cannot be found by intersecting planes.
  std::vector<cv::Vec4f> lines2D =
      checkLinesInBounds(lines2D_in, cloud.cols, cloud.rows);

  find3DlinesRated(cloud, lines2D, &lines3D_cand, &rating);
  // Loop over all 2D lines.
  for (size_t i = 0; i < lines2D.size(); ++i) {
    // If the rating is so high, no valid 3d line was found by the
    // find3DlinesRated function.
    if (rating[i] > max_rating) continue;
    // For both the left and the right side of the line: Find a rectangle
    // defining a patch, find all points within the patch and try to fit a
    // plane to these points.
    getRectanglesFromLine(lines2D[i], &rect_left, &rect_right);
    // Find lines for the left side.
    findPointsInRectangle(rect_left, &points_in_rect);
    if (set_colors) {
      assignColorToLines(image, points_in_rect, &line3D_true);
    }
    plane_point_cand.clear();
    for (size_t j = 0; j < points_in_rect.size(); ++j) {
      if (points_in_rect[j].x < 0 || points_in_rect[j].x >= cloud.cols ||
          points_in_rect[j].y < 0 || points_in_rect[j].y >= cloud.rows) {
        continue;
      }
      if (std::isnan(cloud.at<cv::Vec3f>(points_in_rect[j])[0])) continue;
      plane_point_cand.push_back(cloud.at<cv::Vec3f>(points_in_rect[j]));
    }
    left_found = false;
    if (plane_point_cand.size() > min_points_for_ransac) {
      planeRANSAC(plane_point_cand, &inliers_left);
      if (inliers_left.size() >= min_inliers * plane_point_cand.size()) {
        left_found = true;
      }
    }
    // Find lines for the right side.
    findPointsInRectangle(rect_right, &points_in_rect);
    if (set_colors) {
      assignColorToLines(image, points_in_rect, &line3D_true);
    }
    plane_point_cand.clear();
    for (size_t j = 0; j < points_in_rect.size(); ++j) {
      if (points_in_rect[j].x < 0 || points_in_rect[j].x >= cloud.cols ||
          points_in_rect[j].y < 0 || points_in_rect[j].y >= cloud.rows) {
        continue;
      }
      if (std::isnan(cloud.at<cv::Vec3f>(points_in_rect[j])[0])) continue;
      plane_point_cand.push_back(cloud.at<cv::Vec3f>(points_in_rect[j]));
    }
    right_found = false;
    if (plane_point_cand.size() > min_points_for_ransac) {
      planeRANSAC(plane_point_cand, &inliers_right);
      if (inliers_right.size() >= min_inliers * plane_point_cand.size()) {
        right_found = true;
      }
    }
    // If any of planes were not found, the line is found at a discontinuity.
    // This is a workaround, more efficiently this would be implemented in the
    // function find3DlineOnPlanes.
    bool is_discont = true;
    if ((!right_found) && (!left_found)) {
      continue;
    } else if (!right_found) {
      inliers_right = inliers_left;
    } else if (!left_found) {
      inliers_left = inliers_right;
    } else {
      is_discont = false;
    }
    // If both planes were found, the inliers are handled to the
    // find3DlineOnPlanes function, which takes care of different special
    // cases.
    if (find3DlineOnPlanes(inliers_right, inliers_left, lines3D_cand[i],
                           &line3D_true)) {
      // Only push back the reliably found lines.
      if (is_discont) {
        line3D_true.type = LineType::DISCONT;
        if (right_found) {
          line3D_true.hessians[1] = {0.0f, 0.0f, 0.0f, 0.0f};
        } else {
          line3D_true.hessians[0] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
      }
      lines3D->push_back(line3D_true);
    }
  }
}

void LineDetector::project2Dto3DwithPlanes(
    const cv::Mat& cloud, const cv::Mat& image, const cv::Mat& camera_P,
    const std::vector<cv::Vec4f>& lines2D_in, const bool set_colors,
    std::vector<cv::Vec4f>* lines2D_out, std::vector<LineWithPlanes>* lines3D) {
  CHECK_NOTNULL(lines2D_out);
  CHECK_NOTNULL(lines3D);
  CHECK_EQ(cloud.type(), CV_32FC3);
  lines3D->clear();
  lines2D_out->clear();
  resetStatistics();
  // Declare all variables before the main loop.
  std::vector<cv::Point2f> rect_left, rect_right;
  std::vector<cv::Vec3f> inliers_left, inliers_right;
  std::vector<cv::Vec6f> lines3D_cand;
  std::vector<double> rating;
  cv::Point2i start, end;
  LineWithPlanes line3D_true;

  double max_rating = params_->max_rating_valid_line;
  bool right_found, left_found;

  // This is a first guess of the 3D lines. They are used in some cases, where
  // the lines cannot be found by intersecting planes.
  std::vector<cv::Vec4f> lines2D =
      checkLinesInBounds(lines2D_in, cloud.cols, cloud.rows);

  //LOG(INFO) << "2D lines before shrinkage:";
  //std::vector<cv::Vec4f>::iterator line;
  //for (line = lines2D.begin(); line != lines2D.end(); ++line)
  //  LOG(INFO) << "(" << (*line)[0] << ", " << (*line)[1] << ") -- (" << (*line)[2]
  //            << ", " << (*line)[3] << ")";

  // Shrink 2D lines to lessen the influence of start and end points
  std::vector<cv::Vec4f> lines2D_shrunk;
  constexpr double kShrinkCoff = 0.8;
  constexpr double kMinLengthAfterShrinking = 1.0;
  shrink2Dlines(lines2D, kShrinkCoff, kMinLengthAfterShrinking,
                &lines2D_shrunk);

  //LOG(INFO) << "2D lines after shrinkage:";
  //for (line = lines2D_shrunk.begin(); line != lines2D_shrunk.end(); ++line)
  //  LOG(INFO) << "(" << (*line)[0] << ", " << (*line)[1] << ") -- ("
  //            << (*line)[2]  << ", " << (*line)[3] << ")";

  find3DlinesRated(cloud, lines2D_shrunk, &lines3D_cand, &rating);

  // Loop over all 2D lines.
  for (size_t i = 0; i < lines2D.size(); ++i) {
    // If cannot find valid 3D start and end points for the 2D line.
    if (rating[i] > max_rating) continue;

    findInliersGiven2DLine(lines2D[i], cloud, image, set_colors, &line3D_true,
                          &inliers_right, &inliers_left, &rect_right,
                          &rect_left, &right_found, &left_found);

    bool planes_found = false;
    if ((!right_found) && (!left_found)) {
      continue;
    } else if (!right_found) {
      inliers_right = inliers_left;
    } else if (!left_found) {
      inliers_left = inliers_right;
    } else {
      // Both left and right planes are found.
      planes_found = true;
    }

    cv::Mat image_of_line_with_rectangles;

    if (visualization_mode_on_) {
      //background_image_ = getImageOfLineWithRectangles(lines2D[i],
      //                                    rect_left, rect_right, 1);
      background_image_ = image;
    }

    // Find 3D line on planes.
    if (find3DlineOnPlanes(inliers_right, inliers_left, lines3D_cand[i], cloud,
                           camera_P, planes_found, &line3D_true)) {
      // Only push back the reliably found lines.
      lines3D->push_back(line3D_true);
      lines2D_out->push_back(lines2D[i]);
      if (visualization_mode_on_) {
        // Display 2D image with rectangles
        image_of_line_with_rectangles = getImageOfLineWithRectangles(lines2D[i],
                                            rect_left, rect_right,
                                            background_image_);
        cv::imshow("Line with rectangles", image_of_line_with_rectangles);
        cv::waitKey();
      }
    }
  }
}

void LineDetector::project3DLineTo2D(const cv::Vec3f& start_3D,
                                     const cv::Vec3f& end_3D,
                                     const cv::Mat& camera_P,
                                     cv::Vec4f* line_2D) {
  CHECK_NOTNULL(line_2D);
  cv::Vec2f start_2D, end_2D;
  cv::Vec4f start_3D_homo = {start_3D[0], start_3D[1], start_3D[2], 1.0};
  cv::Vec4f end_3D_homo = {end_3D[0], end_3D[1], end_3D[2], 1.0};

  cv::Mat start_2D_homo = camera_P * cv::Mat(start_3D_homo);
  cv::Mat end_2D_homo = camera_P * cv::Mat(end_3D_homo);
  start_2D = {start_2D_homo.at<float>(0, 0) /
              start_2D_homo.at<float>(2, 0),
              start_2D_homo.at<float>(1, 0) /
              start_2D_homo.at<float>(2, 0)};
  end_2D = {end_2D_homo.at<float>(0, 0) / end_2D_homo.at<float>(2, 0),
            end_2D_homo.at<float>(1, 0) / end_2D_homo.at<float>(2, 0)};
  *line_2D = cv::Vec4f({start_2D[0], start_2D[1], end_2D[0], end_2D[1]});

}

void LineDetector::project3DLineTo2D(const LineWithPlanes& line_3D,
                                     const cv::Mat& camera_P,
                                     cv::Vec4f* line_2D) {
  CHECK_NOTNULL(line_2D);
  cv::Vec3f start_3D({line_3D.line[0], line_3D.line[1], line_3D.line[2]});
  cv::Vec3f end_3D({line_3D.line[3], line_3D.line[4], line_3D.line[5]});

  project3DLineTo2D(start_3D, end_3D, camera_P, line_2D);
}



void LineDetector::findInliersGiven2DLine(const cv::Vec4f& line_2D,
                                          const cv::Mat& cloud,
                                          std::vector<cv::Vec3f>* inliers_right,
                                          std::vector<cv::Vec3f>* inliers_left)
                                          {
  const cv::Mat image;
  LineWithPlanes line_3D;
  std::vector<cv::Point2f> rect_right, rect_left;
  bool right_found, left_found;

  findInliersGiven2DLine(line_2D, cloud, image, false, &line_3D, inliers_right,
                         inliers_left, &rect_right, &rect_left, &right_found,
                         &left_found);
}

void LineDetector::findInliersGiven2DLine(const cv::Vec4f& line_2D,
                                        const cv::Mat& cloud,
                                        const cv::Mat& image,
                                        bool set_colors,
                                        LineWithPlanes* line_3D,
                                        std::vector<cv::Vec3f>* inliers_right,
                                        std::vector<cv::Vec3f>* inliers_left,
                                        std::vector<cv::Point2f>* rect_right,
                                        std::vector<cv::Point2f>* rect_left,
                                        bool* right_found, bool* left_found) {
  CHECK_NOTNULL(line_3D);
  CHECK_NOTNULL(inliers_right);
  CHECK_NOTNULL(inliers_left);
  CHECK_NOTNULL(rect_right);
  CHECK_NOTNULL(rect_left);
  CHECK_NOTNULL(right_found);
  CHECK_NOTNULL(left_found);

  std::vector<cv::Point2i> points_in_rect;
  std::vector<cv::Vec3f> plane_point_cand;
  constexpr size_t min_points_for_ransac = 3;
  // Parameter: Fraction of inlier that must be found for the plane model to
  // be valid.
  double min_inliers = params_->min_inlier_ransac;

  // Clear inliers.
  inliers_right->clear();
  inliers_left->clear();

  // For both the left and the right side of the line: Find a rectangle
  // defining a patch, find all points within the patch and try to fit a plane
  // to these points.
  getRectanglesFromLine(line_2D, rect_left, rect_right);
  // Find lines for the left side.
  findPointsInRectangle(*rect_left, &points_in_rect);
  if (set_colors) {
    assignColorToLines(image, points_in_rect, line_3D);
  }
  plane_point_cand.clear();
  for (size_t j = 0; j < points_in_rect.size(); ++j) {
    if (points_in_rect[j].x < 0 || points_in_rect[j].x >= cloud.cols ||
        points_in_rect[j].y < 0 || points_in_rect[j].y >= cloud.rows) {
      continue;
    }
    if (std::isnan(cloud.at<cv::Vec3f>(points_in_rect[j])[0])) continue;
    plane_point_cand.push_back(cloud.at<cv::Vec3f>(points_in_rect[j]));
  }
  // If the size of plane_point_cand is too small, either the line is too short
  // or the line is near the edge of the image, reject it.
  if (plane_point_cand.size() < params_->min_points_in_rect) {
    *right_found = false;
    *left_found = false;
    return;
  }
  // See if left plane is found by RANSAC.
  *left_found = false;
  if (plane_point_cand.size() > min_points_for_ransac) {
    planeRANSAC(plane_point_cand, inliers_left);
    if (inliers_left->size() >= min_inliers * plane_point_cand.size()) {
      *left_found = true;
    }
  }
  // Find lines for the right side.
  findPointsInRectangle(*rect_right, &points_in_rect);
  if (set_colors) {
    assignColorToLines(image, points_in_rect, line_3D);
  }
  plane_point_cand.clear();
  for (size_t j = 0; j < points_in_rect.size(); ++j) {
    if (points_in_rect[j].x < 0 || points_in_rect[j].x >= cloud.cols ||
        points_in_rect[j].y < 0 || points_in_rect[j].y >= cloud.rows) {
      continue;
    }
    if (std::isnan(cloud.at<cv::Vec3f>(points_in_rect[j])[0])) continue;
    plane_point_cand.push_back(cloud.at<cv::Vec3f>(points_in_rect[j]));
  }

  if (plane_point_cand.size() < params_->min_points_in_rect) {
    *right_found = false;
    *left_found = false;
    return;
  }
  // See if right plane is found by RANSAC.
  *right_found = false;
  if (plane_point_cand.size() > min_points_for_ransac) {
    planeRANSAC(plane_point_cand, inliers_right);
    if (inliers_right->size() >= min_inliers * plane_point_cand.size()) {
      *right_found = true;
    }
  }
}

void LineDetector::find3DlinesByShortest(const cv::Mat& cloud,
                                         const std::vector<cv::Vec4f>& lines2D,
                                         std::vector<cv::Vec6f>* lines3D) {
  std::vector<int> correspondences;
  find3DlinesByShortest(cloud, lines2D, lines3D, &correspondences);
}
void LineDetector::find3DlinesByShortest(const cv::Mat& cloud,
                                         const std::vector<cv::Vec4f>& lines2D,
                                         std::vector<cv::Vec6f>* lines3D,
                                         std::vector<int>* correspondences) {
  CHECK_NOTNULL(lines3D);
  CHECK_NOTNULL(correspondences);
  CHECK_EQ(cloud.type(), CV_32FC3);
  int cols = cloud.cols;
  int rows = cloud.rows;
  // The actual patch size wil be bigger. The number of pixels within a patch
  // is euqal to (2*patch_size + 1)^2. And because for every pixel in the
  // start patch the distance to every pixel in the end patch is computed, the
  // complexity is proportional to (2*patch_size + 1)^4.
  int patch_size = 1;
  int x_opt_start, y_opt_start, x_min_start, x_max_start, y_min_start,
      y_max_start, x_opt_end, y_opt_end, x_min_end, x_max_end, y_min_end,
      y_max_end;
  float dist, dist_opt;
  cv::Vec3f start, end;
  correspondences->clear();
  lines3D->clear();
  for (size_t i = 0u; i < lines2D.size(); ++i) {
    dist_opt = 1e20;
    x_opt_start = lines2D[i][0];
    y_opt_start = lines2D[i][1];
    x_opt_end = lines2D[i][2];
    y_opt_end = lines2D[i][3];
    // This checks are used to make sure, that we do not try to access a
    // point not within the image.
    x_min_start = checkInBoundaryInt(x_opt_start - patch_size, 0, rows - 1);
    x_max_start = checkInBoundaryInt(x_opt_start + patch_size, 0, rows - 1);
    y_min_start = checkInBoundaryInt(y_opt_start - patch_size, 0, cols - 1);
    y_max_start = checkInBoundaryInt(y_opt_start + patch_size, 0, cols - 1);
    x_min_end = checkInBoundaryInt(x_opt_end - patch_size, 0, rows - 1);
    x_max_end = checkInBoundaryInt(x_opt_end + patch_size, 0, rows - 1);
    y_min_end = checkInBoundaryInt(y_opt_end - patch_size, 0, cols - 1);
    y_max_end = checkInBoundaryInt(y_opt_end + patch_size, 0, cols - 1);
    // For every pixel in start patch.
    for (int x_start = x_min_start; x_start <= x_max_start; ++x_start) {
      for (int y_start = y_min_start; y_start <= y_max_start; ++y_start) {
        // For every pixel in end patch.
        for (int x_end = x_min_end; x_end <= x_max_end; ++x_end) {
          for (int y_end = y_min_end; y_end <= y_max_end; ++y_end) {
            // Check that the corresponding 3D point is not NaN.
            if (std::isnan(cloud.at<cv::Vec3f>(y_start, x_start)[0]) ||
                std::isnan(cloud.at<cv::Vec3f>(y_end, x_end)[0])) {
              continue;
            }
            // Compute distance and compare it to the optimal distance found
            // so far.
            start = cloud.at<cv::Vec3f>(y_start, x_start);
            end = cloud.at<cv::Vec3f>(y_end, x_end);
            dist = pow(start[0] - end[0], 2) + pow(start[1] - end[1], 2) +
                   pow(start[2] - end[2], 2);
            if (dist < dist_opt) {
              dist_opt = dist;
              x_opt_end = x_end;
              x_opt_start = x_start;
              y_opt_end = y_end;
              y_opt_start = y_start;
            }
          }
        }
      }
    }
    // Assuming that distances are in meters, we can safely assume that if our
    // optimal distance is still 1e20, no non-NaN points were found.
    if (dist_opt == 1e20) continue;
    // Otherwise, a line was found.
    start = cloud.at<cv::Vec3f>(y_opt_start, x_opt_start);
    end = cloud.at<cv::Vec3f>(y_opt_end, x_opt_end);
    lines3D->push_back(
        cv::Vec6f(start[0], start[1], start[2], end[0], end[1], end[2]));
    correspondences->push_back(i);
  }
}

void LineDetector::find3DlinesRated(const cv::Mat& cloud,
                                    const std::vector<cv::Vec4f>& lines2D,
                                    std::vector<cv::Vec6f>* lines3D,
                                    std::vector<double>* rating) {
  CHECK_NOTNULL(lines3D);
  CHECK_NOTNULL(rating);
  CHECK_EQ(cloud.type(), CV_32FC3);
  int cols = cloud.cols;
  int rows = cloud.rows;
  cv::Vec4f upper_line2D, lower_line2D;
  cv::Point2f line;
  cv::Vec6f lower_line3D, line3D, upper_line3D;
  double rate_low, rate_mid, rate_up;
  lines3D->clear();
  lines3D->reserve(lines2D.size());
  rating->clear();
  rating->reserve(lines2D.size());
  for (size_t i = 0; i < lines2D.size(); ++i) {
    line.x = lines2D[i][2] - lines2D[i][0];
    line.y = lines2D[i][3] - lines2D[i][1];
    double line_normalizer = sqrt(line.x * line.x + line.y * line.y);
    upper_line2D[0] = checkInBoundary(
        floor(lines2D[i][0]) + floor(line.y / line_normalizer + 0.5), 0.0,
        cols - 1);
    upper_line2D[1] = checkInBoundary(
        floor(lines2D[i][1]) + floor(-line.x / line_normalizer + 0.5), 0.0,
        rows - 1);
    upper_line2D[2] = checkInBoundary(
        floor(lines2D[i][2]) + floor(line.y / line_normalizer + 0.5), 0.0,
        cols - 1);
    upper_line2D[3] = checkInBoundary(
        floor(lines2D[i][3]) + floor(-line.x / line_normalizer + 0.5), 0.0,
        rows - 1);
    lower_line2D[0] = checkInBoundary(
        floor(lines2D[i][0]) + floor(-line.y / line_normalizer + 0.5), 0.0,
        cols - 1);
    lower_line2D[1] = checkInBoundary(
        floor(lines2D[i][1]) + floor(line.x / line_normalizer + 0.5), 0.0,
        rows - 1);
    lower_line2D[2] = checkInBoundary(
        floor(lines2D[i][2]) + floor(-line.y / line_normalizer + 0.5), 0.0,
        cols - 1);
    lower_line2D[3] = checkInBoundary(
        floor(lines2D[i][3]) + floor(line.x / line_normalizer + 0.5), 0.0,
        rows - 1);
    rate_low = findAndRate3DLine(cloud, lower_line2D, &lower_line3D);
    rate_mid = findAndRate3DLine(cloud, lines2D[i], &line3D);
    rate_up = findAndRate3DLine(cloud, upper_line2D, &upper_line3D);

    //LOG(INFO) << "Ratings of lines are:\nUp = " << rate_up << "\nMid = "
    //          << rate_mid << "\nLow = " << rate_low << "\n";

    if (rate_up < rate_mid && rate_up < rate_low) {
      lines3D->push_back(upper_line3D);
      rating->push_back(rate_up);
    } else if (rate_low < rate_mid) {
      lines3D->push_back(lower_line3D);
      rating->push_back(rate_low);
    } else {
      lines3D->push_back(line3D);
      rating->push_back(rate_mid);
    }
  }
}
void LineDetector::find3DlinesRated(const cv::Mat& cloud,
                                    const std::vector<cv::Vec4f>& lines2D,
                                    std::vector<cv::Vec6f>* lines3D) {
  std::vector<double> rating;
  std::vector<cv::Vec6f> lines3D_cand;
  find3DlinesRated(cloud, lines2D, &lines3D_cand, &rating);
  for (size_t i = 0; i < lines3D_cand.size(); ++i) {
    if (rating[i] > params_->max_rating_valid_line) {
      continue;
    }
    lines3D->push_back(lines3D_cand[i]);
  }
}

void LineDetector::runCheckOn3DLines(
    const cv::Mat& cloud, const std::vector<LineWithPlanes>& lines3D_in,
    std::vector<LineWithPlanes>* lines3D_out) {
  CHECK_NOTNULL(lines3D_out);
  lines3D_out->clear();
  LineWithPlanes line_cand;
  for (size_t i = 0; i < lines3D_in.size(); ++i) {
    line_cand = lines3D_in[i];
    if (checkIfValidLineBruteForce(cloud, &(line_cand.line))) {
      lines3D_out->push_back(line_cand);
    }
  }
}

void LineDetector::runCheckOn3DLines(
    const cv::Mat& cloud, const cv::Mat& camera_P,
    const std::vector<cv::Vec4f>& lines2D_in,
    const std::vector<LineWithPlanes>& lines3D_in,
    std::vector<cv::Vec4f>* lines2D_out,
    std::vector<LineWithPlanes>* lines3D_out) {
  CHECK_NOTNULL(lines2D_out);
  CHECK_NOTNULL(lines3D_out);
  lines3D_out->clear();
  lines2D_out->clear();
  LineWithPlanes line_cand;
  cv::Vec4f line_cand_2D;
  for (size_t i = 0; i < lines3D_in.size(); ++i) {
    line_cand = lines3D_in[i];
    line_cand_2D = lines2D_in[i];
    if (checkIfValidLineWith2DInfo(cloud, camera_P, line_cand_2D,
                                   &(line_cand.line))) {
      lines3D_out->push_back(line_cand);
      lines2D_out->push_back(line_cand_2D);
    }
  }
}

void LineDetector::runCheckOn2DLines(const cv::Mat& cloud,
                                     const std::vector<cv::Vec4f>& lines2D_in,
                                     std::vector<cv::Vec4f>* lines2D_out) {
  CHECK_NOTNULL(lines2D_out);
  size_t N = lines2D_in.size();
  lines2D_out->clear();
  for (size_t i = 0; i < N; ++i) {
    if (checkIfValidLineDiscont(cloud, lines2D_in[i])) {
      lines2D_out->push_back(lines2D_in[i]);
    }
  }
}

bool LineDetector::checkIfValidLineWith2DInfo(const cv::Mat& cloud,
                                              const cv::Mat& camera_P,
                                              cv::Vec4f& line_2D,
                                              cv::Vec6f* line) {
  CHECK_NOTNULL(line);
  CHECK_EQ(cloud.type(), CV_32FC3);
  CHECK_EQ(camera_P.type(), CV_32FC1);
  // First check: if one of the points near exactly on the origin, get rid of
  // it.
  if ((fabs((*line)[0]) < 1e-3 && fabs((*line)[1]) < 1e-3 &&
       fabs((*line)[2]) < 1e-3) ||
      (fabs((*line)[3]) < 1e-3 && fabs((*line)[4]) < 1e-3 &&
       fabs((*line)[5]) < 1e-3)) {
    return false;
  }

  // If the 2D line is too close to the edges, reject it.
  constexpr double kMinDistanceToEdge = 4.0;
  if (line_2D[0] + line_2D[2] < kMinDistanceToEdge ||
      line_2D[1] + line_2D[3] < kMinDistanceToEdge ||
      line_2D[0] + line_2D[2] > cloud.cols * 2 - kMinDistanceToEdge ||
      line_2D[1] + line_2D[3] > cloud.rows * 2 - kMinDistanceToEdge) {
    return false;
  }

  cv::Vec3f start_3D, end_3D;

  start_3D = {(*line)[0], (*line)[1], (*line)[2]};
  end_3D = {(*line)[3], (*line)[4], (*line)[5]};

  double length = cv::norm(start_3D - end_3D);

  // If the length of the line is too short, reject it
  if (length < params_->min_length_line_3D) {
    return false;
  }

  cv::Vec4f line_3D_reprojected;

  project3DLineTo2D(start_3D, end_3D, camera_P, &line_3D_reprojected);

  cv::Vec2f start_2D({line_3D_reprojected[0], line_3D_reprojected[1]});
  cv::Vec2f end_2D({line_3D_reprojected[2], line_3D_reprojected[3]});

  cv::Vec2f line_dir_true{line_2D[2] - line_2D[0], line_2D[3] - line_2D[1]};
  cv::Vec2f line_dir{end_2D[0] - start_2D[0], end_2D[1] - start_2D[1]};

  // Check difference of length.
  constexpr double kLengthDifference = 1.5;
  if (cv::norm(line_dir) / cv::norm(line_dir_true) > kLengthDifference) {
    return false;
  }

  // Check difference of angle.
  constexpr double kAngleDifference = 0.95;
  if (fabs(line_dir.dot(line_dir_true) /
           (cv::norm(line_dir) * cv::norm(line_dir_true))) < kAngleDifference) {
    return false;
  }

  // Store the line and return.
  *line = {start_3D[0], start_3D[1], start_3D[2],
           end_3D[0],   end_3D[1],   end_3D[2]};
  return true;
}

bool LineDetector::checkIfValidLineBruteForce(const cv::Mat& cloud,
                                              cv::Vec6f* line) {
  CHECK_NOTNULL(line);
  CHECK_EQ(cloud.type(), CV_32FC3);
  // First check: if one of the points near exactly on the origin, get rid of
  // it.
  if ((fabs((*line)[0]) < 1e-3 && fabs((*line)[1]) < 1e-3 &&
       fabs((*line)[2]) < 1e-3) ||
      (fabs((*line)[3]) < 1e-3 && fabs((*line)[4]) < 1e-3 &&
       fabs((*line)[5]) < 1e-3)) {
    return false;
  }
  // Minimum number of inliers for the line to be valid.
  const int num_of_points_required = params_->min_points_in_line;
  // Maximum deviation for a point to count as an inlier.
  double max_deviation = params_->max_deviation_inlier_line_check;
  // This point density measures the where the points lie on the line. It is
  // used to truncate the line on the ends, if one end lies in empty space.
  int point_density[num_of_points_required];
  for (int i = 0; i < num_of_points_required; ++i) {
    point_density[i] = 0;
  }

  double dist;
  cv::Vec3f start, end, point;
  start = {(*line)[0], (*line)[1], (*line)[2]};
  end = {(*line)[3], (*line)[4], (*line)[5]};
  double length = cv::norm(start - end);
  int count_inliers = 0;
  // For every point in the cloud: This is why it is called brute force
  // approach.
  for (int i = 0; i < cloud.rows; ++i) {
    for (int j = 0; j < cloud.cols; ++j) {
      point = cloud.at<cv::Vec3f>(i, j);
      // Check if the distance to the line is below the threshold. This
      // computes the distance to the infinite line.
      if (distPointToLine(start, end, point) < max_deviation) {
        // This is the distance from the start point projected on to the line.
        // If its negative or larger the line length, the point may lie on the
        // line, but not between the start and the end point.
        dist = (end - start).dot(point - start) / length;
        if (dist < 0 || length <= dist) {
          continue;
        }
        // Now the histogramm like point_density is raised at the entry where
        // the point lies.
        point_density[(int)(dist / length * (double)num_of_points_required)] +=
            1;
        ++count_inliers;
      }
    }
  }
  // Only take lines with enough inliers.
  if (count_inliers <= num_of_points_required) {
    return false;
  }
  // Check from the front and the back of the line if the density is zero.
  int front = 0;
  int back = num_of_points_required - 1;
  while (0 == point_density[front]) ++front;
  while (0 == point_density[back]) --back;
  cv::Vec3f direction;
  direction = end - start;
  // This part will truncate the line, if the point_density was zero at either
  // the back or the front. Otherwise it has no influence.
  end = start + direction * back / (double)(num_of_points_required - 1);
  start = start + direction * front / (double)(num_of_points_required - 1);
  // Store the line and return.
  *line = {start[0], start[1], start[2], end[0], end[1], end[2]};
  return true;
}

bool LineDetector::checkIfValidLineDiscont(const cv::Mat& cloud,
                                           const cv::Vec4f& line) {
  CHECK_EQ(cloud.type(), CV_32FC3);
  cv::Point2i start, end, dir;
  start = {static_cast<int>(floor(line[0])), static_cast<int>(floor(line[1]))};
  end = {static_cast<int>(floor(line[2])), static_cast<int>(floor(line[3]))};
  int patch_size = 1;
  // The patch is restricted to be within the rectangle that is spawned by
  // start and end. This has two positive effects: We never try to acces a
  // pixel outside of the image and if a line starts at an discontinuity edge
  // it prevents the algorithm from early stopping.
  int x_max, y_max, x_min, y_min, x_from, x_to, y_from, y_to;
  if (line[0] < line[2]) {
    x_min = line[0];
    x_max = line[2];
  } else {
    x_min = line[2];
    x_max = line[0];
  }
  if (line[1] < line[3]) {
    y_min = line[1];
    y_max = line[3];
  } else {
    y_min = line[3];
    y_max = line[1];
  }
  cv::Vec3f current_mean(0, 0, 0);
  cv::Vec3f last_mean(0, 0, 0);
  double max_mean_diff = 0.1;
  int count;
  bool first_time = true;
  while (start != end) {
    current_mean = {0, 0, 0};
    // This procedure always makes a 1 pixel step towards the end point. It is
    // guaranteed to land on the end point eventually, so the loop will
    // terminate.
    dir = end - start;
    start.x += floor(dir.x / sqrt(dir.x * dir.x + dir.y * dir.y) + 0.5);
    start.y += floor(dir.y / sqrt(dir.x * dir.x + dir.y * dir.y) + 0.5);
    // We need to be within the boundaries defined previously.
    x_from = checkInBoundary(start.x - patch_size, x_min, x_max);
    x_to = checkInBoundary(start.x + patch_size, x_min, x_max);
    y_from = checkInBoundary(start.y - patch_size, y_min, y_max);
    y_to = checkInBoundary(start.y + patch_size, y_min, y_max);
    // Count is used to count the number of pixels that were added to the mean
    // so that we can effectively build the mean from the sum.
    count = 0;
    for (int i = x_from; i <= x_to; ++i) {
      for (int j = y_from; j <= y_to; ++j) {
        current_mean += cloud.at<cv::Vec3f>(j, i);
        ++count;
      }
    }
    current_mean = current_mean / count;
    if (first_time) {
      last_mean = current_mean;
      first_time = false;
      continue;
    }
    if (cv::norm(current_mean - last_mean) > max_mean_diff) return false;
    last_mean = current_mean;
  }
  return true;
}

void LineDetector::shrink2Dlines(const std::vector<cv::Vec4f>& lines2D_in,
                                 const double shrink_coff,
                                 const double min_length,
                                 std::vector<cv::Vec4f>* lines2D_out) {
  CHECK(shrink_coff <= 1 && shrink_coff > 0);
  CHECK_NOTNULL(lines2D_out);
  lines2D_out->clear();

  cv::Vec2f start, end, line_dir;
  for (size_t i = 0; i < lines2D_in.size(); ++i) {
    start[0] = lines2D_in[i][0];
    start[1] = lines2D_in[i][1];
    end[0] = lines2D_in[i][2];
    end[1] = lines2D_in[i][3];

    line_dir = (end - start) / cv::norm(end - start);

    start = start + line_dir * (1 - shrink_coff) / 2;
    end = end - line_dir * (1 - shrink_coff) / 2;

    if (cv::norm(end - start) < 1) {
      lines2D_out->push_back(lines2D_in[i]);
    } else {
      lines2D_out->push_back(cv::Vec4f(start[0], start[1], end[0], end[1]));
    }
  }
}

void LineDetector::getNearestPointToLine(const std::vector<cv::Vec3f>& points,
                                         const cv::Vec3f& start,
                                         const cv::Vec3f& end,
                                         cv::Vec3f* nearest_point) {
  CHECK_NOTNULL(nearest_point);

  cv::Vec3f direction = end - start;
  normalizeVector3D(&direction);

  double min_dist = 1e9;
  for (size_t i = 0; i < points.size(); ++i) {
    double dist = distPointToLine(start, end, points[i]);

    if (dist < min_dist) {
      min_dist = dist;
      *nearest_point = points[i];
    }
  }
}

double LineDetector::getRatioOfPointsAroundCenter(
    const std::vector<double>& points_distribution) {
  size_t points_number = points_distribution.size();
  size_t count = 0;
  for (size_t i = 0; i < points_number; ++i) {
    if (points_distribution[i] < 0.75f && points_distribution[i] > 0.25f) {
      count += 1;
    }
  }
  return count / static_cast<double>(points_number);
}

bool LineDetector::adjustLineUsingInliers(const std::vector<cv::Vec3f>& points,
                                          const cv::Vec3f& start_in,
                                          const cv::Vec3f& end_in,
                                          cv::Vec3f* start_out,
                                          cv::Vec3f* end_out) {
  CHECK_NOTNULL(start_out);
  CHECK_NOTNULL(end_out);

  cv::Vec3f direction = end_in - start_in;
  normalizeVector3D(&direction);

  double dist;
  double dist_min = 1e9;
  double dist_max = -1e9;
  size_t count_inliers = 0u;
  for (size_t i = 0u; i < points.size(); ++i) {
    if (distPointToLine(start_in, end_in, points[i]) >
        params_->max_deviation_inlier_line_check) {
      continue;
    }
    ++count_inliers;
    dist = direction.dot(points[i] - start_in);
    if (dist < dist_min) {
      dist_min = dist;
    }
    if (dist > dist_max) {
      dist_max = dist;
    }
  }

  // Update start and end points of the line.
  *start_out = start_in + direction * dist_min;
  *end_out = start_in + direction * dist_max;

  // Line is not valid since it is not supported by enough 3D points.
  if (count_inliers < params_->min_points_in_line) {
    return false;
  } else {
    return true;
  }
}

void LineDetector::adjustLineOrientationGivenReferenceLine(
    const cv::Vec6f& reference_line, cv::Vec3f* start, cv::Vec3f* end) {
  cv::Vec3f temp_endpoint;
  cv::Vec3f ref_start({reference_line[0], reference_line[1],
                      reference_line[2]});
  cv::Vec3f ref_end({reference_line[3], reference_line[4], reference_line[5]});

  if (cv::norm(*start - ref_end) < cv::norm(*start - ref_start) &&
      cv::norm(*end - ref_start) < cv::norm(*end - ref_end)) {
    LOG(INFO) << "Switching endpoints.";
    temp_endpoint = *start;
    *start = *end;
    *end = temp_endpoint;
  }
}

bool LineDetector::checkIfValidLineUsingInliers(
    const std::vector<cv::Vec3f>& points, const cv::Vec3f& start,
    const cv::Vec3f& end) {
  std::vector<double> positions_on_line;
  double length = cv::norm(start - end);
  cv::Vec3f direction = end - start;
  normalizeVector3D(&direction);
  for (size_t i = 0u; i < points.size(); ++i) {
    if (distPointToLine(start, end, points[i]) >
        params_->max_deviation_inlier_line_check) {
      continue;
    }
    double position_on_line = direction.dot(points[i] - start) / length;
    positions_on_line.push_back(position_on_line);
  }
  const double ratio_mid = getRatioOfPointsAroundCenter(positions_on_line);
  // Most points are near the start and end points, reject this line
  constexpr double kRatioThreshold = 0.25;
  if (ratio_mid < kRatioThreshold) {
    return false;
  } else {
    return true;
  }
}

void LineDetector::displayStatistics() {
  int total_num_lines = num_discontinuity_lines + num_planar_lines +
                        num_intersection_lines + num_edge_lines;
  LOG(INFO) << "Found " << total_num_lines << " total lines, of which:\n* "
            << num_discontinuity_lines << " discontinuity lines\n* "
            << num_planar_lines << " planar lines\n* "
            << num_edge_lines << " edge lines\n* "
            << num_intersection_lines << " intersection lines.";
  LOG(INFO) << num_lines_discarded_for_convexity_concavity << " lines were "
            << "discarded because it was not possible to determine convexity/"
            << "concavity";
  LOG(INFO) << "Among the edge/intersection lines that were assigned to their "
            << "type by looking at the prolonged lines/planes the following "
            << "occurrences for each configuration were found (format: "
            << "before_start [L][R]/[L][R] after end):"
            << "\n* [0][0]/[0][0]: "
            << occurrences_config_prolonged_plane[0][0][0][0]
            << "\n* [0][0]/[0][1], [0][0]/[1][0], [0][1]/[0][0], "
            << "[1][0]/[0][0]: "
            << occurrences_config_prolonged_plane[1][0][0][0]
            << "\n* [1][1]/[0][0], [0][0]/[1][1]: "
            << occurrences_config_prolonged_plane[1][1][0][0]
            << "\n* [1][0]/[1][0], [0][1]/[0][1]: "
            << occurrences_config_prolonged_plane[1][0][1][0]
            << "\n* [1][0]/[0][1], [0][1]/[1][0]: "
            << occurrences_config_prolonged_plane[1][0][0][1]
            << "\n* [1][1]/[1][0], [1][1]/[0][1], [1][0]/[1][1], "
            << "[0][1]/[1][1]: "
            << occurrences_config_prolonged_plane[1][1][1][0]
            << "\n* [1][1]/[1][1]: "
            << occurrences_config_prolonged_plane[1][1][1][1];
}

void LineDetector::resetStatistics() {
  num_discontinuity_lines = 0;
  num_planar_lines = 0;
  num_intersection_lines = 0;
  num_edge_lines = 0;
  num_lines_discarded_for_convexity_concavity = 0;
  occurrences_config_prolonged_plane[0][0][0][0] = 0;
  occurrences_config_prolonged_plane[1][0][0][0] = 0;
  occurrences_config_prolonged_plane[1][1][0][0] = 0;
  occurrences_config_prolonged_plane[1][0][1][0] = 0;
  occurrences_config_prolonged_plane[1][0][0][1] = 0;
  occurrences_config_prolonged_plane[1][1][1][0] = 0;
  occurrences_config_prolonged_plane[1][1][1][1] = 0;
}

}  // namespace line_detection
