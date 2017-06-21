#include "line_detection/line_detection.h"
#include "line_detection/line_detection_inl.h"

namespace line_detection {

bool areLinesEqual2D(const cv::Vec4f line1, const cv::Vec4f line2) {
  // First compute the difference in angle. For easier computation not the
  // actual difference in angle, but cos(theta)^2 is computed.
  float vx1 = line1[0] - line1[2];
  float vx2 = line2[0] - line2[2];
  float vy1 = line1[1] - line1[3];
  float vy2 = line2[1] - line2[3];

  float angle_difference = pow((vx1 * vx2 + vy1 * vy2), 2) /
                           ((vx1 * vx1 + vy1 * vy1) * (vx2 * vx2 + vy2 * vy2));
  // Then compute the distance of the two lines. All distances between both end
  // and start points are computed and the lowest is kept.
  float dist[4], min_dist;
  dist[0] = pow(line1[0] - line2[0], 2) + pow(line1[1] - line2[1], 2);
  dist[1] = pow(line1[0] - line2[2], 2) + pow(line1[1] - line2[3], 2);
  dist[2] = pow(line1[2] - line2[0], 2) + pow(line1[3] - line2[1], 2);
  dist[3] = pow(line1[2] - line2[2], 2) + pow(line1[3] - line2[3], 2);
  min_dist = dist[0];
  for (int i = 1; i < 4; ++i) {
    if (dist[i] < min_dist) min_dist = dist[i];
  }
  // Set the comparison parameters. These need to be adjusted correctly.
  // angle_diffrence > 0.997261 refers is equal to effective angle < 3 deg.
  if (angle_difference > 0.95 && min_dist < 5)
    return true;
  else
    return false;
}

void storeLines3DinMarkerMsg(const std::vector<cv::Vec<float, 6> >& lines3D,
                             visualization_msgs::Marker& disp_lines) {
  disp_lines.points.clear();
  disp_lines.action = visualization_msgs::Marker::ADD;
  disp_lines.type = visualization_msgs::Marker::LINE_LIST;
  disp_lines.scale.x = 0.03;
  disp_lines.scale.y = 0.03;
  disp_lines.color.a = 1;
  disp_lines.color.r = 1;
  disp_lines.color.g = 0;
  disp_lines.color.b = 0;
  disp_lines.id = 1;
  // Fill in the line information. LINE_LIST is an array where the first point
  // is the start and the second is the end of the line. The third is then again
  // the start of the next line, and so on.
  geometry_msgs::Point p;
  for (size_t i = 0; i < lines3D.size(); ++i) {
    p.x = lines3D[i][0];
    p.y = lines3D[i][1];
    p.z = lines3D[i][2];
    disp_lines.points.push_back(p);
    p.x = lines3D[i][3];
    p.y = lines3D[i][4];
    p.z = lines3D[i][5];
    disp_lines.points.push_back(p);
  }
}

void pclFromSceneNetToMat(const pcl::PointCloud<pcl::PointXYZRGB>& pcl_cloud,
                          int width, int height, cv::Mat& mat_cloud) {
  CHECK_EQ(pcl_cloud.points.size(), width * height);
  mat_cloud.create(height, width, CV_32FC3);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      mat_cloud.at<cv::Vec3f>(i, j) = cv::Vec3f(
          pcl_cloud.points[j + i * width].x, pcl_cloud.points[j + i * width].y,
          pcl_cloud.points[j + i * width].z);
    }
  }
}

bool hessianNormalFormOfPlane(const std::vector<cv::Vec3f>& points,
                              cv::Vec4f& hessian_normal_form) {
  const int N = points.size();
  CHECK(N >= 3);
  if (N == 3) {  // In this case an exact solution can be computed.
    cv::Vec3f vec1 = points[1] - points[0];
    cv::Vec3f vec2 = points[2] - points[0];
    // This checks first if the points were too close.
    double norms = normOfVector3D(vec1) * normOfVector3D(vec2);
    if (norms < 1e-6) return false;
    // Then if they lie on a line. The angle between the vectors must at least
    // be 2 degrees.
    double cos_theta = fabs(scalarProduct(vec1, vec2)) / norms;
    if (cos_theta > 0.994) return false;
    // The normal already defines the orientation of the plane (it is
    // perpendicular to both vectors, since they must lie within the plane).
    cv::Vec3f normal = crossProduct(vec1, vec2);
    // Now bring the plane into the hessian normal form.
    hessian_normal_form = cv::Vec4f(normal[0], normal[1], normal[2],
                                    computeDfromPlaneNormal(normal, points[0]));
    hessian_normal_form = hessian_normal_form / normOfVector3D(normal);
    return true;
  } else {  // If there are more than 3 points, the solution is approximate.
    cv::Vec3f mean(0, 0, 0);
    for (int i = 0; i < N; ++i) {
      mean += points[i] / N;
    }
    cv::Mat A(3, N, CV_64FC1);
    for (int i = 0; i < N; ++i) {
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
    hessian_normal_form = cv::Vec4f(normal[0], normal[1], normal[2],
                                    computeDfromPlaneNormal(normal, mean));
    return true;
  }
}

void findYCoordOfPixelsOnVector(const cv::Point2f& start,
                                const cv::Point2f& end, bool left_side,
                                std::vector<int>& x_coord) {
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
    if (left_side)
      x_coord.push_back(floor(start.x));
    else
      x_coord.push_back(ceil(end.x));
    return;
  }
  for (int i = 0; i < height; ++i) {
    x_coord.push_back(int(x_start + i * width / (height - 1)));
  }
}

void findPointsInRectangle(std::vector<cv::Point2f> corners,
                           std::vector<cv::Point2i>& points) {
  CHECK_EQ(corners.size(), 4)
      << "The rectangle must be defined by exactly 4 corner points.";
  // Find the relative positions of the points.
  // int upper, lower, left, right, store_i;
  std::vector<int> idx{0, 1, 2, 3};
  // This part finds out if two of the points have equal x values. This may not
  // be very likely for some data, but if it happens it can produce
  // unpredictibale outcome. If this is the case, the rectangle is rotated by
  // 0.1 degree. This should not make a difference, becaues the pixels have
  // integer values anyway (so a corner point of 100.1 and 100.2 gives the same
  // result).
  bool some_points_have_equal_height = false;
  // Check all x values against all others.
  for (int i = 1; i < 4; ++i) {
    if (corners[0].y == corners[i].y) some_points_have_equal_height = true;
  }
  for (int i = 2; i < 4; ++i) {
    if (corners[1].y == corners[i].y) some_points_have_equal_height = true;
  }
  if (corners[2].y == corners[3].y) some_points_have_equal_height = true;
  // Do the rotation.
  if (some_points_have_equal_height) {
    for (int i = 0; i < 4; ++i)
      corners[i] =
          cv::Point2f(0.99998 * corners[i].x - 0.0017453 * corners[i].y,
                      0.0017453 * corners[i].x + 0.99998 * corners[i].y);
  }
  // The points are set to lowest, highest, most right and most left in this
  // order. It does work because the preprocessing done guarantees that no two
  // points have the same y coordinate.
  cv::Point2f upper, lower, left, right;
  upper = corners[0];
  int j;
  for (int i = 1; i < 4; ++i) {
    if (upper.y > corners[i].y) upper = corners[i];
  }
  lower.y = -1e6;
  for (int i = 0; i < 4; ++i) {
    if (lower.y < corners[i].y && corners[i] != upper) lower = corners[i];
  }
  left.x = 1e6;
  for (int i = 0; i < 4; ++i) {
    if (left.x > corners[i].x && corners[i] != upper && corners[i] != lower)
      left = corners[i];
  }
  for (int i = 0; i < 4; ++i) {
    if (corners[i] != left && corners[i] != upper && corners[i] != lower)
      right = corners[i];
  }
  // With the ordering given, the border pixels can be found as pixels, that lie
  // on the border vectors.
  std::vector<int> left_border;
  std::vector<int> right_border;
  findYCoordOfPixelsOnVector(upper, left, true, left_border);
  findYCoordOfPixelsOnVector(upper, right, false, right_border);
  // Pop_back is used because otherwise the corners[left/right] pixels would be
  // counted twice.
  left_border.pop_back();
  right_border.pop_back();
  findYCoordOfPixelsOnVector(left, lower, true, left_border);
  findYCoordOfPixelsOnVector(right, lower, false, right_border);
  if (left_border.size() > right_border.size())
    left_border.pop_back();
  else if (left_border.size() < right_border.size())
    right_border.pop_back();
  CHECK_EQ(left_border.size(), right_border.size());
  // Iterate over all pixels in the rectangle.
  points.clear();
  int x, y;
  for (int i = 0; i < left_border.size(); ++i) {
    y = floor(upper.y) + i;
    // y = floor(corners[upper].y) + i;
    x = left_border[i];
    do {
      points.push_back(cv::Point2i(x, y));
      ++x;
    } while (x <= right_border[i]);
  }
}

bool getRectanglesFromLine(const cv::Vec4f& line,
                           std::vector<cv::Point2f>& rect_left,
                           std::vector<cv::Point2f>& rect_right) {
  // The offset defines how far away from the line the nearest corner points
  // are.
  double offset = 1;
  // Defines the length of the side perpendicular to the line.
  double relative_rect_size = 0.5;
  // Exactly as above, but defines a numerical maximum.
  double max_rect_size = 10;
  double eff_rect_size = max_rect_size;
  cv::Point2f start(line[0], line[1]);
  cv::Point2f end(line[2], line[3]);
  cv::Point2f line_dir = end - start;
  cv::Point2f go_left(-line_dir.y, line_dir.x);
  cv::Point2f go_right(line_dir.y, -line_dir.x);
  double norm = sqrt(pow(line_dir.x, 2) + pow(line_dir.y, 2));
  if (eff_rect_size > norm * relative_rect_size)
    eff_rect_size = norm * relative_rect_size;
  rect_left.resize(4);
  rect_left[0] = start + offset / norm * go_left;
  rect_left[1] = start + eff_rect_size / norm * go_left;
  rect_left[2] = end + offset / norm * go_left;
  rect_left[3] = end + eff_rect_size / norm * go_left;
  rect_right.resize(4);
  rect_right[0] = start + offset / norm * go_right;
  rect_right[1] = start + eff_rect_size / norm * go_right;
  rect_right[2] = end + offset / norm * go_right;
  rect_right[3] = end + eff_rect_size / norm * go_right;
}

bool getPointOnPlaneIntersectionLine(const cv::Vec4f& hessian1,
                                     const cv::Vec4f& hessian2,
                                     const cv::Vec3f& direction,
                                     cv::Vec3f& x_0) {
  // The problem can be solved with a under determined linear system. See
  // http://mathworld.wolfram.com/Plane-PlaneIntersection.html
  cv::Mat m(2, 2, CV_32FC1);
  cv::Mat b(2, 1, CV_32FC1);
  cv::Mat x_0_mat(2, 1, CV_32FC1);
  int non_zero, count = 0;
  // Because the system is underdetemined, we can set an element of our solution
  // to zero. We just have to check that the corresponding element in the
  // direction vector is non-zero. For numerical stability we check here that
  // the element is greater than 0.1. Given that the vector is normalized, at
  // least one element always meets this condition.
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
      x_0[i] = 0;
      continue;
    }
    x_0[i] = x_0_mat.at<float>(count, 0);
    ++count;
  }
  return success;
}

bool find3DlineOnPlanes(const std::vector<cv::Vec3f>& points1,
                        const std::vector<cv::Vec3f>& points2,
                        const cv::Vec<float, 6>& line_guess,
                        cv::Vec<float, 6>& line) {
  size_t N1 = points1.size();
  size_t N2 = points2.size();
  if (N1 < 3 || N2 < 3) return false;
  cv::Vec3f mean1, mean2, normal1, normal2;
  cv::Vec4f hessian1, hessian2;
  // Fit a plane model to the two sets of points individually.
  if (!hessianNormalFormOfPlane(points1, hessian1))
    ROS_WARN("find3DlineOnPlanes: search for hessian failed.");
  if (!hessianNormalFormOfPlane(points2, hessian2))
    ROS_WARN("find3DlineOnPlanes: search for hessian failed.");
  // Extract the two plane normals.
  normal1 = {hessian1[0], hessian1[1], hessian1[2]};
  normal2 = {hessian2[0], hessian2[1], hessian2[2]};
  // This parameter defines at which point 2 lines are concerned to be near.
  // This distance is computed from the means of the two set of points. If the
  // distance is higher than this value, it is assumed that the line is not the
  // intersection of the two planes, but just lies on the one that is in the
  // foreground.
  double max_mean_dist = 0.5;  // TODO adjust
  mean1 = computeMean(points1);
  mean2 = computeMean(points2);
  if (normOfVector3D(mean1 - mean2) < max_mean_dist) {
    // Checks if the planes ar parallel.
    if (fabs(scalarProduct(normal1, normal2)) > 0.995) {
      return true;
    } else {
      // The line lying on both planes must be perpendicular to both normals, so
      // it can be computed with the cross product.
      cv::Vec3f direction = crossProduct(normal1, normal2);
      normalizeVec3D(direction);
      // Now a point on the intersection line is searched.
      cv::Vec3f x_0;
      getPointOnPlaneIntersectionLine(hessian1, hessian2, direction, x_0);
      // This part searches for start and end point, because so far we only have
      // a line from and to inifinity. The procedure used here projects all
      // points in both sets onto the line and then chooses the pair of point
      // that maximize the distance of the line.
      double dist;
      double dist_min = 1e9;
      double dist_max = -1e9;
      for (size_t i = 0; i < N1; ++i) {
        dist = scalarProduct(direction, points1[i] - x_0);
        if (dist < dist_min) dist_min = dist;
        if (dist > dist_max) dist_max = dist;
      }
      for (size_t i = 0; i < N2; ++i) {
        dist = scalarProduct(direction, points2[i] - x_0);
        if (dist < dist_min) dist_min = dist;
        if (dist > dist_max) dist_max = dist;
      }
      cv::Vec3f start, end;
      start = x_0 + direction * dist_min;
      end = x_0 + direction * dist_max;
      line = {start[0], start[1], start[2], end[0], end[1], end[2]};
      return true;
    }
  } else {
    return false;
  }
}

LineDetector::LineDetector() {
  lsd_detector_ = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
  edl_detector_ =
      cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
  fast_detector_ = cv::ximgproc::createFastLineDetector();
}

void LineDetector::detectLines(const cv::Mat& image, int detector,
                               std::vector<cv::Vec4f>& lines) {
  if (detector == 0)
    detectLines(image, Detector::LSD, lines);
  else if (detector == 1)
    detectLines(image, Detector::EDL, lines);
  else if (detector == 2)
    detectLines(image, Detector::FAST, lines);
  else if (detector == 3)
    detectLines(image, Detector::HOUGH, lines);
  else {
    ROS_ERROR(
        "LineDetetor::detectLines: Detector choice not valid, LSD was "
        "chosen as default.");
    detectLines(image, Detector::LSD, lines);
  }
}
void LineDetector::detectLines(const cv::Mat& image, Detector detector,
                               std::vector<cv::Vec4f>& lines) {
  lines.clear();
  // Check which detector is chosen by user. If an invalid number is given the
  // default (LSD) is chosen without a warning.
  if (detector == Detector::LSD) {
    lsd_detector_->detect(image, lines);
  } else if (detector == Detector::EDL) {  // EDL_DETECTOR
    // The edl detector uses a different kind of vector to store the lines in.
    // The conversion is done later.
    std::vector<cv::line_descriptor::KeyLine> edl_lines;
    edl_detector_->detect(image, edl_lines);

    // Write lines to standard format
    for (size_t i = 0u; i < edl_lines.size(); ++i) {
      lines.push_back(cv::Vec4i(
          edl_lines[i].getStartPoint().x, edl_lines[i].getStartPoint().y,
          edl_lines[i].getEndPoint().x, edl_lines[i].getEndPoint().y));
    }

  } else if (detector == Detector::FAST) {  // FAST_DETECTOR
    fast_detector_->detect(image, lines);
  } else if (detector == Detector::HOUGH) {  // HOUGH_DETECTOR
    cv::Mat output;
    // Parameters of the Canny should not be changed (or better: the result is
    // very likely to get worse);
    cv::Canny(image, output, 50, 200, 3);
    // Here parameter changes might improve the result.
    cv::HoughLinesP(output, lines, 1, CV_PI / 180, 50, 30, 10);
  }
}
void LineDetector::detectLines(const cv::Mat& image,
                               std::vector<cv::Vec4f>& lines) {
  detectLines(image, Detector::LSD, lines);
}

void LineDetector::computePointCloud(
    const cv::Mat image, const cv::Mat& depth, const cv::Mat& K,
    pcl::PointCloud<pcl::PointXYZRGB>& point_cloud) {
  // The points are intermediatelly stored in here (because the function
  // rgbd::depthTo3d is used). This is not the fastest way, so if speed is
  // needed this functoin should be rewriten.
  cv::Mat points3d;
  cv::rgbd::depthTo3d(depth, K, points3d);

  // Here the point could gets filled with the values found by depthTo3d.
  int rows = depth.rows;
  int cols = depth.cols;
  point_cloud.width = cols;
  point_cloud.height = rows;
  point_cloud.is_dense = false;
  point_cloud.reserve(cols * rows);

  pcl::PointXYZRGB pcl_point;
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      pcl_point.x = points3d.at<cv::Vec3f>(j, i)[0];
      pcl_point.y = points3d.at<cv::Vec3f>(j, i)[1];
      pcl_point.z = points3d.at<cv::Vec3f>(j, i)[2];
      pcl_point.r = image.at<cv::Vec3b>(j, i)[0];
      pcl_point.g = image.at<cv::Vec3b>(j, i)[1];
      pcl_point.b = image.at<cv::Vec3b>(j, i)[2];
      point_cloud.push_back(pcl_point);
    }
  }
}

void LineDetector::projectLines2Dto3D(
    const std::vector<cv::Vec4f>& lines2D, const cv::Mat& point_cloud,
    std::vector<cv::Vec<float, 6> >& lines3D) {
  // First check if the point_cloud mat has the right format.
  CHECK_EQ(point_cloud.type(), CV_32FC3)
      << "The input matrix point_cloud must be of type CV_32FC3.";
  int rows = point_cloud.rows;
  int cols = point_cloud.cols;
  lines3D.clear();
  cv::Point2f start, end, line;
  cv::Vec<float, 6> line3D, lower_line3D, upper_line3D;
  cv::Vec4f upper_line2D, lower_line2D;
  double rate_mid, rate_up, rate_low;
  for (size_t i = 0; i < lines2D.size(); ++i) {
    // **************** VARIANT 1 ****************** //
    start.x = floor(lines2D[i][0]);
    start.y = floor(lines2D[i][1]);
    end.x = floor(lines2D[i][2]);
    end.y = floor(lines2D[i][3]);
    if (!std::isnan(point_cloud.at<cv::Vec3f>(start)[0]) &&
        !std::isnan(point_cloud.at<cv::Vec3f>(end)[0])) {
      lines3D.push_back(cv::Vec<float, 6>(point_cloud.at<cv::Vec3f>(start)[0],
                                          point_cloud.at<cv::Vec3f>(start)[1],
                                          point_cloud.at<cv::Vec3f>(start)[2],
                                          point_cloud.at<cv::Vec3f>(end)[0],
                                          point_cloud.at<cv::Vec3f>(end)[1],
                                          point_cloud.at<cv::Vec3f>(end)[2]));
    }
  }
}

void LineDetector::fuseLines2D(const std::vector<cv::Vec4f>& lines_in,
                               std::vector<cv::Vec4f>& lines_out) {
  lines_out.clear();
  // This list is used to remember which lines have already been assigned to a
  // cluster. Every time a line is assigned, the corresponding index is
  // deleted in this list.
  std::list<int> line_index;
  for (size_t i = 0; i < lines_in.size(); ++i) line_index.push_back(i);
  // This vector is used to store the line clusters until they are merged into
  // one line.
  std::vector<cv::Vec4f> line_cluster;
  // Iterate over all lines.
  for (size_t i = 0; i < lines_in.size(); ++i) {
    line_cluster.clear();
    // If this condition does not hold, the line lines_in[i] has already been
    // merged into a new one. If not, the algorithm tries to find lines that
    // are near this line.
    if (*(line_index.begin()) != i) {
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
      lines_out.push_back(line_cluster[0]);
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
      lines_out.push_back(cv::Vec4f(x_min, y_min, x_max, y_max));
    else
      lines_out.push_back(cv::Vec4f(x_min, y_max, x_max, y_min));
  }
}

void LineDetector::paintLines(cv::Mat& image,
                              const std::vector<cv::Vec4f>& lines,
                              cv::Vec3b color) {
  cv::Point2i p1, p2;

  for (int i = 0; i < lines.size(); i++) {
    p1.x = lines[i][0];
    p1.y = lines[i][1];
    p2.x = lines[i][2];
    p2.y = lines[i][3];

    cv::line(image, p1, p2, color, 2);
  }
}

bool LineDetector::find3DLineStartAndEnd(const cv::Mat& point_cloud,
                                         const cv::Vec4f& line2D,
                                         cv::Vec<float, 6>& line3D) {
  CHECK_EQ(point_cloud.type(), CV_32FC3)
      << "The input matrix point_cloud must be of type CV_32FC3.";
  cv::Point2f start, end, line;
  // A floating point value that decribes a position in an image is always
  // within the pixel described through the floor operation.
  start.x = floor(line2D[0]);
  start.y = floor(line2D[1]);
  end.x = floor(line2D[2]);
  end.y = floor(line2D[3]);
  // Search for a non NaN value on the line. Effectivly these two while loops
  // just make unit steps (one pixel) from start to end (first loop) and then
  // from end to start (second loop) until a non NaN point is found.
  while (std::isnan(point_cloud.at<cv::Vec3f>(start)[0])) {
    line = end - start;
    start.x += floor(line.x / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    start.y += floor(line.y / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    if (start.x == end.x && start.y == end.y) break;
  }
  if (start.x == end.x && start.y == end.y) return false;
  while (std::isnan(point_cloud.at<cv::Vec3f>(end)[0])) {
    line = start - end;
    end.x += floor(line.x / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    end.y += floor(line.y / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    if (start.x == end.x && start.y == end.y) break;
  }
  if (start.x == end.x && start.y == end.y) return false;
  // If a non NaN point was found before end was reached, the line is stored.
  line3D = cv::Vec<float, 6>(
      point_cloud.at<cv::Vec3f>(start)[0], point_cloud.at<cv::Vec3f>(start)[1],
      point_cloud.at<cv::Vec3f>(start)[2], point_cloud.at<cv::Vec3f>(end)[0],
      point_cloud.at<cv::Vec3f>(end)[1], point_cloud.at<cv::Vec3f>(end)[2]);
  return true;
}

float LineDetector::findAndRate3DLine(const cv::Mat& point_cloud,
                                      const cv::Vec4f& line2D,
                                      cv::Vec<float, 6>& line3D) {
  CHECK_EQ(point_cloud.type(), CV_32FC3)
      << "The input matrix point_cloud must be of type CV_32FC3.";
  cv::Point2f start, end, line, rate_it;
  // A floating point value that decribes a position in an image is always
  // within the pixel described through the floor operation.
  start.x = floor(line2D[0]);
  start.y = floor(line2D[1]);
  end.x = floor(line2D[2]);
  end.y = floor(line2D[3]);
  // Search for a non NaN value on the line.
  while (std::isnan(point_cloud.at<cv::Vec3f>(start)[0])) {
    line = end - start;
    start.x += floor(line.x / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    start.y += floor(line.y / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    if (start.x == end.x && start.y == end.y) break;
  }
  if (start.x == end.x && start.y == end.y) return 0.0;
  while (std::isnan(point_cloud.at<cv::Vec3f>(end)[0])) {
    line = start - end;
    end.x += floor(line.x / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    end.y += floor(line.y / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    if (start.x == end.x && start.y == end.y) break;
  }
  if (start.x == end.x && start.y == end.y) return 0.0;
  line3D = cv::Vec<float, 6>(
      point_cloud.at<cv::Vec3f>(start)[0], point_cloud.at<cv::Vec3f>(start)[1],
      point_cloud.at<cv::Vec3f>(start)[2], point_cloud.at<cv::Vec3f>(end)[0],
      point_cloud.at<cv::Vec3f>(end)[1], point_cloud.at<cv::Vec3f>(end)[2]);

  // In addition to find3DLineStartAndEnd, this function also rates the line.
  // This is done here, but it does not work very well.
  double rating = 0.0;
  int num_points_rated = 0;
  rate_it = start;
  while (!(rate_it.x == end.x && rate_it.y == end.y)) {
    line = end - rate_it;
    rate_it.x += floor(line.x / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    rate_it.y += floor(line.y / sqrt(line.x * line.x + line.y * line.y) + 0.5);
    if (std::isnan(point_cloud.at<cv::Vec3f>(rate_it)[0])) continue;
    rating += computeDistPointToLine3D(point_cloud.at<cv::Vec3f>(start),
                                       point_cloud.at<cv::Vec3f>(end),
                                       point_cloud.at<cv::Vec3f>(rate_it));
    ++num_points_rated;
  }
  if (num_points_rated > 5)
    rating = rating / double(num_points_rated);
  else
    rating = 0;
  return (1.0 - exp(-rating));
}

double LineDetector::computeDistPointToLine3D(const cv::Vec3f& start,
                                              const cv::Vec3f& end,
                                              const cv::Vec3f& point) {
  return normOfVector3D(crossProduct(point - start, point - end)) /
         normOfVector3D(start - end);
}

bool LineDetector::planeRANSAC(const std::vector<cv::Vec3f>& points,
                               cv::Vec4f& hessian_normal_form) {
  const size_t N = points.size();
  double inlier_fraction_min = 0.5;
  std::vector<cv::Vec3f> inliers;
  planeRANSAC(points, inliers);
  // If we found not enough inlier, return false. This is important because
  // there might not be a solution (and we dont want to propose one if there
  // is none).
  if (inliers.size() <= inlier_fraction_min * N) return false;
  // Now we compute the final model parameters with all the inliers.
  return hessianNormalFormOfPlane(inliers, hessian_normal_form);
}
void LineDetector::planeRANSAC(const std::vector<cv::Vec3f>& points,
                               std::vector<cv::Vec3f>& inliers) {
  // Set parameters and do a sanity check.
  const int N = points.size();
  inliers.clear();
  // ROS_INFO("N = %d", N);
  const int max_it = 50;
  const int number_of_model_params = 3;
  double max_deviation = 0.005;
  double inlier_fraction_max = 0.95;
  CHECK(N > number_of_model_params) << "Not enough points to use RANSAC.";
  // Declare variables that are used for the RANSAC.
  std::vector<cv::Vec3f> random_points, inlier_candidates;
  cv::Vec3f normal;
  cv::Vec4f hessian_normal_form;
  // Set a random seed.
  unsigned seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  // Start RANSAC.
  for (int iter = 0; iter < max_it; ++iter) {
    // Get number_of_model_params unique elements from poitns.
    getNUniqueRandomElements(points, number_of_model_params, generator,
                             random_points);
    // It might happen that the randomly chosen points lie on a line. In this
    // case, hessianNormalFormOfPlane would return false.
    if (!hessianNormalFormOfPlane(random_points, hessian_normal_form)) continue;
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
    if (inlier_candidates.size() > inliers.size()) inliers = inlier_candidates;

    // Usual not part of RANSAC: stop early if we have enough inliers.
    // This feature is here because it might be that we have a very
    // high inlier percentage. In this case RANSAC finds the right
    // model within the first few iterations and all later iterations
    // are just wasted run time.
    if (inliers.size() > inlier_fraction_max * N) break;
  }
}

void LineDetector::project2Dto3DwithPlanes(
    const cv::Mat& cloud, const std::vector<cv::Vec4f>& lines2D,
    std::vector<cv::Vec<float, 6> >& lines3D) {
  CHECK_EQ(cloud.type(), CV_32FC3);
  cv::Vec4f hessianNF_left, hessianNF_right;
  std::vector<cv::Point2f> rect_left, rect_right;
  std::vector<cv::Point2i> points_in_rect;
  std::vector<cv::Vec3f> plane_point_cand, inliers_left, inliers_right;
  std::vector<cv::Vec<float, 6> > lines3D_cand;
  std::vector<int> idx;
  cv::Point2i start, end;
  cv::Vec<float, 6> line3D_true;
  // TODO implement version with correspondeces
  projectLines2Dto3D(lines2D, cloud, lines3D_cand);
  for (size_t i = 0; i < lines2D.size(); ++i) {
    getRectanglesFromLine(lines2D[i], rect_left, rect_right);
    findPointsInRectangle(rect_left, points_in_rect);
    plane_point_cand.clear();
    for (size_t i = 0; i < points_in_rect.size(); ++i) {
      if (std::isnan(cloud.at<cv::Vec3f>(points_in_rect[i])[0])) continue;
      plane_point_cand.push_back(cloud.at<cv::Vec3f>(points_in_rect[i]));
    }
    planeRANSAC(plane_point_cand, inliers_left);
    findPointsInRectangle(rect_right, points_in_rect);
    plane_point_cand.clear();
    for (size_t i = 0; i < points_in_rect.size(); ++i) {
      if (std::isnan(cloud.at<cv::Vec3f>(points_in_rect[i])[0])) continue;
      plane_point_cand.push_back(cloud.at<cv::Vec3f>(points_in_rect[i]));
    }
    planeRANSAC(plane_point_cand, inliers_right);
    if (find3DlineOnPlanes(inliers_right, inliers_left, lines3D_cand[i],
                           line3D_true)) {
      lines3D.push_back(line3D_true);
    }
  }
}

void LineDetector::find3DlinesByShortest(
    const cv::Mat& cloud, const std::vector<cv::Vec4f>& lines2D,
    std::vector<cv::Vec<float, 6> >& lines3D) {
  std::vector<int> correspondeces;
  find3DlinesByShortest(cloud, lines2D, lines3D, correspondeces);
}
void LineDetector::find3DlinesByShortest(
    const cv::Mat& cloud, const std::vector<cv::Vec4f>& lines2D,
    std::vector<cv::Vec<float, 6> >& lines3D,
    std::vector<int>& correspondeces) {
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
  correspondeces.clear();
  lines3D.clear();
  for (size_t i; i < lines2D.size(); ++i) {
    dist_opt = 1e20;
    x_opt_start = lines2D[i][0];
    y_opt_start = lines2D[i][1];
    x_opt_end = lines2D[i][2];
    y_opt_end = lines2D[i][3];
    // This checks are used to make shure, that we do not try to access a
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
                std::isnan(cloud.at<cv::Vec3f>(y_end, x_end)[0]))
              continue;
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
    // Assuming that distances are in meter, we can savely assume that if our
    // optimal distance is still 1e20, no non-NaN points were found.
    if (dist_opt == 1e20) continue;
    // Otherwise, a line was found.
    start = cloud.at<cv::Vec3f>(y_opt_start, x_opt_start);
    end = cloud.at<cv::Vec3f>(y_opt_end, x_opt_end);
    lines3D.push_back(cv::Vec<float, 6>(start[0], start[1], start[2], end[0],
                                        end[1], end[2]));
    correspondeces.push_back(i);
  }
}
}  // namespace line_detection
