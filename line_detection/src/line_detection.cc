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
    if (norms < 1e-9) return false;
    // Then if they lie on a line. The angle between the vectors must at least
    // be 2 degrees.
    double cos_theta = abs(scalarProduct(vec1, vec2)) / norms;
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
    cv::SVD::compute(A, W, U, Vt, cv::SVD::FULL_UV);
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
                                std::vector<int>& y_coord) {
  int top = floor(start.x);
  int bottom = ceil(end.x);
  int height = bottom - top;
  float y_start, width;
  y_start = floor(start.y) + 0.5;
  width = floor(end.y) - floor(start.y);
  CHECK(height > 0) << "Important: the following statement must hold: start.x "
                       "< end.x.";
  if (height == 1) {
    if (left_side)
      y_coord.push_back(floor(start.y));
    else
      y_coord.push_back(ceil(end.y));
    return;
  }
  for (int i = 0; i < height; ++i) {
    y_coord.push_back(int(y_start + i * width / (height - 1)));
  }
}

void findPointsInRectangle(std::vector<cv::Point2f> corners,
                           std::vector<cv::Point2i>& points) {
  CHECK_EQ(corners.size(), 4)
      << "The rectangle must be defined by exactly 4 corner points.";
  // Find the relative positions of the points.
  int upper, lower, left, right, store_i;
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
    if (corners[0].x == corners[i].x) some_points_have_equal_height = true;
  }
  for (int i = 2; i < 4; ++i) {
    if (corners[1].x == corners[i].x) some_points_have_equal_height = true;
  }
  if (corners[2].x == corners[3].x) some_points_have_equal_height = true;
  // Do the rotation.
  if (some_points_have_equal_height) {
    for (int i = 0; i < 4; ++i)
      corners[i] =
          cv::Point2f(0.99998 * corners[i].x - 0.0017453 * corners[i].y,
                      0.0017453 * corners[i].x + 0.99998 * corners[i].y);
  }
  // The points are set to lowest, highest, most right and most left in this
  // order. It does work because the preprocessing done guarantees that no two
  // points have the same x coordinate.
  lower = idx[0];
  for (int i = 1; i < 4; ++i) {
    if (corners[i].x > corners[lower].x) lower = i;
  }
  idx.erase(idx.begin() + lower);

  upper = idx[0];
  store_i = 0;
  for (int i = 0; i < 3; ++i) {
    if (corners[idx[i]].x < corners[upper].x) store_i = i;
  }
  upper = idx[store_i];
  idx.erase(idx.begin() + store_i);

  right = idx[0];
  store_i = 0;
  for (int i = 0; i < 2; ++i) {
    if (corners[idx[i]].y > corners[right].y) store_i = i;
  }
  right = idx[store_i];
  idx.erase(idx.begin() + store_i);

  left = idx[0];
  // With the ordering given, the border pixels can be found as pixels, that lie
  // on the border vectors.
  std::vector<int> left_border;
  std::vector<int> right_border;
  findYCoordOfPixelsOnVector(corners[upper], corners[left], true, left_border);
  findYCoordOfPixelsOnVector(corners[upper], corners[right], false,
                             right_border);
  // Pop_back is used because otherwise the corners[left/right] pixels would be
  // counted twice.
  left_border.pop_back();
  right_border.pop_back();
  findYCoordOfPixelsOnVector(corners[left], corners[lower], true, left_border);
  findYCoordOfPixelsOnVector(corners[right], corners[lower], false,
                             right_border);
  CHECK_EQ(left_border.size(), right_border.size()) << "Something went wrong.";
  // Iterate over all pixels in the rectangle.
  points.clear();
  int x, y;
  for (int i = 0; i < left_border.size(); ++i) {
    x = floor(corners[upper].x) + i;
    y = left_border[i];
    do {
      points.push_back(cv::Point2i(x, y));
      ++y;
    } while (y <= right_border[i]);
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

    // *************** VARIANT 2 ***************** //
    // if (find3DLineStartAndEnd(point_cloud, lines2D[i], line3D))
    //   lines3D.push_back(line3D);

    //*********** VARIANT 3 ****************//
    // line.x = lines2D[i][2] - lines2D[i][0];
    // line.y = lines2D[i][3] - lines2D[i][1];
    // double line_normalizer = sqrt(line.x * line.x + line.y * line.y);
    // upper_line2D[0] = checkInBoundary(
    //     floor(lines2D[i][0]) + floor(line.y / line_normalizer + 0.5), 0.0,
    //     cols);
    // upper_line2D[1] = checkInBoundary(
    //     floor(lines2D[i][1]) + floor(-line.x / line_normalizer + 0.5), 0.0,
    //     rows);
    // upper_line2D[2] = checkInBoundary(
    //     floor(lines2D[i][2]) + floor(line.y / line_normalizer + 0.5), 0.0,
    //     cols);
    // upper_line2D[3] = checkInBoundary(
    //     floor(lines2D[i][3]) + floor(-line.x / line_normalizer + 0.5), 0.0,
    //     rows);
    // lower_line2D[0] = checkInBoundary(
    //     floor(lines2D[i][0]) + floor(-line.y / line_normalizer + 0.5), 0.0,
    //     cols);
    // lower_line2D[1] = checkInBoundary(
    //     floor(lines2D[i][1]) + floor(line.x / line_normalizer + 0.5), 0.0,
    //     rows);
    // lower_line2D[2] = checkInBoundary(
    //     floor(lines2D[i][2]) + floor(-line.y / line_normalizer + 0.5), 0.0,
    //     cols);
    // lower_line2D[3] = checkInBoundary(
    //     floor(lines2D[i][3]) + floor(line.x / line_normalizer + 0.5), 0.0,
    //     rows);
    // rate_low = findAndRate3DLine(point_cloud, lower_line2D, lower_line3D);
    // rate_mid = findAndRate3DLine(point_cloud, lines2D[i], line3D);
    // rate_up = findAndRate3DLine(point_cloud, upper_line2D, upper_line3D);
    // if (rate_up == 0 && rate_low == 0 && rate_mid == 0) continue;
    // if (rate_up > rate_mid && rate_up > rate_low)
    //   lines3D.push_back(upper_line3D);
    // else if (rate_mid > rate_low)
    //   lines3D.push_back(line3D);
    // else
    //   lines3D.push_back(lower_line3D);
  }
}

void LineDetector::fuseLines2D(const std::vector<cv::Vec4f>& lines_in,
                               std::vector<cv::Vec4f>& lines_out) {
  lines_out.clear();
  // This list is used to remember which lines have already been assigned to a
  // cluster. Every time a line is assigned, the corresponding index is deleted
  // in this list.
  std::list<int> line_index;
  for (size_t i = 0; i < lines_in.size(); ++i) line_index.push_back(i);
  // This vector is used to store the line clusters until they are merged into
  // one line.
  std::vector<cv::Vec4f> line_cluster;
  // Iterate over all lines.
  for (size_t i = 0; i < lines_in.size(); ++i) {
    line_cluster.clear();
    // If this condition does not hold, the line lines_in[i] has already been
    // merged into a new one. If not, the algorithm tries to find lines that are
    // near this line.
    if (*(line_index.begin()) != i) {
      continue;
    } else {
      line_cluster.push_back(lines_in[i]);
      line_index.pop_front();
    }
    for (std::list<int>::iterator it = line_index.begin();
         it != line_index.end(); ++it) {
      // This loop checks if the line is near any line in the momentary cluster.
      // If yes, it assignes it to the cluster.
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
  // Set parameters and do a sanity check.
  const int N = points.size();
  const int max_it = 50;
  const int number_of_model_params = 3;
  double max_deviation = 0.05;
  double inlier_fraction_max = 0.8;
  double inlier_fraction_min = 0.5;
  CHECK(N > number_of_model_params) << "Not enough points to use RANSAC.";
  // Declare variables that are used for the RANSAC.
  std::vector<cv::Vec3f> random_points, inliers, inlier_candidates;
  cv::Vec3f normal;
  // Set a random seed.
  unsigned seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  // Start RANSAC.
  for (int iter = 0; iter < max_it; ++iter) {
    // Get number_of_model_params unique elements from poitns.
    getNUniqueRandomElements(points, number_of_model_params, random_points,
                             generator);
    // It might happen that the randomly chosen points lie on a line. In this
    // case, hessianNormalFormOfPlane would return false.
    if (!hessianNormalFormOfPlane(random_points, hessian_normal_form)) continue;
    normal = cv::Vec3f(hessian_normal_form[0], hessian_normal_form[1],
                       hessian_normal_form[2]);
    // Check which of the points are inlier with the current plane model.
    inlier_candidates.clear();
    for (int j = 0; j < N; ++j) {
      if (errorPointToPlane(normal, random_points[0], points[j]) <
          max_deviation) {
        inlier_candidates.push_back(points[j]);
      }
    }
    // If we found more inliers than in any previous run, we store them as
    // global inliers.
    if (inlier_candidates.size() > inliers.size()) inliers = inlier_candidates;
    // Usual not part of RANSAC: stop early if we have enough inliers. This
    // feature is here because it might be that we have a very high inlier
    // percentage. In this case RANSAC finds the right model within the first
    // few iterations and all later iterations are just wasted run time.
    if (inliers.size() > inlier_fraction_max * N) break;
  }
  // If we found not enough inlier, return false. This is important because
  // there might not be a solution (and we dont want to propose one if there is
  // none).
  if (inliers.size() < inlier_fraction_min * N) return false;
  // Now we compute the final model parameters with all the inliers.
  return hessianNormalFormOfPlane(inliers, hessian_normal_form);
}

}  // namespace line_detection
