#ifndef LINE_DETECTION_LINE_DETECTION_H_
#define LINE_DETECTION_LINE_DETECTION_H_

#include "line_detection/common.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/ximgproc.hpp>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <boost/filesystem.hpp>

namespace line_detection {

constexpr double kPi = 3.141592653589793;

enum class Detector : unsigned int { LSD = 0, EDL = 1, FAST = 2, HOUGH = 3 };

// Meaning of line types:
// - DISCONT: discontinuity line. Line on a discontinuity edge of an object,
//            i.e., for which the two planes fitted to the inliers around the
//            line are distant in depth from each other. Example: edge of a
//            chair placed in front of a wall -> One 'inlier plane' belongs to
//            the wall and the other belongs to the chair.
// - PLANE: planar (surface) line. Line on a plane, i.e., for which the two
//          'inlier planes' are close in depth and (almost) parallel to each
//          other. Example: line on a poster or on any planar object.
// - EDGE: edge line. Line that separates two parts of a same object, i.e., for
//         which the two 'inlier planes' are close in depth but not parallel.
//         Example: line that separates the seating cushion of an armchair and
//         the cushion against which the back leans.
// - INTERSECT: intersection line. Line that separates two different objects
//              where they are in contact with each other. Example: line between
//              a box and the surface on which the box is placed. Note: 'inlier
//              planes' have the same properties as the those for an edge line.
//              However, while the inlier points on the two planes around an
//              edge line all belong to the same object, for intersection lines
//              the inlier points on one plane belong to an object (e.g. the
//              box) and those on the other plane belong to another object (e.g.
//              the surface on which the box lies). We later assign the line to
//              belong to the object (between the two) which, if removed, would
//              cause the line to disappear (e.g., removing the box causes the
//              line to disappear, whereas if the surface on which the box lies
//              is removed the line is still present).

enum class LineType : unsigned int {
  DISCONT = 0,
  PLANE = 1,
  EDGE = 2,
  INTERSECT = 3
};

struct LineWithPlanes {
  cv::Vec6f line;
  std::vector<cv::Vec4f> hessians;
  std::vector<cv::Vec3b> colors;
  LineType type;
};

struct LineDetectionParams {
  // default = 0.3: find3DLineOnPlanes
  double max_dist_between_planes = 0.3;
  // default = 0.5: getRectanglesFromLine
  double rectangle_offset_pixels = 0.5;
  // default = 0.5: getRectanglesFromLine
  double max_relative_rect_size = 0.5;
  // default = 5.0: getRectanglesFromLine
  double max_absolute_rect_size = 5.0;
  // default = 20: LineDetector::project2Dto3DwithPlanes
  unsigned int min_points_in_rect = 20;
  // default = 100: LineDetector::planeRANSAC
  unsigned int num_iter_ransac = 300;
  // default = 0.005: LineDetector::planeRANSAC
  double max_error_inlier_ransac = 0.005;
  // default = 0.8: LineDetector::planeRANSAC
  double inlier_max_ransac = 0.8;
  // default = 0.1: LineDetector::project2Dto3DwithPlanes
  double min_inlier_ransac = 0.1;
  // default = 10: LineDetector::checkIfValidLineBruteForce
  unsigned int min_points_in_line = 10;
  // default = 0.01: LineDetector::checkIfValidLineWith2DInfo
  double min_length_line_3D = 0.01;
  // default = 0.1: LineDetector::assignEdgeOrIntersectionLineType
  double extension_length_for_edge_or_intersection = 0.05;
  // default = 10: LineDetetor::checkEdgeOrIntersectionGivenProlongedLine
  double max_points_for_empty_rectangle = 5;
  // default = 0.02: LineDetector::checkIfValidLineBruteForce
  double max_deviation_inlier_line_check = 0.02;
  // default = 1e6: LineDetector::find3DlinesRated
  double max_rating_valid_line = 1e6;
  // default = 1e-6: hessianNormalFormOfPlane
  double min_distance_between_points_hessian = 1e-6;
  // default = 0.994: hessianNormalFormOfPlane
  double max_cos_theta_hessian_computation = 0.994;
  // default = 50: hough line detector
  unsigned int canny_edges_threshold1 = 50;
  // default = 200: hough line detector
  unsigned int canny_edges_threshold2 = 200;
  // default = 3: hough line detector
  unsigned int canny_edges_aperture = 3;
  // default = 1: hough line detector
  double hough_detector_rho = 1.0;
  // default = kPi/180: hough line detector
  double hough_detector_theta = kPi / 180.0;
  // default = 10: hough line detector
  unsigned int hough_detector_threshold = 10;
  // default = 10: hough line detector
  double hough_detector_minLineLength = 10.0;
  // default = 5: hough line detector
  double hough_detector_maxLineGap = 5.0;
};

// Returns true if lines are nearby and could be equal (low difference in angle
// and start or end point).
bool areLinesEqual2D(const cv::Vec4f line1, const cv::Vec4f line2);

// Returns value if: lower < value < upper
// Returns lower if: value < lower
// Returns upper if: upper < value
inline double checkInBoundary(const double value, const double lower,
                              const double upper) {
  if (value < lower) {
    return lower;
  } else if (value > upper) {
    return upper;
  } else {
    return value;
  }
}

inline int checkInBoundaryInt(const int value, const int lower,
                              const int upper) {
  if (value < lower) {
    return lower;
  } else if (value > upper) {
    return upper;
  } else {
    return value;
  }
}

inline cv::Vec3f crossProduct(const cv::Vec3f a, const cv::Vec3f b) {
  return cv::Vec3f(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0]);
}

// Compute the slope depending on start and end point of a line.
inline double computeSlopeOfLine(const cv::Vec4f line) {
  return (line[1] - line[3]) / (line[0] - line[2]);
}

inline void normalizeVector3D(cv::Vec3f* vector) {
  *vector = (*vector) / cv::norm(*vector);
}

inline double degToRad(const double in_deg) { return in_deg / 180.0 * kPi; }

inline double scalarProduct(const cv::Vec3f& a, const cv::Vec3f& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Computes d from the plane equation a*x+b*y+c*z+d=0 given the plane normal and
// a point on the plane. The coefficients a, b and c are then the entries of the
// normal.
inline double computeDfromPlaneNormal(const cv::Vec3f& normal,
                                      const cv::Vec3f& anchor) {
  return -normal.dot(anchor);
}

// Computes the orthogonal square distance from a point to a plane (given by
// normal and point on plane). The distance has only a real meaning if the
// normal is a unit vector.
inline double errorPointToPlane(const cv::Vec3f& normal,
                                const cv::Vec3f& point_on_plane,
                                const cv::Vec3f& point) {
  return fabs((point_on_plane - point).dot(normal));
}
inline double errorPointToPlane(const cv::Vec4f& hessian_n_f,
                                const cv::Vec3f& point) {
  return fabs(
      cv::Vec3f(hessian_n_f[0], hessian_n_f[1], hessian_n_f[2]).dot(point) +
      hessian_n_f[3]);
}

inline double distPointToLine(const cv::Vec3f& start, const cv::Vec3f& end,
                              const cv::Vec3f& point) {
  return cv::norm((point - start).cross(point - end)) / cv::norm(start - end);
}

// Assume normalized direction vector.
inline cv::Vec3f projectPointOnLine(const cv::Vec3f& x_0,
                                    const cv::Vec3f& direction,
                                    const cv::Vec3f& point) {
  return x_0 + direction * direction.dot(point - x_0);
}

inline cv::Vec3f computeMean(const std::vector<cv::Vec3f>& points) {
  const size_t num_points = points.size();
  CHECK(num_points > 0);
  cv::Vec3f mean(0.0f, 0.0f, 0.0f);
  for (size_t i = 0; i < num_points; ++i) {
    mean += (points[i] / float(num_points));
  }
  return mean;
}

// Makes an hessian normal vector point towards a point/the origin, so that the
// point lies in the half-space towards which the normal vector points.
// (Input: point: Point towards which the hessian vector should be pointed.)
//
//  Output: hessian: Hessian directed towards the point/origin.
inline void directHessianTowardsOrigin(cv::Vec4f* hessian) {
  cv::Vec3f origin{0.0f, 0.0f, 0.0f};
  float d = (*hessian)[3];
  if (d < 0)
    *hessian = -(*hessian);
}
inline void directHessianTowardsPoint(const cv::Vec3f& point,
                                      cv::Vec4f* hessian) {
  // Let (x_, y_, z_) be a point in the half-space towards which the hessian
  // of the plane should point. Then, letting the equation of the plane be
  // a * x +  b * y + c * z + d = 0, with WLOG ||(a, b, c)||_2 = 1 (otherwise
  // the equation can be rescaled accordingly) => a^2 + b^2 + c^2 = 1, and
  // letting (x_p, y_p, z_p) be the projection of (x_, y_, z_) on the plane, one
  // has that:
  // - a * x_p +  b * y_p + c * z_p + d = 0 (since the projection belongs to the
  //   plane by definition);
  // - There exists t > 0 s.t. x_ = x_p + a * t, y_ = y_p + b * t,
  //   z_ = z_p + c * t.
  // Therefore a * x_ +  b * y_ + c * z_ + d = a * (x_p + a * t) +
  //           b * (y_p + b * t) + c * (z_p + c * t) + d =
  //         = (a * x_p +  b * y_p + c * z_p + d) + t * (a^2 + b^2 + c^2) =
  //         = 0 + t * (a^2 + b^2 + c^2) > 0, since t > 0.
  // If this is not verified, the orientation of the normal should be reverted.
  cv::Vec4f point_homo({point[0], point[1], point[2], 1.0});
  if (hessian->dot(point_homo) < 0)
    *hessian = -(*hessian);
}

// Returns the projection of a point on the plane given defined by the hessian.
cv::Vec3f projectPointOnPlane(const cv::Vec4f& hessian, const cv::Vec3f& point);

// Returns the intersection point between a plane and a line, if possible.
// Input: plane:               Plane in hessian form.
//
//        line_direction:      Direction of the line.
//
// Output: intersection_point: Intersection point between line and plane.
//
//         return:             True if an intersection was found, i.e., if line
//                             is not parallel to the plane, false otherwise.
bool findIntersectionBetweenPlaneAndLine(const cv::Vec4f& plane,
                                         const cv::Vec3f& line_direction,
                                         cv::Vec3f* intersection_point);

// Returns the hessian form of a plane that contains a given point and a given
// line.
// Input: point:                Point that belongs to the plane.
//
//        line_start/end_start: Endpoints of the line that belongs to the plane.
//
// Output: plane:               Plane that contains both the line and the point,
//                              in Hessian form.
//         return:              True if a plane exists (i.e., if the point does
//                              not belong to the line), false otherwise.
bool findPlaneThroughPointAndLine(const cv::Vec3f& point,
                                  const cv::Vec3f& line_start,
                                  const cv::Vec3f& line_end,
                                  cv::Vec4f* plane);

// Finds the x-coordinates of a line between two points.
// Input: start:      Starting point of line.
//        end:        End point of line. Note that start.y < end.y must
//                    hold.
//        left_side:  For the special case for a near to horizontal line. If
//                    true, it gives back the most left x value of the line,
//                    otherwise the most right.
//
// Output: x_coord    The x coordinates ordered from top to bottom. Example:
//                    If the line spans 3 rows of pixel (so 2 < |end.y -
//                    start.y| < 3), 3 values are stored in x_coord, where
//                    the first corresponds to the x coordinate of the point
//                    with the lowest y (most upper in image coordinates).
void findXCoordOfPixelsOnVector(const cv::Point2f& start,
                                const cv::Point2f& end, bool left_side,
                                std::vector<int>* x_coord);

// Returns all pixels that are within or on the border of a rectangle.
// Input: corners:  These corners define the rectangle. It must contain
//                  only 4 points. The points must be the cornerpoints of a
//                  parallelogram. If that is not given, the outcome of the
//                  function depends on the ordering of the corner point and
//                  might be wrong.
// Output: points:  A vector of pixel coordinates.
void findPointsInRectangle(std::vector<cv::Point2f> corners,
                           std::vector<cv::Point2i>* points);

// Takes two planes and computes the intersection line. This function takes
// already the direction of the line (which could be computed from the two
// planes as well), because you can save computation time with it, if you
// already have computed it beforehand.
// Input: hessian1/2: Hessian normal forms of the two planes.
//
//        direction:  Direction of the line (must lie on both planes).
//
// Output: x_0:       A point on on both planes.
bool getPointOnPlaneIntersectionLine(const cv::Vec4f& hessian1,
                                     const cv::Vec4f& hessian2,
                                     const cv::Vec3f& direction,
                                     cv::Vec3f* x_0);

// Displays (via the script python/display_line_with_points_and_planes.py) a
// line together with its two planes and its two sets of inliers.
// Input: start/end: Endpoints of the line.
//
//        start/end_guess: Endpoints of the guess line.
//
//        inliers1/2: Points inliers to the two planes around the line.
//
//        hessian1/2: Planes around the line.
//
// Output: none.
void displayLineWithPointsAndPlanes(const cv::Vec3f& start,
                                    const cv::Vec3f& end,
                                    const cv::Vec3f& start_guess,
                                    const cv::Vec3f& end_guess,
                                    const std::vector<cv::Vec3f>& inliers1,
                                    const std::vector<cv::Vec3f>& inliers2,
                                    const cv::Vec4f& hessian1,
                                    const cv::Vec4f& hessian2);


class LineDetector {
 public:
  LineDetector();
  LineDetector(LineDetectionParams* params);

  ~LineDetector();

  // Returns the parameter of the line detector.
  // Return: parameters params of the line detector.
  inline LineDetectionParams get_line_detection_params() {
    return *params_;
  }

  // detectLines:
  // Input: image:    The image on which the lines should be detected.
  //
  //        detector: 0-> LSD, 1->EDL, 2->FAST, 3-> HOUGH
  //                  Default is LSD. It is chosen even if an invalid number is
  //                  given
  //
  // Output: lines:   The lines are stored in the following format:
  //                  {start.x, start.y, end.x, end.y}
  void detectLines(const cv::Mat& image, Detector detector,
                   std::vector<cv::Vec4f>* lines);
  void detectLines(const cv::Mat& image, int detector,
                   std::vector<cv::Vec4f>* lines);
  void detectLines(const cv::Mat& image, std::vector<cv::Vec4f>* lines);

  // This function computes the Hessian Normal Form of a plane given points on
  // that plane.
  // Input: points:   Must contain at least have 3 points that do not
  //                  lie on a line. If 3 points are given, the plane normal is
  //                  computed as the cross product. The solution is then exact.
  //                  If more than 3 points are given the function solves a
  //                  minimization problem (min sum (orthogonal dist)^2) through
  //                  SVD.
  //
  // Output: hessian_normal_form: The first 3 entries are the normal vector n,
  //                              the last one is the parameter p
  bool hessianNormalFormOfPlane(const std::vector<cv::Vec3f>& points,
                                cv::Vec4f* hessian_normal_form);

  // So far this function uses a naive approach which is not yet very robust.
  void projectLines2Dto3D(const std::vector<cv::Vec4f>& lines2D,
                          const cv::Mat& point_cloud,
                          std::vector<cv::Vec6f>* lines3D);

  // This function clusters nearby lines in line_in and summarizes it as one
  // line. All new lines are stored in lines_out.
  void fuseLines2D(const std::vector<cv::Vec4f>& lines_in,
                   std::vector<cv::Vec4f>* lines_out);

  // Simply paints all lines to image.
  void paintLines(const std::vector<cv::Vec4f>& lines, cv::Mat* image,
                  cv::Vec3b color = {255, 0, 0});

  // This function checks for all lines, if their start and end points are
  // within the borders of an image. It returns a vector with the same amount of
  // lines in the same order. Only start or end points that were out of the
  // image border are shifted into it.
  std::vector<cv::Vec4f> checkLinesInBounds(
      const std::vector<cv::Vec4f>& lines2D, size_t x_max, size_t y_max);

  // Finds two rectangles left and right of a line defined by 4 corner points.
  // Input:   line:   The 2D line defined as (start, end).
  //
  // Output:  rect_left: A rectangle defined by the 4 corner points.
  //          rect_right: Same as rect_left.
  bool getRectanglesFromLine(const cv::Vec4f& line,
                             std::vector<cv::Point2f>* rect_left,
                             std::vector<cv::Point2f>* rect_right);

  // This function takes a set of points within an image and computes the
  // average color of all pixels at these points. It appends the color in the
  // colors vectors of the line3D.
  void assignColorToLines(const cv::Mat& image,
                          const std::vector<cv::Point2i>& points,
                          LineWithPlanes* line3D);

  // (The two following functions are deprecated.They remain here just for
  // back compatibility concerns.)
  // Uses two sets of points and fit a line,
  // assuming that the two set of points are from a plane left and right of the
  // line.
  // Input: points1/2:  The two set of points.
  //
  //       line_guess:  This is a guess of the line that is used if the two sets
  //                    belong to same plane. In this case a line cannot be
  //                    determined by the intersection of the planes.
  //
  // Output: line:      The fitted 3D line (start, end);
  bool find3DlineOnPlanes(const std::vector<cv::Vec3f>& points1,
                          const std::vector<cv::Vec3f>& points2,
                          const cv::Vec6f& line_guess, cv::Vec6f* line);
  bool find3DlineOnPlanes(const std::vector<cv::Vec3f>& points1,
                          const std::vector<cv::Vec3f>& points2,
                          const cv::Vec6f& line_guess, LineWithPlanes* line);

  // Find 3D line using the points on the planes found by planeRANSAC.
  // Input:  points1/2: Two sets of points.
  //
  //         line_guess: Initial guess of the line.
  //
  //         cloud:    Point cloud as CV_32FC3.
  //
  //         camera_P:  Camera projection matrix.
  //
  //         planes_found: True if and only if both planes of the line are found
  //
  // Output:  line:  The 3D line found.
  bool find3DlineOnPlanes(const std::vector<cv::Vec3f>& points1,
                          const std::vector<cv::Vec3f>& points2,
                          const cv::Vec6f& line_guess, const cv::Mat& cloud,
                          const cv::Mat& camera_P, const bool planes_found,
                          LineWithPlanes* line);

  // Assign the type of line to be either edge or intersection.
  // Input: cloud:    Point cloud as CV_32FC3.
  //
  //        camera_P:  Camera projection matrix.
  //
  //        inliers_1/2: Inlier points for each of the two planes associated to
  //                     the line.
  //
  // Output: line: Input line with a type assigned to it.
  //         return: True if line type assignment could be performed, False
  //                 otherwise.
  bool assignEdgeOrIntersectionLineType(const cv::Mat& cloud,
                                        const cv::Mat& camera_P,
                                        const std::vector<cv::Vec3f>& inliers_1,
                                        const std::vector<cv::Vec3f>& inliers_2,
                                        LineWithPlanes* line);

  // Determines whether the two inlier planes of a line form a convex or concave
  // angle when seen from a given viewpoint. This is done by using the two sets
  // of points inlier to the line.
  // Input: line:                       Input line.
  //
  //        inliers_1/2:                Inlier points.
  //
  //        viewpoint:                  Point from which the line is observed.
  //
  // Output: convex_true_concave_false: True if the planes form a convex angle,
  //                                    false if they form a concave angle.
  //
  //         return:                    True if no errors occurs, false
  //                                    otherwise.
  bool determineConvexityFromViewpointGivenLineAndInlierPoints(
    const LineWithPlanes& line, const std::vector<cv::Vec3f>& inliers_1,
    const std::vector<cv::Vec3f>& inliers_2, const cv::Vec3f& viewpoint,
    bool* convex_true_concave_false);

  // Determines whether the two inlier planes of a line form a convex or concave
  // angle when seen from a given viewpoint. This is done by using the two mean
  // points of the points inlier to the line.
  // Input: line:                       Input line.
  //
  //        mean_point_1/2:             Mean points of the inlier points.
  //
  //        viewpoint:                  Point from which the line is observed.
  //
  // Output: convex_true_concave_false: True if the planes form a convex angle,
  //                                    false if they form a concave angle.
  //
  //         return:                    True if no errors occurs, false
  //                                    otherwise.
  bool determineConvexityFromViewpointGivenLineAndMeanPoints(
    const LineWithPlanes& line, const cv::Vec3f& mean_point_1,
    const cv::Vec3f& mean_point_2, const cv::Vec3f& viewpoint,
    bool* convex_true_concave_false);

  // Given the two endpoints of a prolonged line segment (produced by
  // assignEdgeOrIntersectionLineType) checks if the points around the prolonged
  // line segments are such that the original line is an edge line or rather an
  // intersection line.
  // Input: cloud:    Point cloud as CV_32FC3.
  //
  //        camera_P:  Camera projection matrix.
  //
  //        start: Start endpoint of the prolonged line segment.
  //
  //        end: End endpoint of the prolonged line segment.
  //
  //        hessians: Vector containing the Hessian normal form of the two
  //                  planes around the original line.
  //
  // Output: line_type: either EDGE or INTERSECT, type of line to be assigned to
  //                    the original line according to what found around the
  //                    prolonged line segments.
  //
  //         hessian_of_object_owning_line: in case of intersection line,
  //                                        hessian that describes the plane (of
  //                                        the two associated to the line) that
  //                                        belongs to the object 'owning' the
  //                                        line.
  //
  //         return: True if type of line can be succcessfully determined, False
  //                 otherwise.
  // Overload.
  bool checkEdgeOrIntersectionGivenProlongedLine(const cv::Mat& cloud,
                                                 const cv::Mat& camera_P,
                                                 const cv::Vec3f& start,
                                                 const cv::Vec3f& end,
                                                 const std::vector<cv::Vec4f>&
                                                   hessians,
                                                 LineType* line_type);
  bool checkEdgeOrIntersectionGivenProlongedLine(const cv::Mat& cloud,
                                                 const cv::Mat& camera_P,
                                                 const cv::Vec3f& start,
                                                 const cv::Vec3f& end,
                                                 const std::vector<cv::Vec4f>&
                                                   hessians,
                                                 LineType* line_type,
                                                 cv::Vec4f*
                                                   hessian_of_object_owning_line);

  // Fits a plane to the points using RANSAC.
  bool planeRANSAC(const std::vector<cv::Vec3f>& points,
                   cv::Vec4f* hessian_normal_form);
  void planeRANSAC(const std::vector<cv::Vec3f>& points,
                   std::vector<cv::Vec3f>* inliers);

  // Projects 2D lines to 3D using a plane intersection method.
  // Input: cloud:    Point cloud of type CV_32FC3.
  //
  //        lines2D:  Lines in 2D in pixel coordenates of the cloud.
  //
  // Output: lines3D:  3D lines found.
  void project2Dto3DwithPlanes(const cv::Mat& cloud,
                               const std::vector<cv::Vec4f>& lines2D,
                               std::vector<cv::Vec6f>* lines3D);
  void project2Dto3DwithPlanes(const cv::Mat& cloud,
                               const std::vector<cv::Vec4f>& lines2D,
                               std::vector<LineWithPlanes>* lines3D);

  // (Deprecated.) Old version of project2Dto3DwithPlanes. It's here for
  // backcomptability concerns.
  void project2Dto3DwithPlanes(const cv::Mat& cloud, const cv::Mat& image,
                               const std::vector<cv::Vec4f>& lines2D_in,
                               const bool set_colors,
                               std::vector<LineWithPlanes>* lines3D);
  // Current version of project2Dto3DwithPlanes.
  // Input: cloud:    Point cloud of type CV_32FC3.
  //
  //        lines2D_in:  Lines in 2D in pixel coordinates of the cloud.
  //        set_colors: True if assigning color to lines.
  //
  // Output:  lines_2D_out: 2D lines that correspond to lines3D
  //
  //          lines3D:  3D lines found.
  //
  //
  // Overload: Add output lines2D_out that correspond to lines3D
  void project2Dto3DwithPlanes(const cv::Mat& cloud, const cv::Mat& image,
                               const cv::Mat& camera_P,
                               const std::vector<cv::Vec4f>& lines2D_in,
                               const bool set_colors,
                               std::vector<cv::Vec4f>* lines2D_out,
                               std::vector<LineWithPlanes>* lines3D);

  // Projects 2D to 3D lines with a shortest is the best approach. Works in
  // general better than naive approach, but lines that lie on surfaces tend to
  // be drawn towards the camera.
  // Input: cloud:    A point cloud of type CV_32FC3. Does not need to be dense.
  //
  //        lines2D:  A vector containing lines in 2D. A line:
  //                  cv::Vec4f(start.x, start.y, end.x, end.y), where x,y are
  //                  in pixel coordinates.
  //
  // Output: lines3D: A vector with the lines found in 3D.
  //
  //         correspondences: The i-th element of this vector is the index of
  //                           the element in lines2D that corresponds to the
  //                           i-th element in lines3D.
  void find3DlinesByShortest(const cv::Mat& cloud,
                             const std::vector<cv::Vec4f>& lines2D,
                             std::vector<cv::Vec6f>* lines3D,
                             std::vector<int>* correspondences);
  // Overload: without correspondences
  void find3DlinesByShortest(const cv::Mat& cloud,
                             const std::vector<cv::Vec4f>& lines2D,
                             std::vector<cv::Vec6f>* lines3D);

  // Searches for best lines using a rating system. For every 2D line, it takes
  // two additional lines (one on the left, on the right) and does the following
  // procedure for every one: Compute inliers to line model given by start/end
  // points by computing the distance to all points on the line. Use the mean
  // distance of all inliers as the rating. Then the line with the lowest rating
  // is chosen as the best.
  // Input: cloud:    Point cloud in the format CV_32FC3.
  //
  //        lines2D:  2D lines defined in pixel coordinates.
  //
  // Output: lines3D: 3D lines defined in same coordinates as the cloud.
  //
  //         rating:  The rating for every 3D lines in the same order.
  void find3DlinesRated(const cv::Mat& cloud,
                        const std::vector<cv::Vec4f>& lines2D,
                        std::vector<cv::Vec6f>* lines3D,
                        std::vector<double>* rating);
  // Overload: Does not give a rating as an output and gets rid off 3D lines for
  // which no reasonable rating is given.
  void find3DlinesRated(const cv::Mat& cloud,
                        const std::vector<cv::Vec4f>& lines2D,
                        std::vector<cv::Vec6f>* lines3D);

  // Does a check by applying checkIfValidLineBruteForce to every line (using
  // checkIfValidLineBruteForce function to check).
  // Input: cloud:      Point cloud in the format CV_32FC3.
  //
  //        lines3D_in: 3D lines to be checked.
  //
  // Output: lines3D_out: All 3D lines that are considered as valid
  void runCheckOn3DLines(const cv::Mat& cloud,
                         const std::vector<LineWithPlanes>& lines3D_in,
                         std::vector<LineWithPlanes>* lines3D_out);
  // Overload: Check the validity of 3D lines with the help of the corresponded
  // 2D lines (using checkIfValidLineWith2DInfo function to check)
  // Input: cloud: Point cloud in the format CV_32FC3.
  //
  //        camera_P:  Camera projection matrix
  //
  //        lines2D_in: 2D lines corresponding to lines3D_in.
  //
  //        lines3D_in: 3D lines to be checked.
  //
  //        lines2D_out: 2D lines corresponding to lines3D_out.
  //
  //        lines3D_out: All 3D lines that are considered as valid. after check
  void runCheckOn3DLines(const cv::Mat& cloud, const cv::Mat& camera_P,
                         const std::vector<cv::Vec4f>& lines2D_in,
                         const std::vector<LineWithPlanes>& lines3D_in,
                         std::vector<cv::Vec4f>* lines2D_out,
                         std::vector<LineWithPlanes>* lines3D_out);

  // Does a check by applying checkIfValidLineDiscont on every line. This
  // check was mostly to try it out, it has shown that this way to check if
  // a line is valid is prone to errors.
  void runCheckOn2DLines(const cv::Mat& cloud,
                         const std::vector<cv::Vec4f>& lines2D_in,
                         std::vector<cv::Vec4f>* lines2D_out);

  // Checks if a line is valid with 2D information:

  // Input: cloud:    Point cloud as CV_32FC3
  //
  //        line:     Line in 3D defined by (start, end).
  //
  //        camera_P:  camera projection matrix
  //
  //        line_2D:  Line in 2D correcponded to 3D line
  //
  // Output: return:   True if it is a valid line, false otherwise.
  bool checkIfValidLineWith2DInfo(const cv::Mat& cloud, const cv::Mat& camera_P,
                                  cv::Vec4f& line_2D, cv::Vec6f* line);

  // Checks if a line is valid by brute force approach: It computes the distance
  // between every point in the point cloud and the line and returns true if a
  // sufficiently large number of this distances are below a threshold.
  // Input: cloud:    Point cloud as CV_32FC3
  //
  //        line:     Line in 3D defined by (start, end).
  //
  // Output: return:   True if it is a possible line, false otherwise.
  bool checkIfValidLineBruteForce(const cv::Mat& cloud, cv::Vec6f* line);

  // Checks if a line is valid by looking for discontinouties. It computes the
  // mean of a patch around a pixel and looks for jumps when this mean is given
  // with respect to the line.
  // Input: cloud:    Point cloud as CV_32FC3
  //
  //        line:     Line in 2D defined by (start, end).
  //
  // Output: return:   True if it is a possible line, false otherwise.
  bool checkIfValidLineDiscont(const cv::Mat& cloud, const cv::Vec4f& line);

  // This function does a search for a line with non NaN start and end points in
  // 3D given a line in 2D. It then computes number of points on this line
  // and returns the mean error of all points.
  // Input: point_cloud: Point cloud given as CV_32FC3.
  //
  //        line2D:   2D line in pixel coordinates.
  //
  // Output: line3D:  3D line in coordinates of the cloud.
  //
  //         num_points: Number of points on the line.
  //
  //         return:      Mean error of all inliers.
  double findAndRate3DLine(const cv::Mat& point_cloud, const cv::Vec4f& line2D,
                           cv::Vec6f* line3D, int* num_points);
  // Overload: Same as above, just without number of inliers as output.
  double findAndRate3DLine(const cv::Mat& point_cloud, const cv::Vec4f& line2D,
                           cv::Vec6f* line3D);

  // Shrink 2D lines
  // Input: lines2D_in: 2D lines to be shrinked
  //
  //        shrink_coff: A 2D line will be shrunk to shrink_coff
  //                     times the original length while retaining the center.
  //                     The value should between 0 and 1
  //
  //        min_length: If line length afer shrinking is less than min_length,
  //        keep the original line
  // Output: lines2D_out: 2D lines shrinked
  void shrink2Dlines(const std::vector<cv::Vec4f>& lines2D_in,
                     const double shrink_coff, const double min_length,
                     std::vector<cv::Vec4f>* lines2D_out);

  // Get the nearest point to the 3D line.
  // Input:  points: A set of points.
  //
  //         start: Start point of the line.
  //
  //         end: End point of the line.
  //
  // Output: nearest_point: Nearest point to the line in the points set.
  void getNearestPointToLine(const std::vector<cv::Vec3f>& points,
                             const cv::Vec3f& start, const cv::Vec3f& end,
                             cv::Vec3f* nearest_point);

  // Get the fraction of points that are around the line's center
  // Input: samples: Points distance to line start point divided by the line
  //                 length.
  //
  // Return: Fraction of points that are around center. A sample i is "Around
  //         center" means that samples[i] belongs to [0.25, 0.75].
  double getRatioOfPointsAroundCenter(const std::vector<double>& samples);

  // Adjusts the start and end points of 3D line using the inliers of the line.
  // All points considered as inliers of the line are projected onto the
  // line and then the pair of points that maximizes the distance of
  // the line are chosen.
  //
  //  Input:  points: A set of points.
  //
  //          start_in: Initial start point of the line.
  //
  //          end_in: Initial end point of the line.
  //
  //  Output: start_out: Start point of the line after adjusting.
  //
  //          end_out: End point of the line after adjusting.
  //
  //  Return: False if the number of inliers of the line is less than the
  //          threshold min_points_in_line.
  bool adjustLineUsingInliers(const std::vector<cv::Vec3f>& points,
                              const cv::Vec3f& start_in,
                              const cv::Vec3f& end_in, cv::Vec3f* start_out,
                              cv::Vec3f* end_out);

  // It might happen that when adjusting line using inliers the orientation of
  // the resulting line changes w.r.t. to the original lines. It might, i.e.,
  // happen that what was defined to be the start in the original line is closer
  // to the end of the adjusted line and what was defined to be the end in the
  // original line is closer to the start of the adjusted line. If this is the
  // case, this function switches start and end.
  //
  // Input: reference_line: Original reference line.
  //
  // Output: start,end:     Endpoints of the input line, switched if needed to
  //                        match the orientation of the reference line.
  void adjustLineOrientationGivenReferenceLine(const cv::Vec6f& reference_line,
                                               cv::Vec3f* start,
                                               cv::Vec3f* end);

  // Checks if a line is valid using the inliers of the line. If the ratio of
  // the inliers that lie around the center of the line is smaller than the
  // threshold kRatioThreshold, the line is not valid.
  //
  // Input: points: A set of points.
  //
  //        start: Start point of the line.
  //
  //        end: End point of the line.
  //
  // Return: True if the line if valid, false if not.
  bool checkIfValidLineUsingInliers(const std::vector<cv::Vec3f>& points,
                                    const cv::Vec3f& start,
                                    const cv::Vec3f& end);

 private:
  cv::Ptr<cv::LineSegmentDetector> lsd_detector_;
  cv::Ptr<cv::line_descriptor::BinaryDescriptor> edl_detector_;
  cv::Ptr<cv::ximgproc::FastLineDetector> fast_detector_;
  LineDetectionParams* params_;
  bool params_is_mine_;

  // find3DLineStartAndEnd:
  // Input: point_cloud:    Mat of type CV_32FC3, that stores the 3D points. A
  //                        point can be accessed by
  //                        point_cloud.at<cv::Point3f>(j, i).x.
  //
  //        line2D:         The line that was extracted in 2D.
  //
  // Output: line3D:        If a 3D line is found it is push_backed to this
  //                        vector. It is stored: (x_s, y_s, z_s, x_e, y_e,
  //                        z_e), where _s/_e means resp. start and end point.
  //
  //         start:         2D start point that corresponds to start point of
  //                        the 3D line.
  //
  //         end:           2D end point that corresponds to end point of the
  //                        3D line.
  //
  // Working principle of the function: It starts at the starting point of the
  // 2D line and looks if the values in the point_cloud are not NaN there. If
  // they are not, this value is stored as the starting point. If they are NaN,
  // the function jumps to a neighbouring pixel in the direction of the end
  // point and repeats the check. This procedure is done until a non NaN point
  // is found or the search reaches the end point. If a starting point was
  // found, then the same procedure is redone from the end point. It returns
  // true if a line was found and false otherwise.
  bool find3DLineStartAndEnd(const cv::Mat& point_cloud,
                             const cv::Vec4f& line2D, cv::Vec6f* line3D,
                             cv::Point2f* start, cv::Point2f* end);
};
}  // namespace line_detection

#include "line_detection/line_detection_inl.h"

#endif  // LINE_DETECTION_LINE_DETECTION_H_
