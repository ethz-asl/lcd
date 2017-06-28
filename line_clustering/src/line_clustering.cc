#include "line_clustering/line_clustering.h"

namespace line_clustering {
double computePerpendicularDistanceLines(const cv::Vec<float, 6>& line1,
                                         const cv::Vec<float, 6>& line2) {
  cv::Vec3f start1(line1[0], line1[1], line1[2]);
  cv::Vec3f end1(line1[3], line1[4], line1[5]);
  cv::Vec3f start2(line2[0], line2[1], line2[2]);
  cv::Vec3f end2(line2[3], line2[4], line2[5]);
  cv::Vec3f temp_vec = crossProduct(start1 - end1, start2 - end2);
  double divisor = normOfVector3D(temp_vec);
  if (divisor < 1e-6) {
    return line_detection::distPointToLine(start1, end1, start2);
  } else {
    return fabs(scalarProduct(temp_vec, start2 - start1)) / divisor;
  }
}

double computeSquareMeanDifferenceLines(const cv::Vec<float, 6>& line1,
                                        const cv::Vec<float, 6>& line2) {
  return pow((line1[0] + line1[3]) / 2 - (line2[0] + line2[3]) / 2, 2) +
         pow((line1[1] + line1[4]) / 2 - (line2[1] + line2[4]) / 2, 2) +
         pow((line1[2] + line1[5]) / 2 - (line2[2] + line2[5]) / 2, 2);
}

double computeSquareNearestDifferenceLines(const cv::Vec<float, 6>& line1,
                                           const cv::Vec<float, 6>& line2) {
  double dist[4];
  dist[0] = pow(line1[0] - line2[0], 2) + pow(line1[1] - line2[1], 2) +
            pow(line1[2] - line2[2], 2);
  dist[1] = pow(line1[0] - line2[3], 2) + pow(line1[1] - line2[4], 2) +
            pow(line1[2] - line2[5], 2);
  dist[2] = pow(line1[3] - line2[0], 2) + pow(line1[4] - line2[1], 2) +
            pow(line1[5] - line2[2], 2);
  dist[3] = pow(line1[3] - line2[3], 2) + pow(line1[4] - line2[4], 2) +
            pow(line1[5] - line2[5], 2);
  double min_dist = dist[0];
  for (int i = 1; i < 4; ++i)
    if (dist[i] < min_dist) min_dist = dist[i];
  return min_dist;
}

}  // namespace line_clustering
