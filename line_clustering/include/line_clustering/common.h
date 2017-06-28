#ifndef LINE_CLUSTERING_COMMON_H_
#define LINE_CLUSTERING_COMMON_H_

#include <stddef.h>
#include <opencv2/core.hpp>

namespace line_clustering {

inline double normOfVector3D(const cv::Vec3f& vector) {
  return sqrt(pow(vector[0], 2) + pow(vector[1], 2) + pow(vector[2], 2));
}

inline void normalizeVec3D(cv::Vec3f& vector) {
  vector = vector / normOfVector3D(vector);
}

inline cv::Vec3f crossProduct(const cv::Vec3f a, const cv::Vec3f b) {
  return cv::Vec3f(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0]);
}

inline double scalarProduct(const cv::Vec3f& a, const cv::Vec3f& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

}  // namespace line_clustering

#endif  // LINE_CLUSTERING_COMMON_H_
