#ifndef LINE_CLUSTERING_LINE_CLUSTERING_H_
#define LINE_CLUSTERING_LINE_CLUSTERING_H_

#include <ros/ros.h>

#include "line_clustering/common.h"
#include "line_detection/line_detection.h"

namespace line_clustering {

double computePerpendicularDistanceLines(const cv::Vec<float, 6>& line1,
                                         const cv::Vec<float, 6>& line2);

double computeSquareMeanDifferenceLines(const cv::Vec<float, 6>& line1,
                                        const cv::Vec<float, 6>& line2);

double computeSquareNearestDifferenceLines(const cv::Vec<float, 6>& line1,
                                           const cv::Vec<float, 6>& line2);

class KMeansCluster {
 public:
  KMeansCluster();
  KMeansCluster(const std::vector<cv::Vec<float, 6> >& lines3D);
  KMeansCluster(const std::vector<cv::Vec<float, 6> >& lines3D,
                unsigned int num_clusters);

  void setNumberOfClusters(unsigned int num_clusters);
  void computeMeansOfLines();
  void runOnMeansOfLines();
  std::vector<int> cluster_idx_;

 private:
  unsigned int K_;
  std::vector<cv::Vec3f> line_means_;
  std::vector<cv::Vec<float, 6> > lines_;
};
}  // namespace line_clustering

#include "line_clustering/line_clustering_inl.h"

#endif  // LINE_CLUSTERING_LINE_CLUSTERING_H_
