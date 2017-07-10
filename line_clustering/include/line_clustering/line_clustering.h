#ifndef LINE_CLUSTERING_LINE_CLUSTERING_H_
#define LINE_CLUSTERING_LINE_CLUSTERING_H_

#include "line_clustering/common.h"
#include "line_detection/line_detection.h"

namespace line_clustering {

double computePerpendicularDistanceLines(const cv::Vec6f& line1,
                                         const cv::Vec6f& line2);

double computeSquareMeanDifferenceLines(const cv::Vec6f& line1,
                                        const cv::Vec6f& line2);

double computeSquareNearestDifferenceLines(const cv::Vec6f& line1,
                                           const cv::Vec6f& line2);

// A class that performs clustering of lines with kmeans.
class KMeansCluster {
 public:
  KMeansCluster();
  KMeansCluster(const std::vector<line_detection::LineWithPlanes>& lines3D);
  KMeansCluster(const std::vector<line_detection::LineWithPlanes>& lines3D,
                unsigned int num_clusters);
  KMeansCluster(const std::vector<cv::Vec6f>& lines3D);
  KMeansCluster(const std::vector<cv::Vec6f>& lines3D,
                unsigned int num_clusters);

  void setNumberOfClusters(unsigned int num_clusters);
  void setLines(const std::vector<cv::Vec6f>& lines3D);
  void setLines(const std::vector<line_detection::LineWithPlanes>& lines3D);
  // Computes the means of the lines that are used to cluster them.
  void computeLineMeans();
  // Performs the clustering on the means of lines.
  void runLineMeans();
  // This initializes clustering with the additional information of the planes
  // adjecent to the lines.
  void initClusteringWithHessians(double scale_hessians);
  void runOnLinesAndHessians();
  // Returns the lines.
  std::vector<cv::Vec6f> getLines();
  // This array contains the labels of the lines.
  std::vector<int> cluster_idx_;

 private:
  bool lines_set_, k_set_, hessians_set_, cluster_with_hessians_init_ = false;
  unsigned int K_;
  double mean_;
  std::vector<cv::Vec3f> line_means_;
  std::vector<cv::Vec6f> lines_;
  std::vector<cv::Vec<float, 8> > hessians_;
  std::vector<cv::Vec<float, 14> > lines_and_hessians_;
};

}  // namespace line_clustering

#include "line_clustering/line_clustering_inl.h"

#endif  // LINE_CLUSTERING_LINE_CLUSTERING_H_
