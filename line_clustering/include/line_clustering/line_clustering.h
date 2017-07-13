#ifndef LINE_CLUSTERING_LINE_CLUSTERING_H_
#define LINE_CLUSTERING_LINE_CLUSTERING_H_

#include "line_clustering/common.h"
#include "line_detection/line_detection.h"
#include "line_detection/line_detection_inl.h"

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

// A class that performs clustering of features based on the kmediods algorithm.
// The advantage of this method is, that it can use a precomputed distance
// matrix, that stores the distance between all nodes. This means, an arbitrary
// distance measure can be used.
class KMedoidsCluster {
 public:
  KMedoidsCluster();
  KMedoidsCluster(const cv::Mat& dist_mat, size_t K);
  void setDistanceMatrix(const cv::Mat& dist_mat);
  void setK(size_t K);
  // Run the clustering.
  void cluster();
  std::vector<size_t> getLabels();

 protected:
  // Initialize clustering.
  void init();
  // Assign every node to its nearest center.
  void assignDataPoints();
  // Within a cluster, choose the node as a center so that the sum of all
  // distances to this center is minimized.
  void reasssignMediods();
  // Reads out the distance matrix. Additionally guarantees that only the upper
  // triangle of the matrix is accessed.
  double dist(size_t i, size_t j);
  // Stores the cluster centers.
  std::vector<size_t> centers_;
  // For every features, this vector stores the index of the if its
  // corresponding center in the centers_ vector. By storing the vector index,
  // and not the actual node index, this labeling is guaranteed to be zero
  // based.
  std::vector<size_t> labels_;
  // Stores the clusters.
  std::vector<std::vector<size_t> > clusters_;
  // Number of clusters.
  size_t K_;
  // Distance matrix. dist_mat(i, j) denotes the distance between node i and j.
  // It is sufficient if the its a upper triangular matrix.
  cv::Mat dist_mat_;
  // Number of points equals number of nodes.
  size_t num_points_;
  bool k_set_, dist_mat_set_;
};
}  // namespace line_clustering

#include "line_clustering/line_clustering_inl.h"

#endif  // LINE_CLUSTERING_LINE_CLUSTERING_H_
