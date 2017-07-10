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
  // Performs the clustering.
  void runLineMeans();
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

// // This class helps publishing several different clusters of lines in
// different
// // colors, so that they are visualized by rviz.
// // IMPORTANT: This function cannot display more clusters than there are
// colors
// //            defined in the constructor. If more clusters are given to the
// //            object, only the one with the highest labels are published.
// class DisplayClusters {
//  public:
//   DisplayClusters();
//   // Frame id of the marker message.
//   void setFrameID(const std::string& frame_id);
//
//   // Is used as input for the clusters to the class:
//   // lines3D:   Vector of 3D lines.
//   //
//   // labels:    Vector of equal size as lines3D. Every entry labels the
//   cluster
//   //            the 3D line with the same index belongs to. The lables should
//   be
//   //            continous ((0,1,2,3 -> good), (0,4,8,16 -> bad)), because the
//   //            highest label defines how many clusters are created (in the
//   //            latter case of the example 17 clusters will be created, but
//   only
//   //            4 will contain information).
//   void setClusters(const std::vector<cv::Vec6f>& lines3D,
//                    const std::vector<int>& labels);
//
//   // This functions advertises the message.
//   void initPublishing(ros::NodeHandle& node_handle);
//   void publish();
//
//  private:
//   bool frame_id_set_, clusters_set_, initialized_;
//   std::vector<visualization_msgs::Marker> marker_lines_;
//   std::vector<std::vector<cv::Vec6f> > line_clusters_;
//   std::vector<ros::Publisher> pub_;
//   std::string frame_id_;
//   std::vector<cv::Vec3f> colors_;
// };

}  // namespace line_clustering

#include "line_clustering/line_clustering_inl.h"

#endif  // LINE_CLUSTERING_LINE_CLUSTERING_H_
