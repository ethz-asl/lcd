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

KMeansCluster::KMeansCluster() {
  lines_set_ = false;
  k_set_ = false;
};
KMeansCluster::KMeansCluster(const std::vector<cv::Vec<float, 6> >& lines3D) {
  lines_ = lines3D;
  lines_set_ = true;
  k_set_ = false;
};
KMeansCluster::KMeansCluster(const std::vector<cv::Vec<float, 6> >& lines3D,
                             unsigned int num_clusters) {
  lines_ = lines3D;
  K_ = num_clusters;
  lines_set_ = true;
  k_set_ = true;
}

void KMeansCluster::setNumberOfClusters(unsigned int num_clusters) {
  K_ = num_clusters;
  k_set_ = true;
}
void KMeansCluster::setLines(const std::vector<cv::Vec<float, 6> >& lines3D) {
  lines_ = lines3D;
  lines_set_ = true;
}
void KMeansCluster::setLines(
    const std::vector<line_detection::LineWithPlanes>& lines3D) {
  lines_.resize(lines3D.size());
  for (size_t i = 0; i < lines3D.size(); ++i) lines_.push_back(lines3D[i].line);
  lines_set_ = true;
}

void KMeansCluster::computeMeansOfLines() {
  CHECK(lines_set_)
      << "You have to set the lines before computing their means.";
  line_means_.clear();
  line_means_.reserve(lines_.size());
  for (size_t i = 0; i < lines_.size(); ++i) {
    line_means_.push_back(cv::Vec3f((lines_[i][0] + lines_[i][3]) / 2,
                                    (lines_[i][1] + lines_[i][4]) / 2,
                                    (lines_[i][2] + lines_[i][5]) / 2));
  }
}
void KMeansCluster::runOnMeansOfLines() {
  if (lines_.size() != line_means_.size()) computeMeansOfLines();
  cv::kmeans(line_means_, K_, cluster_idx_,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              100, 0.01),
             3, cv::KMEANS_PP_CENTERS);
}

std::vector<cv::Vec<float, 6> > KMeansCluster::getLines() { return lines_; }

DisplayClusters::DisplayClusters() {
  colors_.push_back({1, 0, 0});
  colors_.push_back({0, 1, 0});
  colors_.push_back({0, 0, 1});
  colors_.push_back({1, 1, 0});
  colors_.push_back({1, 0, 1});
  colors_.push_back({0, 1, 1});
  colors_.push_back({1, 0.5, 0});
  colors_.push_back({1, 0, 0.5});
  colors_.push_back({0.5, 1, 0});
  colors_.push_back({0, 1, 0.5});
  colors_.push_back({0.5, 0, 1});
  colors_.push_back({0, 0.5, 1});

  frame_id_set_ = false;
  clusters_set_ = false;
  initialized_ = false;
}

void DisplayClusters::setFrameID(const std::string& frame_id) {
  frame_id_ = frame_id;
  frame_id_set_ = true;
}

void DisplayClusters::setClusters(
    const std::vector<cv::Vec<float, 6> >& lines3D,
    const std::vector<int>& labels) {
  CHECK_EQ(lines3D.size(), labels.size());
  size_t N = 0;
  line_clusters_.clear();
  for (size_t i = 0; i < lines3D.size(); ++i) {
    // This if clause sets the number of clusters. This works well as long the
    // clusters are indexed as an array (0,1,2,3). In any other case, it creates
    // to many clusters (which is not that bad, because empty clusters do not
    // need a lot of memory nor a lot of time to allocate), but if one label is
    // higher than the number of colors defined in the constructor (which
    // defines the number of labels that can be displayed), some clusters might
    // not be displayed.
    if (labels[i] >= N) {
      N = 1 + labels[i];
      line_clusters_.resize(N);
    }
    if (labels[i] < 0) {
      ROS_WARN_ONCE(
          "line_clustering::DisplayClusters::setClusters: A negative label has "
          "been detected and ignored.");
      continue;
    }
    line_clusters_[labels[i]].push_back(lines3D[i]);
  }
  marker_lines_.resize(line_clusters_.size());
  size_t n;
  for (size_t i = 0; i < line_clusters_.size(); ++i) {
    n = i % colors_.size();
    line_detection::storeLines3DinMarkerMsg(line_clusters_[i], marker_lines_[i],
                                            colors_[n]);
    marker_lines_[i].header.frame_id = frame_id_;
  }
  clusters_set_ = true;
}

void DisplayClusters::initPublishing(ros::NodeHandle& node_handle) {
  pub_.resize(colors_.size());
  size_t n;
  for (size_t i = 0; i < colors_.size(); ++i) {
    std::stringstream topic;
    topic << "/visualization_marker_" << i;
    pub_[i] =
        node_handle.advertise<visualization_msgs::Marker>(topic.str(), 1000);
  }
  initialized_ = true;
}

void DisplayClusters::publish() {
  CHECK(initialized_)
      << "You need to call initPublishing to advertise before publishing.";
  CHECK(frame_id_set_) << "You need to set the frame_id before publishing.";
  CHECK(clusters_set_) << "You need to set the clusters before publishing.";
  int n;
  for (size_t i = 0; i < marker_lines_.size(); ++i) {
    n = i % pub_.size();
    pub_[n].publish(marker_lines_[i]);
  }
}

}  // namespace line_clustering
