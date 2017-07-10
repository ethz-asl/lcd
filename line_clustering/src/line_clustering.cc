#include "line_clustering/line_clustering.h"

namespace line_clustering {

double computePerpendicularDistanceLines(const cv::Vec6f& line1,
                                         const cv::Vec6f& line2) {
  cv::Vec3f start1(line1[0], line1[1], line1[2]);
  cv::Vec3f end1(line1[3], line1[4], line1[5]);
  cv::Vec3f start2(line2[0], line2[1], line2[2]);
  cv::Vec3f end2(line2[3], line2[4], line2[5]);
  cv::Vec3f temp_vec = (start1 - end1).cross(start2 - end2);
  const double divisor = cv::norm(temp_vec);
  constexpr double min_divisor = 1e-6;
  if (divisor < min_divisor) {
    return line_detection::distPointToLine(start1, end1, start2);
  } else {
    return fabs(temp_vec.dot(start2 - start1)) / divisor;
  }
}

double computeSquareMeanDifferenceLines(const cv::Vec6f& line1,
                                        const cv::Vec6f& line2) {
  return pow((line1[0] + line1[3]) / 2 - (line2[0] + line2[3]) / 2, 2) +
         pow((line1[1] + line1[4]) / 2 - (line2[1] + line2[4]) / 2, 2) +
         pow((line1[2] + line1[5]) / 2 - (line2[2] + line2[5]) / 2, 2);
}

double computeSquareNearestDifferenceLines(const cv::Vec6f& line1,
                                           const cv::Vec6f& line2) {
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
  for (size_t i = 1; i < 4; ++i) {
    if (dist[i] < min_dist) {
      min_dist = dist[i];
    }
  }
  return min_dist;
}

KMeansCluster::KMeansCluster() {
  lines_set_ = false;
  k_set_ = false;
  hessians_set_ = false;
}
KMeansCluster::KMeansCluster(
    const std::vector<line_detection::LineWithPlanes>& lines3D) {
  setLines(lines3D);
  k_set_ = false;
}
KMeansCluster::KMeansCluster(
    const std::vector<line_detection::LineWithPlanes>& lines3D,
    unsigned int num_clusters) {
  setLines(lines3D);
  setNumberOfClusters(num_clusters);
}
KMeansCluster::KMeansCluster(const std::vector<cv::Vec6f>& lines3D) {
  lines_ = lines3D;
  lines_set_ = true;
  k_set_ = false;
  hessians_set_ = false;
};
KMeansCluster::KMeansCluster(const std::vector<cv::Vec6f>& lines3D,
                             unsigned int num_clusters) {
  lines_ = lines3D;
  K_ = num_clusters;
  lines_set_ = true;
  k_set_ = true;
  hessians_set_ = false;
}

void KMeansCluster::setNumberOfClusters(unsigned int num_clusters) {
  K_ = num_clusters;
  k_set_ = true;
}
void KMeansCluster::setLines(const std::vector<cv::Vec6f>& lines3D) {
  lines_ = lines3D;
  lines_set_ = true;
}
void KMeansCluster::setLines(
    const std::vector<line_detection::LineWithPlanes>& lines3D) {
  lines_.resize(lines3D.size());
  hessians_.resize(lines3D.size());
  int n;
  for (size_t i = 0; i < lines3D.size(); ++i) {
    lines_[i] = lines3D[i].line;
    if (lines3D[i].hessians.size() == 2) {
      n = 1;
    } else {
      n = 0;
    }
    hessians_[i] = {lines3D[i].hessians[0][0], lines3D[i].hessians[0][1],
                    lines3D[i].hessians[0][2], lines3D[i].hessians[0][3],
                    lines3D[i].hessians[n][0], lines3D[i].hessians[n][1],
                    lines3D[i].hessians[n][2], lines3D[i].hessians[n][3]};
  }
  lines_set_ = true;
  hessians_set_ = true;
}

void KMeansCluster::computeLineMeans() {
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

void KMeansCluster::runLineMeans() {
  if (lines_.size() != line_means_.size()) {
    computeLineMeans();
  }
  CHECK(k_set_) << "You need to set K before clustering.";
  if (K_ >= line_means_.size()) {
    cluster_idx_ = std::vector<int>(line_means_.size(), 0);
    return;
  }
  constexpr size_t max_iter = 100;
  constexpr double epsilon = 0.01;
  constexpr size_t num_attempts = 3;
  cv::kmeans(line_means_, K_, cluster_idx_,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              max_iter, epsilon),
             num_attempts, cv::KMEANS_PP_CENTERS);
}

void KMeansCluster::initClusteringWithHessians(double scale_hessians) {
  CHECK(lines_set_ && hessians_set_)
      << "You have to set the lines before initializing clustering.";
  CHECK(lines_.size() == hessians_.size());
  mean_ = 0;
  for (size_t i = 0; i < lines_.size(); ++i) {
    for (int j = 0; j < 6; ++j) mean_ += lines_[i][j];
  }
  mean_ = mean_ / (double)(lines_.size() * 6);
  lines_and_hessians_.resize(lines_.size());
  for (size_t i = 0; i < lines_.size(); ++i) {
    for (int j = 0; j < 6; ++j) {
      lines_and_hessians_[i][j] = lines_[i][j] / mean_;
    }
    for (int j = 0; j < 8; ++j) {
      lines_and_hessians_[i][j + 6] = hessians_[i][j] * mean_ * scale_hessians;
    }
  }
  cluster_with_hessians_init_ = true;
}

void KMeansCluster::runOnLinesAndHessians() {
  if (!cluster_with_hessians_init_) initClusteringWithHessians(0.5);
  CHECK(k_set_) << "You need to set K before clustering.";
  constexpr size_t max_iter = 100;
  constexpr double epsilon = 0.01;
  constexpr size_t num_attempts = 3;
  if (K_ >= lines_and_hessians_.size()) {
    cluster_idx_ = std::vector<int>(lines_and_hessians_.size(), 0);
    return;
  }
  cv::kmeans(lines_and_hessians_, K_, cluster_idx_,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              max_iter, epsilon),
             num_attempts, cv::KMEANS_PP_CENTERS);
}

std::vector<cv::Vec6f> KMeansCluster::getLines() { return lines_; }

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

void DisplayClusters::setClusters(const std::vector<cv::Vec6f>& lines3D,
                                  const std::vector<int>& labels) {
  CHECK_EQ(lines3D.size(), labels.size());
  CHECK(frame_id_set_) << "line_clustering::DisplayClusters::setClusters: You "
                          "need to set the frame_id before setting the "
                          "clusters.";
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
    CHECK(labels[i] >= 0) << "line_clustering::DisplayClusters::setClusters: "
                             "Negative lables are not allowed.";
    line_clusters_[labels[i]].push_back(lines3D[i]);
  }
  marker_lines_.resize(line_clusters_.size());
  size_t n;
  for (size_t i = 0; i < line_clusters_.size(); ++i) {
    n = i % colors_.size();
    line_detection::storeLines3DinMarkerMsg(line_clusters_[i],
                                            &marker_lines_[i], colors_[n]);
    marker_lines_[i].header.frame_id = frame_id_;
  }
  clusters_set_ = true;
}

void DisplayClusters::initPublishing(ros::NodeHandle& node_handle) {
  pub_.resize(colors_.size());
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
  size_t n;
  for (size_t i = 0; i < marker_lines_.size(); ++i) {
    n = i % pub_.size();
    pub_[n].publish(marker_lines_[i]);
  }
}

}  // namespace line_clustering
