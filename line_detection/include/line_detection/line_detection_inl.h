#ifndef LINE_DETECTION_LINE_DETECTION_INL_H_
#define LINE_DETECTION_LINE_DETECTION_INL_H_

#include "line_detection/line_detection.h"

#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>

#include <functional>
#include <queue>
#include <set>

namespace line_detection {

// This function samples a specific number of unique elements from an array.
// Input: in:         From this vector the elements are sampled.
//        N:          The number of samples taken.
//        generator:  A random engine that is handled to the uniform number
//                    sampler (std::uniform_int_distribution). It is used as an
//                    input here, so that the seed may be set outside of this
//                    function. This is useful if you run the function in a fast
//                    loop. Proposed engine: std::default_random_engine.
//
// Output: out:       A vector containing N unique samples of in.
template <typename T>
void getNUniqueRandomElements(const std::vector<T>& in, size_t num_samples,
                              std::default_random_engine* generator,
                              std::vector<T>* out) {
  CHECK_NOTNULL(generator);
  CHECK_NOTNULL(out);
  CHECK(in.size() > num_samples);
  out->clear();
  out->reserve(num_samples);
  // The algorithm uses the Fisher-Yates Shuffle to guarantee that no element is
  // sampled twice.
  size_t max = in.size();
  int idx;
  // From this array the indices of an element is sampled.
  int indices[max];
  for (size_t i = 0; i < max; ++i) {
    indices[i] = i;
  }
  std::uniform_int_distribution<int>* distribution;
  for (size_t i = max; i > max - num_samples; --i) {
    distribution = new std::uniform_int_distribution<int>(0, i - 1);
    idx = (*distribution)(*generator);
    out->push_back(in[indices[idx]]);
    indices[idx] = indices[i - 1];
  }
  delete distribution;
}

// An overload, that allows the use without specifyng an random engine. Be
// careful if this is used in a loop (because the seed might be the same for
// very small time differences between calls).
template <typename T>
void getNUniqueRandomElements(const std::vector<T>& in, size_t num_samples,
                              std::vector<T>* out) {
  unsigned seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  getNUniqueRandomElements(in, num_samples, &generator, out);
}

// Union-find data structure for efficient clustering of the points used in
// planeRANSAC.
class ClusterUnionFind {
 public:
   ClusterUnionFind(double distance_threshold) {
     distance_threshold_ = distance_threshold;
   }
   // Clears the entire structure.
   void clear() {
     points_.clear();
     parents_.clear();
     ranks_.clear();
   }

   // Adds a point to the union-find data structure.
   void addPoint(cv::Vec3f point) {
     size_t point_index = points_.size();
     points_.push_back(point);
     parents_.push_back(point_index);
     ranks_.push_back(1);
     // Compare the point to all previous points and merges sets if two points
     // are at a distance less than the specified threshold.
     for (size_t i = 0; i < point_index; ++i) {
       if (cv::norm(points_[i] - point) < distance_threshold_) {
         // Points should belong to the same set.
         unionSets(i, point_index);
       }
     }
   }

   // Counts the number of separated sets.
   size_t countSeparatedSets() {
     std::set<size_t> separated_sets;
     for (size_t i = 0; i < points_.size(); ++i) {
       separated_sets.insert(findSet(i));
     }
     return separated_sets.size();
   }

 private:
   // Stores the points.
   std::vector<cv::Vec3f> points_;
   // Stores the parent of each point.
   std::vector<size_t> parents_;
   // Stores the rank of each set.
   std::vector<size_t> ranks_;
   // Threshold to classify two points as belonging to the same set.
   double distance_threshold_;

   // Finds the (index of the) set to which the point with index idx belongs.
   size_t findSet(size_t idx) {
     CHECK(idx < points_.size());
     if (parents_[idx] == idx) {
       return idx;
     } else {
       return (parents_[idx] = findSet(parents_[idx]));
     }
   }

   // Merges the sets to which the points with indices index_point_1 and
   // index_point_2 belong.
   void unionSets(size_t index_point_1, size_t index_point_2) {
     CHECK(index_point_1 < points_.size());
     CHECK(index_point_2 < points_.size());
     size_t index_set_1 = findSet(index_point_1);
     size_t index_set_2 = findSet(index_point_2);
     if (index_set_1 != index_set_2) {
       // The two sets are yet to be merged.
       if (ranks_[index_set_1] >= ranks_[index_set_2]) {
         // Merge set 2 into set 1.
         parents_[index_set_2] = index_set_1;
         ranks_[index_set_1] += ranks_[index_set_2];
       } else {
         // Merge set 1 into set 2.
         parents_[index_set_1] = index_set_2;
         ranks_[index_set_2] += ranks_[index_set_1];
       }
     }
   }
};

// Priority-queue-like structure to cluster the points based on their sorted
// pairwise distances.
class ClusterPriorityQueue {
 public:
   ClusterPriorityQueue(double distance_threshold) {
     distance_threshold_ = distance_threshold;
   }
   // Clears the entire structure.
   void clear() {
     while(!distances_.empty()) {
       distances_.pop();
     }
     points_.clear();
   }

   // Adds a point to the queue data structure.
   void addPoint(cv::Vec3f point) {
     size_t point_index = points_.size();
     points_.push_back(point);
     // Compute the distance from all previous points and insert the computed
     // distance into the priority queue.
     for (size_t i = 0; i < point_index; ++i) {
       distances_.push(cv::norm(points_[i] - point));
     }
   }

   // Counts the number of separated sets.
   size_t countSeparatedSets() {
     if (points_.size() == 0) {
       return 0;
     }
     // At least one component.
     size_t num_separated_sets = 1;
     double current_distance;
     // Obtain first element.
     double previous_distance = distances_.top();
     distances_.pop();
     while(!distances_.empty()) {
       // Obtains current distance.
       current_distance = distances_.top();
       distances_.pop();
       if (previous_distance - current_distance > distance_threshold_) {
         // The difference in distance is such that they identify two separated
         // components.
         num_separated_sets++;
       }
       previous_distance = current_distance;
     }

     return num_separated_sets;
   }

 private:
   // Stores the pairwise distances in descending order.
   std::priority_queue<double> distances_;
   // Stores the points.
   std::vector<cv::Vec3f> points_;
   // Threshold for two distances to still identify the same cluster.
   double distance_threshold_;
};

// Octree structure to cluster the points based on their pairwise distances.
class ClusterOctree {
 public:
   ClusterOctree(double distance_threshold) {
     distance_threshold_ = distance_threshold;
     // Initialize the point cloud.
     cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(
         new pcl::PointCloud<pcl::PointXYZ>);
     // Initialize the octree.
     octree_ = new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> (
         distance_threshold_);
   }
   ~ClusterOctree() {
     delete octree_;
   }
   // Clears the entire structure.
   void clear() {
     cloud_->clear();
     octree_->deleteTree();
   }

   // Adds a point to the union-find data structure.
   void addPoints(std::vector<cv::Vec3f>& points) {
     // Clear the point cloud.
     cloud_->clear();
     // Resize the point cloud.
     cloud_->width = points.size();
     cloud_->height = 1;
     cloud_->points.resize(cloud_->width * cloud_->height);
     // Add the points to the cloud.
     for (size_t i = 0; i < points.size(); ++i) {
       cloud_->points[i].x = points[i][0];
       cloud_->points[i].y = points[i][1];
       cloud_->points[i].z = points[i][2];
     }
     // Add points to octree.
     octree_->setInputCloud(cloud_);
     octree_->addPointsFromInputCloud();
   }

   // True if the points form a single connected component, false otherwise.
   bool singleConnectedComponent() {
     // Partly based on
     // http://www.pointclouds.org/documentation/tutorials/octree.php.

     // Set each point to be non-visited yet.
     std::vector<bool> visited(cloud_->points.size(), false);
     size_t num_non_visited_points = cloud_->points.size();
     std::queue<size_t> queue;
     std::vector<int> point_indices;
     std::vector<float> point_distances;
     size_t current_point_index;
     // Start visiting trasversing the tree from any point in the cloud and keep
     // doing so until nearby points are found.
     queue.push(0);
     while(!queue.empty()) {
       current_point_index = queue.front();
       queue.pop();
       if (!visited[current_point_index]) {
         visited[current_point_index] = true;
         num_non_visited_points--;
         // Find neighbouring points and push them into the queue if they have
         // not been visited.
         if (octree_->radiusSearch(cloud_->points[current_point_index],
                                  distance_threshold_, point_indices,
                                  point_distances) > 0) {
           for (auto& index : point_indices) {
             if (!visited[index]) {
               queue.push(index);
             }
           }
         }
       }
     }
     return num_non_visited_points == 0;
   }

 private:
   // Stores the octree.
   pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>* octree_;
   // Stores the points.
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
   // Threshold for two distances to still identify the same cluster.
   double distance_threshold_;
};

// Priority-queue-like structure to cluster the points based on their
// distances from their mean point.
class ClusterDistanceFromMean {
 public:
   ClusterDistanceFromMean(double distance_threshold) {
     distance_threshold_ = distance_threshold;
   }
   // Clears the entire structure.
   void clear() {
     while(!distances_.empty()) {
       distances_.pop();
     }
   }

   // Adds the points to the data structure.
   void addPoints(std::vector<cv::Vec3f>& points) {
     // Compute mean point.
     cv::Vec3f mean = {0.0f, 0.0f, 0.0f};
     for (auto& point : points) {
       mean += (point / float(points.size()));
     }
     // Compute the distance of all points from the mean point and insert them
     // in the priority queue.
     for (auto& point : points) {
       distances_.push(cv::norm(mean - point));
     }
   }

   // True if the points form a single connected component, false otherwise.
   bool singleConnectedComponent() {
     if (distances_.size() == 0) {
       return false;
     }
     double current_distance;
     // Obtain first element.
     double previous_distance = distances_.top();
     distances_.pop();
     // If the smallest distance from the mean point is more than 10 cm, then
     // the points do not form a single connected component.
     if (previous_distance > 0.1) {
       return false;
     }
     while(!distances_.empty()) {
       // Obtains current distance.
       current_distance = distances_.top();
       distances_.pop();
       if (current_distance - previous_distance > distance_threshold_) {
         // The difference in distance is such that they identify two separated
         // components.
         return false;
       }
       previous_distance = current_distance;
     }
     return true;
   }

 private:
   // Stores the pairwise distances in ascending order.
   std::priority_queue<double, std::vector<double>, std::greater<double>>
       distances_;
   // Threshold for two distances to still identify the same cluster.
   double distance_threshold_;
};

}  // namespace line_detection

#endif  // LINE_DETECTION_LINE_DETECTION_INL_H_
