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
}  // namespace line_clustering

#include "line_clustering/line_clustering_inl.h"

#endif  // LINE_CLUSTERING_LINE_CLUSTERING_H_
