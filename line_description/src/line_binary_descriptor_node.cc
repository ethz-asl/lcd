#include <line_description/line_description.h>

#include "line_description/KeyLineToBinaryDescriptor.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <ros/ros.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

line_description::LineDescriber line_describer_(
    line_description::DescriptorType::BINARY);

bool callback(line_description::KeyLineToBinaryDescriptor::Request& req,
              line_description::KeyLineToBinaryDescriptor::Response& res) {
  cv::line_descriptor::KeyLine keyline;
  cv::Mat image;
  cv_bridge::CvImageConstPtr cv_image_ptr;
  line_description::Descriptor descriptor;
  // Convert input line into KeyLine.
  keyline.angle = req.keyline.angle;
  keyline.class_id = req.keyline.class_id;
  keyline.endPointX = req.keyline.endPointX;
  keyline.endPointY = req.keyline.endPointY;
  keyline.ePointInOctaveX = req.keyline.ePointInOctaveX;
  keyline.ePointInOctaveY = req.keyline.ePointInOctaveY;
  keyline.lineLength = req.keyline.lineLength;
  keyline.numOfPixels = req.keyline.numOfPixels;
  keyline.octave = req.keyline.octave;
  keyline.pt.x = req.keyline.pt.x;
  keyline.pt.y = req.keyline.pt.y;
  keyline.response = req.keyline.response;
  keyline.size = req.keyline.size;
  keyline.sPointInOctaveX = req.keyline.sPointInOctaveX;
  keyline.sPointInOctaveY = req.keyline.sPointInOctaveY;
  keyline.startPointX = req.keyline.startPointX;
  keyline.startPointY = req.keyline.startPointY;
  // Convert input image into OpenCV image.
  cv_image_ptr = cv_bridge::toCvCopy(req.image, "rgb8");
  image = cv_image_ptr->image;
  // Retrieve descriptor.
  line_describer_.describeLine(keyline, image, &descriptor);
  // Send descriptor as response.
  for (size_t i = 0; i < res.descriptor.size(); ++i) {
    res.descriptor[i] = descriptor[i];
  }
  return true;
}

int main(int argc, char** argv) {
  // Initialize node.
  ros::init(argc, argv, "line_binary_descriptor");
  ros::NodeHandle node_handle;
  ros::ServiceServer server_keyline_to_binary_descriptor_;
  // Initialize service server.
  server_keyline_to_binary_descriptor_ =
     node_handle.advertiseService("keyline_to_binary_descriptor", callback);
  ros::spin();
}
