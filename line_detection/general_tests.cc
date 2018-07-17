#include <line_detection/line_detection.h>

#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "general_test");
  ros::NodeHandle node_handle;

  int algorithm = 0;
  if (argc < 2 || argc > 3) {
    ROS_INFO("usage: general_test <imagepath> <algorithm>");
    return -1;
  }
  if (argc == 3) {
    algorithm = atoi(argv[2]);
  }

  std::string path = argv[1];
  cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  if (!image.data) {
    ROS_INFO(
        "The path \"%s\" is not a valid image. Please pass a valid path as "
        "an argument to the function.",
        path.c_str());
    return -1;
  }

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;

  cv::Mat img_gray;
  cvtColor(image, img_gray, CV_BGR2GRAY);

  line_detection::LineDetector line_detector;
  std::vector<cv::Vec4f> lines2D;
  std::vector<cv::Vec4f> lines2D_fused;

  start = std::chrono::system_clock::now();
  line_detector.detectLines(img_gray, algorithm, &lines2D);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  ROS_INFO("Time to detect: %f", elapsed_seconds.count());

  start = std::chrono::system_clock::now();
  line_detector.fuseLines2D(lines2D, &lines2D_fused);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  ROS_INFO("Time to fuse: %f", elapsed_seconds.count());
  ROS_INFO("Number of old lines: %d", (int)lines2D.size());
  ROS_INFO("Number of new lines: %d", (int)lines2D_fused.size());

  // Save cropped images for each line (fused)
  const bool save_cropped_images = false;

  if(save_cropped_images){

    line_detection::getCroppedImageForLines2D(lines2D_fused, image);
    // for(size_t i = 0; i < (int)lines2D_fused.size(); ++i){
    //   cv::Point2f top_left(lines2D_fused[i][0], lines2D_fused[i][1]);
    //   cv::Point2f bottom_right(lines2D_fused[i][2], lines2D_fused[i][3]);
    //   //
    //   // cv::Point2f top_left(320, 240);
    //   // cv::Point2f bottom_right(0, 0);
    //   //
    //   // std::cout << lines2D_fused[i][0] << "  " << lines2D_fused[i][1] << " " << lines2D_fused[i][2] << " " <<lines2D_fused[i][3] << std::endl;
    //   cv::Rect line_ROI(top_left, bottom_right);
    //   cv::Mat cropped_image = image(line_ROI);
    //   cv::imwrite("/home/chengkun/InternASL/catkin_ws/src/line_tools/data/lines_cropped_images/line_" + std::to_string(i) + ".jpg", cropped_image);
    // }
  }

  cv::Mat image_new = image.clone();
  line_detector.paintLines(lines2D, &image);
  line_detector.paintLines(lines2D_fused, &image_new, cv::Vec3b(0, 255, 0));
  cv::imshow("original lines", image);
  cv::imshow("fused lines", image_new);
  cv::waitKey();
  return 0;
}
