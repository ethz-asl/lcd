The following package is used to detect lines in 2D, backproject them in 3D using the depth information, fit planes around them, assign them line types and readjust them in 3D.
The core part of the package is ROS-independent. Two ROS nodes are added to allow usage of the package via ROS topics/services.

### Libraries
- **line_detection**: Library to detect lines in 2D/3D, readjust them, fit planes around them and assign them a line type. **Please set the variable `kLineToolsRootPath` in `include/common.h` to be the absolute path of the root folder of this repository (e.g., `/home/user/catkin_extended_ws/src/line_tools/`).**

  _Classes_:
  - `LineDetector`: Implements all the functionalities of the library.
    - Detects lines in 2D and fuses those that can be linked to one another;
    - Backprojects them in 3D;
    - Fits planes around them;
    - Readjusts them using inliers;
    - Assigns a type to the lines;
    - Performs checks on the lines;
    - Displays lines the extracted lines in 2D or 3D => Set `visualization_mode_on_` to `true`. The visualization of lines in 3D with the planes fitted around them is done via a Python script in the package `python`. This requires the variable `kLineToolsRootPath` (cf. above) to be set correctly.
    - Displays statistics about the extracted lines. => Set `verbose_mode_on_` to `true`.


### ROS nodes
- `src/line_extractor_node.cc`: Uses the complete line-detection pipeline (from 2D detection to line readjustment). Handles the ROS service `extract_lines`, by means of which the lines extracted can be retrieved without using auxiliary `.txt` files (as done in `line_ros_utility` instead).

- `src/detector_node.cc`: [_Currently not used_].

### ROS messages
- `msg/Line3DWithHessians.msg`: Stores a 3D line with the Hessian parameters of the two planes fitted around them and the line type;
- `msg/KeyLine.msg`: Used to handle the OpenCV struct `cv::line_descriptor::KeyLine` [_Used only as a comparison for_ `line_ros_utility/line_detect_describe_and_match`_, but it is not meant to be currently used_].

### ROS services
- `srv/ExtractLines.srv`: Given an image, a point cloud, camera info and a detector type, returns the line extracted from the image, both in 2D and 3D;
- `srv/ExtractKeyLines.srv`: Given an image, returns the `cv::line_descriptor::KeyLine`s detected. [_Used only as a comparison for_ `line_ros_utility/line_detect_describe_and_match`_, but it is not meant to be currently used_];
- `srv/RequestLineDetection.srv`: [_Currently not used_].
