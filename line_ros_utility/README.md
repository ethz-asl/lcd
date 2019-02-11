The following package contains ROS utilities mainly to make the first steps of the place-recognition pipeline compact: lines are extracted and labelled with ground-truth instance labels. The extracted lines with 2D and 3D info are saved to disk for later use in the pipeline.

### Libraries

* **line_ros_utility**: Main library to generate the data about the lines detected for later use in the place recognition pipeline.

  _Classes_:
  - `ListenAndPublish`: Main class.
    - Uses the utilities from the package `line_detection` to detect lines and project them to 3D.
    - Discards/performs checks on the lines.
    - Clusters them [_Currently not used. NOTE: The clustering is now performed differently_].
    - Labels lines with ground-truth instance labels.
    - Displays lines with the pixels that have the same instance label in the ground-truth instance image => Set `labelled_line_visualization_mode_on_` to `true`.
    - Displays the points that are inliers to the planes fitted around the line =>
    Set `inliers_visualization_mode_on_` to `true`.
    - Displays detailed statistics about the lines detected => Set `verbose_mode_on_` to `true`.
    - Saves the lines, with their 2D/3D info, line types and instance label to file. Three types of `.txt` files are generated:
      - `lines_with_labels_<num_iteration>.txt`: Contains a line for each detected line, in the following format (22 total values per line):

        `<Start point in 3D (3x)> <End point in 3D (3x)> <Hessian parameter of first inlier plane (4x)> <Hessian parameter of second inlier plane (4x)> <Colour assigned to first inlier plane (3x)> <Colour assigned to second inlier plane (3x)> <Line type (1x)> <Instance label (1x)>`

      - `lines_2D_<num_iteration>.txt`: Contains a line for each 2D line detected, in the following format (4 total values per line):

        `<Start point in 2D (2x)> <End point in 2D (2x)>`

      - `lines_2D_kept_<num_iteration>.txt`: Contains a line for each 2D line kept after the checks performed, in the following format (4 total values per line):

        `<Start point in 2D (2x)> <End point in 2D (2x)>`

  - `DisplayClusters`: Publishes lines for visualization in RViz.

  - `InliersWithLabels`: Handles inlier points with their labels. Needed to retrieve the ground-truth label associated to a line.

  - `TreeClassifier`: [_Currently not used_].

  - `EvalData`: [_Currently not used_].

* **line_detect_describe_and_match**: 'Test' library to handle the entire pipeline with ROS, from line detection, virtual-camera image generation, retrieval of embeddings from a previoùsly-trained network and matching of lines (_the latter not coherent with the current implementation of the embeddings anymore_).

  _Classes_:
  - `LineDetectorDescriptorAndMatcher`: Implements all the functionalities of the class.


* **histogram_line_lengths_builder**: Library to handle the entire line-detection pipeline with ROS to build a histogram of the lengths of the lines.

  _Classes_:
  - `HistogramLineLengthsBuilder`: Implements all the functionalities of the class.



### ROS nodes
* **src/detect\_and\_save\_lines\_node.cc**: Simply creates an instance of the class `ListenAndPublish` from `src/line_ros_utility.cc` with the correct parameters.

  _Input arguments_:
  - `{1}`: Number (index in the original dataset) of the trajectory from which to generate data;
  - `{2}`: Path where to store the output data (`.txt` files containing line info);
  - `{3}`: Index of the start frame. Needed because ROS bags from SceneNN start from frame 2 instead of frame 0 (cf. `scenenn_ros_tools/scenenn_to_rosbag.py`).
  - `{4}:` Frame step. It indicates the step (in number of frames of the original trajectory) between a frame in the ROS bag and the subsequent one. For instance, with `frame_step = 3` and `start_frame = 1`, the actual indices of the frames received are 1, 4, 7, etc.


* **src/matching_visualizer_node.cc**: Simply creates an instance of the class `LineDetectorDescriptorAndMatcher` from `src/line_detect_describe_and_match.cc` with the correct parameters.

  _Input arguments_:
  - `{1}`: Detector type (0 -> LSD, 1 -> EDL, 2 -> FAST, 3 -> HOUGH).
  - `{2}`: Descriptor type (0 -> Neural-network embeddings, 1 -> Binary descriptor).


* **src/histogram_line_lengths_node.cc**: Simply creates an instance of the class `HistogramLineLengthsBuilder` from `histogram_line_lengths_builder` with the correct parameters.

  _Input arguments_:
  - `{1}`: Detector type (0 -> LSD, 1 -> EDL, 2 -> FAST, 3 -> HOUGH).


* **nodes/histogram_line_lengths_builder_node.py**: Subscribes to the topic `/line_lengths` and creates a histogram with the length of the lines by using an instance of the class `LineLengthHistogram` from `nodes/tools/histogram_line_lengths.py`.

* **src/scenenet_to_line_tools_node.cc**: Subscribes to the topics from a `scenenet_ros_tools` ROS bag (see below) and republishes the topics in a friendly format for `line_tools` (i.e., it republishes the point cloud, the RGB image, the depth image, the camera info, the instance image and the camera-to-world matrix).

* **src/scenenn_to_line_tools_node.cc**: Subscribes to the topics from a `scenenn_ros_tools` ROS bag (see below) and republishes the topics in a friendly format for `line_tools` (i.e., it republishes the point cloud, the RGB image, the depth image, the camera info, the instance image and the camera-to-world matrix).


* **src/freiburg_to_line_tools_node.cc**: [_Currently not used_]


### ROS launch files (`launch/` folder)
The package contains four launch files:
* **detect\_and\_save\_lines.launch**: Detects lines,  backprojects them in 3D using the depth information, fits planes around them, assigns them line types and readjusts them in 3D and labels them with ground-truth instance labels.
  - The lines obtained can be saved as `.txt` files for later use in the pipeline, by setting `write_labeled_lines` to `true` in `src/line_ros_utility.cc`. The path where the output is stored is defined by the `write_path` argument of the launch file.
  - Lines can be also displayed as they get extracted. Open RViz after starting the launch file to display the lines overlapped with the point cloud frame by frame and coloured by line type.

  Works when run with a ROS bag generated by  [scenenet\_ros\_tools](https://github.com/ethz-asl/scenenet_ros_tools) or by [scenenn\_ros\_tools](https://github.com/ethz-asl/scenenn_ros_tools) (make sure `scenenet_ros_tools` complies with the version at https://github.com/ethz-asl/scenenet_ros_tools/pull/8 and that `publish_instances_color` is set to `False` in `nodes/scenenet_to_rosbag.py`).
* **detect\_describe\_and\_match.launch**: Example _online_ pipeline that detects (and backprojects, readjusts, etc.) lines,  with the _data being solely transferred through ROS_, i.e., without saving `.txt` files to disk. Currently, the 'final' part of this pipeline performs matching between the lines based on the embeddings retrieved. However, this does not correspond to the current implementation of the embeddings, that are not feature descriptors, but are rather needed to form clusters associated to instances in the embedding space. For these reasons, the last part might be replaced and used for any other purposes, by simply calling the same services and subscribing to the same topics as the current pipeline.  
  Works when run with a ROS bag generated by  [scenenet\_ros\_tools](https://github.com/ethz-asl/scenenet_ros_tools) or by [scenenn\_ros\_tools](https://github.com/ethz-asl/scenenn_ros_tools) (make sure `scenenet_ros_tools` complies with the version at https://github.com/ethz-asl/scenenet_ros_tools/pull/8 and that `publish_instances_color` is set to `False` in `nodes/scenenet_to_rosbag.py`).

* **build_histogram_line_lengths.launch**: _Online_ pipeline that detects (and backprojects, readjusts, etc.) lines and creates a histogram of the lengths of the lines.  
  Works when run with a ROS bag generated by  [scenenet\_ros\_tools](https://github.com/ethz-asl/scenenet_ros_tools) or by [scenenn\_ros\_tools](https://github.com/ethz-asl/scenenn_ros_tools) (make sure `scenenet_ros_tools` complies with the version at https://github.com/ethz-asl/scenenet_ros_tools/pull/8 and that `publish_instances_color` is set to `False` in `nodes/scenenet_to_rosbag.py`).


* **freiburg.launch**: Works when run with a ROS bag from the Freiburg data set. _[Currently not used]_.
