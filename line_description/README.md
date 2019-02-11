The following package is used to 'describe' lines. This can be meant in two different ways:
- Assigning a regular line-feature descriptor to each line [_Not the current purpose of this project, as we want to have embeddings that can be used to form clusters associated to instances, cf. main README_];
- Retrieving an **embedding** associated to each line, by obtaining the virtual-camera image associated to the line and feeding this image to a previously-trained neural network. *NOTE: This can be done 'offline', i.e., without transferring data between ROS nodes that run concurrently, but by storing them to disk => Use the utilities in the* `python` *package*.  
This functionality is currently _ROS-dependent_ and depends on Python scripts and rospy nodes. Combined with the nodes from the other packages, it allows to run the entire pipeline (from line detection to retrieval of the embeddings) 'online'.

### Libraries
- **line_description**: Library to 'describe' the lines, as explained above.

  _Classes_:
  - `LineDescriber`: Similarly what `LineDetector` from `line_detection` does for line detection, it allows to 'describe' lines with different descriptors:
    - A binary descriptor from OpenCV (`cv::line_descriptor::KeyLine`) [_Used only as a comparison for_ `line_ros_utility/line_detect_describe_and_match`_, but not meant to be currently used_];
    - Neural-network embeddings. [_Currently not implemented in C++. TODO: Python-to-C++ bindings would need to be generated_].


### ROS nodes
- `src/line_binary_descriptor_node.cc`: Handles the ROS service `keyline_to_binary_descriptor`, by means of which a detected `cv::line_descriptor::KeyLine` can be associated to its binary descriptor. [_Not meant to be currently used, cf. above_];

- `nodes/image_to_embeddings_node.py`: Handles the ROS service `image_to_embeddings`, by means an embedding can be retrieved from a line and its virtual-camera image, using a previously-trained network. Set `log_files_folder` properly, as well as the checkpoint and meta file path, to choose the previously-trained model to use;

- `nodes/line_to_virtual_camera_image_node.py`: Handles the ROS service `line_to_virtual_camera_image`, by means of which a virtual-camera image can be associated to an input line.

### ROS services
- `srv/EmbeddingsRetrieverReady.srv`: Internal service, used by the embedding-retriever node to inform the main node that the previously-trained model has been loaded and that embeddings can therefore be retrieved;
- `srv/ImageToEmbeddings.srv`: Given the virtual-camera image (both color- and depth-), the line type and the endpoints of a line (in camera-frame coordinates), as well as the camera-to-world matrix, returns the embedding associated to the line;
- `srv/KeyLineToBinaryDescriptor.srv`: Given a detected `cv::line_descriptor::KeyLine`, as well as the image from which the line was extracted, returns the associated 32-dimensional binary descriptor [_Not meant to be currently used, cf. above_];
- `srv/LineToVirtualCameraImage.srv`: Given a detected line in 3D, with the planes fitted around it and its line type, as well as the color image and the point cloud from which the line was extracted, returns the virtual-camera images (both color- and depth-) associated to the line.
