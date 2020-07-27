# Line Tools
The following repository contains packages to detect lines and use them for the purpose of place recognition. In particular, the pipeline for place recognition works as follows:
1. We detect lines in the RGB images from a training set and backproject them to 3D using depth information. We then fit two planes around the lines (using the points around them), readjust the lines in 3D (to cope with noise) and assign them a _line type_.
2. We label the lines in the training set by assigning them the ground-truth instance label of the object that they belong to.
3. For each line detected we obtain a virtual-camera image by reprojecting the point cloud on a virtual-camera image plane. The pose of the virtual camera depends on the line and on its type.
4. We feed the virtual-camera images (and, depending on the model, also the type of the line and geometric information, such as the center point and unit direction vector of the line) to a neural network. The latter is trained so as to minimize a _triplet loss_, in which triplets are formed on the basis of the ground-truth instance labels of the lines: for each _anchor_ line in a batch, another line in the batch is considered _positive_ if it is different from the anchor, but has its same ground-truth instance label, and _negative_ if it has a different ground-truth instance label. With this procedure, each line is assigned an _embedding_, that can obtained from the output of the neural network. If the optimization is performed correctly, the embeddings obtained should be such that clusters can be formed out of them, with **each cluster corresponding to a different instance**.
5. Steps 1 and 3 are repeated for a _test set_ and the virtual-camera images obtained are fed to the network previously-trained in step 4. The embeddings of the test set are also clustered.
6. _[TODO]_ Using a descriptor for the clusters obtained in the embeddings space, the clusters from the test set can be matched with those from the training set. The task of **place recognition reduces to finding scenes in the training set that have the same (or a similar) configuration of clusters as those in the test set**.



The repository consists of the following packages, *each of which is described in more detail in the associated folder*:
1. `line_detection`: Package to detect lines in 2D, backproject them in 3D using the depth information, fit planes around them and readjust them in 3D. 
2. `line_description`: Package to describe lines (either with _neural-network embeddings_ or with a binary descriptor). It is a test-like package mainly developed to allow execution of the entire pipeline (from line extraction, to virtual-camera images extraction and retrieval of neural-network embeddings from a previously-trained model) 'online', i.e., without saving data to disk, but transferring them through ROS. Please note that the virtual-camera images and embeddings can currently only be retrieved 'offline' (i.e., saving data to disk and subsequently reading them, without using ROS), by using the scripts in `python` (cf. below). The binary descriptor, instead, was only introduced as an initial test for comparison, but it is not meant to be used currently. Indeed, the latter is a regular feature descriptor, whereas _the neural-network embeddings are not descriptors for the line features, but are rather needed to define the clusters for the instances in the embedding space_ (cf. above). This package is currently not in use.
3. `line_ros_utility`: Package containing ROS utilities mainly to allow extraction of lines and geometric information from ROS messages generated from the InteriorNet dataset (cf. [Datasets](#datasets)). The lines extracted are labelled with ground-truth instance labels and saved (with 2D and 3D info) for later use in the pipeline. The package also includes a node to create a histogram of the line lengths and one to launch a test 'online' pipeline to detect, describe and match lines.
4. `python`: Collection of Python scripts to serve several purposes:
  - Generate virtual-camera images for the detected lines;
  - Generate data using the full pipeline for a large amount of InteriorNet scenes;
  - Train the neural network for clustering of lines;
  - Train the neural network for the description of clusters;
  - Evaluate the full place recognition pipeline;
  - Visualization.

The following 'auxiliary' packages are also part of the repository:
1. `line_clustering`: Deprecated package developed to perform clustering of lines.
2. `line_detection_python`: Currently-unused (and not up-to-date) C++-to-Python bindings of the package `line_detection`.
3. `line_matching`: Deprecated package to match lines extracted in one frame to lines extracted in another frame. 


## Build instructions for ROS packages
You may need to install `python_catkin_tools` and `autoconf`:
```
$ sudo apt-get install python-catkin-tools
$ sudo apt-get install autoconf
```
Create a catkin workspace:
```
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws
$ catkin init
```

`git clone` the following packages to your `src` directory in catkin workspace `catkin_ws`

* [catkin_simple](https://github.com/catkin/catkin_simple)
* [eigen_catkin](https://github.com/ethz-asl/eigen_catkin)
* [eigen_checks](https://github.com/ethz-asl/eigen_checks)
* [gflags_catkin](https://github.com/ethz-asl/gflags_catkin)
* [glog_catkin](https://github.com/ethz-asl/glog_catkin/)
* [line_tools](https://github.com/ethz-asl/line_tools/)
* [opencv3_catkin](https://github.com/ethz-asl/opencv3_catkin/)
* [pcl_catkin](https://github.com/ethz-asl/pcl_catkin/)
* [interior_net_to_rosbag](https://github.com/ethz-asl/interiornet_to_rosbag)
* [vision_opencv](https://github.com/ethz-asl/vision_opencv)

You also need to edit the file `CMakeLists.txt` in the `opencv3_catkin` directory by:
- modifying the following three lines:
```
-DBUILD_opencv_python3=ON
...
-DBUILD_opencv_line_descriptor=ON
...
-DBUILD_opencv_ximgproc=ON
```
- adding the following line:
```
-DBUILD_opencv_nonfree=OFF
```
- adding the following two libraries in the list under `cs_export`:
```
opencv_line_descriptor
opencv_ximgproc
```

(For a safer operation, just make sure that `opencv3_catkin` points to commit [bd876bc](https://github.com/ethz-asl/opencv3_catkin/commit/bd876bcf5ad393190e6c771be3f19e9e0df6470d)).

Build them (it could take up to 1 hour for the first time):
```
$ cd ~/catkin_ws/
$ catkin config --isolate-devel
$ caktin config -DCMAKE_BUILD_TYPE=Release
$ catkin build
```


## Datasets
- The main dataset used is `InteriorNet`.  
Please download the [dataset](https://interiornet.org/), specifically the HD7 scenes. Not all scenes have to be downloaded. In the python/dataset_utils directory there is a python script (`download_interiornet.py`) that automatically downloads scenes from the HD7 directory. The path were the dataset is stored can be set in the script.
- Other datasets can be used as well, as long as they are converted into the InteriorNet format. As an example, scripts for converting the DIML dataset and NYU dataset are located in the python/dataset_utils/ directory. The files used from the InteriorNet scene directories include `cam0.render` and the images in `cam0/data/` (or `random_lighting_cam0/data/`, if set in python/generate_raw_data.py), `label0/data/`, `depth0/data/`. Note that cam0.render is currently only used to determine the frame count for the `interiornet_to_rosbag` node. Also, no ground truth instancing labels are required for the place recognition pipeline to work. However, "fake" instance and semantic masks need to be created in the label0/data/ directory. Lastly, the InteriorNet dataset depth images are stored as euclidean ray lengths from the camera center. If the depth data is stored in the z coordinate (as it is the case with the NYU and DIML dataset), it needs to be converted accordingly.

## Data generation
The following section explains how to obtain the data that is fed to the neural-network, starting from the dataset downloaded above. In particular:
- The dataset is published as ROS messages;
- Using the utilities in `line_ros_utility`, lines and their geometric information are extracted from the ROS messages previously generated and are saved to disk;
- The lines data previoùsly saved are used to generate the virtual-camera images, which are also saved to disk;
- After some post processing, the geometric information and the virtual-camera image of the line are merged and saved to disk. Each directory corresponds to one scene from the dataset. These files are used for training later;

Data generation can be performed automatically by using the bash script `generate_data.sh`. Before using it, make sure you properly set the _paths_ and _variables_ in the python/**generate_raw_data.py** script file (the meaning of each variable is explained in detail in the scripts themselves). Note that the dataset needs to be downloaded prior to this step.


### Training the model
To train the models with the data previously generated and to perform place recognition experiments, please take a look at the package `python`.


### Utility files
Please look at `line_ros_utility` and `python` for a detailed explanation of the available utility files.
