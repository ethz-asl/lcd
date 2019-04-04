# Line Tools
The following repository contains packages to detect lines and use them for the purpose of place recognition. In particular, the pipeline for place recognition works as follows:
1. We detect lines in the RGB images from a training set and backproject them to 3D using depth information. We then fit two planes around the lines (using the points around them), readjust the lines in 3D (to cope with noise) and assign them a _line type_.
2. We label the lines in the training set by assigning them the ground-truth instance label of the object that they belong to.
3. For each line detected we obtain a virtual-camera image by reprojecting the point cloud on a virtual-camera image plane. The pose of the virtual camera depends on the line and on its type.
4. We feed the virtual-camera images (and, depending on the model, also the type of the line and geometric information, such as the center point and unit direction vector of the line) to a neural network. The latter is trained so as to minimize a _triplet loss_, in which triplets are formed on the basis of the ground-truth instance labels of the lines: for each _anchor_ line in a batch, another line in the batch is considered _positive_ if it is different from the anchor, but has its same ground-truth instance label, and _negative_ if it has a different ground-truth instance label. With this procedure, each line is assigned an _embedding_, that can obtained from the output of the neural network. If the optimization is performed correctly, the embeddings obtained should be such that clusters can be formed out of them, with **each cluster corresponding to a different instance**.
5. Steps 1 and 3 are repeated for a _test set_ and the virtual-camera images obtained are fed to the network previously-trained in step 4. The embeddings of the test set are also clustered.
6. _[TODO]_ Using a descriptor for the clusters obtained in the embeddings space, the clusters from the test set can be matched with those from the training set. The task of **place recognition reduces to finding scenes in the training set that have the same (or a similar) configuration of clusters as those in the test set**.



The repository consists of the following packages, *each of which is described in more detail in the associated folder*:
1. `line_detection`: Package to detect lines in 2D, backproject them in 3D using the depth information, fit planes around them, assign them line types and readjust them in 3D.
2. `line_description`: Package to describe lines (either with _neural-network embeddings_ or with a binary descriptor). It is a test-like package mainly developed to allow execution of the entire pipeline (from line extraction, to virtual-camera images extraction and retrieval of neural-network embeddings from a previously-trained model) 'online', i.e., without saving data to disk, but transferring them through ROS. Please note that virtual-camera images and embeddings can be retrieved also 'offline' (i.e., saving data to disk and subsequently reading them, without using ROS), by using the scripts in `python` (cf. below). The binary descriptor, instead, was only introduced as an initial test for comparison, but it is not meant to be currently used. Indeed, the latter is a regular feature descriptor, whereas _the neural-network embeddings are not descriptors for the line features, but are rather needed to define the clusters for the instances in the embedding space_ (cf. above).
3. `line_ros_utility`: Package containing ROS utilities mainly to allow extraction of lines from ROS bags generated from the SceneNetRGBD or SceneNN datasets (cf. [Datasets](#datasets)). The lines extracted are labelled with ground-truth instance labels and saved (with 2D and 3D info) for later use in the pipeline. The package also includes a node to create a histogram of the line lengths and one to launch a test 'online' pipeline to detect, describe and match lines.
4. `python`: Collection of Python scripts to serve several purposes:
  - Generate virtual-camera images for the detected lines;
  - Generate data (*pickle files*) that can be used to either train the neural network that generates the embeddings or to obtain the embeddings, using a previously-trained model;
  - Train the neural network that generates the embeddings;
  - 'Test' the neural network, i.e., retrieve the embeddings for a certain dataset;
  - Visualization.

The following 'auxiliary' packages are also part of the repository:
1. `line_clustering`: Deprecated package developed to perform clustering of lines.
2. `line_detection_python`: Currently-unused (and not up-to-date) C++-to-Python bindings of the package `line_detection`.
3. `line_matching`: Package to match lines extracted in one frame to lines extracted in another frame. Since the neural-network embeddings are not meant to be line-features descriptors (cf. above), one should currently not expect to obtain line-to-line matching, but rather (line from one instance)-to-(line from the same instance).


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
* [scenenet_ros_tools](https://github.com/ethz-asl/scenenet_ros_tools/) (make sure you use the [following version](https://github.com/ethz-asl/scenenet_ros_tools/pull/8))
* [scenenn_ros_tools](https://github.com/ethz-asl/scenenn_ros_tools/) (make sure you use the [following version](https://github.com/ethz-asl/scenenn_ros_tools/pull/4))
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

Build them (it could take 1 hour for the first time):
```
$ cd ~/catkin_ws/
$ catkin config --isolate-devel
$ caktin config -DCMAKE_BUILD_TYPE=Release
$ catkin build
```


## Datasets
- The main dataset used is `SceneNetRGBD`.  
Please download the [dataset](https://robotvault.bitbucket.io/scenenet-rgbd.html) (or at least the 'subset' `train_0`) and the protobuf files (available at the main page of the dataset, or, e.g., from [here](www.doc.ic.ac.uk/~ahanda/train_protobufs.tar.gz), for the training set). Also clone the repository [pySceneNetRGBD](https://github.com/jmccormac/pySceneNetRGBD), that contains scripts needed to handle the data from SceneNetRGBD. You may use the following steps (replace `<folder_where_the_data_will_be_stored>` with the path where you will store the data you download):
```bash
cd <folder_where_the_data_will_be_stored>
git clone https://github.com/jmccormac/pySceneNetRGBD.git
cd pySceneNetRGBD
mkdir data && cd data
wget http://www.doc.ic.ac.uk/~ahanda/train_split/train_0.tar.gz train_0.tar.gz
wget http://www.doc.ic.ac.uk/~ahanda/train_protobufs.tar.gz train_protobufs.tar.gz
tar -xvzf train_0.tar.gz train_protobufs.tar.gz
cd .. && make
```
- The pipeline is also configured to use the `SceneNN` dataset, although the latter has not been used in testing yet.  
Please download the [dataset](http://www.scenenn.net/) with _instance annotation_ (currently [this](https://drive.google.com/drive/folders/0B-aa7y5Ox4eZWE8yMkRkNkU4Tk0) is the direct link) or even only few scenes from it. You may want to follow the instructions on the [scenenn repository](https://github.com/scenenn/scenenn). Make sure you also download the `.oni` files, as well as the `intrinsic/` folder, and that the structure of your folder matches the one described in the `scenenn` repository. Use the tool in the `playback/` folder of the [scenenn repository](https://github.com/scenenn/scenenn) to extract depth and color images from the `.oni` files.

## Data generation
The following section explains how to obtain the data that should be fed to the neural-network, starting from the datasets downloaded above. In particular:
- ROS bags are generated from the datasets;
- Using the utilities in `line_ros_utility`, lines are extracted from the ROS bags previously generated and are saved to disk;
- The lines data previoùsly saved are used to generate the virtual-camera images, which are also saved to disk;
- Lines are converted from camera-frame coordinates to world-frame coordinates (using the camera-to-world matrices obtained from the datasets) and are split in a training, testing and validation set;
- Lines and virtual-camera images are compactly stored in _pickle files_ (one file for the training set, one file for the testing set and one for the validation set);
- All the generated data can be stored in a `.tar.gz` archive file.


Before starting generating the data, make sure that the following variables are set as indicated below:
- In `scenenet_ros_tools/nodes/scenenet_to_rosbag.py`, function `convert`:
  - `write_scene_pcl = True`
  - `write_rgbd = True`
  - `write_instances = True`
  - `write_instances_color = False`
  - You may also set `write_object_segments = False`, as object segments are not used for this project, but make the ROS bags heavier.
- In `scenenn_ros_tools/nodes/scenenn_to_rosbag.py`, function `publish`:
  - `publish_scene_pcl = True`
  - `publish_rgbd = True`
  - `publish_instances = True`
  - `publish_instances_color = False`
  - It is suggested that `publish_object_segments` be set to `False`, as object segments are not used for this project, but make the ROS bags much heavier and largely increase the time required for generation.

Data generation can be performed automatically by using the Bash script `generate_trajectory_files.sh`. Before using it, make sure you properly set the _paths_ and _variables_ in the two following configuration scripts (the meaning of each variable is explained in detail in the scripts themselves):
* **config_paths_and_variables.sh**: Stores the paths where the datasets are located, where the intermediate data generated by the 'offline' pipeline are saved and where the auxiliary scripts are located. This configuration script is required to coordinate the handling of paths for all the scripts and nodes in the pipeline (i.e., both C++ executables and Python scripts).  
_For the purpose of data generation_ it also contains two variables, that specify for which trajectory and which dataset the data should be generated:
  - `TRAJ_NUM`: index of the trajectory (called 'scene' in SceneNN) in the original dataset (e.g., `0`, `1`, `2`, etc. for any subset of SceneNetRGBD, or `005` or any other three-digit ID for SceneNN);
  - `DATASET_NAME`: name of the dataset that contains the trajectory.
    - For SceneNetRGBD: either `train_<NUM>` or `val` (where `<NUM>` is an integer between 0 and 16), indicates respectively one of the 17 subsets in which the original SceneNetRGBD training set is split or the original SceneNetRGBD validation set.
    - For SceneNN: `scenenn`.

    Please note that you need to have previously downloaded the dataset `DATASET_NAME` (as indicated in [Datasets](#datasets)). For SceneNetRGBD, you also need to store the corresponding protobuf file in `config_protobuf_paths` (cf. below).
* **config_protobuf_paths**: 'Dictionary' of the protobuf files for all the subsets of SceneNetRGBD that you intend to use. The information in the protobuf file is required to associate a trajectory index to its location on disk (called _render path_), as well as to retrieve the camera-to-world matrix.

After editing the two configuration files above, you can start generating the data with `generate_trajectory_files.sh`, as follows:
```bash
chmod +x generate_trajectory_files.sh
./generate_trajectory_files.sh
```


Data generation can also be performed manually by following the same steps as those in the script `generate_trajectory_files.sh`. However, this is not suggested, as the handling of the intermediate data generated, as well as of the right paths and variables to use, is rather intricate. If you intend to manually execute all the steps, make sure you use the correct arguments for each script (cf. usage in `generate_trajectory_files.sh` and description in the README of the packages).


### Training the model
To train the model with the data previously generated, as well as to obtain the embeddings for a test set and cluster them, please look at the package `python`.


### Utility files
Please look at `line_ros_utility` and `python` for a detailed explanation of the available utility files.
