# Line Tools

Toolbox containing packages for line detection and clustering. The main functionality is within the two packages line\_detection and line\_clustering. The two libraries implemented in this do only depend on the standard library, OpenCV and glog. But there are some ros nodes within the line\_detection package that depend on ROS.
All other ROS dependencies are implemented within a third library in the package line\_ros\_utility. This includes conversion for different datasets, file input, communication with python nodes, displaying in rviz and putting the actual work together in a pipeline.

## Running the pipeline
There are two launch files within the line\_ros\_utility package:
+ **detect\_cluster\_show.launch**: Needs the package [scenenet\_ros\_tools](https://github.com/ethz-asl/scenenet_ros_tools) installed to work properly.
+ **freiburg.launch**: Works when run with a rosbag from the freiburg data set.

For both of these nodes, the script random\_forest.py must be running. This script takes input data as it is written by the line\_ros\_utility::printToFile() function. You can change the kWritePath parameter in line\_ros\_utility.cc to the path where you want to store these files. Then let the main node run on a data set (we used the SceneNet, see above) where there exist ground truth instances images (depicting objects). Now you just have to make sure that the random\_forest.py reads these files correctly.

## Build instructions
You may need to install `python_catkin_tools` and `autoconf`:
```
$ sudo apt-get install python-catkin-tools
$ sudo apt-get install autoconf
```
Creat a catkin workspace:
```
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
$ source devel/setup.bash   # overlay this workspace on top of your environment
```

`git clone` the following packages to your `src` directory in catkin workspace `catkin_ws`

* catkin_simple
* eigen_catkin
* eigen_checks
* gflags_catkin
* glog_catkin
* line_tools
* opencv3_catkin
* pcl_catkin

You also need to modify the following two lines of `CMakeLists.txt` in `opencv3_catkin` directory:
```
-DBUILD_opencv_line_descriptor=ON
...
-DBUILD_opencv_ximgproc=ON
```

Build them (it could take 1 hour for the first time):
```
$ cd ~/catkin_ws/
$ caktin config -DCMAKE_BUILD_TYPE=Release
$ catkin build
```
