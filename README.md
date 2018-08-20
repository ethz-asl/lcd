# Line Tools

The repository contains two main parts:
1. ROS packages for detecting lines in rgb-d image.
2. Python scripts for processing lines data and training on the data to get embeddings for detected lines.

## Build instructions for ROS packages
You may need to install `python_catkin_tools` and `autoconf`:
```
$ sudo apt-get install python-catkin-tools
$ sudo apt-get install autoconf
```
Creat a catkin workspace:
```
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws
$ catkin init
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
