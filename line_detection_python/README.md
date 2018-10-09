# Python bindings

This repository is an example for generating python bindings for `line_detection` using **pybind11**.

The steps are quite straight-forward except the type conversion between `cv::Mat` and numpy array. Here the type conversion is borrowed from https://github.com/edmBernard/pybind11_opencv_numpy.

One can extend the functions to bind in `src/module.cc`.
