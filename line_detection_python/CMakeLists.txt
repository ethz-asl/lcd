cmake_minimum_required(VERSION 2.8.3)
project(line_detection_python)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(PythonLibs 2 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(src/pybind11/include)
include_directories(/usr/local/lib/python2.7/dist-packages/numpy/core/include)

SET(SOURCES
  src/module.cc
  src/ndarray_converter.cc
)

cs_add_library(py_line_detection MODULE  ${SOURCES})
target_link_libraries(py_line_detection ${PYTHON_LIBRARIES})
set_target_properties(py_line_detection PROPERTIES PREFIX "")

cs_install()
cs_export()
