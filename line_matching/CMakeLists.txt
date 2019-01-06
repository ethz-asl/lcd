cmake_minimum_required(VERSION 2.8.3)
project(line_matching)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME}
  src/line_matching.cc
)

cs_install()
cs_export()
