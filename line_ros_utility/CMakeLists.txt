cmake_minimum_required(VERSION 2.8.3)
project(line_ros_utility)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME}
  src/line_ros_utility.cc
)

add_executable(detect_cluster_show src/detect_cluster_show_node.cc)
target_link_libraries(detect_cluster_show ${PROJECT_NAME})

add_executable(scenenet_to_line_tools src/scenenet_to_line_tools_node.cc)
target_link_libraries(scenenet_to_line_tools ${PROJECT_NAME})

cs_install()
cs_export()
