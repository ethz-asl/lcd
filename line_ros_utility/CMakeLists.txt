cmake_minimum_required(VERSION 2.8.3)
project(line_ros_utility)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME}
  src/line_ros_utility.cc
)

cs_add_library(line_detect_describe_and_match
  src/line_detect_describe_and_match.cc
)

cs_add_library(histogram_line_lengths_builder
  src/histogram_line_lengths_builder.cc
)

catkin_install_python(PROGRAMS
  nodes/histogram_line_lengths_builder_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

add_executable(detect_and_save_lines src/detect_and_save_lines_node.cc)
target_link_libraries(detect_and_save_lines ${PROJECT_NAME})

add_executable(scenenet_to_line_tools src/scenenet_to_line_tools_node.cc)
target_link_libraries(scenenet_to_line_tools ${PROJECT_NAME})

add_executable(scenenn_to_line_tools src/scenenn_to_line_tools_node.cc)
target_link_libraries(scenenn_to_line_tools ${PROJECT_NAME})

add_executable(freiburg_to_line_tools src/freiburg_to_line_tools_node.cc)
target_link_libraries(freiburg_to_line_tools ${PROJECT_NAME})

add_executable(matching_visualizer_node src/matching_visualizer_node.cc)
target_link_libraries(matching_visualizer_node line_detect_describe_and_match)

add_executable(histogram_line_lengths_node src/histogram_line_lengths_node.cc)
target_link_libraries(histogram_line_lengths_node histogram_line_lengths_builder)

cs_install()
cs_export()
