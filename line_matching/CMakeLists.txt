cmake_minimum_required(VERSION 2.8.3)
project(line_matching)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME}
  src/line_matching.cc
)

catkin_add_gtest(test_line_matching test/test_line_matching.cc)
target_link_libraries(test_line_matching ${PROJECT_NAME} pthread)

cs_install()
cs_export()
