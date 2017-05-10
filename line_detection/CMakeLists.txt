cmake_minimum_required(VERSION 2.8.3)
project(line_detection)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME}
  src/line_detection.cc
)

add_executable(detector detector.cc)
target_link_libraries(detector ${PROJECT_NAME} pthread)

add_executable(send_to_detector send_to_detector.cc)
target_link_libraries(send_to_detector ${catkin_LIBRARIES})

add_executable(test_point_cloud_view test_point_cloud_view.cc)
target_link_libraries(test_point_cloud_view ${PROJECT_NAME} pthread)


add_custom_target(test_data)
add_custom_command(TARGET test_data
                   COMMAND rm -rf test_data
                   COMMAND mkdir -p test_data
                   COMMAND cp -r ${CMAKE_SOURCE_DIR}/../test_data/line_detection/*
                                 test_data/ || :)

catkin_add_gtest(test_line_detection test/test_line_detection.cc)
target_link_libraries(test_line_detection ${PROJECT_NAME} pthread)
add_dependencies(test_line_detection test_data)


cs_install()
cs_export()
