cmake_minimum_required(VERSION 2.8.3)
project(line_detection)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(PCL 1.8 REQUIRED)

cs_add_library(${PROJECT_NAME}
  src/line_detection.cc
)

add_executable(line_extractor_node src/line_extractor_node.cc)
target_link_libraries(line_extractor_node ${PROJECT_NAME})

add_executable(general_test general_tests.cc)
target_link_libraries(general_test ${PROJECT_NAME})

add_executable(send_to_detector send_to_detector.cc)
target_link_libraries(send_to_detector ${catkin_LIBRARIES})

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
