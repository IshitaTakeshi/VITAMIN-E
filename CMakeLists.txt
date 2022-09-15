cmake_minimum_required(VERSION 3.14)

project(curvature)

add_compile_options(-pg -g -O3 -Wall -Wextra -Wpedantic -Werror)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(include)

add_library(curvature_lib src/curvature.cpp)
add_executable(curvature src/main.cpp)

target_link_libraries(curvature_lib ${OpenCV_LIBS} ${GTEST_LIBRARIES} Eigen3::Eigen)
target_link_libraries(curvature curvature_lib)

function(add_testcase filepath)
  get_filename_component(filename ${filepath} NAME)
  string(REGEX REPLACE ".cpp" "" test_name ${filename})

  add_executable(${test_name} ${filepath})
  target_link_libraries(${test_name} curvature_lib GTest::gtest_main)
  gtest_discover_tests(${test_name})
endfunction()

set(TEST_FILES tests/test_curvature.cpp)

foreach(filepath ${TEST_FILES})
  add_testcase(${filepath})
endforeach()
