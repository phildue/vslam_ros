cmake_minimum_required(VERSION 3.8)
project(least_squares VERSION 1.0.0)

set( sources
  src/Loss.cpp
  src/Scaler.cpp
  src/NormalEquations.cpp
  src/GaussNewton.cpp )

set( headers
  src/GaussNewton.h
  src/NormalEquations.h
  src/Scaler.h
  src/Prior.h
  src/Loss.h
  src/Solver.h
  include/least_squares/least_squares.h )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)
target_link_libraries(${PROJECT_NAME} core utils)

if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

option(VSLAM_LOG_MINIMAL Off)
if(NOT VSLAM_LOG_MINIMAL)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_DEBUG_LOGS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_INFO_LOGS)
endif()
# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
  enable_testing()

  find_package(GTest CONFIG REQUIRED)

  pd_add_test(loss least_squares)

  option(VSLAM_TEST_VISUALIZE "Enable visuals in unit test" Off)
  if(VSLAM_TEST_VISUALIZE)
    target_compile_definitions(lossTest PUBLIC TEST_VISUALIZE)
  endif()
endif()

# Install
install( DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cfg/
    DESTINATION share/vslam/cfg/
    FILES_MATCHING # install only matched files
    PATTERN "*" )# select all files