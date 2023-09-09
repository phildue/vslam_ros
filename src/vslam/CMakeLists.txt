cmake_minimum_required(VERSION 3.8)
project(vslam VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools ${CMAKE_BINARY_DIR}/)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
set(VSLAM_CONFIG_DIR "${CMAKE_CURRENT_LIST_DIR}/config")
include(Macros)

set( sources
    src/core/Camera.cpp
    src/core/Frame.cpp
    src/core/Feature2D.cpp
    src/core/Point3D.cpp
    src/core/Trajectory.cpp
    src/core/Pose.cpp
    src/core/random.cpp
    src/utils/log.cpp
    src/utils/visuals.cpp
    src/utils/utils.cpp
    src/features/keypoint_selection.cpp
    #src/features/FeatureSelection.cpp
    src/features/overlays.cpp
    src/evaluation/evaluation.cpp
    src/evaluation/tum.cpp
    src/direct/odometry/AlignmentRgb.cpp
    src/direct/odometry/AlignmentRgbd.cpp
    src/direct/overlays.cpp
    src/direct/NormalEquations.cpp
    src/pose_prediction/ConstantVelocityModel.cpp
    src/descriptor_matching/bundle_adjustment/BundleAdjustment.cpp
    src/descriptor_matching/overlays.cpp
    src/descriptor_matching/FeatureTracking.cpp
    src/descriptor_matching/FeatureTrackingOcv.cpp
    src/descriptor_matching/Matcher.cpp
    src/pose_graph_optimization/PoseGraphOptimizer.cpp
    src/loop_closure_detection/CandidateProposal.cpp
    src/loop_closure_detection/DifferentialEntropy.cpp
    #src/mapping/Map.cpp
    /usr/local/include/easylogging++.cc
   )

set( headers
    src/core/Trajectory.h
    src/core/Pose.h
    src/core/Camera.h
    src/core/Frame.h
    src/core/Feature2D.h
    src/core/Point3D.h
    src/core/macros.h
    src/core/types.h
    src/core/random.h
    src/utils/utils.h
    src/utils/visuals.h
    src/utils/log.h
    src/features/keypoint_selection.h
    src/features/FeatureSelection.h
    src/features/overlays.h
    src/evaluation/evaluation.h
    src/evaluation/tum.h
    src/direct/NormalEquations.h
    src/direct/interpolate.h
    src/direct/weights.h
    src/direct/overlays.h
    src/direct/odometry/AlignmentRgb.h
    src/direct/odometry/AlignmentRgbd.h
    src/pose_prediction/ConstantVelocityModel.h
    src/descriptor_matching/bundle_adjustment/BundleAdjustment.h
    src/descriptor_matching/overlays.h
    src/descriptor_matching/FeatureTracking.h
    src/descriptor_matching/FeatureTrackingOcv.h
    src/descriptor_matching/Matcher.h
    src/pose_graph_optimization/PoseGraphOptimizer.h
    src/loop_closure_detection/CandidateProposal.h
    src/loop_closure_detection/DifferentialEntropy.h

    /usr/local/include/easylogging++.h 
    include/vslam/core.h
    include/vslam/direct.h 
    include/vslam/evaluation.h 
    include/vslam/utils.h 
    include/vslam/descriptor_matching.h
    include/vslam/features.h 
    include/vslam/pose_graph_optimization.h 
    include/vslam/loop_closure_detection.h 

  )
pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)
add_library(pd::vslam ALIAS vslam)

if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
  target_link_libraries(${PROJECT_NAME} pthread)

endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

# While format is not implemented in gcc we use this thirdparty
find_package(fmt 9.0.0 REQUIRED)
target_link_libraries(${PROJECT_NAME} fmt::fmt)


# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

#find_package(Matplot++ REQUIRED)
#target_link_libraries(${PROJECT_NAME} Matplot++::matplot)

find_package(TBB REQUIRED)
target_link_libraries(${PROJECT_NAME} TBB::tbb)

find_package(Ceres REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})


target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_FEATURE_PERFORMANCE_TRACKING)
target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_PERFORMANCE_MICROSECONDS)
option(VSLAM_LOG_PERFORMANCE_TRACKING ON)
if(NOT VSLAM_LOG_PERFORMANCE_TRACKING)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_PERFORMANCE_TRACKING)
endif()
option(VSLAM_LOG_MINIMAL Off)
if(VSLAM_LOG_MINIMAL)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_PERFORMANCE_TRACKING)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_DEBUG_LOGS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_INFO_LOGS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC LOG_DISABLE_IMAGE_LOGS)

endif()


# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
  SET(VSLAM_TEST_BINARY_DIR "${CMAKE_BINARY_DIR}/test" CACHE PATH "Location of test binaries")
  set(VSLAM_TEST_RESOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/test/resource" CACHE PATH "Location of test resources")
  enable_testing()
  find_package(GTest CONFIG REQUIRED)
  option(VSLAM_TEST_VISUALIZE "Enable test visualizations." ON)
  if(VSLAM_TEST_VISUALIZE)
      set(TEST_VISUALIZE true)
  else()
      set(TEST_VISUALIZE false)
  endif()
  pd_add_experiment(rgbd_odometry_simple ${CMAKE_CURRENT_LIST_DIR}/test/direct/rgbd_odometry_simple.cpp)
  pd_add_experiment(rgb_odometry_simple ${CMAKE_CURRENT_LIST_DIR}/test/direct/rgb_odometry_simple.cpp)
  pd_add_experiment(rgbd_odometry_entropy ${CMAKE_CURRENT_LIST_DIR}/test/direct/rgbd_odometry_entropy.cpp)
  
  pd_add_experiment(direct_icp ${CMAKE_CURRENT_LIST_DIR}/test/direct/test_direct_icp.cpp)

  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test/pose_graph_optimization)

  pd_add_experiment(rgbd_odometry_local_map ${CMAKE_CURRENT_LIST_DIR}/test/direct/rgbd_odometry_local_map.cpp)
  
  pd_add_experiment(matching ${CMAKE_CURRENT_LIST_DIR}/test/descriptor_matching/test_matching.cpp)
  pd_add_experiment(descriptor_odometry ${CMAKE_CURRENT_LIST_DIR}/test/descriptor_matching/test_odometry.cpp)
  pd_add_experiment(match_and_optimize ${CMAKE_CURRENT_LIST_DIR}/test/descriptor_matching/test_match_and_optimize.cpp)

  pd_add_test(frame ${CMAKE_CURRENT_LIST_DIR}/test/core/test_frame.cpp)
  pd_add_test(camera ${CMAKE_CURRENT_LIST_DIR}/test/core/test_camera.cpp)
  pd_add_test(random ${CMAKE_CURRENT_LIST_DIR}/test/core/test_random.cpp)
  pd_add_test(trajectory ${CMAKE_CURRENT_LIST_DIR}/test/core/test_trajectory.cpp)

  pd_add_test(ba ${CMAKE_CURRENT_LIST_DIR}/test/bundle_adjustment/test_ba.cpp)
  pd_add_test(ceres_ba ${CMAKE_CURRENT_LIST_DIR}/test/bundle_adjustment/test_ceres_ba.cpp)
  pd_add_test(jacobians ${CMAKE_CURRENT_LIST_DIR}/test/direct/test_jacobians.cpp)
endif()

# Installation
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
    
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION include/
    FILES_MATCHING # install only matched files
    PATTERN "*.h")# select header files

install(EXPORT ${PROJECT_NAME}Targets
    DESTINATION lib/cmake/${PROJECT_NAME}
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE pd::
    DESTINATION share/${PROJECT_NAME}/cmake)

include(CMakePackageConfigHelpers)

write_basic_package_version_file( ${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

# install export target and config for find_package
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/tools/${PROJECT_NAME}Config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "share/${PROJECT_NAME}/cmake/"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION "share/${PROJECT_NAME}/cmake/")


