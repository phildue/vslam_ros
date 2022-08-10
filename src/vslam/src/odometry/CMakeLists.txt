cmake_minimum_required(VERSION 3.8)
project(odometry VERSION 1.0.0)

set(sources
        src/MotionPrediction.cpp
        src/KeyFrameSelection.cpp
        src/Map.cpp
        src/Odometry.cpp
        src/IterativeClosestPoint.cpp
        src/IterativeClosestPointOcv.cpp
        src/SE3Alignment.cpp
        src/RgbdAlignmentOpenCv.cpp
        src/BundleAdjustment.cpp
        src/FeatureTracking.cpp
        src/MapOptimization.cpp
        )

set(headers
        src/MotionPrediction.h
        src/KeyFrameSelection.h
        src/Odometry.h
        src/IterativeClosestPoint.h
        src/IterativeClosestPointOcv.h
        src/SE3Alignment.h
        src/RgbdAlignmentOpenCv.h
        src/BundleAdjustment.h
        src/FeatureTracking.h
        src/MapOptimization.h
        include/odometry/odometry.h
        )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

target_compile_definitions(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:ODOMETRY_CFG_DIR="${CMAKE_CURRENT_LIST_DIR}/cfg">
$<INSTALL_INTERFACE:ODOMETRY_CFG_DIR="/share/vslam/cfg">)

target_link_libraries(${PROJECT_NAME} kalman lukas_kanade utils core)

target_link_libraries(${PROJECT_NAME} core utils least_squares)
if (UNIX)
	target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

find_package(PCL REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${PCL_INCLUDE_DIRS})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

find_package(Ceres REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})

# Add tests
option(VSLAM_ODOMETRY_BUILD_TESTS "Enable to build odometry tests" ON)
if(VSLAM_ODOMETRY_BUILD_TESTS)
	enable_testing()
        find_package(GTest CONFIG REQUIRED)

        pd_add_test(frame ${PROJECT_NAME})
        target_compile_definitions(frameTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")

        pd_add_test(alignment_se3 ${PROJECT_NAME})
        target_compile_definitions(alignment_se3Test PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")

        pd_add_test(alignment_se3_multi_frame ${PROJECT_NAME})
        target_compile_definitions(alignment_se3_multi_frameTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")


        pd_add_test(odometry_icp ${PROJECT_NAME})
        target_compile_definitions(odometry_icpTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")

        pd_add_test(odometry_rgbd_cv ${PROJECT_NAME})
        target_compile_definitions(odometry_rgbd_cvTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")

        pd_add_test(trajectory ${PROJECT_NAME})
        target_compile_definitions(trajectoryTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")
       
        pd_add_test(odometry ${PROJECT_NAME})
        target_compile_definitions(odometry PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")
        
        pd_add_test(ceres_ba ${PROJECT_NAME})
        target_compile_definitions(ceres_baTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")

        pd_add_test(ba ${PROJECT_NAME})
        target_compile_definitions(baTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")
        
        pd_add_test(tracking ${PROJECT_NAME})
        target_compile_definitions(trackingTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")

        option(VSLAM_TEST_VISUALIZE "Enable visuals in unit test" Off)
        if(VSLAM_TEST_VISUALIZE)
                target_compile_definitions(odometry_icpTest TEST_VISUALIZE)
                target_compile_definitions(odometry_rgbd_cvTest TEST_VISUALIZE)
                target_compile_definitions(trajectoryTest TEST_VISUALIZE)
                target_compile_definitions(odometry TEST_VISUALIZE)
                target_compile_definitions(ceres_baTest TEST_VISUALIZE)
                target_compile_definitions(ceres_baTest TEST_VISUALIZE)
                target_compile_definitions(baTest TEST_VISUALIZE)
                target_compile_definitions(trackingTest TEST_VISUALIZE)
        endif()

endif()
