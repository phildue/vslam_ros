cmake_minimum_required(VERSION 3.8)
project(vslam_ros_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2 REQUIRED)
find_package(OpenCV 4 QUIET)
find_package(cv_bridge REQUIRED)

if (NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif ()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                EXPECTED_HASH SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
                TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(
  REQUIRES sophus/1.0.0 easyloggingpp/9.97.0 eigen/3.4.0 gtest/cci.20210126
  GENERATORS cmake_find_package_multi
)

conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE .
                    BUILD missing
                    REMOTE conancenter
                    SETTINGS ${settings})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
find_package(vslam_ros REQUIRED)
find_package(GTest CONFIG REQUIRED)


find_package(vslam REQUIRED)

add_executable(test_lukas_kanade_se3_synth src/test_lukas_kanade_se3_synth.cpp)
target_link_libraries(test_lukas_kanade_se3_synth pd::vslam)
target_link_libraries(test_lukas_kanade_se3_synth GTest::gtest_main)

ament_target_dependencies(test_lukas_kanade_se3_synth 
            std_msgs 
            sensor_msgs 
            nav_msgs 
            stereo_msgs
            vslam_ros
            rosbag2_cpp
            rosbag2
            OpenCV
            cv_bridge
            )
                     

install(
  TARGETS test_lukas_kanade_se3_synth
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(test_lukas_kanade_se3_real src/test_lukas_kanade_se3_real.cpp)
target_link_libraries(test_lukas_kanade_se3_real pd::vslam)
target_link_libraries(test_lukas_kanade_se3_real GTest::gtest_main)

ament_target_dependencies(test_lukas_kanade_se3_real
            std_msgs 
            sensor_msgs 
            nav_msgs 
            tf2
            tf2_msgs
            stereo_msgs
            vslam_ros
            rosbag2_cpp
            rosbag2
            OpenCV
            cv_bridge
            )
                     

install(
  TARGETS test_lukas_kanade_se3_real
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
