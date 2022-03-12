cmake_minimum_required(VERSION 3.8)
project(vslam_ros_evaluation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2 REQUIRED)
find_package(OpenCV 4 QUIET)
find_package(cv_bridge REQUIRED)
find_package(rclcpp_components REQUIRED)
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
  GENERATORS cmake_find_package_multi )

conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE .
                    BUILD missing
                    REMOTE conancenter
                    SETTINGS ${settings})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
find_package(vslam_ros REQUIRED)
find_package(GTest CONFIG REQUIRED)


find_package(vslam REQUIRED)


add_library(${PROJECT_NAME} SHARED src/NodeGtLoader.cpp src/NodeResultWriter.cpp src/NodeReplayer.cpp
)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(${PROJECT_NAME} pd::vslam)

rclcpp_components_register_nodes(
  ${PROJECT_NAME}
    "NodeResultWriter"
    "NodeReplayer"
    "NodeGtLoader"
)

ament_target_dependencies(${PROJECT_NAME}
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
            rclcpp_components
            rclcpp
            std_srvs
            )
install(DIRECTORY
            launch
            DESTINATION share/${PROJECT_NAME}/
          )
  install(
    TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
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
