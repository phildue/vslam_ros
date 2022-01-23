cmake_minimum_required(VERSION 3.8)
project(vslam_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
find_package(Eigen3 REQUIRED)
# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif ()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
find_package(cv_bridge REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)

find_package(tf2_geometry_msgs REQUIRED)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                EXPECTED_HASH SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
                TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_autodetect(settings)
#set (settings "build_type=Release;compiler=gcc;compiler.version=9;compiler.libcxx=libstdc++11")
message("Detected settings: ${settings}")

conan_cmake_install(PATH_OR_REFERENCE ${CMAKE_CURRENT_LIST_DIR}/src/vslam
                   # BUILD missing
                    REMOTE conancenter
                    SETTINGS ${settings})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(src/vslam)
add_library(${PROJECT_NAME} SHARED
    src/ros/converter.cpp 
    src/ros/Queue.cpp
    src/ros/nodes/RgbdAlignmentNode.cpp
)

target_link_libraries(${PROJECT_NAME} pd::vslam)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/ros/>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}/>
    )
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:include/>
    )

rclcpp_components_register_nodes(
    ${PROJECT_NAME}
    "vslam_ros::RgbdAlignmentNode"
)

ament_target_dependencies(${PROJECT_NAME}
                                         OpenCV
                                         Eigen3
                                         rclcpp
                                         rclcpp_components
                                         std_msgs 
                                         sensor_msgs 
                                         nav_msgs 
                                         stereo_msgs
                                         cv_bridge
                                         message_filters
                                         geometry_msgs
                                         tf2
                                         tf2_ros
                                         tf2_geometry_msgs
                                         tf2_msgs
                                         )
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(vslam)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/ros/
    DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING # install only matched files
    PATTERN "*.h*" # select header files
    )
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION include/
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


#TODO we should also install the conan lib files
    
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

#ament_package(CONFIG_EXTRAS "${CMAKE_CURRENT_BINARY_DIR}/conan_paths.cmake")
ament_package()