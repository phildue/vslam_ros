#ifndef VSLAM_ROS2_CONVERTER_H__
#define VSLAM_ROS2_CONVERTER_H__
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <vslam/vslam.h>
#include <sophus/se3.hpp>

namespace vslam_ros2{
pd::vision::Camera::ShPtr convert(const sensor_msgs::msg::CameraInfo& msg);
geometry_msgs::msg::Pose convert(const Sophus::SE3d& se3);

}
#endif