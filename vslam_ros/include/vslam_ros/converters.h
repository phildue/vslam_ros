#ifndef VSLAM_ROS2_CONVERTER_H__
#define VSLAM_ROS2_CONVERTER_H__
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <vslam/vslam.h>
#include <sophus/se3.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace vslam_ros{
pd::vision::Camera::ShPtr convert(const sensor_msgs::msg::CameraInfo& msg);
geometry_msgs::msg::Pose convert(const Sophus::SE3d& se3);
Sophus::SE3d convert(const geometry_msgs::msg::Pose& ros);
Sophus::SE3d convert(const geometry_msgs::msg::TransformStamped& tf);
void convert(const Sophus::SE3d& sophus, geometry_msgs::msg::TransformStamped& tf);

}
#endif