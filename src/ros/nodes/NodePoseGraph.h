#ifndef NODE_POSE_GRAPH_H__
#define NODE_POSE_GRAPH_H__
#include <iostream>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <string>

#include "vslam/vslam.h"
#include "vslam_ros/visibility_control.h"
namespace vslam_ros {
class NodePoseGraph : public rclcpp::Node {
public:
  COMPOSITION_PUBLIC
  NodePoseGraph(const rclcpp::NodeOptions &options);

private:
  void callbackOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void callbackOdomLc(const nav_msgs::msg::Odometry::ConstSharedPtr msg);

  const rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _subOdom, _subLoop;
  const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pub;
  vslam::PoseGraph _poseGraph;
  std::map<std::string, vslam::Timestamp> _keyframes;
};
}  // namespace vslam_ros
#endif