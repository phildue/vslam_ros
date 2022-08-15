#ifndef NODE_RESULT_WRITER_H__
#define NODE_RESULT_WRITER_H__
#include <vslam/vslam.h>

#include <fstream>
#include <iostream>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <string>

#include "vslam_ros/visibility_control.h"

namespace vslam_ros
{
class NodeResultWriter : public rclcpp::Node
{
public:
  COMPOSITION_PUBLIC
  NodeResultWriter(const rclcpp::NodeOptions & options);
  void callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  ~NodeResultWriter() { _algoFile.close(); }

private:
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _sub;
  std::fstream _algoFile;
  std::string _outputFile;
};
}  // namespace vslam_ros
#endif