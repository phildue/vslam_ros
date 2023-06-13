#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <iostream>
#include <nav_msgs/msg/path.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sstream>
#include <std_srvs/srv/set_bool.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_cpp/readers/sequential_reader.hpp"
#include "vslam/evaluation.h"
#include "vslam_ros/visibility_control.h"
#include "vslam_ros/vslam_ros.h"
#ifndef VSLAM_ROS2_NODE_REPLAYER_EXTRACTED_H__
#define VSLAM_ROS2_NODE_REPLAYER_EXTRACTED_H__
namespace vslam_ros
{
class NodeReplayerExtracted : public rclcpp::Node
{
public:
  COMPOSITION_PUBLIC

  NodeReplayerExtracted(const rclcpp::NodeOptions & options);

  void replayNext();
  void servicePlayCallback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);

private:
  void publish(int fId);

  vslam::evaluation::tum::DataLoader::UnPtr _dl;
  int _fId, _fEnd;
  bool _visualize = true;
  int _idx = 0;
  int _fNoOut = 0;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr _pubTf;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> _pubImg;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pubDepth;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr> _pubCamInfo;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr _servicePlay;
  std::condition_variable _cond;
  std::mutex _mutex;
  bool _play;
  double _duration;
  rclcpp::TimerBase::SharedPtr _periodicTimer;
  rcl_time_point_value_t _tWaiting, _period;
};
}  // namespace vslam_ros
#endif