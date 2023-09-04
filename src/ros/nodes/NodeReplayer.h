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
#include "vslam_ros/visibility_control.h"
#include "vslam_ros/vslam_ros.h"
#include "vslam_ros_interfaces/srv/replayer_play.hpp"
#ifndef VSLAM_ROS2_NODE_REPLAYER_H__
#define VSLAM_ROS2_NODE_REPLAYER_H__
namespace vslam_ros {
class NodeReplayer : public rclcpp::Node {
public:
  COMPOSITION_PUBLIC

  NodeReplayer(const rclcpp::NodeOptions &options);

  void replayNext();
  void servicePlayCallback(
    const std::shared_ptr<rmw_request_id_t> requestHeader,
    const std::shared_ptr<vslam_ros_interfaces::srv::ReplayerPlay::Request> request,
    std::shared_ptr<vslam_ros_interfaces::srv::ReplayerPlay::Response> response);

private:
  rcl_time_point_value_t seek(rcl_time_point_value_t t);
  void publish(rosbag2_storage::SerializedBagMessageSharedPtr msg);

  std::unique_ptr<rosbag2_cpp::readers::SequentialReader> open(const std::string &bagName) const;
  std::unique_ptr<rosbag2_cpp::readers::SequentialReader> _reader;
  bool _visualize = true;
  int _idx = 0;
  int _fNoOut = 0;
  std::chrono::system_clock::time_point _tReplay0;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr _pubTf;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> _pubImg;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pubDepth;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr> _pubCamInfo;
  rclcpp::Service<vslam_ros_interfaces::srv::ReplayerPlay>::SharedPtr _servicePlay;
  std::condition_variable _cond;
  std::mutex _mutex;
  bool _play;
  std::map<std::string, bool> _playRequest;
  std::string _bagName;
  std::string _sequenceId;
  std::string _syncTopic;
  double _duration;
  rosbag2_storage::BagMetadata _meta;
  std::map<std::string, unsigned long int> _msgCtr;
  std::map<std::string, unsigned long int> _nMessages;
  rcl_time_point_value_t getStartingTime(const rosbag2_storage::BagMetadata &meta);
  rcl_time_point_value_t getEndTime(rcl_time_point_value_t tStart, const rosbag2_storage::BagMetadata &meta);
  rcl_time_point_value_t _tStart, _tEnd, _tLast;
  rclcpp::TimerBase::SharedPtr _periodicTimer;
  rcl_time_point_value_t _tWaiting, _period;
  void setReplay(bool pause);
};
}  // namespace vslam_ros
#endif