#ifndef NODE_LOOP_CLOSURES_H__
#define NODE_LOOP_CLOSURES_H__
#include "vslam/loop_closure_detection.h"
#include "vslam_ros/visibility_control.h"
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <iostream>
#include <memory>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <vslam_ros_interfaces/srv/replayer_play.hpp>

#include <functional>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <vector>
namespace vslam_ros {
class NodeLoopClosures : public rclcpp::Node {
public:
  COMPOSITION_PUBLIC
  NodeLoopClosures(const rclcpp::NodeOptions &options);

private:
  // Sub-Pub
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _subCamInfo;
  image_transport::SubscriberFilter _subImage, _subDepth;
  const rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _subOdom;
  const rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr _subPath;
  const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _pub;

  // Clients
  rclcpp::Client<vslam_ros_interfaces::srv::ReplayerPlay>::SharedPtr _cliReplayer;

  // Synchronization
  using ExactPolicy = message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;
  std::shared_ptr<ExactSync> _sync;

  void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth);
  void callbackOdom(nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void callbackPath(nav_msgs::msg::Path::ConstSharedPtr msg);
  void cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

  vslam::Frame::UnPtr createFrame(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth) const;
  void setReplay(bool play);
  const vslam::loop_closure_detection::DifferentialEntropy _loopClosureDetection;
  const vslam::FeatureSelection<vslam::FiniteGradient> _featureSelection;
  struct Entropy {
    vslam::Timestamp t;
    double entropy;
  };
  std::map<vslam::Timestamp, std::vector<Entropy>> _childFrames;

  vslam::Frame::VecShPtr _keyframes;
  std::vector<vslam::loop_closure_detection::LoopClosure::ConstShPtr> _loopClosures;
  std::function<bool(vslam::Frame::ConstShPtr, vslam::Frame::ConstShPtr)> _isCandidate;

  vslam::Trajectory::UnPtr _trajectory;
  vslam::Camera::ShPtr _camera;
};
}  // namespace vslam_ros
#endif