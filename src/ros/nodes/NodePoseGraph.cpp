#include <vslam_ros/converters.h>

#include <Eigen/Dense>
#include <iostream>

#include "NodePoseGraph.h"
#include "vslam_ros/visibility_control.h"
using namespace vslam;
namespace vslam_ros {
NodePoseGraph::NodePoseGraph(const rclcpp::NodeOptions &options) :
    rclcpp::Node("NodePoseGraph", options),
    _subOdom(create_subscription<nav_msgs::msg::Odometry>(
      "/odom/keyframe2frame", 10, std::bind(&NodePoseGraph::callbackOdom, this, std::placeholders::_1))),
    _subLoop(create_subscription<nav_msgs::msg::Odometry>(
      "/loop_closures/odom", 10, std::bind(&NodePoseGraph::callbackOdomLc, this, std::placeholders::_1))),

    _pub(create_publisher<nav_msgs::msg::Path>("/pose_graph/path", 10)) {
  if (declare_parameter("replay", true)) {
    _cliReplayer = create_client<vslam_ros_interfaces::srv::ReplayerPlay>("togglePlay");
  }
}

void NodePoseGraph::callbackOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
  vslam::Pose pose;
  vslam::Timestamp t;
  vslam_ros::convert(msg->pose, pose);
  vslam_ros::convert(msg->header.stamp, t);

  if (_poseGraph.hasMeasurement(std::stoull(msg->header.frame_id), std::stoull(msg->child_frame_id))) {
    RCLCPP_WARN(get_logger(), "Odometry constraint is already present. Ignoring..");
    return;
  }
  _poseGraph.addMeasurement(std::stoull(msg->header.frame_id), std::stoull(msg->child_frame_id), pose.inverse());

  nav_msgs::msg::Path path;
  path.header.stamp = msg->header.stamp;
  path.header.frame_id = "odom";  // todo make configurable
  vslam_ros::convert(Trajectory(_poseGraph.poses()).inverse(), path);
  _pub->publish(path);
}

void NodePoseGraph::callbackOdomLc(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {

  RCLCPP_INFO(get_logger(), "Loop closure received between: [%s] --> [%s]", msg->header.frame_id.c_str(), msg->child_frame_id.c_str());
  vslam::Pose pose;
  vslam::Timestamp t;
  vslam_ros::convert(msg->pose, pose);
  vslam_ros::convert(msg->header.stamp, t);

  if (_poseGraph.hasMeasurement(std::stoull(msg->header.frame_id), std::stoull(msg->child_frame_id))) {
    RCLCPP_WARN(get_logger(), "Loop closure constraint is already present. Ignoring..");
    return;
  }
  _poseGraph.addMeasurement(std::stoull(msg->header.frame_id), std::stoull(msg->child_frame_id), pose.inverse());
  setReplay(false);
  _poseGraph.optimize();

  nav_msgs::msg::Path path;
  path.header.stamp = msg->header.stamp;
  path.header.frame_id = "odom";  // todo make configurable
  vslam_ros::convert(Trajectory(_poseGraph.poses()).inverse(), path);
  _pub->publish(path);
  setReplay(true);
}

void NodePoseGraph::setReplay(bool ready) {
  using namespace std::chrono_literals;
  while (!_cliReplayer->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      throw std::runtime_error("Interrupted while waiting for the service. Exiting.");
    }
    RCLCPP_INFO(get_logger(), "Replayer Service not available, waiting again...");
  }
  auto request = std::make_shared<vslam_ros_interfaces::srv::ReplayerPlay::Request>();
  request->play = ready;
  request->requester = get_name();
  using ServiceResponseFuture = rclcpp::Client<vslam_ros_interfaces::srv::ReplayerPlay>::SharedFuture;
  auto response_received_callback = [ready, request, this](ServiceResponseFuture future) {
    if (!request->play) {
      RCLCPP_WARN(get_logger(), "Try stopping replayer ... [%s]", future.get()->isplaying ? "Failed" : "Success");
    }
  };
  _cliReplayer->async_send_request(request, response_received_callback);
}

}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodePoseGraph)
