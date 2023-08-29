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

  // TODO add callback for keyframes
  // TODO on new key frame search for loop closures and optimize, repeat until no new loop closures are found

  // TODO could we have a different node that takes care of the loop closures?
  // It could publish relative transformations using frame_id->child_frame_id in the odometry message
}

void NodePoseGraph::callbackOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
  vslam::Pose pose;
  vslam::Timestamp t;
  vslam_ros::convert(msg->pose, pose);
  vslam_ros::convert(msg->header.stamp, t);

  _poseGraph.addMeasurement(std::stoull(msg->header.frame_id), t, pose.inverse());

  nav_msgs::msg::Path path;
  path.header.stamp = msg->header.stamp;
  path.header.frame_id = "odom";  // todo make configurable
  vslam_ros::convert(Trajectory(_poseGraph.poses()).inverse(), path);
  _pub->publish(path);
}

void NodePoseGraph::callbackOdomLc(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
  vslam::Pose pose;
  vslam::Timestamp t;
  vslam_ros::convert(msg->pose, pose);
  vslam_ros::convert(msg->header.stamp, t);

  _poseGraph.addMeasurement(std::stoull(msg->header.frame_id), t, pose.inverse());

  _poseGraph.optimize();

  nav_msgs::msg::Path path;
  path.header.stamp = msg->header.stamp;
  path.header.frame_id = "odom";  // todo make configurable
  vslam_ros::convert(Trajectory(_poseGraph.poses()).inverse(), path);
  _pub->publish(path);
}

}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodePoseGraph)
