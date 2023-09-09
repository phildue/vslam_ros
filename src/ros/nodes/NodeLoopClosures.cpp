#include <cv_bridge/cv_bridge.h>
#include <vslam_ros/converters.h>

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <numeric>

#include "NodeLoopClosures.h"
#include "vslam_ros/visibility_control.h"
using namespace vslam;
namespace vslam_ros {
NodeLoopClosures::NodeLoopClosures(const rclcpp::NodeOptions &options) :
    rclcpp::Node("NodeLoopClosures", options),
    _subCamInfo(create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/rgb/camera_info", 10, std::bind(&NodeLoopClosures::cameraCallback, this, std::placeholders::_1))),
    _subOdom(create_subscription<nav_msgs::msg::Odometry>(
      "/odom/keyframe2frame", 10, std::bind(&NodeLoopClosures::callbackOdom, this, std::placeholders::_1))),
    _subPath(create_subscription<nav_msgs::msg::Path>(
      "/pose_graph/path", 10, std::bind(&NodeLoopClosures::callbackPath, this, std::placeholders::_1))),
    _pub(create_publisher<nav_msgs::msg::Odometry>("/loop_closures/odom", 10)),
    _loopClosureDetection{
      declare_parameter("min_entropy_ratio.fine", 0.9),
      declare_parameter("min_entropy_ratio.coarse", 0.9),
      {static_cast<int>(declare_parameter("aligner.fine.n_levels", 4)),
       static_cast<int>(declare_parameter("aligner.fine.max_iterations", 50)),
       declare_parameter("aligner.fine.min_parameter_update", 0.0001),
       declare_parameter("aligner.fine.max_error_increase", 10.0)},
      {std::vector<int>({3}),
       static_cast<int>(declare_parameter("aligner.coarse.max_iterations", 50)),
       declare_parameter("aligner.coarse.min_parameter_update", 0.0001),
       declare_parameter("aligner.coarse.max_error_increase", 10.0)}},
    _featureSelection{
      FiniteGradient{
        static_cast<float>(declare_parameter("features.intensity_gradient_min", 5.0)),
        static_cast<float>(declare_parameter("features.depth_gradient_min", 0.01)),
        static_cast<float>(declare_parameter("features.depth_gradient_max", 0.3)),
        static_cast<float>(declare_parameter("features.depth_min", 0.0)),
        static_cast<float>(declare_parameter("features.depth_max", 8.0))},
      static_cast<float>(declare_parameter("features.grid_size", 10.0)),
      static_cast<int>(declare_parameter("features.n_levels", 4))},
    _trajectory(std::make_unique<Trajectory>()) {
  if (declare_parameter("replay", true)) {
    _cliReplayer = create_client<vslam_ros_interfaces::srv::ReplayerPlay>("togglePlay");
  }
  auto minTrans = std::make_shared<loop_closure_detection::candidate_proposal::RelativePose>(
    declare_parameter("candidate_selection.max_translation", 0.5));
  auto minFt = std::make_shared<loop_closure_detection::candidate_proposal::ReprojectedFeatures>(
    declare_parameter("candidate_selection.min_features", 500));

  _isCandidate = [minTrans, minFt](auto f0, auto f1) { return minTrans->isCandidate(f0, f1) && minFt->isCandidate(f0, f1); };
  log::config("LoopClosures")->show = declare_parameter("logimg.LoopClosures.show", -1);
  log::config("LoopClosuresRatioTest")->show = declare_parameter("logimg.LoopClosuresRatioTest.show", -1);
}

void NodeLoopClosures::imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth) {
  Frame::ShPtr kfn = createFrame(msgImg, msgDepth);
  _featureSelection.select(kfn);

  std::for_each(_keyframes.begin(), _keyframes.end(), [&](auto kf) {
    if (!_isCandidate(kf, kfn)) {
      return;
    }
    auto lc = _loopClosureDetection.isLoopClosure(kf, kfn);

    if (lc) {
      RCLCPP_INFO(get_logger(), "Loop closure detected between [%ld] and [%ld]", lc->t0, lc->t1);
      nav_msgs::msg::Odometry odom;
      odom.header.stamp = msgImg->header.stamp;
      odom.header.frame_id = std::to_string(lc->t0);
      odom.child_frame_id = std::to_string(lc->t1);

      vslam_ros::convert(lc->relativePose.inverse(), odom.pose);
      _pub->publish(odom);
      _loopClosures.push_back(loop_closure_detection::LoopClosure::ConstShPtr(std::move(lc)));
    }
  });

  _keyframes.push_back(kfn);
}

void NodeLoopClosures::cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
  auto camera = vslam_ros::convert(*msg);

  if (camera->fx() > 0.0 && camera->cx() > 0.0) {
    _subCamInfo.reset();
    _camera = camera;

    image_transport::TransportHints hints(this, "raw");
    rclcpp::QoS image_sub_qos = rclcpp::SensorDataQoS();
    const auto image_sub_rmw_qos = image_sub_qos.get_rmw_qos_profile();
    _subImage.subscribe(this, "/keyframe/image", hints.getTransport(), image_sub_rmw_qos);
    _subDepth.subscribe(this, "/keyframe/depth", hints.getTransport(), image_sub_rmw_qos);

    _sync.reset(new ExactSync(ExactPolicy(declare_parameter("sync.queue_size", 10)), _subImage, _subDepth));
    _sync->registerCallback(&NodeLoopClosures::imageCallback, this);
    RCLCPP_INFO(get_logger(), "Valid camera calibration received: %s \n", camera->toString().c_str());
  } else {
    RCLCPP_ERROR(get_logger(), "Invalid camera calibration received: %s \n.", camera->toString().c_str());
  }
}
vslam::Frame::UnPtr
NodeLoopClosures::createFrame(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth) const {
  namespace enc = sensor_msgs::image_encodings;
  auto cv_ptr = cv_bridge::toCvShare(msgImg);
  if (enc::isColor(msgImg->encoding)) {
    cv_ptr = cv_bridge::cvtColor(cv_ptr, "mono8");
  }
  Timestamp t;
  vslam_ros::convert(msgImg->header.stamp, t);
  cv::Mat depth;
  cv_bridge::toCvShare(msgDepth)->image.convertTo(depth, CV_32FC1);
  auto f = std::make_unique<Frame>(cv_ptr->image.clone(), depth.clone(), _camera, t, *_trajectory->poseAt(t, false));
  /*TODO do we have to recompute it all? */
  f->computePyramid(4);
  f->computeDerivatives();
  f->computePcl();

  return f;
}

void NodeLoopClosures::callbackOdom(nav_msgs::msg::Odometry::ConstSharedPtr msg) {
  /*Receives odometry, mostly relevant for current keyframe pose and covariance*/
  vslam::Pose pose;
  vslam::Timestamp t;
  vslam_ros::convert(msg->pose, pose);
  vslam_ros::convert(msg->header.stamp, t);
  if (!_trajectory->poseAt(t)) {
    _trajectory->append(t, pose);
  }
  _loopClosureDetection.update(std::stoull(msg->header.frame_id), std::stoull(msg->child_frame_id), pose);
}

void NodeLoopClosures::callbackPath(nav_msgs::msg::Path::ConstSharedPtr msg) {
  /*Receives potentially optimized graph. Hence sync local trajectory and key frame poses*/
  for (const auto &poseRos : msg->poses) {
    vslam::SE3d pose = convert(poseRos.pose);
    vslam::Timestamp t;
    convert(poseRos.header.stamp, t);
    if (!_trajectory->poseAt(t)) {
      _trajectory->append(t, pose.inverse());
    } else {
      _trajectory->poseAt(t)->SE3() = pose.inverse();
    }
  }
  for (const auto &kf : _keyframes) {
    kf->pose().SE3() = _trajectory->poseAt(kf->t(), false)->SE3();
  }
}
void NodeLoopClosures::setReplay(bool ready) {
  using namespace std::chrono_literals;

  if (!_cliReplayer) {
    return;
  }
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
  auto response_received_callback = [&](ServiceResponseFuture future) {
    if (!ready) {
      RCLCPP_WARN(get_logger(), "Try stopping replayer ... [%s]", future.get()->isplaying ? "Failed" : "Success");
    }
  };
  _cliReplayer->async_send_request(request, response_received_callback);
}

}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeLoopClosures)
