// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//
// Created by phil on 07.08.21.
//

#include <cv_bridge/cv_bridge.h>

#include "NodeRgbdAlignment.h"
#include "vslam_ros/converters.h"

using namespace vslam;
using namespace std::chrono_literals;

namespace vslam_ros {
NodeRgbdAlignment::NodeRgbdAlignment(const rclcpp::NodeOptions &options) :
    rclcpp::Node("NodeRgbdAlignment", options),
    _queueSizeMin(declare_parameter("sync.queue_size_min", 10)),
    _queueSizeMax(declare_parameter("sync.queue_size_max", 50)),
    _replay(declare_parameter("replayMode", false)),
    _fNo(0),
    _pubOdom(create_publisher<nav_msgs::msg::Odometry>("/odom", 10)),
    _pubPath(create_publisher<nav_msgs::msg::Path>("/path", 10)),
    _pubPoseGraph(create_publisher<nav_msgs::msg::Path>("/path/pose_graph", 10)),
    _pubTf(std::make_shared<tf2_ros::TransformBroadcaster>(this)),
    _pubPclMap(create_publisher<sensor_msgs::msg::PointCloud2>("/pcl/map", 10)),
    _subCamInfo(create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/rgb/camera_info", 10, std::bind(&NodeRgbdAlignment::cameraCallback, this, std::placeholders::_1))),
    _tfAvailable(false),
    _frameId(declare_parameter("tf.frame_id", _frameId)),
    _fixedFrameId(declare_parameter("tf.base_link_id", _fixedFrameId)),
    _tfBuffer(std::make_unique<tf2_ros::Buffer>(get_clock())),
    _subTf(std::make_shared<tf2_ros::TransformListener>(*_tfBuffer)) {
  if (_replay) {
    _cliReplayer = create_client<std_srvs::srv::SetBool>("togglePlay");
  }
  _featureSelection = std::make_unique<FeatureSelection<FiniteGradient>>(
    FiniteGradient{
      declare_parameter<float>("features.intensity_gradient_min", 5.0f),
      declare_parameter<float>("features.depth_gradient_min", 0.01f),
      declare_parameter<float>("features.depth_gradient_max", 0.3f),
      declare_parameter<float>("features.depth_min", 0.0f),
      declare_parameter<float>("features.depth_max", 8.0f)},
    declare_parameter<float>("features.grid_size", 10.0f),
    declare_parameter("features.n_levels", 4));

  const std::string odometryMethod = declare_parameter("odometry.method", "rgbd");
  if (odometryMethod == "rgb") {
    auto aligner = std::make_shared<AlignmentRgb>(
      declare_parameter("odometry.n_levels", 4),
      declare_parameter("odometry.max_iterations", 50),
      declare_parameter("odometry.min_parameter_update", 0.0001),
      declare_parameter("odometry.max_error_increase", 10));
    _odom = [aligner](Frame::ConstShPtr f0, Frame::ConstShPtr f1) {
      auto r = aligner->align(f0, f1);
      return r->pose;
    };
  } else if (odometryMethod == "rgbd") {
    auto aligner = std::make_shared<AlignmentRgbd>(
      declare_parameter("odometry.n_levels", 4),
      declare_parameter("odometry.max_iterations", 50),
      declare_parameter("odometry.min_parameter_update", 0.0001),
      declare_parameter("odometry.max_error_increase", 10));
    _odom = [aligner](Frame::ConstShPtr f0, Frame::ConstShPtr f1) {
      auto r = aligner->align(f0, f1);
      return r->pose;
    };
  } else {
    throw std::runtime_error(format("Unknown odometry method: {}", odometryMethod));
  }

  _nLevels = std::max(get_parameter("features.n_levels").as_int(), get_parameter("odometry.n_levels").as_int());
  _motionModel = std::make_shared<ConstantVelocityModel>(declare_parameter("motion_model.information", 10.0), INFd, INFd);

  _maxEntropyReduction = declare_parameter("keyframe_selection.max_entropy_reduction", 0.05);
  vslam::log::configure(declare_parameter("log.config_directory", "/home/ros/vslam_ros/config/log/"));
#ifdef USE_ROS2_SYNC

  image_transport::TransportHints hints(this, "raw");
  rclcpp::QoS image_sub_qos = rclcpp::SensorDataQoS();
  const auto image_sub_rmw_qos = image_sub_qos.get_rmw_qos_profile();
  _subImage.subscribe(this, "/camera/rgb/image_color", hints.getTransport(), image_sub_rmw_qos);
  _subDepth.subscribe(this, "/camera/depth/image", hints.getTransport(), image_sub_rmw_qos);

  if (declare_parameter("sync.max_interval", 0.2) > 0.0) {
    _approximateSync.reset(new ApproximateSync(ApproximatePolicy(queue_size), _subImage, _subDepth));
    _approximateSync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(get_parameter("sync.max_interval").as_double));
    _approximateSync->registerCallback(&NodeRgbdAlignment::imageCallback, this);
  } else {
    _exactSync.reset(new ExactSync(ExactPolicy(queue_size), _subImage, _subDepth));
    _exactSync->registerCallback(&NodeRgbdAlignment::imageCallback, this);
  }
#else
  _queue = std::make_unique<vslam_ros::Associator>(_queueSizeMax, declare_parameter("sync.max_interval", 0.02) * 1e9);
  _subImage = create_subscription<sensor_msgs::msg::Image>(
    "/camera/rgb/image_color", 10, [&](sensor_msgs::msg::Image::ConstSharedPtr msg) { _queue->pushImage(msg); });
  _subDepth = create_subscription<sensor_msgs::msg::Image>(
    "/camera/depth/image", 10, [&](sensor_msgs::msg::Image::ConstSharedPtr msg) { _queue->pushDepth(msg); });

#endif

  RCLCPP_INFO(get_logger(), "Ready.");
}

void NodeRgbdAlignment::cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
  auto camera = vslam_ros::convert(*msg);

  if (camera->fx() > 0.0 && camera->cx() > 0.0) {
    _subCamInfo.reset();
    _camera = camera;
    _processFrameTimer = create_wall_timer(std::chrono::milliseconds(10), [&]() { initialize(); });

    _periodicTimer = create_wall_timer(std::chrono::seconds(1), [this, msg]() { this->timerCallback(); });

    _cameraFrameId = msg->header.frame_id;
    RCLCPP_INFO(get_logger(), "Valid camera calibration received: %s \n. Node Ready.", camera->toString().c_str());

  } else {
    RCLCPP_ERROR(get_logger(), "Invalid camera calibration received: %s \n.", camera->toString().c_str());
  }
}

void NodeRgbdAlignment::initialize() {
  if (std::min(_queue->nImages(), _queue->nDepth()) > _queueSizeMin) {
    TIMED_SCOPE(timerF, "initialize");
    auto const [_, depth, img] = _queue->pop();

    _lf = _cf;
    _cf = createFrame(img, depth);

    _cf->computePyramid(_nLevels);

    _kf = _cf;
    _lf = _cf;
    _kf->computeDerivatives();
    _kf->computePcl();
    _featureSelection->select(_kf);
    _motionModel->update(_cf->pose(), _cf->t());
    _motion = _cf->pose() * _lf->pose().inverse();

    _trajectory.append(_cf->t(), _cf->pose());

    publish(img->header.stamp);
    _fNo++;
    _processFrameTimer = create_wall_timer(std::chrono::milliseconds(10), [&]() { process(); });
  }
}
void NodeRgbdAlignment::process() {
  if (std::min(_queue->nImages(), _queue->nDepth()) > _queueSizeMin) {
    TIMED_SCOPE(timerF, "processFrame");
    auto const [_, depth, img] = _queue->pop();
    _lf = _cf;
    _cf = createFrame(img, depth);

    _cf->computePyramid(_nLevels);

    _cf->pose() = _motionModel->predict(_cf->t());

    _cf->pose() = _odom(_kf, _cf);
    if (1.0 - std::log(_cf->pose().cov().determinant()) / _entropyRef > _maxEntropyReduction) {
      _kf->removeFeatures();
      _kf = _lf;
      _kf->computeDerivatives();
      _kf->computePcl();
      _featureSelection->select(_kf);
      _cf->pose() = _motionModel->predict(_cf->t());
      _cf->pose() = _odom(_kf, _cf);
      ;
      _entropyRef = std::log(_cf->pose().cov().determinant());
    }
    _motionModel->update(_cf->pose(), _cf->t());
    _motion = _cf->pose() * _lf->pose().inverse();

    _trajectory.append(_cf->t(), _cf->pose());

    publish(img->header.stamp);
    _fNo++;
  }
}

Frame::UnPtr
NodeRgbdAlignment::createFrame(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth) const {
  namespace enc = sensor_msgs::image_encodings;
  auto cv_ptr = cv_bridge::toCvShare(msgImg);
  if (enc::isColor(msgImg->encoding)) {
    cv_ptr = cv_bridge::cvtColor(cv_ptr, "mono8");
  }
  Timestamp t;
  convert(msgImg->header.stamp, t);
  cv::Mat depth;
  cv_bridge::toCvShare(msgDepth)->image.convertTo(depth, CV_32FC1);
  return std::make_unique<Frame>(cv_ptr->image.clone(), depth.clone(), _camera, t);
}

void NodeRgbdAlignment::timerCallback() {
  const int maxQueue = static_cast<int>(0.75 * _queueSizeMax);
  const bool queueAlmostFull = std::max(_queue->nDepth(), _queue->nImages()) > maxQueue;
  if (_replay) {
    setReplay(!queueAlmostFull);
  }
  if (queueAlmostFull) {
    RCLCPP_WARN(get_logger(), "Warning input queue exceeds threshold: [%d]/[%d]. Cannot process all frames.", _queue->nDepth(), maxQueue);
  }

  if (!_tfAvailable) {
    lookupTf();
  }
  RCLCPP_INFO(
    get_logger(),
    format(
      "Egomotion: {} m/s {:.2f}°/s", _motionModel->velocity().translation().transpose(), _motionModel->velocity().totalRotationDegrees())
      .c_str());
}

void NodeRgbdAlignment::lookupTf() {
  try {
    _world2origin = _tfBuffer->lookupTransform(_fixedFrameId, _cameraFrameId.substr(1), tf2::TimePointZero);
    _tfAvailable = true;

  } catch (tf2::TransformException &ex) {
    RCLCPP_WARN(get_logger(), "%s", ex.what());
  }
}
void NodeRgbdAlignment::setReplay(bool ready) {
  while (!_cliReplayer->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      throw std::runtime_error("Interrupted while waiting for the service. Exiting.");
    }
    RCLCPP_INFO(get_logger(), "Replayer Service not available, waiting again...");
  }
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = ready;
  using ServiceResponseFuture = rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture;
  auto response_received_callback = [this](ServiceResponseFuture future) {
    if (!future.get()->success) {
      RCLCPP_WARN(get_logger(), "Last replayer signal result was not valid.");
    }
  };
  _cliReplayer->async_send_request(request, response_received_callback);
}

void NodeRgbdAlignment::publish(const rclcpp::Time &t) {
  // tf from origin (odom) to another fixed frame (world)
  geometry_msgs::msg::TransformStamped tfOrigin = _world2origin;
  tfOrigin.header.stamp = t;
  tfOrigin.header.frame_id = _fixedFrameId;
  tfOrigin.child_frame_id = _frameId;

  // Send current camera pose as estimate for pose of optical frame
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = t;
  tf.header.frame_id = _frameId;
  tf.child_frame_id = "camera";  // camera name?
  vslam_ros::convert(_cf->pose().SE3().inverse(), tf);
  _pubTf->sendTransform({tf, tfOrigin});

  // Send pose, twist and path in optical frame
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = t;
  odom.header.frame_id = _frameId;
  vslam_ros::convert(_cf->pose().inverse(), odom.pose);
  vslam_ros::convert(_motion.inverse(), odom.twist);
  _pubOdom->publish(odom);

  nav_msgs::msg::Path path;
  path.header = odom.header;
  vslam_ros::convert(_trajectory.inverse(), path);
  _pubPath->publish(path);
}

}  // namespace vslam_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeRgbdAlignment)
