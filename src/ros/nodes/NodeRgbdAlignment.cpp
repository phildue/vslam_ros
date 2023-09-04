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
    _pubOdomKf(create_publisher<nav_msgs::msg::Odometry>("/odom/keyframe", 10)),
    _pubOdomKf2f(create_publisher<nav_msgs::msg::Odometry>("/odom/keyframe2frame", 10)),
    _pubKeyImg(create_publisher<sensor_msgs::msg::Image>("/keyframe/image", 10)),
    _pubKeyDepth(create_publisher<sensor_msgs::msg::Image>("/keyframe/depth", 10)),
    _pubPath(create_publisher<nav_msgs::msg::Path>("/path", 10)),
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
    _cliReplayer = create_client<vslam_ros_interfaces::srv::ReplayerPlay>("togglePlay");
  }
  _featureSelection = std::make_unique<FeatureSelection>(
    declare_parameter("features.intensity_gradient_min", 5),
    declare_parameter("features.depth_gradient_min", 0.01),
    declare_parameter("features.depth_gradient_max", 0.3),
    declare_parameter("features.depth_min", 0),
    declare_parameter("features.depth_max", 8.0),
    declare_parameter("features.grid_size", 10.0),
    declare_parameter("features.n_levels", 4));

  _alignmentRgbd = std::make_shared<AlignmentRgbd>(
    declare_parameter("odometry.n_levels", 4),
    declare_parameter("odometry.max_iterations", 50),
    declare_parameter("odometry.min_parameter_update", 0.0001),
    declare_parameter("odometry.max_error_increase", 10));

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

  _processFrameTimer = create_wall_timer(std::chrono::milliseconds(10), [&]() {
    /*Check periodically if there are enough frames and process if so.
     */
    if (_camera && std::min(_queue->nImages(), _queue->nDepth()) > _queueSizeMin) {
      auto const [_, depth, img] = _queue->pop();
      RCLCPP_DEBUG(
        get_logger(),
        "fNo: %d Processing Z: %d.%d I: %d.%d %f",
        _fNo,
        depth->header.stamp.sec,
        depth->header.stamp.nanosec,
        img->header.stamp.sec,
        img->header.stamp.nanosec,
        (rclcpp::Time(depth->header.stamp).nanoseconds() - rclcpp::Time(img->header.stamp).nanoseconds()) / 1e9);
      this->imageCallback(img, depth);

      _fNo++;
    }
  });
#endif

  RCLCPP_INFO(get_logger(), "Ready.");
}

void NodeRgbdAlignment::cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
  auto camera = vslam_ros::convert(*msg);

  if (camera->fx() > 0.0 && camera->cx() > 0.0) {
    _subCamInfo.reset();
    _camera = camera;

    _periodicTimer = create_wall_timer(std::chrono::seconds(1), [this, msg]() { this->timerCallback(); });

    _cameraFrameId = msg->header.frame_id;
    RCLCPP_INFO(get_logger(), "Valid camera calibration received: %s \n. Node Ready.", camera->toString().c_str());

  } else {
    RCLCPP_ERROR(get_logger(), "Invalid camera calibration received: %s \n.", camera->toString().c_str());
  }
}

void NodeRgbdAlignment::imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth) {
  TIMED_SCOPE(timerF, "processFrame");
  _lf = _cf;
  _cf = createFrame(msgImg, msgDepth);

  if (_kf) {
    process();
  } else {
    initialize();
  }

  publish(msgImg->header.stamp);
}
void NodeRgbdAlignment::initialize() {
  _cf->computePyramid(_alignmentRgbd->nLevels());

  _kf = _cf;
  _lf = _cf;
  _kf->computeDerivatives();
  _kf->computePcl();
  _featureSelection->select(_kf, true);

  _motionModel->update(_cf->pose(), _cf->t());
  _motion = _cf->pose() * _lf->pose().inverse();

  _trajectory.append(_cf->t(), _cf->pose());
}
void NodeRgbdAlignment::process() {
  _cf->computePyramid(_alignmentRgbd->nLevels());

  _cf->pose() = _motionModel->predict(_cf->t());

  auto results = _alignmentRgbd->align(_kf, _cf);
  _cf->pose() = results->pose;

  // TODO keyframe handling should be in own class
  _newKeyFrame = _lf != _kf && 1.0 - std::log(_cf->pose().cov().determinant()) / _entropyRef > _maxEntropyReduction;
  if (_newKeyFrame) {
    _kf->removeFeatures();
    _kf = _lf;
    _kf->computeDerivatives();
    _kf->computePcl();
    _featureSelection->select(_kf, true);
    _cf->pose() = _motionModel->predict(_cf->t());
    _cf->pose() = _alignmentRgbd->align(_kf, _cf)->pose;
    _entropyRef = std::log(_cf->pose().cov().determinant());
  }
  _motionModel->update(_cf->pose(), _cf->t());
  _motion = _cf->pose() * _lf->pose().inverse();

  _trajectory.append(_cf->t(), _cf->pose());
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
    get_logger(), format("Pose: {} m {:.2f}°", _cf->pose().translation().transpose(), _cf->pose().totalRotationDegrees()).c_str());
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

void NodeRgbdAlignment::publish(const rclcpp::Time &t) {

  // TODO since we measure the relative pose only,
  //  can we simply, only publish the transformation to the last keyframe, while another node publishs that "global transformation"
  //  At least this could be optional when a backend is available
  //  Then the backend would take care of optimizing the global transformation

  // tf from origin (odom) to another fixed frame (world)
  geometry_msgs::msg::TransformStamped tfOrigin = _world2origin;
  tfOrigin.header.stamp = t;
  tfOrigin.header.frame_id = _fixedFrameId;
  tfOrigin.child_frame_id = _frameId;

  // tf from current frame to origin
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = t;
  tf.header.frame_id = _frameId;
  tf.child_frame_id = "camera";  // camera name?
  vslam_ros::convert(_cf->pose().SE3().inverse(), tf);

  nav_msgs::msg::Odometry odom;
  odom.header.stamp = t;
  odom.header.frame_id = _frameId;
  odom.child_frame_id = "camera";  // camera name?
  vslam_ros::convert(_cf->pose().inverse(), odom.pose);
  vslam_ros::convert(_motion.inverse(), odom.twist);
  _pubOdom->publish(odom);

  // tf from keyframe to origin
  geometry_msgs::msg::TransformStamped tfKey;
  tfKey.header.stamp = t;
  tfKey.header.frame_id = std::to_string(_kf->t());
  tfKey.child_frame_id = "camera";  // camera name?
  vslam_ros::convert(_cf->pose().SE3().inverse(), tf);

  nav_msgs::msg::Odometry odomKf;
  odomKf.header.stamp = t;
  odomKf.header.frame_id = _frameId;
  odomKf.child_frame_id = std::to_string(_kf->t());
  _pubOdomKf->publish(odomKf);

  _pubTf->sendTransform({tf, tfOrigin, tfKey});

  if (_cf && _kf && _cf != _kf) {
    // tf from current frame to keyframe
    nav_msgs::msg::Odometry odomKf2f;
    odomKf2f.header.stamp = t;
    odomKf2f.header.frame_id = std::to_string(_kf->t());
    odomKf2f.child_frame_id = std::to_string(_cf->t());
    vslam_ros::convert(_kf->pose().inverse(), odomKf.pose);
    Pose kf2f(_cf->pose().SE3() * _kf->pose().SE3().inverse(), _cf->pose().cov());
    vslam_ros::convert(kf2f.inverse(), odomKf2f.pose);
    _pubOdomKf2f->publish(odomKf2f);
  }

  if (_newKeyFrame) {
    RCLCPP_INFO(get_logger(), "New Keyframe: %ld", _kf->id());
    namespace enc = sensor_msgs::image_encodings;
    std_msgs::msg::Header header;
    header.frame_id = "camera";
    vslam_ros::convert(_kf->t(), header.stamp);
    cv_bridge::CvImage img(header, enc::MONO8, _kf->intensity());
    cv_bridge::CvImage depth(header, enc::TYPE_32FC1, _kf->depth());
    _pubKeyImg->publish(*img.toImageMsg());
    _pubKeyDepth->publish(*depth.toImageMsg());
  }

  nav_msgs::msg::Path path;
  path.header = odom.header;
  vslam_ros::convert(_trajectory.inverse(), path);
  _pubPath->publish(path);
}

}  // namespace vslam_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeRgbdAlignment)
