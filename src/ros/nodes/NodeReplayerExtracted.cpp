#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;
#include <cv_bridge/cv_bridge.h>

#include <filesystem>

#include "NodeReplayerExtracted.h"
#include "vslam/core.h"
namespace fs = std::filesystem;
using namespace vslam;
using namespace std::chrono_literals;

namespace vslam_ros
{
NodeReplayerExtracted::NodeReplayerExtracted(const rclcpp::NodeOptions & options)
: rclcpp::Node("NodeReplayer", options),
  _servicePlay(create_service<std_srvs::srv::SetBool>(
    "togglePlay", std::bind(
                    &NodeReplayerExtracted::servicePlayCallback, this, std::placeholders::_1,
                    std::placeholders::_2))),
  _play(true)
{
  _dl = std::make_unique<evaluation::tum::DataLoader>(
    declare_parameter("dataset_root", "/mnt/dataset/tum_rgbd/"),
    declare_parameter("sequence_id", "rgbd_dataset_freiburg1_desk"));

  RCLCPP_INFO(get_logger(), "Opening: %s", _dl->extracteDataPath().c_str());
  _fId = declare_parameter("start_random", false) ? random::U(0, _dl->nFrames() - 1) : 0;
  int desiredDuration = static_cast<int>(declare_parameter("duration", -1.0));

  _fEnd = std::min(desiredDuration < 0 ? _dl->nFrames() - 1 : desiredDuration, _dl->nFrames() - 1);
  declare_parameter("timeout", 1.0);
  _duration = _dl->duration(_fId, _fEnd);
  RCLCPP_INFO(
    get_logger(), "Bag has length of [%.2fs] and contains [%ld]", _duration, _dl->nFrames() - _fId);

  _pubImg["/camera/rgb/image_color"] =
    create_publisher<sensor_msgs::msg::Image>("/camera/rgb/image_color", 100);
  _pubCamInfo["/camera/rgb/camera_info"] =
    create_publisher<sensor_msgs::msg::CameraInfo>("/camera/rgb/camera_info", 100);
  _pubTf = create_publisher<tf2_msgs::msg::TFMessage>("/tf", 100);
  _pubDepth = create_publisher<sensor_msgs::msg::Image>("/camera/depth/image", 100);
  RCLCPP_INFO(
    get_logger(), "%s",
    format(
      "Will play from t0=[{:%Y-%m-%d %H:%M:%S}] to t1=[{:%Y-%m-%d %H:%M:%S}] for [{:.3f}]s.",
      vslam::time::to_time_point(_dl->timestamps()[_fId]),
      vslam::time::to_time_point(_dl->timestamps()[_fEnd]), _duration)
      .c_str());
  _period = 50;
  _tWaiting = 0;
  _periodicTimer = create_wall_timer(
    std::chrono::milliseconds(_period), std::bind(&NodeReplayerExtracted::replayNext, this));
}
void NodeReplayerExtracted::publish(int fId)
{
  auto rgb = _dl->loadIntensity(fId);
  auto z = _dl->loadDepth(fId);
  namespace enc = sensor_msgs::image_encodings;
  auto img = std::make_shared<sensor_msgs::msg::Image>();
  img->header.frame_id = "/camera";
  vslam_ros::convert(_dl->timestampsImage()[_fId], img->header.stamp);
  cv_bridge::CvImage cv_image(img->header, enc::MONO8, rgb);
  _pubImg["/camera/rgb/image_color"]->publish(*cv_image.toImageMsg());
  auto camInfo = std::make_shared<sensor_msgs::msg::CameraInfo>();
  camInfo->header = img->header;
  vslam_ros::convert(_dl->cam(), *camInfo);
  _pubCamInfo["/camera/rgb/camera_info"]->publish(*camInfo);

  auto depth = std::make_shared<sensor_msgs::msg::Image>();
  depth->header.frame_id = "/camera";
  vslam_ros::convert(_dl->timestampsDepth()[_fId], depth->header.stamp);
  cv_bridge::CvImage cv_depth(depth->header, enc::TYPE_32FC1, z);
  _pubDepth->publish(*cv_depth.toImageMsg());
}

void NodeReplayerExtracted::replayNext()
{
  if (_fId >= _fEnd) {
    _tWaiting = 10;
    _periodicTimer = create_wall_timer(std::chrono::seconds(1), [&]() -> void {
      RCLCPP_INFO(
        get_logger(), "Replay of [%s] has ended. Will shutdown in %ld seconds..",
        _dl->sequenceId().c_str(), _tWaiting);
      _tWaiting--;
      if (_tWaiting <= 0) {
        rclcpp::shutdown();
      }
    });

    return;
  }

  if (!_play) {
    _periodicTimer = create_wall_timer(
      std::chrono::nanoseconds(static_cast<int64_t>(get_parameter("timeout").as_double() * 1e9)),
      [&]() {
        RCLCPP_WARN(get_logger(), "Timed out during waiting for node to be ready. Continuing..");
        std::unique_lock<std::mutex> lk(_mutex);
        _play = true;
        _periodicTimer = create_wall_timer(
          std::chrono::milliseconds(_period), std::bind(&NodeReplayerExtracted::replayNext, this));
      });
    return;
  }

  if (_fId % (_dl->nFrames() / 100) == 0) {
    RCLCPP_INFO(
      get_logger(), "Replayed [%d]%s, [%d/%ld] msgs, t=[%ld]",
      static_cast<int>(static_cast<float>(_fId) / static_cast<float>(_dl->nFrames()) * 100), "%",
      _fId, _dl->nFrames(), _dl->timestamps()[_fId]);
  }

  publish(_fId);
  _fId++;
}

void NodeReplayerExtracted::servicePlayCallback(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  {
    std::unique_lock<std::mutex> lk(_mutex);
    _play = request->data;
  }

  _cond.notify_all();
  response->success = true;
}

}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeReplayerExtracted)