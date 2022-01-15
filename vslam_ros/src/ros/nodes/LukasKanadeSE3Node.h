#ifndef VSLAM_ROS2_LUKAS_KANADE_SE3_NODE_H__
#define VSLAM_ROS2_LUKAS_KANADE_SE3_NODE_H__

#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <nav_msgs/msg/path.hpp>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include "vslam_ros/Queue.h"
#include "vslam_ros/visibility_control.h"
#include "vslam/vslam.h"
namespace vslam_ros{
class LukasKanadeSE3Node : public rclcpp::Node
{
    public:
    COMPOSITION_PUBLIC
    LukasKanadeSE3Node(const rclcpp::NodeOptions& options);
    
    bool ready();
    void processFrame(sensor_msgs::msg::Image::ConstPtr msgImg, sensor_msgs::msg::Image::ConstPtr msgDepth);

    void depthCallback(sensor_msgs::msg::Image::ConstPtr msgDepth);

    void imageCallback(sensor_msgs::msg::Image::ConstPtr msgImg);
    void dropCallback(sensor_msgs::msg::Image::ConstPtr msgImg, sensor_msgs::msg::Image::ConstPtr msgDepth);

    void cameraCallback(sensor_msgs::msg::CameraInfo::ConstPtr msg);

    private:

    bool _camInfoReceived;
    const double _scale = 0.5;
    nav_msgs::msg::Path _pathImu,_pathVo;
    Sophus::SE3d _pose;
    pd::vision::Image _lastImg;
    pd::vision::DepthMap _lastDepth;
    rclcpp::Time _lastT;
    pd::vision::Camera::ShPtr _camera;
    int _fNo;
    std::string _cameraName;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _pubOdom;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pubPathVo;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camInfoSub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _imageSub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _depthSub;
    const std::shared_ptr<vslam_ros::Queue> _queue;
    
};
}

#endif //VSLAM_ROS2_LUKAS_KANADE_SE3_NODE