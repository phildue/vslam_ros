#ifndef VSLAM_ROS2_LUKAS_KANADE_SE3_NODE_H__
#define VSLAM_ROS2_LUKAS_KANADE_SE3_NODE_H__

#include <chrono>
#include <memory>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_srvs/srv/set_bool.hpp>

#include "vslam_ros/Queue.h"
#include "vslam_ros/visibility_control.h"
#include "vslam/vslam.h"



namespace vslam_ros{
class RgbdAlignmentNode : public rclcpp::Node
{
    public:
    COMPOSITION_PUBLIC
    RgbdAlignmentNode(const rclcpp::NodeOptions& options);
    
    bool ready();
    void processFrame(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth);

    void depthCallback(sensor_msgs::msg::Image::ConstSharedPtr msgDepth);

    void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg);
    void dropCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth);

    void cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    pd::vslam::FrameRgbd::ShPtr createFrame(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth) const;

    private:
    const bool _includeKeyFrame;
    bool _camInfoReceived;
    bool _tfAvailable;
    int _fNo;
    std::string _frameId;
    std::string _fixedFrameId;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _pubOdom;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pubPath;
    std::shared_ptr<tf2_ros::TransformBroadcaster> _pubTf;

    std::unique_ptr<tf2_ros::Buffer> _tfBuffer;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _subCamInfo;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subImage;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subDepth;
    std::shared_ptr<tf2_ros::TransformListener> _subTf;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr _cliReplayer;
    
    const std::shared_ptr<vslam_ros::Queue> _queue;

    pd::vslam::Odometry::ShPtr _odometry;
    pd::vslam::KeyFrameSelection::ShPtr _keyFrameSelection;
    pd::vslam::MotionPrediction::ShPtr _prediction;
    pd::vslam::Map::ShPtr _map;

    pd::vslam::Camera::ShPtr _camera;
    geometry_msgs::msg::TransformStamped _world2origin; //transforms from fixed frame to initial pose of optical frame
    nav_msgs::msg::Path _path;


    void publish(sensor_msgs::msg::Image::ConstSharedPtr msgImg);
    void lookupTf(sensor_msgs::msg::Image::ConstSharedPtr msgImg);
    void signalReplayer();

};
}

#endif //VSLAM_ROS2_LUKAS_KANADE_SE3_NODE