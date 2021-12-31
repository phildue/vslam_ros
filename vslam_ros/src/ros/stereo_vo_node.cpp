//
// Created by phil on 07.08.21.
//

#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <stereo_msgs/msg/disparity_image.hpp>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include "vslam/vslam.h"

class StereoAlignmentROS : public rclcpp::Node
{
    public:
    StereoAlignmentROS()
    : rclcpp::Node("StereoAligner")
    , sync(image_sub,depth_sub,10)
    , ready(false)
    , fNo(0)
    {
        _cameraName = this->declare_parameter<std::string>("camera","/cam_stereo/left");
        

        RCLCPP_INFO(this->get_logger(),"Setting up for camera: %s ..",_cameraName.c_str());

        //rclcpp::topic::waitForMessage<sensor_msgs::CameraInfo>(cameraTopic);

        //ROS_INFO("Waiting for images to be published...");

        //rclcpp::wait::waitForMessage<sensor_msgs::Image>(imageTopic);
        //rclcpp::topic::waitForMessage<stereo_msgs::DisparityImage>(depthMapTopic);

        image_sub.subscribe( this, _cameraName + "/image_rect");
        depth_sub.subscribe( this, _cameraName + "/disparity");
        pub = this->create_publisher<nav_msgs::msg::Odometry>(_cameraName + "/odom", 10);

        config.levelMax = 0;
        config.levelMin = 0;
        config.desiredFeatures = 200;
        config.minGradient = 50;
        config.patchSize = 7;
        camInfoSub = this->create_subscription<sensor_msgs::msg::CameraInfo>(_cameraName+"/camera_info",1,std::bind(&StereoAlignmentROS::cameraCallback,this,std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(),"Ready.");

    }
    ~StereoAlignmentROS()
    {
        RCLCPP_INFO(this->get_logger(),"Done.");

    }
    void imageCallback(sensor_msgs::msg::Image::ConstPtr msgImg, stereo_msgs::msg::DisparityImage::ConstPtr msgDisp)
    {
        if ( !ready )
        {
            return;
        }
        auto cvImage = cv_bridge::toCvCopy(msgImg);
        cv::Mat matGray;
        cv::resize(cvImage->image,matGray,cv::Size(640,480));
//    cv::cvtColor(matGray,matGray,cv::COLOR_RGB2GRAY);
        Image img;
        cv::cv2eigen(matGray,img);

        auto cvDisp = cv_bridge::toCvCopy(msgDisp->image);
        cv::Mat matDisp;
        cv::resize(cvImage->image,matDisp,cv::Size(640,480));
        Eigen::MatrixXd disp;
        cv::cv2eigen(matDisp,disp);
        Eigen::MatrixXd depth = msgDisp->f * msgDisp->t /disp.array();

        const auto ts = msgImg->header.stamp.nanosec;
        Sophus::SE3d result = alignment->align(img,depth,ts);
        //pd::vision::StereoAlignment aligner(config);
        //auto result = aligner.align(img,depth,ts);
        nav_msgs::msg::Odometry msgOut;
        msgOut.header.stamp = msgImg->header.stamp;
        msgOut.header.frame_id = _cameraName;
        const auto t = result.translation();
        const auto q = result.unit_quaternion();
        msgOut.pose.pose.position.x = t.x();
        msgOut.pose.pose.position.y = t.y();
        msgOut.pose.pose.position.z = t.z();
        msgOut.pose.pose.orientation.w = q.w();
        msgOut.pose.pose.orientation.x = q.x();
        msgOut.pose.pose.orientation.y = q.y();
        msgOut.pose.pose.orientation.z = q.z();
        pub->publish(msgOut);
        fNo++;
    }

    void cameraCallback(sensor_msgs::msg::CameraInfo::ConstPtr msg)
    {
        if ( ready )
        {
            return;
        }
        config.fx = msg->k[0*3 + 0];
        config.fy = msg->k[1*3 + 1];
        config.cx = msg->k[0*3 + 2];
        config.cy = msg->k[1*3 + 2];

        alignment = std::make_shared<pd::vision::StereoAlignment>(config);
        sync.registerCallback(std::bind(&StereoAlignmentROS::imageCallback, this,std::placeholders::_1, std::placeholders::_2));
        RCLCPP_INFO(this->get_logger(),"Camera calibration received. Alignment initialized.");
        ready = true;
    }
    bool ready;
    pd::vision::StereoAlignment::Config config;
    std::shared_ptr<pd::vision::StereoAlignment> alignment;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camInfoSub;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub;
    message_filters::Subscriber<stereo_msgs::msg::DisparityImage> depth_sub;
    message_filters::TimeSynchronizer<sensor_msgs::msg::Image,stereo_msgs::msg::DisparityImage> sync;
    int fNo;
    std::string _cameraName;
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
  
    auto stereoAlignmentRos = std::make_shared<StereoAlignmentROS>();

    rclcpp::spin(stereoAlignmentRos);
    rclcpp::shutdown();

}




