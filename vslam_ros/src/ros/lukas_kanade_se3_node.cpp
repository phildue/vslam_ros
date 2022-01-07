//
// Created by phil on 07.08.21.
//

#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "approximate_time_synchronizer.h"
#include <stereo_msgs/msg/disparity_image.hpp>
#include <nav_msgs/msg/path.hpp>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include "vslam_ros/vslam_ros.h"

using namespace pd::vision;

class StereoAlignmentROS : public rclcpp::Node
{
    public:
    StereoAlignmentROS()
    : rclcpp::Node("StereoAligner")
    , sync(image_sub,depth_sub,_queueSize)
    , ready(false)
    , _fNo(0)
    {
       // _cameraName = this->declare_parameter<std::string>("camera","/camera/rgb");
        

        RCLCPP_INFO(this->get_logger(),"Setting up for camera: %s ..",_cameraName.c_str());

        //rclcpp::topic::waitForMessage<sensor_msgs::CameraInfo>(cameraTopic);

        //ROS_INFO("Waiting for images to be published...");

        //rclcpp::wait::waitForMessage<sensor_msgs::Image>(imageTopic);
        //rclcpp::topic::waitForMessage<stereo_msgs::DisparityImage>(depthMapTopic);

        image_sub.subscribe( this, "/camera/rgb/image_color");
        depth_sub.subscribe( this, "/camera/depth/image");
        _pubOdom = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
        _pubPathVo = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
        sync.registerCallback(std::bind(&StereoAlignmentROS::imageCallback, this,std::placeholders::_1, std::placeholders::_2));
        sync.setMaxIntervalDuration(rclcpp::Duration::from_nanoseconds(10000000));
        //sync.registerDropCallback(std::bind(&StereoAlignmentROS::dropCallback, this,std::placeholders::_1, std::placeholders::_2));
        camInfoSub = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/rgb/camera_info",1,std::bind(&StereoAlignmentROS::cameraCallback,this,std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(),"Ready.");
        Log::_blockLevel = Level::Unknown;
        Log::_showLevel = Level::Info;
        Log::getImageLog("Image Warped")->_block = false;
        //Log::getPlotLog("SolverGN",Level::Debug)->_block = true;
        Log::getPlotLog("SolverGN",Level::Debug)->_show = false;
        Log::getPlotLog("ErrorDistribution",Level::Debug)->_show = false;
        Log::getPlotLog("ErrorDistribution",Level::Debug)->_block = false;
        
        Log::getImageLog("Depth")->_block = false;

    }
    ~StereoAlignmentROS()
    {
        RCLCPP_INFO(this->get_logger(),"Done.");

    }
    void imageCallback(sensor_msgs::msg::Image::ConstPtr msgImg, sensor_msgs::msg::Image::ConstPtr msgDepth)
    {
        if ( !ready )
        {
            return;
        }
        auto cvImage = cv_bridge::toCvCopy(*msgImg);
        cv::Mat mat = cvImage->image;
        cv::cvtColor(mat,mat,cv::COLOR_RGB2GRAY);
        Image img;
        cv::cv2eigen(mat,img);
        img = algorithm::resize(img,_scale);
        auto cvDepth = cv_bridge::toCvCopy(*msgDepth);
        cv::imshow("CvDepth",cvDepth->image);
        cv::imshow("CvImg",cvImage->image);

        Eigen::MatrixXd depth;
        cv::cv2eigen(cvDepth->image,depth);
        depth = depth.array().isNaN().select(0,depth);
        depth = algorithm::resize(depth,_scale);
        

        if (_fNo < 1 )
        {
            _lastImg = img;
            _depth = depth;
        }else{
            RCLCPP_INFO(this->get_logger(),"Aligning %d to %d.",_fNo-1,_fNo);
            Log::getImageLog("Image")->append(img);
            Log::getImageLog("Template")->append(_lastImg);
            Log::getImageLog("Depth")->append(depth);

            auto w = std::make_shared<WarpSE3>(Eigen::Vector6d::Zero(),depth,_camera);
            auto l = std::make_shared<HuberLoss>(10);
            auto solver = std::make_shared<GaussNewton<LukasKanadeInverseCompositional<WarpSE3>>> ( 
                            1.0,
                            1e-4,
                            20);
            
            for(int i = 4; i > 0; i--)
            {
                    const auto s = 1.0/(double)i;
                    
                    auto templScaled = algorithm::resize(_lastImg,s);
                    auto imageScaled = algorithm::resize(img,s);
                    auto wScaled = std::make_shared<WarpSE3>(w->x(),algorithm::resize(_depth,s),Camera::resize(_camera,s));

                    auto lk = std::make_shared<LukasKanadeInverseCompositional<WarpSE3>> (
                            templScaled,
                            imageScaled,
                            wScaled,l);

                    solver->solve(lk);
                    
                    w->setX(wScaled->x());
                
            }

            _lastImg = img;
            _depth = depth;
            _pose = w->pose() * _pose;
            auto x = _pose.log();
            RCLCPP_INFO(this->get_logger(),"Pose: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",x(0),x(1),x(2),x(3),x(4),x(5));

            auto pose = vslam_ros2::convert(_pose);
            nav_msgs::msg::Odometry odom;
            odom.header = msgImg->header;
            odom.pose.pose = pose;
            geometry_msgs::msg::PoseStamped poseStamped;
            poseStamped.header = odom.header;
            poseStamped.pose = pose;
            _pathVo.header = odom.header;
            _pathVo.poses.push_back(poseStamped);
                    
            _pubOdom->publish(odom);
            _pubPathVo->publish(_pathVo);
        }
       
        _fNo++;
    }
    void dropCallback(sensor_msgs::msg::Image::ConstPtr msgImg, sensor_msgs::msg::Image::ConstPtr msgDepth)
    {
        RCLCPP_INFO(this->get_logger(), "Message dropped.");
        if(msgImg)
        {
            const auto ts = msgImg->header.stamp.nanosec;
            RCLCPP_INFO(this->get_logger(), "Image: %10.0f",(double)ts);
        }
        if(msgDepth)
        {
            const auto ts = msgDepth->header.stamp.nanosec;
            RCLCPP_INFO(this->get_logger(), "Depth: %10.0f",(double)ts);
        }
    }

    void cameraCallback(sensor_msgs::msg::CameraInfo::ConstPtr msg)
    {
        if ( ready )
        {
            return;
        }
        _camera = vslam_ros2::convert(*msg);

        RCLCPP_INFO(this->get_logger(),"Camera calibration received. Alignment initialized.");
        ready = true;
    }
    bool ready;
    const double _scale = 1.0;
    nav_msgs::msg::Path _pathImu,_pathVo;
    Sophus::SE3d _pose;
    Image _lastImg;
    pd::vision::Camera::ShPtr _camera;
    int _fNo;
    std::string _cameraName;
    int _queueSize = 1000;
    DepthMap _depth;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _pubOdom;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pubPathVo;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camInfoSub;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub;
    message_filters::ApproximateTimeSynchronizer<sensor_msgs::msg::Image,sensor_msgs::msg::Image> sync;
    
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
  
    auto stereoAlignmentRos = std::make_shared<StereoAlignmentROS>();

    rclcpp::spin(stereoAlignmentRos);
    rclcpp::shutdown();

}




