//
// Created by phil on 07.08.21.
//

#include "RgbdAlignmentNode.h"
#include "vslam_ros/converters.h"
using namespace pd::vision;
namespace vslam_ros{

    RgbdAlignmentNode::RgbdAlignmentNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("RgbdAlignmentNode",options)
    , _camInfoReceived(false)
    , _fNo(0)
    , _queue(std::make_shared<vslam_ros::Queue>(1000,10000000))
    , _minGradient(30)
    , _rgbdOdometry (std::make_shared<pd::vision::RgbdOdometry>(_minGradient,4,20,1e-3))
    , _tfBuffer(std::make_unique<tf2_ros::Buffer>(this->get_clock()))
    , _tfListener(std::make_shared<tf2_ros::TransformListener>(*_tfBuffer))
    , _pubTf(std::make_shared<tf2_ros::TransformBroadcaster>(this))
    , _frameId("odom")
    , _baseLinkId("world")

    {
       // _cameraName = this->declare_parameter<std::string>("camera","/camera/rgb");
        RCLCPP_INFO(get_logger(),"Setting up for camera: %s ..",_cameraName.c_str());

        _camInfoSub = create_subscription<sensor_msgs::msg::CameraInfo>("/camera/rgb/camera_info",10,std::bind(&RgbdAlignmentNode::cameraCallback,this,std::placeholders::_1));
        _imageSub = create_subscription<sensor_msgs::msg::Image>("/camera/rgb/image_color",10,std::bind(&RgbdAlignmentNode::imageCallback,this,std::placeholders::_1));
        _depthSub = create_subscription<sensor_msgs::msg::Image>("/camera/depth/image",10,std::bind(&RgbdAlignmentNode::depthCallback,this,std::placeholders::_1));

        _pubOdom = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
        _pubPathVo = create_publisher<nav_msgs::msg::Path>("/path", 10);
        //sync.registerDropCallback(std::bind(&StereoAlignmentROS::dropCallback, this,std::placeholders::_1, std::placeholders::_2));
        
        Log::_blockLevel = Level::Unknown;
        Log::_showLevel = Level::Unknown;
        LOG_IMG("Image Warped")->_block = false;
        //LOG_PLT("SolverGN",Level::Debug)->_block = true;
        LOG_PLT("SolverGN")->_show = false;
        LOG_PLT("ErrorDistribution")->_show = false;
        LOG_PLT("ErrorDistribution")->_block = false;
        
        LOG_IMG("Depth")->_block = false;
        LOG_IMG("Residual")->_show = false;
        LOG_IMG("ImageWarped")->_show = false;


        RCLCPP_INFO(get_logger(),"Ready.");
    }

    bool RgbdAlignmentNode::ready(){
        return _queue->size() >= 1;
    } 
    void RgbdAlignmentNode::processFrame(sensor_msgs::msg::Image::ConstSharedPtr msgImg,sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
    {
        TIMED_FUNC(timerF);

        try{

            auto cvImage = cv_bridge::toCvShare(msgImg);
            cv::Mat mat = cvImage->image;
            cv::cvtColor(mat,mat,cv::COLOR_RGB2GRAY);
            Image img;
            cv::cv2eigen(mat,img);
            img = algorithm::resize(img,_scale);
            auto cvDepth = cv_bridge::toCvShare(msgDepth);

            Eigen::MatrixXd depth;
            cv::cv2eigen(cvDepth->image,depth);
            depth = depth.array().isNaN().select(0,depth);
            depth = algorithm::resize(depth,_scale);
            const Timestamp t = rclcpp::Time(msgImg->header.stamp.sec,msgImg->header.stamp.nanosec).nanoseconds();
            const long int dT = _lastFrame ? t - _lastFrame->t() : 0;
            
            PoseWithCovariance::ConstShPtr pose = std::make_shared<PoseWithCovariance>(Vec6d::Zero(),Matd<6,6>::Identity());
            if ( _lastFrame )
            {
                auto curFrame = std::make_shared<const pd::vision::FrameRgb>(img, _camera,
                rclcpp::Time(msgImg->header.stamp.sec,msgImg->header.stamp.nanosec).nanoseconds(),_lastFrame->pose());
            
                RCLCPP_INFO(get_logger(),"Aligning %d to %d.",_fNo-1,_fNo);

                LOG_IMG("Image") << img;
                LOG_IMG("Template") << _lastFrame->rgb();
                LOG_IMG("Depth") << _lastFrame->depth();
                pose = _rgbdOdometry->align(_lastFrame, curFrame);
          
                if(!_tfAvailable)
                {
                    lookupTf(msgImg);
                }
                if(_tfAvailable)
                {
                    publish(msgImg, pose);
                }
                
            }

            auto x = pose->pose().log();
            RCLCPP_INFO(get_logger(),"Dt: %ld Pose: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",dT,x(0),x(1),x(2),x(3),x(4),x(5));
              
            _lastFrame = std::make_shared<const pd::vision::FrameRgbd>(img, depth, _camera,
             rclcpp::Time(msgImg->header.stamp.sec,msgImg->header.stamp.nanosec).nanoseconds(), *pose);
           
            _fNo++;

        }catch(const std::runtime_error& e)
        {
            RCLCPP_WARN(this->get_logger(),"%s",e.what());
        }

    }

    void RgbdAlignmentNode::lookupTf(sensor_msgs::msg::Image::ConstSharedPtr msgImg)
    {
        try{
            _camera2base = \
            _tfBuffer->lookupTransform(_baseLinkId,msgImg->header.frame_id.substr(1),tf2::TimePointZero);
            _tfAvailable = true;
            
        }catch (tf2::TransformException &ex) {
            RCLCPP_INFO(get_logger(),"%s",ex.what());
        }
    }

    void RgbdAlignmentNode::publish(sensor_msgs::msg::Image::ConstSharedPtr msgImg, const pd::vision::PoseWithCovariance::ConstShPtr poseEst)
    {
        PoseWithCovariance::UnPtr odomEst = std::make_unique<PoseWithCovariance>(
            algorithm::computeRelativeTransform(_lastFrame->pose().pose(),poseEst->pose()),poseEst->cov());
        Timestamp t = rclcpp::Time(msgImg->header.stamp.sec,msgImg->header.stamp.nanosec).nanoseconds();
        long int dT = _lastFrame ? t - _lastFrame->t() : 0;
            
        auto twistBase = vslam_ros::convert(_camera2base) * PoseWithCovariance( pd::vision::SE3d::exp(odomEst->mean()/dT), odomEst->cov()/dT );
        auto poseBase = vslam_ros::convert(_camera2base) * (*poseEst);

        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = msgImg->header.stamp;
        tf.header.frame_id = "world";
        tf.child_frame_id = msgImg->header.frame_id + "/est";
        vslam_ros::convert(poseBase.pose(),tf);

        nav_msgs::msg::Odometry odom;
        odom.header = msgImg->header;
        odom.header.frame_id = _baseLinkId;
        vslam_ros::convert(poseBase,odom.pose);
        vslam_ros::convert(twistBase,odom.twist);

        geometry_msgs::msg::PoseStamped poseStamped;
        poseStamped.header = odom.header;

        poseStamped.pose = vslam_ros::convert(poseBase.pose());
        _pathVo.header = odom.header;
        _pathVo.poses.push_back(poseStamped);
                
        _pubOdom->publish(odom);
        _pubPathVo->publish(_pathVo);
        _pubTf->sendTransform(tf);
    }

    void RgbdAlignmentNode::depthCallback(sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
    {
        if ( _camInfoReceived )
        {
            _queue->pushDepth(msgDepth);

            if(ready())
            {
                auto img = _queue->popClosestImg();
                processFrame(img,_queue->popClosestDepth(rclcpp::Time(img->header.stamp.sec,img->header.stamp.nanosec).nanoseconds()));
            }
        }
    }

    void RgbdAlignmentNode::imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg)
    {
        if ( _camInfoReceived )
        {
            _queue->pushImage(msgImg);
        }
      
    }
    void RgbdAlignmentNode::dropCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
    {
        RCLCPP_INFO(get_logger(), "Message dropped.");
        if(msgImg)
        {
            const auto ts = msgImg->header.stamp.nanosec;
            RCLCPP_INFO(get_logger(), "Image: %10.0f",(double)ts);
        }
        if(msgDepth)
        {
            const auto ts = msgDepth->header.stamp.nanosec;
            RCLCPP_INFO(get_logger(), "Depth: %10.0f",(double)ts);
        }
    }

    void RgbdAlignmentNode::cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
    {
        if ( _camInfoReceived )
        {
            return;
        }
        _camera = vslam_ros::convert(*msg);
        _camera = Camera::resize(_camera,_scale);
        
        
        RCLCPP_INFO(get_logger(),"Camera calibration received. Alignment initialized.");
        _camInfoReceived = true;
    }
    
}


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::RgbdAlignmentNode)