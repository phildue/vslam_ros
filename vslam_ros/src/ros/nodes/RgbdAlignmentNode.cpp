//
// Created by phil on 07.08.21.
//

#include "RgbdAlignmentNode.h"
#include "vslam_ros/converters.h"
using namespace pd::vision;
using namespace std::chrono_literals;

namespace vslam_ros{

    RgbdAlignmentNode::RgbdAlignmentNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("RgbdAlignmentNode",options)
    , _camInfoReceived(false)
    , _tfAvailable(false)
    , _fNo(0)
    , _frameId("odom")
    , _fixedFrameId("world")
    , _pubOdom(create_publisher<nav_msgs::msg::Odometry>("/odom", 10))
    , _pubPath(create_publisher<nav_msgs::msg::Path>("/path", 10))
    , _pubTf(std::make_shared<tf2_ros::TransformBroadcaster>(this))
    , _tfBuffer(std::make_unique<tf2_ros::Buffer>(get_clock()))
    , _subCamInfo(create_subscription<sensor_msgs::msg::CameraInfo>("/camera/rgb/camera_info",10,std::bind(&RgbdAlignmentNode::cameraCallback,this,std::placeholders::_1)))
    , _subImage(create_subscription<sensor_msgs::msg::Image>("/camera/rgb/image_color",10,std::bind(&RgbdAlignmentNode::imageCallback,this,std::placeholders::_1)))
    , _subDepth(create_subscription<sensor_msgs::msg::Image>("/camera/depth/image",10,std::bind(&RgbdAlignmentNode::depthCallback,this,std::placeholders::_1)))
    , _subTf(std::make_shared<tf2_ros::TransformListener>(*_tfBuffer))
    , _cliReplayer(create_client<std_srvs::srv::SetBool>("set_ready"))
    , _queue(std::make_shared<vslam_ros::Queue>(10,0.20*1e9))
    {   
        declare_parameter("frame.base_link_id",_fixedFrameId);
        declare_parameter("frame.frame_id",_frameId);
        declare_parameter("features.min_gradient",1);
        declare_parameter("pyramid.levels.max",4);
        declare_parameter("pyramid.levels.min",0);
        declare_parameter("solver.max_iterations",100);
        declare_parameter("solver.min_step_size",1e7);
        declare_parameter("loss.function","None");
        declare_parameter("loss.huber.c",10.0);
        Log::_blockLevel = Level::Unknown;
        Log::_showLevel = Level::Unknown;

        const std::vector<std::string> imageLogs =
        {
            "ImageWarped",
            "Residual",
            "Weights",
            "Template",
            "Image",
            "Depth"
        };
        const std::vector<std::string> plotLogs =
        {
            "ErrorDistribution"
        };
        for (const auto& imageLog : imageLogs)
        {
            declare_parameter("log.image." + imageLog + ".show", false);
            declare_parameter("log.image." + imageLog + ".block",false);
            LOG_IMG(imageLog)->_show = get_parameter("log.image." + imageLog + ".show").as_bool();
            LOG_IMG(imageLog)->_block = get_parameter("log.image." + imageLog + ".block").as_bool();

        }
        for (const auto& plotLog : plotLogs)
        {
            declare_parameter("log.plot." + plotLog + ".show",false);
            declare_parameter("log.plot." + plotLog + ".block",false);
            LOG_PLT(plotLog)->_show = get_parameter("log.plot." + plotLog + ".show").as_bool();
            LOG_PLT(plotLog)->_block = get_parameter("log.plot." + plotLog + ".block").as_bool();

        }
        declare_parameter("prediction.model","NoMotion");

        RCLCPP_INFO(get_logger(),"Setting up..");

        Loss::ShPtr loss;
        auto paramLoss = get_parameter("loss.function").as_string();
        if (paramLoss == "Tukey")
        {
            loss = std::make_shared<TukeyLoss>();
        }else if(paramLoss == "Huber")
        {
            loss = std::make_shared<HuberLoss>(get_parameter("loss.huber.c").as_double());
        }else{
            loss = std::make_shared<QuadraticLoss>();
        }
        _rgbdOdometry = std::make_shared<pd::vision::RgbdOdometry>(get_parameter("features.min_gradient").as_int(),
        get_parameter("pyramid.levels.max").as_int(),get_parameter("pyramid.levels.min").as_int(),
        get_parameter("solver.max_iterations").as_int(),get_parameter("solver.min_step_size").as_double(),loss);
       // _cameraName = this->declare_parameter<std::string>("camera","/camera/rgb");
        //sync.registerDropCallback(std::bind(&StereoAlignmentROS::dropCallback, this,std::placeholders::_1, std::placeholders::_2));
        
        
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
            auto cvDepth = cv_bridge::toCvShare(msgDepth);

            Eigen::MatrixXd depth;
            cv::cv2eigen(cvDepth->image,depth);
            depth = depth.array().isNaN().select(0,depth);
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
        signalReplayer();
    }

    void RgbdAlignmentNode::lookupTf(sensor_msgs::msg::Image::ConstSharedPtr msgImg)
    {
        try{
            _camera2base = \
            _tfBuffer->lookupTransform(_fixedFrameId,msgImg->header.frame_id.substr(1),tf2::TimePointZero);
            _tfAvailable = true;
            
        }catch (tf2::TransformException &ex) {
            RCLCPP_INFO(get_logger(),"%s",ex.what());
        }
    }
    void RgbdAlignmentNode::signalReplayer()
    {
        if(get_parameter("use_sim_time").as_bool())
        {
            while (!_cliReplayer->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                throw std::runtime_error("Interrupted while waiting for the service. Exiting.");
            }
                RCLCPP_INFO(get_logger(), "Replayer Service not available, waiting again...");
            }
            auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
            request->data = true;
            auto result = _cliReplayer->async_send_request(request);
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
        auto x = poseBase.pose().log();
        RCLCPP_INFO(get_logger(),"Dt: %ld PoseBase: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",dT,x(0),x(1),x(2),x(3),x(4),x(5));
              
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = msgImg->header.stamp;
        tf.header.frame_id = "world";
        tf.child_frame_id = msgImg->header.frame_id + "/est";
        vslam_ros::convert(poseBase.pose(),tf);
        _pubTf->sendTransform(tf);

        nav_msgs::msg::Odometry odom;
        odom.header = msgImg->header;
        odom.header.frame_id = _fixedFrameId;
        vslam_ros::convert(poseBase,odom.pose);
        vslam_ros::convert(twistBase,odom.twist);
        _pubOdom->publish(odom);

        geometry_msgs::msg::PoseStamped poseStamped;
        poseStamped.header = odom.header;
        poseStamped.pose = vslam_ros::convert(poseBase.pose());
        _path.header = odom.header;
        _path.poses.push_back(poseStamped);
        _pubPath->publish(_path);
    }

    void RgbdAlignmentNode::depthCallback(sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
    {
        if ( _camInfoReceived )
        {
            _queue->pushDepth(msgDepth);

            if(ready())
            {
                try{
                    auto img = _queue->popClosestImg();
                    processFrame(img,_queue->popClosestDepth(rclcpp::Time(img->header.stamp.sec,img->header.stamp.nanosec).nanoseconds()));
                }catch(const std::runtime_error& e)
                {
                    RCLCPP_WARN(get_logger(),"%s",e.what());
                }
                
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
        if ( _camInfoReceived ){return;}

        _camera = vslam_ros::convert(*msg);
        _camInfoReceived = true;
  
        RCLCPP_INFO(get_logger(),"Camera calibration received. Alignment initialized.");
    }
    
}


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::RgbdAlignmentNode)