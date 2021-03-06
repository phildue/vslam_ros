//
// Created by phil on 07.08.21.
//

#include "RgbdAlignmentNode.h"
#include "vslam_ros/converters.h"
using namespace pd::vslam;
using namespace std::chrono_literals;

namespace vslam_ros{

    RgbdAlignmentNode::RgbdAlignmentNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("RgbdAlignmentNode",options)
    , _includeKeyFrame(false)
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
        declare_parameter("pyramid.levels",std::vector<double>({0.25,0.5,1.0}));
        declare_parameter("solver.max_iterations",100);
        declare_parameter("solver.min_step_size",1e-7);
        declare_parameter("loss.function","None");
        declare_parameter("loss.huber.c",10.0);
        declare_parameter("loss.tdistribution.v",5.0);
        declare_parameter("keyframe_selection.method","idx");
        declare_parameter("keyframe_selection.idx.period",5);
        declare_parameter("prediction.model","NoMotion");
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

        RCLCPP_INFO(get_logger(),"Setting up..");

        least_squares::Loss::ShPtr loss = nullptr;
        least_squares::Scaler::ShPtr scaler;
        auto paramLoss = get_parameter("loss.function").as_string();
        if (paramLoss == "Tukey")
        {
            loss = std::make_shared<least_squares::TukeyLoss>( std::make_shared<least_squares::MedianScaler>());
        }else if(paramLoss == "Huber")
        {
            loss = std::make_shared<least_squares::HuberLoss>(std::make_shared<least_squares::MedianScaler>(),get_parameter("loss.huber.c").as_double());
        }else if(paramLoss == "tdistribution")
        {
            loss = std::make_shared<least_squares::LossTDistribution>(std::make_shared<least_squares::ScalerTDistribution>(get_parameter("loss.tdistribution.v").as_double()),get_parameter("loss.tdistribution.v").as_double());
        }
        
        auto solver = std::make_shared<least_squares::GaussNewton<6>>(
            get_parameter("solver.min_step_size").as_double(),
            get_parameter("solver.max_iterations").as_int()
        );
        _map = std::make_shared<Map>();
        _odometry = std::make_shared<OdometryRgbd>(
            get_parameter("features.min_gradient").as_int(),
            solver, loss, _map);
       // _odometry = std::make_shared<pd::vision::OdometryIcp>(1,10,_map);
        _prediction = MotionPrediction::make(get_parameter("prediction.model").as_string());
        _keyFrameSelection = std::make_shared<KeyFrameSelectionIdx>(get_parameter("keyframe_selection.idx.period").as_int());
       // _cameraName = this->declare_parameter<std::string>("camera","/camera/rgb");
        //sync.registerDropCallback(std::bind(&StereoAlignmentROS::dropCallback, this,std::placeholders::_1, std::placeholders::_2));
        
        
        RCLCPP_INFO(get_logger(),"Ready.");
    }

    bool RgbdAlignmentNode::ready(){
        return _queue->size() >= 1;
    }
    
    void RgbdAlignmentNode::processFrame(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
    {
        TIMED_FUNC(timerF);

        try{

            auto frame = createFrame(msgImg,msgDepth);

            frame->set(*_prediction->predict(frame->t()));

            _odometry->update(frame);
            
            frame->set(*_odometry->pose());

            _prediction->update(_odometry->pose(),frame->t());

            _keyFrameSelection->update(frame);
            
            _map->update(frame, _keyFrameSelection->isKeyFrame());

            publish(msgImg);

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
            _world2origin = \
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

    FrameRgbd::ShPtr  RgbdAlignmentNode::createFrame(sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth) const
    {
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
        
        return std::make_shared<FrameRgbd>(img,depth,_camera,get_parameter("pyramid.levels").as_double_array().size(),t);
    }
    void RgbdAlignmentNode::publish(sensor_msgs::msg::Image::ConstSharedPtr msgImg)
    {
        if(!_tfAvailable)
        {
            lookupTf(msgImg);
            return;
        }
        
        auto x = _odometry->pose()->pose().inverse().log();
        RCLCPP_INFO(get_logger(),"Pose: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",x(0),x(1),x(2),x(3),x(4),x(5));

        // Send the transformation from fixed frame to origin of optical frame
        // TODO possibly only needs to be sent once
        geometry_msgs::msg::TransformStamped tfOrigin = _world2origin;
        tfOrigin.header.stamp = msgImg->header.stamp;
        tfOrigin.header.frame_id = _fixedFrameId;
        tfOrigin.child_frame_id = _frameId;
        _pubTf->sendTransform(tfOrigin);

        // Send current camera pose as estimate for pose of optical frame
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = msgImg->header.stamp;
        tf.header.frame_id = _frameId;
        tf.child_frame_id = "camera"; //camera name?
        vslam_ros::convert(_odometry->pose()->pose().inverse(),tf);
        _pubTf->sendTransform(tf);

        // Send pose, twist and path in optical frame
        nav_msgs::msg::Odometry odom;
        odom.header = msgImg->header;
        odom.header.frame_id = _frameId;
        vslam_ros::convert(_odometry->pose()->inverse(),odom.pose);
        vslam_ros::convert(_odometry->speed()->inverse(),odom.twist);
        _pubOdom->publish(odom);

        geometry_msgs::msg::PoseStamped poseStamped;
        poseStamped.header = odom.header;
        poseStamped.pose = vslam_ros::convert(_odometry->pose()->pose().inverse());
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