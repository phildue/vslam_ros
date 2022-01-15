//
// Created by phil on 07.08.21.
//

#include "LukasKanadeSE3Node.h"
#include "vslam_ros/converters.h"
using namespace pd::vision;
namespace vslam_ros{

    LukasKanadeSE3Node::LukasKanadeSE3Node(const rclcpp::NodeOptions& options)
    : rclcpp::Node("LukasKanadeSE3Node",options)
    , _camInfoReceived(false)
    , _fNo(0)
    , _queue(std::make_shared<vslam_ros::Queue>(1000,10000000))
    {
       // _cameraName = this->declare_parameter<std::string>("camera","/camera/rgb");
        RCLCPP_INFO(this->get_logger(),"Setting up for camera: %s ..",_cameraName.c_str());

        _camInfoSub = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/rgb/camera_info",10,std::bind(&LukasKanadeSE3Node::cameraCallback,this,std::placeholders::_1));
        _imageSub = this->create_subscription<sensor_msgs::msg::Image>("/camera/rgb/image_color",10,std::bind(&LukasKanadeSE3Node::imageCallback,this,std::placeholders::_1));
        _depthSub = this->create_subscription<sensor_msgs::msg::Image>("/camera/depth/image",10,std::bind(&LukasKanadeSE3Node::depthCallback,this,std::placeholders::_1));

        _pubOdom = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
        _pubPathVo = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
        //sync.registerDropCallback(std::bind(&StereoAlignmentROS::dropCallback, this,std::placeholders::_1, std::placeholders::_2));
        
        Log::_blockLevel = Level::Unknown;
        Log::_showLevel = Level::Info;
        Log::getImageLog("Image Warped")->_block = false;
        //Log::getPlotLog("SolverGN",Level::Debug)->_block = true;
        Log::getPlotLog("SolverGN",Level::Debug)->_show = false;
        Log::getPlotLog("ErrorDistribution",Level::Debug)->_show = false;
        Log::getPlotLog("ErrorDistribution",Level::Debug)->_block = false;
        
        Log::getImageLog("Depth")->_block = false;
        LOG_IMAGE_DEBUG("Residual")->_show = false;
        LOG_IMAGE_DEBUG("ImageWarped")->_show = true;

        RCLCPP_INFO(this->get_logger(),"Ready.");

    }

    bool LukasKanadeSE3Node::ready(){
        return _queue->size() >= 1;
    } 
    void LukasKanadeSE3Node::processFrame(sensor_msgs::msg::Image::ConstPtr msgImg,sensor_msgs::msg::Image::ConstPtr msgDepth)
    {
        try{

            auto cvImage = cv_bridge::toCvCopy(*msgImg);
            cv::Mat mat = cvImage->image;
            cv::cvtColor(mat,mat,cv::COLOR_RGB2GRAY);
            Image img;
            cv::cv2eigen(mat,img);
            img = algorithm::resize(img,_scale);
            auto cvDepth = cv_bridge::toCvCopy(*msgDepth);

            Eigen::MatrixXd depth;
            cv::cv2eigen(cvDepth->image,depth);
            depth = depth.array().isNaN().select(0,depth);
            depth = algorithm::resize(depth,_scale);
        

            if (_fNo > 1 )
            {
                RCLCPP_INFO(this->get_logger(),"Aligning %d to %d.",_fNo-1,_fNo);

                Log::getImageLog("Image")->append(img);
                Log::getImageLog("Template")->append(_lastImg);
                Log::getImageLog("Depth")->append(_lastDepth);

                Sophus::SE3d dPose;
                auto l = std::make_shared<HuberLoss>(10);
                auto solver = std::make_shared<GaussNewton<LukasKanadeInverseCompositional<WarpSE3>>> ( 
                                1.0,
                                1e-4,
                                10);
                
                for(int i = 4; i > 1; i--)
                {
                        const auto s = 1.0/(double)i;
                        
                        auto templScaled = algorithm::resize(_lastImg,s);
                        auto depthScaled = algorithm::resize(_lastDepth,s);
                        auto imageScaled = algorithm::resize(img,s);
                        auto w = std::make_shared<WarpSE3>(dPose.log(),depthScaled,Camera::resize(_camera,s));

                        auto lk = std::make_shared<LukasKanadeInverseCompositional<WarpSE3>> (
                                templScaled,
                                imageScaled,
                                w,l);

                        solver->solve(lk);
                        
                        dPose = w->pose();
                    
                }
                _pose = dPose * _pose;

            }

            auto x = _pose.log();
            rclcpp::Time t(msgImg->header.stamp.sec,msgImg->header.stamp.nanosec);
            long int dT = (t - _lastT).nanoseconds();
            RCLCPP_INFO(this->get_logger(),"Dt: %ld Pose: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",dT,x(0),x(1),x(2),x(3),x(4),x(5));
                
            _lastImg = img;
            _lastDepth = depth;
            _lastT = t;

            auto pose = vslam_ros::convert(_pose);
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
            
        
            _fNo++;

        }catch(const std::runtime_error& e)
        {
            RCLCPP_WARN(this->get_logger(),"%s",e.what());
        }

    }

    void LukasKanadeSE3Node::depthCallback(sensor_msgs::msg::Image::ConstPtr msgDepth)
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

    void LukasKanadeSE3Node::imageCallback(sensor_msgs::msg::Image::ConstPtr msgImg)
    {
        if ( _camInfoReceived )
        {
            _queue->pushImage(msgImg);
        }
      
    }
    void LukasKanadeSE3Node::dropCallback(sensor_msgs::msg::Image::ConstPtr msgImg, sensor_msgs::msg::Image::ConstPtr msgDepth)
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

    void LukasKanadeSE3Node::cameraCallback(sensor_msgs::msg::CameraInfo::ConstPtr msg)
    {
        if ( _camInfoReceived )
        {
            return;
        }
        _camera = vslam_ros::convert(*msg);
        _camera = Camera::resize(_camera,_scale);

        RCLCPP_INFO(this->get_logger(),"Camera calibration received. Alignment initialized.");
        _camInfoReceived = true;
    }
    
}


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::LukasKanadeSE3Node)