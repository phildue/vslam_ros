//
// Created by phil on 07.08.21.
//

#include <ros/time.h>
#include <ros/publisher.h>
#include <nav_msgs/Odometry.h>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/rate.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <stereo_msgs/DisparityImage.h>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include "vslam/vslam.h"



class DataExtractorROS
{
public:
    DataExtractorROS(std::shared_ptr<ros::NodeHandle> n, const std::string& imageTopic, const std::string& depthMapTopic, const std::string& cameraTopic)
    : image_sub(*n, imageTopic, 1)
    , depth_sub(*n, depthMapTopic,1)
    , sync(image_sub,depth_sub,10)
    , camInfoSub(n->subscribe(cameraTopic,1,&DataExtractorROS::cameraCallback,this))
    , pub(n->advertise<nav_msgs::Odometry>("/odom", 10))
    , ready(false)
    , fNo(0)
    {

    }
    void imageCallback(sensor_msgs::ImageConstPtr msgImg, stereo_msgs::DisparityImageConstPtr msgDisp)
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
        Eigen::MatrixXd depth = msgDisp->f * msgDisp->T /disp.array();

        const auto ts = msgImg->header.stamp.toNSec();
        std::string path = "/home/phil/code/bot_ws/log/";
        std::stringstream ss;

        ss << path << std::to_string(ts);
//        pd::vision::utils::saveImage(img,ss.str());
//        pd::vision::utils::saveDepth(depth, ss.str());
        fNo++;
    }

    void cameraCallback(sensor_msgs::CameraInfoConstPtr msg)
    {
        auto fx = msg->K[0*3 + 0];
        auto fy = msg->K[1*3 + 1];
        auto cx = msg->K[0*3 + 2];
        auto cy = msg->K[1*3 + 2];

        ROS_INFO("%s", ("fx:" + std::to_string(fx)).c_str());
        ROS_INFO("%s", ("fy:" + std::to_string(fy)).c_str());
        ROS_INFO("%s", ("cx:" + std::to_string(cx)).c_str());
        ROS_INFO("%s", ("cy:" + std::to_string(cy)).c_str());
        ROS_INFO("fx : %f, fy: %f, cx: %f, cy: %f",fx,fy,cx,cy);

        sync.registerCallback(boost::bind(&DataExtractorROS::imageCallback, this,_1, _2));
        ready = true;
    }
    bool ready;
    ros::Publisher pub;
    ros::Subscriber camInfoSub;
    message_filters::Subscriber<sensor_msgs::Image> image_sub;
    message_filters::Subscriber<stereo_msgs::DisparityImage> depth_sub;
    message_filters::TimeSynchronizer<sensor_msgs::Image,stereo_msgs::DisparityImage> sync;
    int fNo;
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "data_extractor_node");
    auto n = std::make_shared<ros::NodeHandle> ("~");
    std::string cameraTopic,imageTopic,depthMapTopic;
    n->getParam("camera_info_topic",cameraTopic);
    n->getParam("image_topic",imageTopic);
    n->getParam("disparity_topic",depthMapTopic);

    ROS_INFO("Waiting for camera topic to be published...");

    ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cameraTopic);

    ROS_INFO("Waiting for images to be published...");

    ros::topic::waitForMessage<sensor_msgs::Image>(imageTopic);
    ros::topic::waitForMessage<stereo_msgs::DisparityImage>(depthMapTopic);

    DataExtractorROS stereoAlignmentRos(n,imageTopic,depthMapTopic,cameraTopic);

    ros::Rate loop_rate(50);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

}




