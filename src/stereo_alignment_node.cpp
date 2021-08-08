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



class StereoAlignmentROS
{
public:
    StereoAlignmentROS(std::shared_ptr<ros::NodeHandle> n, const std::string& imageTopic, const std::string& depthMapTopic, const std::string& cameraTopic)
    : image_sub(*n, imageTopic, 1)
    , depth_sub(*n, depthMapTopic,1)
    , sync(image_sub,depth_sub,10)
    , camInfoSub(n->subscribe(cameraTopic,1,&StereoAlignmentROS::cameraCallback,this))
    , config()
    , pub(n->advertise<nav_msgs::Odometry>("/odom", 10))
    , ready(false)
    {
        config.levelMax = 1;
        config.levelMin = 0;
        config.desiredFeatures = 50;
        config.minGradient = 0;
        config.patchSize = 7;

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
        Eigen::MatrixXi img;
        cv::cv2eigen(matGray,img);

        auto cvDisp = cv_bridge::toCvCopy(msgDisp->image);
        cv::Mat matDisp;
        cv::resize(cvImage->image,matDisp,cv::Size(640,480));
        Eigen::MatrixXd disp;
        cv::cv2eigen(matDisp,disp);
        Eigen::MatrixXd depth = msgDisp->f * msgDisp->T /disp.array();

        const auto ts = msgImg->header.stamp.toNSec();
        Sophus::SE3d result = alignment->align(Eigen::Matrix<int,640,480>::Ones(),Eigen::Matrix<double,640,480>::Ones(),ts);
        //pd::vision::StereoAlignment aligner(config);
        //auto result = aligner.align(img,depth,ts);
        nav_msgs::Odometry msgOut;
        msgOut.header.stamp = msgImg->header.stamp;
        const auto t = result.translation();
        const auto q = result.unit_quaternion();
        msgOut.pose.pose.position.x = t.x();
        msgOut.pose.pose.position.y = t.y();
        msgOut.pose.pose.position.z = t.z();
        msgOut.pose.pose.orientation.w = q.w();
        msgOut.pose.pose.orientation.x = q.x();
        msgOut.pose.pose.orientation.y = q.y();
        msgOut.pose.pose.orientation.z = q.z();
        pub.publish(msgOut);
    }

    void cameraCallback(sensor_msgs::CameraInfoConstPtr msg)
    {
        if ( ready )
        {
            return;
        }
        config.fx = msg->K[0*3 + 0];
        config.fy = msg->K[1*3 + 1];
        config.cx = msg->K[0*3 + 2];
        config.cy = msg->K[1*3 + 2];

        alignment = std::make_shared<pd::vision::StereoAlignment>(config);
        sync.registerCallback(boost::bind(&StereoAlignmentROS::imageCallback, this,_1, _2));
        ROS_INFO("Camera calibration received. Alignment initialized.");
        ready = true;
    }
    bool ready;
    pd::vision::StereoAlignment::Config config;
    std::shared_ptr<pd::vision::StereoAlignment> alignment;
    ros::Publisher pub;
    ros::Subscriber camInfoSub;
    message_filters::Subscriber<sensor_msgs::Image> image_sub;
    message_filters::Subscriber<stereo_msgs::DisparityImage> depth_sub;
    message_filters::TimeSynchronizer<sensor_msgs::Image,stereo_msgs::DisparityImage> sync;

};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_alignment");
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

    StereoAlignmentROS stereoAlignmentRos(n,imageTopic,depthMapTopic,cameraTopic);

    ros::Rate loop_rate(50);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

}




