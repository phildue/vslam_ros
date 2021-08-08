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
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "vslam/vslam.h"


std::shared_ptr<pd::vision::StereoAlignment> alignment;
ros::Publisher pub;

void imageCallback(sensor_msgs::ImageConstPtr msg, sensor_msgs::ImageConstPtr msgDepth)
{
    Eigen::Map<Eigen::Matrix<uint8_t ,768,480>> mat((uint8_t*) msg->data.data(),msg->height,msg->width);
    Eigen::Map<Eigen::Matrix<uint8_t ,768,480>> matDepth((uint8_t*) msgDepth->data.data(),msgDepth->height,msgDepth->width);
    Sophus::SE3d result = alignment->align(mat.cast<int>(),matDepth.cast<double>(),msg->header.stamp.toNSec());
    nav_msgs::Odometry msgOut;
    msgOut.header.stamp = msg->header.stamp;
    //msgOut.pose.pose.position = result.translation();
    //msgOut.pose.pose.orientation = result.unit_quaternion();
    pub.publish(msgOut);
}


int main(int argc, char **argv)
{

    const std::string imageTopic = "/cam00/image/raw";
    const std::string depthMapTopic = "/cam00/image/depth";
    ros::init(argc, argv, "image_alignment");
    ros::NodeHandle n("~");

    pd::vision::StereoAlignment::Config config;
    alignment = std::make_shared<pd::vision::StereoAlignment>(config);

    pub = n.advertise<nav_msgs::Odometry>("/odom", 10);

    ros::topic::waitForMessage<sensor_msgs::Image>(imageTopic);
    ros::topic::waitForMessage<sensor_msgs::Image>(depthMapTopic);
    message_filters::Subscriber<sensor_msgs::Image> image_sub(n, imageTopic, 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(n, depthMapTopic, 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image,sensor_msgs::Image> sync(image_sub, depth_sub,10);

    sync.registerCallback(boost::bind(&imageCallback, _1, _2));

    ros::Rate loop_rate(50);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

}




