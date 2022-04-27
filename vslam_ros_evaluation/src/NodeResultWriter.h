#ifndef NODE_RESULT_WRITER_H__
#define NODE_RESULT_WRITER_H__
#include <iostream>
#include <fstream>
#include <string>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <vslam/vslam.h>
#include "visibility_control.h"


class NodeResultWriter : public rclcpp::Node
{
        public:
        COMPOSITION_PUBLIC
        NodeResultWriter(const rclcpp::NodeOptions& options);
        void callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
        ~NodeResultWriter(){_algoFile.close();}
        private:
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _sub;
        std::fstream _algoFile;
        std::string _outputFile;

};
#endif