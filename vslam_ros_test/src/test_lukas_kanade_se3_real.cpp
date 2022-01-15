//
#include "rosbag2_cpp/readers/sequential_reader.hpp"
#include "rosbag2_cpp/reader.hpp"
#include <iostream>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <nav_msgs/msg/path.hpp>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "vslam_ros/vslam_ros.h"


using namespace testing;
using namespace pd;
using namespace pd::vision;

class ImuNode : public rclcpp::Node
{
        public:
        ImuNode()
        :rclcpp::Node("ImuNode")
        ,_pubPath(this->create_publisher<nav_msgs::msg::Path>("/path/imu",10))
        {}
        void publish(const nav_msgs::msg::Path& path){_pubPath->publish(path);}
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pubPath;
};
class LukasKanadeSE3Test : public testing::Test{
        public:
        std::unique_ptr<rosbag2_cpp::readers::SequentialReader> _reader;
        int _nFrames;
        rcutils_time_point_value_t _tStart = 0;
        bool _visualize = true;
        int _idx = 0;

        std::string _gtTrajectoryFile;
        std::map<std::string,Sophus::SE3d> _gtTrajectory;
        nav_msgs::msg::Path _pathImu;
        std::shared_ptr<vslam_ros::LukasKanadeSE3Node> _node;
        std::shared_ptr<ImuNode> _nodeImu;

        LukasKanadeSE3Test(){
                _reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
                rosbag2_storage::StorageOptions storageOptions;
                storageOptions.uri = "/media/data/dataset/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz.db3";
                storageOptions.storage_id = "sqlite3";
                rosbag2_cpp::ConverterOptions converterOptions;
                _reader->open(storageOptions,converterOptions);
                _gtTrajectoryFile = "/media/data/dataset/rgbd_dataset_freiburg1_xyz/groundtruth.txt";
                std::vector<rosbag2_storage::TopicMetadata> meta = _reader->get_all_topics_and_types();
                //std::cout << "Found: " << _reader->get_metadata().topics_with_message_count.size() << " meta entries.";
              
                for (const auto& mc : _reader->get_metadata().topics_with_message_count)
                {
                      //  std::cout << mc.topic_metadata.name << ": " << mc.topic_metadata.type << std::endl;

                        if(mc.topic_metadata.name == "/camera/rgb/image_color")
                        {
                                _nFrames = mc.message_count;
                        }
                }
                
        }
        ~LukasKanadeSE3Test(){
                _reader->close();
        }

        void run()
        {
                _pathImu.header.frame_id = "openni_rgb_optical_frame";
                std::ifstream gtFile;
                gtFile.open(_gtTrajectoryFile);
                if(!gtFile.is_open())
                {
                        std::runtime_error("Could not open file at: " + _gtTrajectoryFile);
                }
                Sophus::SE3d pose0;
                int nPoses = 0;
                std::string line;
                while(getline(gtFile, line)) {
                        
                        std::vector<std::string> elements;
                        std::string s;
                        std::istringstream lines(line);    
                        while (getline(lines, s, ' ')) {
                                elements.push_back(s);
                        }
                        if(elements[0] == "#")
                        {//skip comments
                                continue;
                        }
                        Eigen::Vector3d t;
                        t << std::stod(elements[1]),std::stod(elements[2]),std::stod(elements[3]);
                        Eigen::Quaterniond q(std::stod(elements[7]),std::stod(elements[4]),std::stod(elements[5]),std::stod(elements[6]));
                        Sophus::SE3d pose(q,t);

                        if(nPoses == 0)
                        {
                                pose0 = pose;
                        }
                        pose = pose0.inverse() * pose;

                        std::vector<std::string> tElements;
                        std::string st;
                        std::istringstream tLine(elements[0]);    
                        while (getline(tLine, st, '.')) {
                                tElements.push_back(st);
                        }
                        geometry_msgs::msg::PoseStamped pStamped;
                        pStamped.header.frame_id = "/openni_rgb_optical_frame";
                        pStamped.header.stamp.sec = std::stoull(tElements[0]);
                        pStamped.header.stamp.nanosec = std::stoull(tElements[1])*100000;

                        pStamped.pose = vslam_ros::convert(pose);
                        _pathImu.poses.push_back(pStamped);
                        nPoses++;

                        
                } 

                int fNo = 0;
                std::shared_ptr<sensor_msgs::msg::CameraInfo> camInfo = nullptr;
                std::shared_ptr<sensor_msgs::msg::Image> img = nullptr;
                std::shared_ptr<sensor_msgs::msg::Image> depth = nullptr;

                while(_reader->has_next() )
                {

                        rosbag2_storage::SerializedBagMessageSharedPtr msg =  _reader->read_next();
                        std::cout << fNo << "/" << _nFrames << " t: " << msg->time_stamp << " topic: " << msg->topic_name << std::endl;
                     
                        if(msg->topic_name == "/camera/rgb/image_color")
                        {
                                fNo++;
                                img = std::make_shared<sensor_msgs::msg::Image>();
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, img.get());
                        }

                        if (msg->topic_name == "/camera/rgb/camera_info")
                        {
                                camInfo = std::make_shared<sensor_msgs::msg::CameraInfo>();
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<sensor_msgs::msg::CameraInfo> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, camInfo.get());
                                _node->cameraCallback(camInfo);
                                
                        }
                     
                        if(msg->topic_name == "/camera/depth/image")
                        {

                                depth = std::make_shared<sensor_msgs::msg::Image>();
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, depth.get());
                                if (img && camInfo)
                                {
                                        _node->processFrame(img, depth);
                                        img = nullptr;
                                        depth = nullptr;
                                }

                        }
                        if(msg->topic_name == "/imu" && fNo > 1)
                        {

                                auto imu = std::make_shared<sensor_msgs::msg::Imu>();
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<sensor_msgs::msg::Imu> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, imu.get());

                                _pathImu.header = imu->header;
                                _pathImu.header.frame_id = "/openni_rgb_optical_frame";
                     

                                _nodeImu->publish(_pathImu);
                        }
                        
                } 

        }

        
};



TEST_F(LukasKanadeSE3Test, LukasKanadeSE3InverseCompositionalGNMultiLevel)
{
        rclcpp::init(0, nullptr);
        _node = std::make_shared<vslam_ros::LukasKanadeSE3Node>(rclcpp::NodeOptions());
        _nodeImu = std::make_shared<ImuNode>();
        run();
}
