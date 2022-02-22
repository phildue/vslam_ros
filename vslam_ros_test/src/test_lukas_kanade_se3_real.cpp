#include <iostream>
#include <sstream>

#include "rosbag2_cpp/readers/sequential_reader.hpp"
#include "rosbag2_cpp/reader.hpp"

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <nav_msgs/msg/path.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
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

class NodeImage : public rclcpp::Node
{
        public:
        NodeImage()
        :rclcpp::Node("NodeImage")
        ,_pubPath(this->create_publisher<sensor_msgs::msg::Image>("/camera/rgb/image_color",10))
        {}
        void publish(const sensor_msgs::msg::Image& path){_pubPath->publish(path);}
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pubPath;
};

class NodeTf : public rclcpp::Node
{
        public:
        NodeTf()
        :rclcpp::Node("TfListener")
        ,_pubTf(this->create_publisher<tf2_msgs::msg::TFMessage>("/tf",10))
        {}
        void publish(const tf2_msgs::msg::TFMessage& tf){_pubTf->publish(tf);}
        rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr _pubTf;
};
class NodeOdom : public rclcpp::Node
{
        public:
        NodeOdom(const std::string& outputFile)
        :rclcpp::Node("OdomListener")
        ,_sub(this->create_subscription<nav_msgs::msg::Odometry>("/odom",10,std::bind(&NodeOdom::callback,this,std::placeholders::_1)))
        , _outputFile(outputFile)
        {
                algoFile.open(_outputFile,std::ios_base::out);
        }
        void callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg){
               
                if(!algoFile.is_open())
                {
                        std::runtime_error("Could not open file at: " + _outputFile);
                }

                algoFile << msg->header.stamp.sec << "." << msg->header.stamp.nanosec << " "
                << msg->pose.pose.position.x << " " << msg->pose.pose.position.y << " " << msg->pose.pose.position.z << " "
                << msg->pose.pose.orientation.w << " " << msg->pose.pose.orientation.x << " " << msg->pose.pose.orientation.y << " " << msg->pose.pose.orientation.z;

                for (int i = 0; i < 36; i++)
                {
                        algoFile << " " << msg->pose.covariance[i];
                }
                algoFile << std::endl;

        }
        ~NodeOdom(){
                algoFile.close();

        }
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _sub;
        std::string _outputFile;
        std::fstream algoFile;

};
class LukasKanadeSE3Test : public testing::Test{
        public:
        std::unique_ptr<rosbag2_cpp::readers::SequentialReader> _reader;
        int _nFrames;
        rcutils_time_point_value_t _tStart = 0;
        bool _visualize = true;
        int _idx = 0;

        std::string _gtTrajectoryFile, _algoTrajectoryFile;
        std::map<std::string,Sophus::SE3d> _gtTrajectory;
        nav_msgs::msg::Path _pathImu;
        std::shared_ptr<vslam_ros::RgbdAlignmentNode> _node;
        std::shared_ptr<ImuNode> _nodeImu;
        std::shared_ptr<NodeTf> _nodeTf;
        std::shared_ptr<NodeImage> _nodeImage;
        std::shared_ptr<NodeOdom> _nodeOdom;

        LukasKanadeSE3Test(){
                _reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
                rosbag2_storage::StorageOptions storageOptions;
                storageOptions.uri = "/media/data/dataset/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz.db3";
                storageOptions.storage_id = "sqlite3";
                rosbag2_cpp::ConverterOptions converterOptions;
                _reader->open(storageOptions,converterOptions);
                _gtTrajectoryFile = "/media/data/dataset/rgbd_dataset_freiburg1_xyz/groundtruth.txt";
                _algoTrajectoryFile = "/media/data/dataset/rgbd_dataset_freiburg1_xyz/algo.txt";

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
                        //pose = pose0.inverse() * pose;

                        std::vector<std::string> tElements;
                        std::string st;
                        std::istringstream tLine(elements[0]);    
                        while (getline(tLine, st, '.')) {
                                tElements.push_back(st);
                        }
                        geometry_msgs::msg::PoseStamped pStamped;
                        pStamped.header.frame_id = "/world";
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
                        //std::cout << fNo << "/" << _nFrames << " t: " << msg->time_stamp << " topic: " << msg->topic_name << std::endl;
                     
                        if(msg->topic_name == "/camera/rgb/image_color")
                        {
                                fNo++;
                                img = std::make_shared<sensor_msgs::msg::Image>();
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, img.get());
                                _nodeImage->publish(*img);
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
                                _pathImu.header.frame_id = "/world";
                     

                                _nodeImu->publish(_pathImu);
                        }
                        if(msg->topic_name == "/tf" && fNo > 1)
                        {

                                auto tf = std::make_shared<tf2_msgs::msg::TFMessage>();
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<tf2_msgs::msg::TFMessage> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, tf.get());


                                _nodeTf->publish(*tf);
                        }
                        rclcpp::spin_some(_nodeOdom);
                        
                } 

        }

        
};



TEST_F(LukasKanadeSE3Test, LukasKanadeSE3InverseCompositionalGNMultiLevel)
{
        rclcpp::init(0, nullptr);
        _node = std::make_shared<vslam_ros::RgbdAlignmentNode>(rclcpp::NodeOptions());
        _nodeImu = std::make_shared<ImuNode>();
        _nodeTf = std::make_shared<NodeTf>();
        _nodeImage = std::make_shared<NodeImage>();
        _nodeOdom = std::make_shared<NodeOdom>(_algoTrajectoryFile);
        run();
}
