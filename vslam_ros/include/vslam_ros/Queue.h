#ifndef VSLAM_ROS_QUEUE_H__
#define VSLAM_ROS_QUEUE_H__
#include <map>
#include <mutex>
#include <sensor_msgs/msg/image.hpp>

namespace vslam_ros{

class Queue
{
        public:
        Queue(int queueSize, int maxDiff):_maxDiff(maxDiff),_queueSize(queueSize){}
        int size() const;
        void pushImage(sensor_msgs::msg::Image::ConstSharedPtr img);
        void pushDepth(sensor_msgs::msg::Image::ConstSharedPtr depth);
        sensor_msgs::msg::Image::ConstSharedPtr popClosestImg(std::uint64_t t = 0U);
        sensor_msgs::msg::Image::ConstSharedPtr popClosestDepth(std::uint64_t t = 0U);
        private:
        std::map<std::uint64_t,sensor_msgs::msg::Image::ConstSharedPtr> _images;
        std::map<std::uint64_t,sensor_msgs::msg::Image::ConstSharedPtr> _depths;
        mutable std::mutex _mutex;
        const int _maxDiff;
        const size_t _queueSize;

};
}
#endif