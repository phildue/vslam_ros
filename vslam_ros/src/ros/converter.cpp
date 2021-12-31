
#include "converters.h"
namespace vslam_ros2{
pd::vision::Camera convert(const sensor_msgs::msg::CameraInfo& msg)
{
        const double fx = msg.k[0*3 + 0];
        const double fy = msg.k[1*3 + 1];
        const double cx = msg.k[0*3 + 2];
        const double cy = msg.k[1*3 + 2];

        return pd::vision::Camera(fx,cx,cy);
}

geometry_msgs::msg::Pose convert(const Sophus::SE3d& se3)
{
        geometry_msgs::msg::Pose pose;

        const auto t = se3.translation();
        const auto q = se3.unit_quaternion();
        pose.position.x = t.x();
        pose.position.y = t.y();
        pose.position.z = t.z();
        pose.orientation.w = q.w();
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        return pose;
}
}