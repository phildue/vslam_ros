// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef VSLAM_ROS_ASSOCIATOR_H__
#define VSLAM_ROS_ASSOCIATOR_H__
#include <map>
#include <mutex>
#include <sensor_msgs/msg/image.hpp>

namespace vslam_ros
{
class Associator
{
public:
  struct Association
  {
    int64_t diff;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::Image::ConstSharedPtr img;
  };
  Associator(int queueSize, int64_t maxDiff);
  int nImages() const;
  int nDepth() const;
  void pushImage(sensor_msgs::msg::Image::ConstSharedPtr img);
  void pushDepth(sensor_msgs::msg::Image::ConstSharedPtr depth);
  Association pop();

private:
  std::map<int64_t, sensor_msgs::msg::Image::ConstSharedPtr> _images;
  std::map<int64_t, sensor_msgs::msg::Image::ConstSharedPtr> _depths;
  mutable std::recursive_mutex _mutex;
  const int64_t _maxDiff;
  const size_t _queueSize;
};
}  // namespace vslam_ros
#endif
