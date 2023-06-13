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

#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "vslam/utils.h"
#include "vslam_ros/Associator.h"
namespace vslam_ros
{
Associator::Associator(int queueSize, int64_t maxDiff) : _maxDiff(maxDiff), _queueSize(queueSize) {}

int Associator::nImages() const { return _images.size(); }
int Associator::nDepth() const { return _depths.size(); }

void Associator::pushImage(sensor_msgs::msg::Image::ConstSharedPtr img)
{
  const int64_t t = rclcpp::Time(img->header.stamp).nanoseconds();
  std::lock_guard<std::recursive_mutex> g(_mutex);

  if (_images.size() >= _queueSize) {
    _images.erase(_images.begin());
  }

  _images[t] = img;
}
void Associator::pushDepth(sensor_msgs::msg::Image::ConstSharedPtr depth)
{
  const int64_t t = rclcpp::Time(depth->header.stamp).nanoseconds();
  std::lock_guard<std::recursive_mutex> g(_mutex);

  if (_depths.size() >= _queueSize) {
    _depths.erase(_depths.begin());
  }

  _depths[t] = depth;
}
Associator::Association Associator::pop()
{
  /*This mimicks tum assoc.py script: 
  we try to find the frame pairs with the smallest distance to each other*/
  std::vector<Association> potentialMatches;
  for (auto z : _depths) {
    for (auto i : _images) {
      const int64_t diff = std::abs<int64_t>((int64_t)z.first - (int64_t)i.first);
      if (diff < _maxDiff) {
        potentialMatches.emplace_back(Association{diff, z.second, i.second});
      }
    }
  }
  std::sort(potentialMatches.begin(), potentialMatches.end(), [](auto a, auto b) {
    return a.diff < b.diff;
  });
  if (potentialMatches.empty()) {
    throw std::runtime_error("No matching pair within threshold available");
  }
  auto depths = _depths;
  auto images = _images;

  std::vector<Association> matches;
  for (auto m : potentialMatches) {
    auto d = depths.find(rclcpp::Time(m.depth->header.stamp).nanoseconds());
    auto i = images.find(rclcpp::Time(m.img->header.stamp).nanoseconds());
    if (d != depths.end() && i != images.end()) {
      matches.push_back(m);
      depths.erase(d);
      images.erase(i);
    }
  }
  std::sort(matches.begin(), matches.end(), [](auto a, auto b) {
    return rclcpp::Time(a.depth->header.stamp).nanoseconds() <
           rclcpp::Time(b.depth->header.stamp).nanoseconds();
  });
  std::erase_if(_depths, [&](const auto & item) {
    auto const & [t, msg] = item;
    return t <= rclcpp::Time(matches[0].depth->header.stamp).nanoseconds();
  });
  std::erase_if(_images, [&](const auto & item) {
    auto const & [t, msg] = item;
    return t <= rclcpp::Time(matches[0].img->header.stamp).nanoseconds();
  });
  return matches[0];
}

}  // namespace vslam_ros
