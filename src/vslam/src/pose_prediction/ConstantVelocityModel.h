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

#ifndef VSLAM_MOTION_MODEL
#define VSLAM_MOTION_MODEL

#include <map>

#include "core/Pose.h"
#include "core/Trajectory.h"
#include "core/macros.h"
#include "core/types.h"
namespace vslam::pose_prediction {
class ConstantVelocityModel {
public:
  TYPEDEF_PTR(ConstantVelocityModel)
  static std::map<std::string, double> defaultParameters() {
    return {{"information", 0.05}, {"maxTranslationalVelocity", 10.0}, {"maxAngularVelocity", 180.0}};
  }

  ConstantVelocityModel(const std::map<std::string, double> &params);

  ConstantVelocityModel(double information, double maxTranslationalVelocity, double maxAngularVelocity);
  void update(const Pose &pose, Timestamp t);
  Pose predict(Timestamp timestamp) const;
  Pose predict(Timestamp from, Timestamp to) const;

  Pose velocity() const { return Pose(SE3d::exp(_velocity * 1e9), _covariance); }

private:
  const double _maxTranslationalVelocity, _maxAngularVelocity;
  const Mat6d _covariance;
  Vec6d _velocity = Vec6d::Zero();
  bool exceedsThresholds(const Vec6d &speed) const;
  Trajectory::ShPtr _trajectory;
};

}  // namespace vslam
#endif  // VSLAM_MOTION_MODEL
