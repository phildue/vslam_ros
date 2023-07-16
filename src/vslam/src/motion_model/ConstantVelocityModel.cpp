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

#include "ConstantVelocityModel.h"
#include "utils/log.h"
#include "utils/utils.h"

#define LOG_NAME "motion_model"
#define MLOG(level) CLOG(level, LOG_NAME)
namespace vslam {
ConstantVelocityModel::ConstantVelocityModel(const std::map<std::string, double> &params) :
    ConstantVelocityModel(params.at("information"), params.at("maxTranslationalVelocity"), params.at("maxAngularVelocity")) {
  log::create(LOG_NAME);
}

ConstantVelocityModel::ConstantVelocityModel(double information, double maxTranslationalVelocity, double maxAngularVelocity) :
    _maxTranslationalVelocity(maxTranslationalVelocity),
    _maxAngularVelocity(maxAngularVelocity / 180.0 * M_PI),
    _covariance((information * Mat6d::Identity()).inverse()),
    _lastT(0) {}

bool ConstantVelocityModel::exceedsThresholds(const Vec6d &velocity) const {
  const double translationalVelocity = velocity.block(0, 0, 3, 1).norm() * 1e9;
  const double angularVelocity = velocity.block(3, 0, 3, 1).norm() * 1e9;
  const bool exceeds = translationalVelocity > _maxTranslationalVelocity || angularVelocity > _maxAngularVelocity;

  if (exceeds) {
    using time::to_time_point;
    MLOG(WARNING) << format(
      "Velocity exceeded, ignoring t={:%Y-%m-%d %H:%M:%S} with vt = {}/{} m/s, va = {}/{} deg/s",
      to_time_point(_lastT),
      translationalVelocity,
      _maxTranslationalVelocity,
      angularVelocity * 180.0 / M_PI,
      _maxAngularVelocity * 180.0 / M_PI);
  }
  return exceeds;
}

void ConstantVelocityModel::update(const Pose &pose, Timestamp timestamp) {
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT));
  if (_lastT > 0 && dT > 0) {
    const Vec6d velocity = (pose.SE3() * _lastPose.inverse()).log() / dT;

    if (exceedsThresholds(velocity))
      return;

    _velocity = velocity;
  }
  _lastPose = pose.SE3();
  _lastT = timestamp;
}

Pose ConstantVelocityModel::predict(Timestamp timestamp) const {
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT));
  return Pose(SE3d::exp(_velocity * dT) * _lastPose, _covariance);
}

Pose ConstantVelocityModel::predict(Timestamp from, Timestamp to) const {
  const double dT = (static_cast<double>(to) - static_cast<double>(from));
  return Pose(SE3d::exp(_velocity * dT), _covariance);
}

}  // namespace vslam
