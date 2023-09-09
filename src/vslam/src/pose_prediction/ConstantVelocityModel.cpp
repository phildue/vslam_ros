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
namespace vslam::pose_prediction {
ConstantVelocityModel::ConstantVelocityModel(const std::map<std::string, double> &params) :
    ConstantVelocityModel(params.at("information"), params.at("maxTranslationalVelocity"), params.at("maxAngularVelocity")) {
  log::create(LOG_NAME);
}

ConstantVelocityModel::ConstantVelocityModel(double information, double maxTranslationalVelocity, double maxAngularVelocity) :
    _maxTranslationalVelocity(maxTranslationalVelocity),
    _maxAngularVelocity(maxAngularVelocity / 180.0 * M_PI),
    _covariance((information * Mat6d::Identity()).inverse()),
    _trajectory(std::make_unique<Trajectory>()) {}

bool ConstantVelocityModel::exceedsThresholds(const Vec6d &velocity) const {
  const double translationalVelocity = velocity.block(0, 0, 3, 1).norm() * 1e9;
  const double angularVelocity = velocity.block(3, 0, 3, 1).norm() * 1e9;
  const bool exceeds = translationalVelocity > _maxTranslationalVelocity || angularVelocity > _maxAngularVelocity;

  if (exceeds) {
    using time::to_time_point;
    MLOG(WARNING) << format(
      "Velocity exceeded, ignoring t={:%Y-%m-%d %H:%M:%S} with vt = {}/{} m/s, va = {}/{} deg/s",
      to_time_point(0),
      translationalVelocity,
      _maxTranslationalVelocity,
      angularVelocity * 180.0 / M_PI,
      _maxAngularVelocity * 180.0 / M_PI);
  }
  return exceeds;
}

void ConstantVelocityModel::update(const Pose &pose, Timestamp timestamp) { _trajectory->append(timestamp, pose); }

Pose ConstantVelocityModel::predict(Timestamp timestamp) const {
  if (_trajectory->poses().empty()) {
    return Pose(SE3d(), _covariance);
  }
  if (_trajectory->poses().size() < 2) {
    return Pose(_trajectory->poses().begin()->second->SE3(), _covariance);
  }
  if (timestamp > _trajectory->tEnd()) {
    auto it = _trajectory->poses().rbegin();
    const Timestamp t1 = it->first;
    const SE3d pose1 = it->second->SE3();
    it++;
    const Timestamp t0 = it->first;
    const SE3d pose0 = it->second->SE3();
    const Vec6d velocity = (pose1 * pose0.inverse()).log() / (double)(t1 - t0);
    return Pose(SE3d::exp(velocity * (timestamp - t1)) * pose1, _covariance);
  }
  if (timestamp < _trajectory->tStart()) {
    auto it = _trajectory->poses().begin();
    const Timestamp t1 = it->first;
    const SE3d pose1 = it->second->SE3();
    it++;
    const Timestamp t0 = it->first;
    const SE3d pose0 = it->second->SE3();
    const Vec6d velocity = (pose1 * pose0.inverse()).log() / (double)(t1 - t0);
    return Pose(SE3d::exp(velocity * (timestamp - t1)) * pose1, _covariance);
  }
  return Pose(_trajectory->poseAt(timestamp)->SE3(), _covariance);
}

Pose ConstantVelocityModel::predict(Timestamp from, Timestamp to) const {
  return Pose(predict(to).SE3() * predict(from).SE3().inverse(), _covariance);
}

}  // namespace vslam::pose_prediction
