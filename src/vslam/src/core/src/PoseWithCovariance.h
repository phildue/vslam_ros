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

#ifndef POSE_WITH_COVARIANCE_H__
#define POSE_WITH_COVARIANCE_H__
#include <memory>

#include "types.h"
namespace pd::vslam
{
class PoseWithCovariance
{
public:
  typedef std::shared_ptr<PoseWithCovariance> ShPtr;
  typedef std::unique_ptr<PoseWithCovariance> UnPtr;
  typedef std::shared_ptr<const PoseWithCovariance> ConstShPtr;
  typedef std::unique_ptr<const PoseWithCovariance> ConstUnPtr;

  PoseWithCovariance(
    const Vec6d & x = Vec6d::Zero(), const Matd<6, 6> & cov = Matd<6, 6>::Identity())
  : _x(x), _cov(cov)
  {
  }
  //PoseWithCovariance( const Vec3d& t, const Vec4d& q, const Matd<6,6>& cov):_x(SE3d(q,t).log()),_cov(cov){}
  PoseWithCovariance(const SE3d & pose, const Matd<6, 6> & cov) : _x(pose.log()), _cov(cov) {}

  SE3d pose() const { return SE3d::exp(_x); }
  Matd<6, 6> cov() const { return _cov; }
  Vec6d mean() const { return _x; }
  PoseWithCovariance inverse() const
  {
    return PoseWithCovariance(SE3d::exp(_x).inverse().log(), _cov);
  }

private:
  Vec6d _x;
  Matd<6, 6> _cov;
};
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance & p0);
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance::ConstUnPtr & p0);
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance::ConstShPtr & p0);
}  // namespace pd::vslam
#endif
