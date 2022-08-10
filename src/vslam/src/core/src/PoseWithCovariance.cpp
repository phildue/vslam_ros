#include "PoseWithCovariance.h"
namespace pd::vslam
{
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance & p0)
{
  //https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance
  const auto C = p0.cov();
  Matd<6, 6> R = Matd<6, 6>::Zero();
  R.block(0, 0, 3, 3) = p1.rotationMatrix();
  R.block(3, 3, 3, 3) = p1.rotationMatrix();

  return PoseWithCovariance(p1 * p0.pose(), R * C * R.transpose());
}
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance::ConstUnPtr & p0)
{
  return p1 * (*p0);
}
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance::ConstShPtr & p0)
{
  return p1 * (*p0);
}
}
