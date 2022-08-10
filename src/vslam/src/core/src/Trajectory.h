#ifndef VSLAM_TRAJECTORY_H__
#define VSLAM_TRAJECTORY_H__
#include <map>
#include <core/core.h>

namespace pd::vslam
{
  class Trajectory {
public:
    typedef std::shared_ptr < Trajectory > ShPtr;
    typedef std::unique_ptr < Trajectory > UnPtr;
    typedef std::shared_ptr < const Trajectory > ConstShPtr;
    typedef std::unique_ptr < const Trajectory > ConstUnPtr;

    Trajectory();
    Trajectory(const std::map < Timestamp, PoseWithCovariance::ConstShPtr > &poses);
    Trajectory(const std::map < Timestamp, SE3d > &poses);
    PoseWithCovariance::ConstShPtr poseAt(Timestamp t, bool interpolate = true) const;
    PoseWithCovariance::ConstShPtr motionBetween(
      Timestamp t0, Timestamp t1,
      bool interpolate = true) const;
    void append(Timestamp t, PoseWithCovariance::ConstShPtr pose);
    const std::map < Timestamp, PoseWithCovariance::ConstShPtr > & poses() const {return _poses;}

private:
    PoseWithCovariance::ConstShPtr interpolateAt(Timestamp t) const;

    std::map < Timestamp, PoseWithCovariance::ConstShPtr > _poses;

  };
} // namespace pd::vslam

#endif
