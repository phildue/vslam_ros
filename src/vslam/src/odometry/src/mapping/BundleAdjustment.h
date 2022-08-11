#ifndef VSLAM_BUNDLE_ADJUSTMENT_H__
#define VSLAM_BUNDLE_ADJUSTMENT_H__
#include <ceres/ceres.h>
#include <vector>
#include "core/core.h"

namespace pd::vslam::mapping {
  class BundleAdjustment
  {
public:
    typedef std::shared_ptr < BundleAdjustment > ShPtr;
    typedef std::unique_ptr < BundleAdjustment > UnPtr;
    typedef std::shared_ptr < const BundleAdjustment > ConstShPtr;
    typedef std::unique_ptr < const BundleAdjustment > ConstUnPtr;

    BundleAdjustment();
    void optimize();

    void setFrame(std::uint64_t frameId, const SE3d & pose, const Mat3d & K);
    void setPoint(std::uint64_t pointId, const Vec3d & point);
    void setObservation(std::uint64_t pointId, std::uint64_t frameId, const Vec2d & observation);

    SE3d getPose(std::uint64_t frameId) const;
    Vec3d getPoint(std::uint64_t pointId) const;
    double computeError() const;

private:
    //compute error is not const?
    mutable ceres::Problem _problem;
    std::map < std::uint64_t, SE3d > _poses;
    std::map < std::uint64_t, Mat3d > _Ks;
    std::map < std::uint64_t, Vec3d > _points;

  };
}

#endif
