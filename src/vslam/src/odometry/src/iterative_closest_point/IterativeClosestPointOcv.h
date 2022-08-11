#ifndef VSLAM_ICP_OPENCV_H__
#define VSLAM_ICP_OPENCV_H__

#include "core/core.h"
#include "AlignmentSE3.h"


namespace pd::vslam {
  class IterativeClosestPointOcv: public AlignmentSE3 {
public:
    typedef std::shared_ptr < IterativeClosestPointOcv > ShPtr;
    typedef std::unique_ptr < IterativeClosestPointOcv > UnPtr;
    typedef std::shared_ptr < const IterativeClosestPointOcv > ConstShPtr;
    typedef std::unique_ptr < const IterativeClosestPointOcv > ConstUnPtr;

    IterativeClosestPointOcv(size_t level, int maxIterations)
      : _level(level),
      _maxIterations(maxIterations) {
      Log::get("odometry", ODOMETRY_CFG_DIR "/log/odometry.conf");

    };

    PoseWithCovariance::UnPtr align(
      FrameRgbd::ConstShPtr from,
      FrameRgbd::ConstShPtr to) const override;

protected:
    size_t _level;
    int _maxIterations;

  };
}
#endif// VSLAM_ICP_H__
