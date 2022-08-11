#ifndef VSLAM_ICP_H__
#define VSLAM_ICP_H__

#include "core/core.h"
#include "AlignmentSE3.h"
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc

namespace pd::vslam
{
  class IterativeClosestPoint: public AlignmentSE3
  {
public:
    typedef std::shared_ptr < IterativeClosestPoint > ShPtr;
    typedef std::unique_ptr < IterativeClosestPoint > UnPtr;
    typedef std::shared_ptr < const IterativeClosestPoint > ConstShPtr;
    typedef std::unique_ptr < const IterativeClosestPoint > ConstUnPtr;

    IterativeClosestPoint(size_t level, int maxIterations)
      : _level(level),
      _maxIterations(maxIterations)
    {
      Log::get("odometry", ODOMETRY_CFG_DIR "/log/odometry.conf");

    }

    PoseWithCovariance::UnPtr align(
      FrameRgbd::ConstShPtr from,
      FrameRgbd::ConstShPtr to) const override;

protected:
    size_t _level;
    int _maxIterations;

  };
}
#endif// VSLAM_ICP_H__
