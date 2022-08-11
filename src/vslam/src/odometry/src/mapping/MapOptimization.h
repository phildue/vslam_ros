#ifndef VSLAM_MAP_OPTIMIZATION_H__
#define VSLAM_MAP_OPTIMIZATION_H__
#include "core/core.h"
#include "BundleAdjustment.h"
#include <vector>

namespace pd::vslam::mapping {
  class MapOptimization
  {
public:
    typedef std::shared_ptr < MapOptimization > ShPtr;
    typedef std::unique_ptr < MapOptimization > UnPtr;
    typedef std::shared_ptr < const MapOptimization > ConstShPtr;
    typedef std::unique_ptr < const MapOptimization > ConstUnPtr;

    MapOptimization();

    void optimize(
      const std::vector < FrameRgbd::ShPtr > & frames,
      const std::vector < Point3D::ShPtr > & points) const;

private:
  };
}

#endif
