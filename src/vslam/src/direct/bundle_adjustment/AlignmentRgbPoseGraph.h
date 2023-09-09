#pragma once

#include "core/Frame.h"
#include <memory>
namespace vslam {
class AlignmentRgbPoseGraph {
public:
  typedef std::shared_ptr<AlignmentRgbPoseGraph> ShPtr;
  typedef std::unique_ptr<AlignmentRgbPoseGraph> UnPtr;

  AlignmentRgbPoseGraph(int nLevels, int maxIterations, double tukeyA);
  void align(Frame::VecShPtr frames, Frame::VecShPtr framesFixed);

private:
  const int _nLevels, _maxIterations;
  const double _tukeyA;
};
}  // namespace vslam
