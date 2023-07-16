#pragma once

#include "core/Frame.h"
namespace vslam {
class AlignmentRgbPoseGraph {
public:
  AlignmentRgbPoseGraph(int nLevels, int maxIterations, bool avoidDuplicates = true);
  void align(Frame::VecShPtr frames, Frame::VecShPtr framesFixed);

private:
  const int _nLevels, _maxIterations;
  const bool _avoidDuplicates;
};
}  // namespace vslam
