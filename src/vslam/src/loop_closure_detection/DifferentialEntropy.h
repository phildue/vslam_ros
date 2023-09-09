#pragma once
#include "core/Frame.h"
#include "core/macros.h"
#include "core/types.h"
#include "odometry/AlignmentRgbd.h"  //TODO generic interface for odometry?

namespace vslam::loop_closure_detection {
struct LoopClosure {
  TYPEDEF_PTR(LoopClosure)
  Timestamp t0, t1;
  Pose relativePose;
};
class DifferentialEntropy {
public:
  TYPEDEF_PTR(DifferentialEntropy)

  DifferentialEntropy(double minRatio, AlignmentRgbd::UnPtr fineAligner, AlignmentRgbd::UnPtr coarseAligner);
  LoopClosure::UnPtr isLoopClosure(Frame::ConstShPtr f, double entropyRef, Frame::ConstShPtr cf) const;

private:
  LoopClosure::UnPtr
  isLoopClosure(Frame::ConstShPtr f, double entropyRef, Frame::ConstShPtr cf, double minEntropy, const AlignmentRgbd::UnPtr &aligner) const;

  const double _minRatio;
  const AlignmentRgbd::UnPtr _coarseAligner, _fineAligner;

  static constexpr const char LOG_NAME[] = "loop_closure_detection";
};
}  // namespace vslam::loop_closure_detection
