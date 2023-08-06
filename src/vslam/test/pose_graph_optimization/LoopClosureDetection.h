#pragma once
#include "core/Frame.h"
#include "core/macros.h"
#include "core/types.h"
#include "direct/AlignmentRgbd.h"

namespace vslam {
class LoopClosureDetection {
  TYPEDEF_PTR(LoopClosureDetection)

public:
  struct Result {
    TYPEDEF_PTR(Result)
    size_t from, to;
    double entropyRatio;
    Pose relativePose;
  };
  LoopClosureDetection(
    double maxTranslation, double maxAngle, double minRatio, AlignmentRgbd::UnPtr fineAligner, AlignmentRgbd::UnPtr coarseAligner);
  Result::VecConstUnPtr detect(Frame::ConstShPtr f, double entropyRef, const Frame::VecConstShPtr &frames) const;
  Result::UnPtr align(Frame::ConstShPtr f, double entropyRef, Frame::ConstShPtr cf) const;

private:
  const double _maxTranslation, _maxAngle, _minRatio;
  const AlignmentRgbd::UnPtr _coarseAligner, _fineAligner;

  Frame::VecConstShPtr selectCandidates(Frame::ConstShPtr f, const Frame::VecConstShPtr &frames) const;
  bool isCandidate(Frame::ConstShPtr f, Frame::ConstShPtr cf) const;
  static constexpr const char LOG_NAME[] = "loop_closure_detection";
};
}  // namespace vslam
