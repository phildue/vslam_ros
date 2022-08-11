#ifndef VSLAM_SE3_ALIGNMENT
#define VSLAM_SE3_ALIGNMENT

#include "core/core.h"
#include "lukas_kanade/lukas_kanade.h"
#include "least_squares/least_squares.h"
#include "AlignmentSE3.h"
namespace pd::vslam
{
  class SE3Alignment: public AlignmentSE3
  {
public:
    typedef std::shared_ptr < SE3Alignment > ShPtr;
    typedef std::unique_ptr < SE3Alignment > UnPtr;
    typedef std::shared_ptr < const SE3Alignment > ConstShPtr;
    typedef std::unique_ptr < const SE3Alignment > ConstUnPtr;

    SE3Alignment(
      double minGradient,
      vslam::least_squares::Solver::ShPtr solver,
      vslam::least_squares::Loss::ShPtr loss,
      bool includePrior = false
    );

    PoseWithCovariance::UnPtr align(
      FrameRgbd::ConstShPtr from,
      FrameRgbd::ConstShPtr to) const override;
    PoseWithCovariance::UnPtr align(
      const std::vector < FrameRgbd::ConstShPtr > & from,
      FrameRgbd::ConstShPtr to) const;

protected:
    const double _minGradient2;
    const vslam::least_squares::Loss::ShPtr _loss;
    const vslam::least_squares::Solver::ShPtr _solver;
    const bool _includePrior;

  };
}
#endif// VSLAM_SE3_ALIGNMENT
