#ifndef VSLAM_ALIGNER_H__
#define VSLAM_ALIGNER_H__

#include <core/core.h>
namespace pd::vslam {
  class AlignmentSE3 {
public:
    virtual PoseWithCovariance::UnPtr align(
      FrameRgbd::ConstShPtr from,
      FrameRgbd::ConstShPtr to) const = 0;
  };
}


#endif //VSLAM_ALIGNER_H__
