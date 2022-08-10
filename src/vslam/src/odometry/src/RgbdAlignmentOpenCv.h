#ifndef VSLAM_RGBD_ALIGNMENT_OPENCV
#define VSLAM_RGBD_ALIGNMENT_OPENCV

#include "core/core.h"
#include "lukas_kanade/lukas_kanade.h"
#include "AlignmentSE3.h"

namespace pd::vslam {
  class RgbdAlignmentOpenCv: public AlignmentSE3 {
public:
    typedef std::shared_ptr < RgbdAlignmentOpenCv > ShPtr;
    typedef std::unique_ptr < RgbdAlignmentOpenCv > UnPtr;
    typedef std::shared_ptr < const RgbdAlignmentOpenCv > ConstShPtr;
    typedef std::unique_ptr < const RgbdAlignmentOpenCv > ConstUnPtr;

    RgbdAlignmentOpenCv();

    PoseWithCovariance::UnPtr align(FrameRgbd::ConstShPtr from, FrameRgbd::ConstShPtr to) const;

protected:
  };
}
#endif// VSLAM_SE3_ALIGNMENT
