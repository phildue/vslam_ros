#ifndef VSLAM_FEATURE_TRACKING_H__
#define VSLAM_FEATURE_TRACKING_H__
#include "core/core.h"
namespace pd::vslam {
  class FeatureTracking {
public:
    typedef std::shared_ptr < FeatureTracking > ShPtr;
    typedef std::unique_ptr < FeatureTracking > UnPtr;
    typedef std::shared_ptr < const FeatureTracking > ConstShPtr;
    typedef std::unique_ptr < const FeatureTracking > ConstUnPtr;

    std::vector < Point3D::ShPtr > track(
      FrameRgbd::ShPtr frameCur,
      const std::vector < FrameRgbd::ShPtr > &framesRef);

    void extractFeatures(FrameRgbd::ShPtr frame) const;
    std::vector < Point3D::ShPtr > match(
      FrameRgbd::ShPtr frameCur,
      const std::vector < Feature2D::ShPtr > &featuresRef) const;

    std::vector < Feature2D::ShPtr > selectCandidates(
      FrameRgbd::ConstShPtr frameCur,
      const std::vector < FrameRgbd::ShPtr >
      &framesRef) const;

private:
    const size_t _nFeatures = 100;
  };
}

#endif //VSLAM_FEATURE_TRACKING_H__
