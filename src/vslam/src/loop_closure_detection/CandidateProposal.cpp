#include "CandidateProposal.h"

namespace vslam::loop_closure_detection::candidate_proposal {

bool RelativePose::isCandidate(Frame::ConstShPtr f0, Frame::ConstShPtr f1) const {
  return (f1->pose().SE3() * f0->pose().SE3().inverse()).translation().norm() < _maxTranslation;
}

bool ReprojectedFeatures::isCandidate(Frame::ConstShPtr f0, Frame::ConstShPtr f1) const {
  int nFeatures = 0;
  double opticalFlow = 0.;

  for (const auto &ft : f1->features()) {
    Vec2d uv1 = f1->world2image(f0->p3dWorld(ft->v(), ft->u()));
    if (uv1.allFinite() && f1->withinImage(uv1)) {
      nFeatures++;
      opticalFlow += (uv1 - ft->position()).norm();
    }
  }
  opticalFlow /= (double)nFeatures;
  return nFeatures > _minFeatures;
}
}  // namespace vslam::loop_closure_detection::candidate_proposal
