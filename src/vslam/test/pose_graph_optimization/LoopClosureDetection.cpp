#include "LoopClosureDetection.h"
#include "utils/log.h"
#include "utils/visuals.h"
namespace vslam {
LoopClosureDetection::LoopClosureDetection(
  double maxTranslation, double maxAngle, double minRatio, AlignmentRgbd::UnPtr fineAligner, AlignmentRgbd::UnPtr coarseAligner) :
    _maxTranslation(maxTranslation),
    _maxAngle(maxAngle),
    _minRatio(minRatio),
    _coarseAligner(std::move(coarseAligner)),
    _fineAligner(std::move(fineAligner)) {
  log::create(LOG_NAME);
}

LoopClosureDetection::Result::VecConstUnPtr
LoopClosureDetection::detect(Frame::ConstShPtr f, double entropyRef, const Frame::VecConstShPtr &frames) const {
  const Frame::VecConstShPtr candidates = selectCandidates(f, frames);
  CLOG(INFO, LOG_NAME) << format("Found: {} candidates.", candidates.size());
  Result::VecConstUnPtr results(candidates.size());
  // TODO make aligners thread safe and do it in parallel here ?
  std::transform(candidates.begin(), candidates.end(), results.begin(), [&](auto c) { return align(f, entropyRef, c); });
  results.erase(std::remove_if(results.begin(), results.end(), [&](const auto &r) { return r->entropyRatio < _minRatio; }), results.end());

  if (results.empty()) {
    CLOG(INFO, LOG_NAME) << "No loop closures detected.";
    return results;
  }
  for (const auto &lc : results) {
    CLOG(INFO, LOG_NAME) << format(
      "Detected loop closure between {} and {} with: t={:.3f}m r={:.3f}° c={:.2f}%\n",
      lc->from,
      lc->to,
      lc->relativePose.translation().norm(),
      lc->relativePose.totalRotationDegrees(),
      lc->entropyRatio * 100);
  }
  return results;
}

LoopClosureDetection::Result::UnPtr LoopClosureDetection::align(Frame::ConstShPtr f, double entropyRef, Frame::ConstShPtr cf) const {
  Result::UnPtr result = std::make_unique<Result>();
  result->from = f->id();
  result->to = cf->id();
  auto r = _coarseAligner->align(f, cf);
  result->entropyRatio = std::log(r->pose.cov().determinant()) / entropyRef;
  result->relativePose.SE3() = r->pose.SE3() * f->pose().SE3().inverse();
  result->relativePose.cov() = r->pose.cov();
  if (result->entropyRatio > _minRatio) {
    r = _fineAligner->align(f, cf);
    result->entropyRatio = std::log(r->pose.cov().determinant()) / entropyRef;
    result->relativePose.SE3() = r->pose.SE3() * f->pose().SE3().inverse();
    result->relativePose.cov() = r->pose.cov();
  }
  const bool passed = result->entropyRatio > _minRatio;
  CLOG(DEBUG, LOG_NAME) << format(
    "Ratio test between {} and {}: t={:.3f}m r={:.3f}° c={:.2f}% [{}]",
    result->from,
    result->to,
    result->relativePose.translation().norm(),
    result->relativePose.totalRotationDegrees(),
    result->entropyRatio * 100,
    passed ? "PASSED" : "NOT PASSED");
  if (passed) {
    log::append("LoopClosures", [&]() { return overlay::frames(Frame::VecConstShPtr{f, cf}); });
  }
  return result;
}

Frame::VecConstShPtr LoopClosureDetection::selectCandidates(Frame::ConstShPtr f, const Frame::VecConstShPtr &frames) const {
  Frame::VecConstShPtr candidates;
  std::copy_if(frames.begin(), frames.end(), std::back_inserter(candidates), [&](auto cf) { return isCandidate(f, cf); });
  return candidates;
}
bool LoopClosureDetection::isCandidate(Frame::ConstShPtr f, Frame::ConstShPtr cf) const {
  int nFeatures = 0;
  double opticalFlow = 0.;
  for (const auto &ft : f->features()) {
    Vec2d uv1 = cf->world2image(f->p3dWorld(ft->v(), ft->u()));
    if (uv1.allFinite() && f->withinImage(uv1)) {
      nFeatures++;
      opticalFlow += (uv1 - ft->position()).norm();
    }
  }
  opticalFlow /= nFeatures;
  const SE3d relativePose = f->pose().SE3() * cf->pose().SE3().inverse();
  const double t = relativePose.translation().norm();
  const double r = relativePose.log().block(3, 1, 3, 1).norm() / M_PI * 180.0;
  const bool passed = nFeatures / (double)f->features().size() > 0.95 && opticalFlow < 100;
  CLOG(DEBUG, LOG_NAME) << format(
    "Evaluting candidate for {}, {}: Features: {:.2f} OpticalFlow: {:.2f} t={:.2f} r={:.2f}: [{}]",
    f->id(),
    cf->id(),
    nFeatures / (double)f->features().size(),
    opticalFlow,
    t,
    r,
    passed ? "PASSED" : "NOT PASSED");

  return passed;
}
}  // namespace vslam
