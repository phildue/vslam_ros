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
  results.erase(std::remove_if(results.begin(), results.end(), [&](const auto &r) { return !r->isLoopClosure; }), results.end());

  if (results.empty()) {
    CLOG(INFO, LOG_NAME) << "No loop closures detected.";
    return results;
  }
  for (const auto &lc : results) {
    CLOG(INFO, LOG_NAME) << format(
      "Detected loop closure between {} and {} with: t={:.3f}m r={:.3f}°\n",
      lc->from,
      lc->to,
      lc->relativePose.translation().norm(),
      lc->relativePose.totalRotationDegrees());
  }
  return results;
}

LoopClosureDetection::Result::UnPtr LoopClosureDetection::align(Frame::ConstShPtr f0, double entropyRef, Frame::ConstShPtr f1) const {
  Result::UnPtr result = std::make_unique<Result>();
  result->from = f0->id();
  result->to = f1->id();
  const Mat6d priorCov = Mat6d::Identity() * std::numeric_limits<double>::quiet_NaN();
  auto r01 = _fineAligner->align(f0, f1, Pose(f1->pose().SE3(), priorCov));
  double entropyRatio = std::log(r01->pose.cov().determinant()) / entropyRef;
  result->relativePose.SE3() = r01->pose.SE3() * f0->pose().SE3().inverse();
  result->relativePose.cov() = r01->pose.cov();
  SE3d diff;
  Mat6d diffCov;
  // if (result->entropyRatio > _minRatio) {
  // r01 = _fineAligner->align(f0, f1, Pose(f1->pose().SE3(), priorCov));
  //  auto r10 = _fineAligner->align(f1, f0, Pose(f0->pose().SE3(), priorCov));
  SE3d pose01 = r01->pose.SE3() * f0->pose().SE3().inverse();
  SE3d pose10 = f0->pose().SE3() * f1->pose().SE3().inverse();
  diff = pose01 * pose10;
  diffCov = r01->pose.cov() + pose01.Adj() * f0->pose().cov() * pose01.Adj().transpose();

  entropyRatio = std::log(r01->pose.cov().determinant()) / entropyRef;
  result->relativePose.SE3() = r01->pose.SE3() * f0->pose().SE3().inverse();
  result->relativePose.cov() = r01->pose.cov();
  //}
  result->isLoopClosure = std::isfinite(entropyRatio) && entropyRatio > _minRatio;
  const double diffNorm = (diff.log().transpose() * diffCov.inverse() * diff.log());
  CLOG(DEBUG, LOG_NAME) << format(
    "Ratio test between {} and {}: t={:.3f}m r={:.3f}° c={:.2f}% dt={:.3f} da={:.3f} |d|={:.3f} [{}]",
    result->from,
    result->to,
    result->relativePose.translation().norm(),
    result->relativePose.totalRotationDegrees(),
    entropyRatio * 100,
    diff.translation().norm(),
    (diff.angleX() + diff.angleY() + diff.angleZ()) / M_PI * 180.0,
    diffNorm,
    result->isLoopClosure ? "PASSED" : "NOT PASSED");
  log::append("LoopClosuresRatioTest", [&]() { return overlay::frames(Frame::VecConstShPtr{f0, f1}); });

  if (result->isLoopClosure) {
    log::append("LoopClosures", [&]() { return overlay::frames(Frame::VecConstShPtr{f0, f1}); });
  }
  return result;
}

Frame::VecConstShPtr LoopClosureDetection::selectCandidates(Frame::ConstShPtr f, const Frame::VecConstShPtr &frames) const {
  Frame::VecConstShPtr candidates;
  std::copy_if(frames.begin(), frames.end(), std::back_inserter(candidates), [&](auto cf) { return isCandidate(f, cf); });
  return candidates;
}
bool LoopClosureDetection::isCandidate(Frame::ConstShPtr f, Frame::ConstShPtr cf) const {

  if (f->id() == cf->id()) {
    return false;
  }

  const SE3d relativePose = f->pose().SE3() * cf->pose().SE3().inverse();
  const double t = relativePose.translation().norm();
  const double r = relativePose.log().block(3, 1, 3, 1).norm() / M_PI * 180.0;

  int nFeatures = 0;
  double opticalFlow = 0.;
  /*
    for (const auto &ft : f->features()) {
      Vec2d uv1 = cf->world2image(f->p3dWorld(ft->v(), ft->u()));
      if (uv1.allFinite() && f->withinImage(uv1)) {
        nFeatures++;
        opticalFlow += (uv1 - ft->position()).norm();
      }
    }
    opticalFlow /= nFeatures;
    const bool passed = nFeatures / (double)f->features().size() > 0.75 && opticalFlow < 200;
    */
  const bool passed = t < 0.5;

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
