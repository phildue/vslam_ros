#include "DifferentialEntropy.h"
#include "features/overlays.h"
#include "utils/log.h"
#include "utils/visuals.h"

namespace vslam::loop_closure_detection {
DifferentialEntropy::DifferentialEntropy(double minRatio, AlignmentRgbd::UnPtr fineAligner, AlignmentRgbd::UnPtr coarseAligner) :
    _minRatio(minRatio),
    _coarseAligner(std::move(coarseAligner)),
    _fineAligner(std::move(fineAligner)) {
  log::create(LOG_NAME);
}
LoopClosure::UnPtr DifferentialEntropy::isLoopClosure(Frame::ConstShPtr f0, double entropyRef, Frame::ConstShPtr f1) const {
  // TODO can't we align on level 4 first and then continue on level 3 and so forth?..
  return isLoopClosure(f0, entropyRef, f1, _minRatio * 0.9, _coarseAligner) ? isLoopClosure(f0, entropyRef, f1, _minRatio, _fineAligner)
                                                                            : nullptr;
  // return isLoopClosure(f0, entropyRef, f1, _fineAligner);
}
LoopClosure::UnPtr DifferentialEntropy::isLoopClosure(
  Frame::ConstShPtr f0, double entropyRef, Frame::ConstShPtr f1, double minEntropyRatio, const AlignmentRgbd::UnPtr &aligner) const {
  LoopClosure::UnPtr result = std::make_unique<LoopClosure>();
  result->t0 = f0->t();
  result->t1 = f1->t();
  const Mat6d priorCov = Mat6d::Identity() * std::numeric_limits<double>::quiet_NaN();
  auto r01 = aligner->align(f0, f1, Pose(f1->pose().SE3(), priorCov));
  if (r01->constraints[r01->levels.back()].size() < 100) {
    return nullptr;
  }
  double entropyRatio = std::log(r01->pose.cov().determinant()) / entropyRef;
  result->relativePose.SE3() = r01->pose.SE3() * f0->pose().SE3().inverse();
  result->relativePose.cov() = r01->pose.cov();
  // auto r10 = aligner->align(f1, f0, Pose(f0->pose().SE3(), priorCov));
  SE3d diff;
  Mat6d diffCov;
  // if (result->entropyRatio > _minRatio) {
  // r01 = _fineAligner->align(f0, f1, Pose(f1->pose().SE3(), priorCov));
  //  auto r10 = _fineAligner->align(f1, f0, Pose(f0->pose().SE3(), priorCov));
  SE3d pose01 = r01->pose.SE3() * f0->pose().SE3().inverse();
  SE3d pose10 = f0->pose().SE3() * f1->pose().SE3().inverse();
  diff = pose01 * pose10;
  diffCov = f0->pose().cov() + pose01.Adj() * f0->pose().cov() * pose01.Adj().transpose();

  entropyRatio = std::log(r01->pose.cov().determinant()) / entropyRef;
  result->relativePose.SE3() = r01->pose.SE3() * f0->pose().SE3().inverse();
  result->relativePose.cov() = r01->pose.cov();
  //}
  bool isLoopClosure = std::isfinite(entropyRatio) && entropyRatio > minEntropyRatio;
  const double diffNorm = (diff.log().transpose() * diffCov.inverse() * diff.log());
  CLOG(DEBUG, LOG_NAME) << format(
    "Ratio test between {} and {}: t={:.3f}m r={:.3f}° c={:.2f}% dt={:.3f} da={:.3f} dc={:.3f} |d|={:.3f} [{}]",
    f0->id(),
    f1->id(),
    result->relativePose.translation().norm(),
    result->relativePose.totalRotationDegrees(),
    entropyRatio * 100,
    diff.translation().norm(),
    (diff.angleX() + diff.angleY() + diff.angleZ()) / M_PI * 180.0,
    std::log(r01->pose.cov().determinant()) / std::log(f1->pose().cov().determinant()),
    diffNorm,
    isLoopClosure ? "PASSED" : "NOT PASSED");

  log::append(
    "LoopClosuresRatioTest",
    overlay::Hstack(overlay::Features(f0, 1), overlay::ReprojectedFeatures(f0, f1, 1, 0, std::make_shared<Pose>(result->relativePose))));

  if (isLoopClosure) {
    log::append(
      "LoopClosures",
      overlay::Hstack(overlay::Features(f0, 1), overlay::ReprojectedFeatures(f0, f1, 1, 0, std::make_shared<Pose>(result->relativePose))));
  }
  return isLoopClosure ? std::move(result) : nullptr;
}

}  // namespace vslam::loop_closure_detection
