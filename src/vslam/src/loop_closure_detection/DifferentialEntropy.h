#pragma once
#include "core/Frame.h"
#include "core/Pose.h"
#include "core/macros.h"
#include "core/types.h"
#include "utils/visuals.h"
namespace vslam::loop_closure_detection {
struct LoopClosure {
  TYPEDEF_PTR(LoopClosure)
  Timestamp t0, t1;
  Pose relativePose;
};

template <typename Aligner> class DifferentialEntropy {
public:
  TYPEDEF_PTR(DifferentialEntropy)

  DifferentialEntropy(double minRatioFine, double minRatioCoarse, Aligner &&fineAligner, Aligner &&coarseAligner) :
      _minRatioFine(minRatioFine),
      _minRatioCoarse(minRatioCoarse),
      _fineAligner(fineAligner),
      _coarseAligner(coarseAligner) {
    log::create(LOG_NAME);
  }

  void update(Timestamp tkf, Timestamp tf, const Pose &pose) { _childFrames[tkf].push_back({tf, std::log(pose.cov().determinant())}); }

  LoopClosure::UnPtr isLoopClosure(Frame::ConstShPtr f0, Frame::ConstShPtr f1) {

    const auto &entropiesTrack = _childFrames[f0->t()];
    if (entropiesTrack.empty()) {
      CLOG(DEBUG, LOG_NAME) << format("No track available to compute base entropy for [{}]", f0->t());
      return nullptr;
    }
    if (std::find_if(entropiesTrack.begin(), entropiesTrack.end(), [&](auto cf) { return cf.t == f1->t(); }) != entropiesTrack.end()) {
      CLOG(DEBUG, LOG_NAME) << format("[{}] is already a child frame of [{}]", f1->t(), f0->t());
      return nullptr;
    }
    const double entropyRef =
      std::transform_reduce(entropiesTrack.begin(), entropiesTrack.end(), 0.0, std::plus<double>{}, [](auto cf) { return cf.entropy; }) /
      entropiesTrack.size();
    auto lc = isLoopClosure(f0, f1, {f1->pose().SE3()}, entropyRef * _minRatioCoarse, _coarseAligner);
    if (lc) {
      lc = isLoopClosure(f0, f1, {lc->relativePose * f0->pose().SE3()}, entropyRef * _minRatioFine, _fineAligner);
      if (lc) {
        log::append(
          "LoopClosures",
          overlay::Hstack(overlay::Features(f0, 1), overlay::ReprojectedFeatures(f0, f1, 1, 0, std::make_shared<Pose>(lc->relativePose))));
      }
    }
    return lc;
  }

private:
  LoopClosure::UnPtr isLoopClosure(Frame::ConstShPtr f0, Frame::ConstShPtr f1, Pose prior, double minEntropy, Aligner &aligner) {
    LoopClosure::UnPtr result = std::make_unique<LoopClosure>();
    result->t0 = f0->t();
    result->t1 = f1->t();
    auto r01 = aligner.align(f0, f1, prior);
    if (r01->constraints[r01->levels.back()].size() < 100) {
      return nullptr;
    }
    const double entropy = std::log(r01->pose.cov().determinant());
    result->relativePose.SE3() = r01->pose.SE3() * f0->pose().SE3().inverse();
    result->relativePose.cov() = r01->pose.cov();
    bool isLoopClosure = std::isfinite(entropy) && entropy < minEntropy;
    CLOG(DEBUG, LOG_NAME) << format(
      "Ratio test between {} and {}: t={:.3f}m r={:.3f}° c={:.2f}/{:.2f} [{}]",
      f0->t(),
      f1->t(),
      result->relativePose.translation().norm(),
      result->relativePose.totalRotationDegrees(),
      entropy,
      minEntropy,
      isLoopClosure ? "PASSED" : "NOT PASSED");

    log::append(
      "LoopClosuresRatioTest",
      overlay::Hstack(overlay::Features(f0, 1), overlay::ReprojectedFeatures(f0, f1, 1, 0, std::make_shared<Pose>(result->relativePose))));

    return isLoopClosure ? std::move(result) : nullptr;
  }
  const double _minRatioFine, _minRatioCoarse;
  Aligner _fineAligner, _coarseAligner;
  struct Entropy {
    vslam::Timestamp t;
    double entropy;
  };
  std::map<vslam::Timestamp, std::vector<Entropy>> _childFrames;

  static constexpr const char LOG_NAME[] = "loop_closure_detection";
};
}  // namespace vslam::loop_closure_detection
