

#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "vslam/core.h"
#include "vslam/direct.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/loop_closure_detection.h"
#include "vslam/motion_model.h"
#include "vslam/pose_graph_optimization.h"
#include "vslam/utils.h"
using namespace vslam;

/*
Attempt:

If we know that we reobserve the same location
we can compute the relative transformation to previous frames
and employ pose graph optimization to improve the overall trajectory and reduce drift.

Here we want to see if we can detect these loop closures.

Result:

Explanation:

*/
class Map {
public:
  Map() {
    _lc = std::make_unique<loop_closure_detection::DifferentialEntropy>(
      0.9,
      std::make_unique<AlignmentRgbd>(std::vector<int>{3}, 20, 1e-4, 1.1),
      std::make_unique<AlignmentRgbd>(AlignmentRgbd::defaultParameters()));
    _poseGraph = std::make_unique<PoseGraphOptimizer>(100);
  }
  void addKeyframe(Frame::ShPtr kf) {
    _keyFrames.push_back(kf);
    _entropiesTrack[kf->id()] = {};
  }
  void addFrame(Frame::ShPtr f) {
    _entropiesTrack[_keyFrames.back()->id()].push_back(std::log(f->pose().cov().determinant()));
    _poseGraph->addMeasurement(
      _keyFrames.back()->t(), f->t(), Pose(f->pose().SE3() * _keyFrames.back()->pose().SE3().inverse(), f->pose().cov()));
  }

  int detectLoopClosures(Frame::ConstShPtr kf) {

    if (_entropiesTrack.find(kf->id()) == _entropiesTrack.end() || _entropiesTrack[kf->id()].empty()) {
      LOG(WARNING) << format("No track found for frame: {} at {}", kf->id(), kf->t());
      return 0;
    }

    const double meanEntropyTrack =
      std::accumulate(_entropiesTrack[kf->id()].begin(), _entropiesTrack[kf->id()].end(), 0.0) / _entropiesTrack[kf->id()].size();

    Frame::VecConstShPtr candidates;
    std::copy_if(_keyFrames.begin(), _keyFrames.end(), std::back_inserter(candidates), [&](auto cf) {
      return cf->id() != kf->id() && !_poseGraph->hasMeasurement(kf->t(), cf->t());
    });

    if (candidates.empty()) {
      return 0;
    }
    int nLoopClosures = 0;
    for (const auto &cf : candidates) {
      int nFeatures = 0;
      double opticalFlow = 0.;

      for (const auto &ft : cf->features()) {
        Vec2d uv1 = kf->world2image(cf->p3dWorld(ft->v(), ft->u()));
        if (uv1.allFinite() && kf->withinImage(uv1)) {
          nFeatures++;
          opticalFlow += (uv1 - ft->position()).norm();
        }
      }
      opticalFlow /= nFeatures;
      if ((kf->pose().SE3() * cf->pose().SE3().inverse()).translation().norm() > 0.5 || nFeatures < 500) {
        continue;
      }
      auto lc = _lc->isLoopClosure(kf, meanEntropyTrack, cf);
      if (lc) {
        nLoopClosures++;
        _poseGraph->addMeasurement(lc->t0, lc->t1, lc->relativePose);
      }
    }
    return nLoopClosures;
  }
  void optimize() {
    _poseGraph->optimize();
    for (const auto &kf : _keyFrames) {
      kf->pose().SE3() = _poseGraph->poses().at(kf->t());
    }
  }
  Trajectory trajectory() const { return Trajectory(_poseGraph->poses()); }
  const Frame::VecShPtr &keyframes() const { return _keyFrames; }

private:
  Frame::VecShPtr _keyFrames;
  std::map<uint64_t, std::vector<double>> _entropiesTrack;
  std::unique_ptr<loop_closure_detection::DifferentialEntropy> _lc;
  PoseGraphOptimizer::UnPtr _poseGraph;
};

int main(int argc, char **argv) {
  const std::string filename = argv[0];
  const std::string experimentId = filename.substr(filename.find_last_of("/") + 1);
  const std::vector<std::string> sequences = evaluation::tum::sequencesTraining();
  random::init(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::string sequenceId = argc > 1 ? argv[1] : "";
  if (sequenceId.empty() || sequenceId == "random") {
    sequenceId = sequences[random::U(0, sequences.size() - 1)];
  }
  auto dl = std::make_unique<evaluation::tum::DataLoader>("/mnt/dataset/tum_rgbd/", sequenceId);

  Trajectory::ShPtr trajGt = evaluation::tum::loadTrajectory(dl->pathGt());
  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  const size_t nFrames = std::min(-1UL, dl->nFrames());
  const size_t fNo0 = 0;  // nFrames >= dl->nFrames() ? 0UL : random::U(0UL, dl->nFrames() - nFrames);

  log::configure(TEST_RESOURCE "/log/");
  log::config("Frame")->show = 1;
  log::config("FrameFeatures")->show = -1;
  log::config("LoopClosures")->show = 0;
  log::config("LoopClosuresRatioTest")->show = 1;
  log::config("KeyFrame")->show = -1;
  log::config("GraphBefore")->show = 1;
  log::config("GraphAfter")->show = 1;

  auto directIcp = std::make_shared<AlignmentRgbd>(AlignmentRgbd::defaultParameters());
  auto motionModel = std::make_shared<ConstantVelocityModel>(10.0, INFd, INFd);
  auto featureSelection = std::make_shared<FeatureSelection<FiniteGradient>>(FiniteGradient{5, 0.01, 0.3, 0, 8.0}, 10, 1);
  auto map = std::make_shared<Map>();
  log::initialize(outPath, true);

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = dl->timestamps().size();
  Frame::ShPtr kf = dl->loadFrame(0);
  kf->computePyramid(directIcp->nLevels());
  kf->computeDerivatives();
  kf->computePcl();
  map->addKeyframe(kf);
  featureSelection->select(kf);
  motionModel->update(kf->pose(), kf->t());

  Frame::ShPtr lf = kf;
  traj->append(kf->t(), kf->pose());
  double entropyRef = 0.;
  double nConstraintsRef = INFd;
  double errorRef = 0.;
  std::map<size_t, double> meanEntropies;
  int nLoopClosures = 0;
  for (size_t fId = fNo0 + 1; fId < fNo0 + nFrames; fId++) {
    try {

      Frame::ShPtr f = dl->loadFrame(fId);
      f->computePyramid(directIcp->nLevels());
      f->pose() = motionModel->predict(f->t());
      log::append("Frame", [&]() { return overlay::frames({kf, lf, f}); });
      log::append("FrameFeatures", overlay::Hstack(overlay::Features(kf, 10), overlay::ReprojectedFeatures(kf, f, 10)));
      auto results = directIcp->align(kf, f);
      f->pose() = results->pose;
      log::append("FrameFeatures", overlay::Hstack(overlay::Features(kf, 10), overlay::ReprojectedFeatures(kf, f, 10)));
      const double entropyRatio = std::log(results->pose.cov().determinant()) / entropyRef;
      const double nConstraintsRatio = (double)results->constraints.size() / (double)nConstraintsRef;
      const double errorRatio = results->normalEquations[0].error / errorRef;
      print(
        "{}/{}: {} m, {:.3f}° err: {:.3f} #: {} |H|={:.3f}, /#: {:.4} /err: {:.2f}\n",
        fId,
        fEnd,
        f->pose().translation().transpose(),
        f->pose().totalRotationDegrees(),
        results->normalEquations[0].error,
        results->normalEquations[0].nConstraints,
        entropyRatio,
        nConstraintsRatio,
        errorRatio);

      if (lf != kf && entropyRatio < 0.9) {
        print("Keyframe selected.\n");
        kf = lf;
        map->addKeyframe(kf);

        kf->computeDerivatives();
        kf->computePcl();
        featureSelection->select(kf);

        log::append("KeyFrame", [&]() { return overlay::frames({kf, lf, f}); });

        f->pose() = motionModel->predict(f->t());
        results = directIcp->align(kf, f);
        f->pose() = results->pose;
        log::append("Frame", overlay::Hstack(overlay::Features(kf, 10), overlay::ReprojectedFeatures(kf, f, 10)));

        errorRef = results->normalEquations[0].error;
        nConstraintsRef = results->constraints.size();
        entropyRef = std::log(f->pose().cov().determinant());
      }
      map->addFrame(f);
      motionModel->update(f->pose(), f->t());
      traj->append(f->t(), f->pose());
      if (lf != kf) {
        lf->removeFeatures();
      }
      lf = f;

    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }
  evaluation::tum::writeTrajectory(traj->inverse(), trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);

  int n = 0;
  do {
    n = 0;
    for (auto kf : map->keyframes()) {
      n += map->detectLoopClosures(kf);
    }
    nLoopClosures += n;
    if (n > 0) {
      map->optimize();
      evaluation::tum::writeTrajectory(map->trajectory().inverse(), trajectoryAlgoPath);
      evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
    }
  } while (n > 0);

  LOG(INFO) << format("Created keyframes {}, Detected loop closures: {}", map->keyframes().size(), nLoopClosures);
  /*
    for (const auto &[t, pose] : traj->poses()) {
      const SE3d diff = map->trajectory().poseAt(t, false)->SE3() * pose->SE3().inverse();
      LOG(INFO) << format(
        "t={} |diff|={:.3f}m rx={:.3f} ry={:.3f} rz={:.3f}",
        t,
        diff.translation().norm(),
        diff.angleX() / M_PI * 180.0,
        diff.angleY() / M_PI * 180.0,
        diff.angleZ() / M_PI * 180.0);
    }
    */
}
