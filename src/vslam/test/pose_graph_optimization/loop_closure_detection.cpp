

#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "LoopClosureDetection.h"
#include "PoseGraph.h"
#include "vslam/core.h"
#include "vslam/direct.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/motion_model.h"
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
  const size_t fNo0 = nFrames >= dl->nFrames() ? 0UL : random::U(0UL, dl->nFrames() - nFrames);

  const int tRmse = 200;
  std::thread thread;
  log::configure(TEST_RESOURCE "/log/");
  log::config("Frame")->show = 1;
  log::config("LoopClosures")->show = 0;

  auto directIcp = std::make_shared<AlignmentRgbd>(AlignmentRgbd::defaultParameters());
  auto motionModel = std::make_shared<ConstantVelocityModel>(10.0, INFd, INFd);
  auto featureSelection = std::make_shared<FeatureSelection>(5, 0.01, 0.3, 0, 8.0, 10, 4);
  auto loopClosureDetection = std::make_unique<LoopClosureDetection>(
    0.1,
    5.0 / 180.0 * M_PI,
    0.9,
    std::make_unique<AlignmentRgbd>(std::vector<int>{3}, 20, 1e-4, 1.1),
    std::make_shared<AlignmentRgbd>(AlignmentRgbd::defaultParameters()));
  auto poseGraph = std::make_unique<PoseGraph>();
  log::initialize(outPath, true);

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = dl->timestamps().size();
  Frame::ShPtr kf = dl->loadFrame(0);
  kf->computePyramid(directIcp->nLevels());
  kf->computeDerivatives();
  kf->computePcl();
  featureSelection->select(kf, true);
  motionModel->update(kf->pose(), kf->t());

  Frame::ShPtr lf = kf;
  traj->append(kf->t(), kf->pose().inverse());
  double entropyRef = 0.;
  double nConstraintsRef = INFd;
  double errorRef = 0.;
  std::vector<Frame::ShPtr> kfs;
  kfs.push_back(kf);
  std::vector<double> entropiesTrack;
  std::map<size_t, Timestamp> timestamps;
  timestamps[kf->id()] = kf->t();

  for (size_t fId = fNo0 + 1; fId < fNo0 + nFrames; fId++) {
    try {

      Frame::ShPtr f = dl->loadFrame(fId);
      timestamps[f->id()] = f->t();
      f->computePyramid(directIcp->nLevels());
      f->pose() = motionModel->predict(f->t());
      log::append("Frame", [&]() { return overlay::frames({kf, lf, f}); });
      auto results = directIcp->align(kf, f);
      f->pose() = results->pose;
      entropiesTrack.push_back(std::log(f->pose().cov().determinant()));
      const double entropyRatio = entropiesTrack.back() / entropyRef;
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

        kf->computeDerivatives();
        kf->computePcl();
        featureSelection->select(kf, true);
        const double meanEntropy =
          std::accumulate(entropiesTrack.begin(), entropiesTrack.end(), 0.0, [&](auto s, auto e) { return s + e / entropiesTrack.size(); });
        auto loopClosures = loopClosureDetection->detect(kf, meanEntropy, {kfs.begin(), kfs.end()});
        for (const auto &lc : loopClosures) {
          poseGraph->addMeasurement(lc->from, lc->to, lc->relativePose);
        }
        if (!loopClosures.empty()) {
          poseGraph->optimize();
          for (const auto &[id, pose] : poseGraph->poses()) {
            traj->append(timestamps.at(id), Pose(pose));
            motionModel->update(Pose(pose), timestamps.at(id));
          }
        }

        log::append("KeyFrame", [&]() { return overlay::frames({kf, lf, f}); });
        kfs.push_back(kf);

        f->pose() = motionModel->predict(f->t());
        results = directIcp->align(kf, f);
        f->pose() = results->pose;
        errorRef = results->normalEquations[0].error;
        nConstraintsRef = results->constraints.size();
        entropyRef = std::log(f->pose().cov().determinant());
      }
      poseGraph->addMeasurement(kf->id(), f->id(), Pose(f->pose().SE3() * kf->pose().SE3().inverse(), f->pose().cov()));
      motionModel->update(f->pose(), f->t());
      traj->append(dl->timestamps()[fId], f->pose().inverse());
      lf = f;

    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }

    if (fId > tRmse && fId % tRmse == 0) {
      if (thread.joinable()) {
        thread.join();
      }
      thread = std::thread([&]() {
        evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
        evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
      });
    };
  }
  for (const auto &_kf : kfs) {
    _kf->removeFeatures();
  }
  if (thread.joinable()) {
    thread.join();
  }
  evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
}
