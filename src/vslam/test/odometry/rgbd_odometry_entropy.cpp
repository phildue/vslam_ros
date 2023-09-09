

#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "vslam/core.h"
#include "vslam/odometry.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/pose_prediction.h"
#include "vslam/utils.h"
using namespace vslam;

/*
Attempt:
Track against keyframes to improve odometry (e.g. more displacement leads to better pose) and reduce overhead on multiframe optimization.
Keyframe selection based on DVOs "differential entropy method":
1. Use the entropy (det|H|) between the keyframe and first following frame as reference entropy (det|H|_kf)
2. Compare follow up entropies against the reference entropy ratio = det|H|_ff/det|H|_kf
3. As the ratio drops below some threshold, use the previous frames as new keyframe repeat again with 1.
This assumes directly neighboring frames can be aligned well.
When further following frames can be aligned better than the direct neighbor, the ratio should increase (>1).
As the further following frames diverge (fov, appearance,..) they can be aligned more poorly and the ratiou would decrease.
Note that H as A.inverse() can not be used as a global uncertainty.

Result:
- Good improvements in some scenes e.g. tum_rgbd_freiburg1_desk
- Results get worse in other scenes e.g. tum_rgbd_freiburg1_desk2
- Increase of runtime as for each keyframes we do the alignment 2x now
- Results more or less match dvo paper

Explanation:
- For direct methods it's ideal to have very closely resembling frames, as appearance changes can have quite high impact
- Only if there are frames which are not suitable for tracking at all, e.g. due to motion blur it can help to use another reference frame
- The uncertainty measure based on fishers criterion makes some assumptions on the underlying functions which are not met in our case

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
  log::initialize(outPath, true);
  log::configure(TEST_RESOURCE "/log/");
  log::config("Frame")->show = 1;

  auto directIcp = std::make_shared<odometry::AlignmentRgbd>(odometry::AlignmentRgbd::defaultParameters());
  auto motionModel = std::make_shared<pose_prediction::ConstantVelocityModel>(10.0, INFd, INFd);
  auto featureSelection = std::make_shared<FeatureSelection<FiniteGradient>>(FiniteGradient{5, 0.01, 0.3, 0, 8.0}, 10, 4);

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = dl->timestamps().size();
  Frame::ShPtr kf = dl->loadFrame(0);
  kf->computePyramid(directIcp->nLevels());
  kf->computeDerivatives();
  kf->computePcl();
  featureSelection->select(kf);
  motionModel->update(kf->pose(), kf->t());

  Frame::ShPtr lf = kf;
  traj->append(kf->t(), kf->pose().inverse());
  double entropyRef = 0.;
  double nConstraintsRef = INFd;
  double errorRef = 0.;
  std::vector<Frame::ShPtr> kfs;
  kfs.push_back(kf);
  for (size_t fId = fNo0; fId < fNo0 + nFrames; fId++) {
    try {

      Frame::ShPtr f = dl->loadFrame(fId);
      f->computePyramid(directIcp->nLevels());
      f->pose() = motionModel->predict(f->t());
      log::append("Frame", [&]() { return overlay::frames({kf, lf, f}); });
      auto results = directIcp->align(kf, f);
      f->pose() = results->pose;

      const double entropyRatio = std::log(f->pose().cov().determinant()) / entropyRef;
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

      if (entropyRatio < 0.95) {
        print("Keyframe selected.\n");
        kf = lf;
        kfs.push_back(kf);

        kf->computeDerivatives();
        kf->computePcl();
        featureSelection->select(kf);
        f->pose() = motionModel->predict(f->t());
        results = directIcp->align(kf, f);
        f->pose() = results->pose;
        errorRef = results->normalEquations[0].error;
        nConstraintsRef = results->constraints.size();
        entropyRef = std::log(f->pose().cov().determinant());
      }
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
