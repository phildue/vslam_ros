#include <gtest/gtest.h>
using namespace testing;

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include <opencv2/highgui.hpp>
#include <sophus/ceres_manifold.hpp>
#include <thread>

#include "direct/overlay.h"
#include "vslam/core.h"
#include "vslam/direct.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/utils.h"
using namespace vslam;
#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "vslam/core.h"
#include "vslam/direct.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/pose_prediction.h"
#include "vslam/utils.h"
using namespace vslam;

/*
Attempt:
- Use multi frame (ceres-based) aligner to estimate pose of current frame, for comparison against own implementation
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
  const size_t nFrames = std::min(-1UL, dl->nFrames());
  const size_t fNo0 = nFrames >= dl->nFrames() ? 0UL : random::U(0UL, dl->nFrames() - nFrames);

  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  const int tRmse = 200;
  std::thread thread;
  log::initialize(outPath, true);
  log::config("Frame")->show = 1;

  auto aligner = std::make_shared<AlignmentRgbPoseGraph>(4, 50, false);
  auto motionModel = std::make_shared<pose_prediction::ConstantVelocityModel>(10.0, INFd, INFd);
  auto featureSelection = std::make_shared<FeatureSelection>(5, 0.01, 0.3, 0, 8.0, 20, 4);

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = dl->timestamps().size();
  Frame::ShPtr kf = dl->loadFrame(fNo0);
  kf->computePyramid(4);
  kf->computeDerivatives();
  kf->computePcl();
  featureSelection->select(kf, true);
  motionModel->update(kf->pose(), kf->t());

  traj->append(kf->t(), kf->pose().inverse());

  for (size_t fId = fNo0 + 1; fId < fNo0 + nFrames; fId++) {
    try {

      Frame::ShPtr f = dl->loadFrame(fId);
      f->computePyramid(4);
      f->pose() = motionModel->predict(f->t());
      log::append("Frame", [&]() { return overlay::frames({kf, f}); });
      aligner->align({kf, f}, {kf});

      print("{}/{}: {} m, {:.3f}° \n", fId, fEnd, f->pose().translation().transpose(), f->pose().totalRotationDegrees());
      kf->removeFeatures();
      kf = f;
      kf->computeDerivatives();
      kf->computePcl();
      featureSelection->select(kf, true);
      motionModel->update(f->pose(), f->t());
      traj->append(f->t(), f->pose().inverse());

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
  if (thread.joinable()) {
    thread.join();
  }
  evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
}