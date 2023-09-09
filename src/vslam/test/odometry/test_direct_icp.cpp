

#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "vslam/core.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/odometry.h"
#include "vslam/pose_prediction.h"
#include "vslam/utils.h"
using namespace vslam;

/*
Attempt:

Result:

Explanation:

*/
int main(int argc, char **argv) {
  const std::string filename = argv[0];
  const std::string experimentId = filename.substr(filename.find_last_of("/") + 1);
  const std::string sequenceId = argc > 1 ? argv[1] : "rgbd_dataset_freiburg1_desk";
  auto dl = std::make_unique<evaluation::tum::DataLoader>("/mnt/dataset/tum_rgbd/", sequenceId);

  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  const int tRmse = 200;
  std::thread thread;
  log::initialize(outPath, true);
  log::config("Frame")->show = 1;

  log::initialize(outPath, true);
  log::configure(TEST_RESOURCE "/log.conf");
  auto directIcp = std::make_shared<AlignmentRgbd>(AlignmentRgbd::defaultParameters());
  auto motionModel = std::make_shared<pose_prediction::ConstantVelocityModel>(INFd, INFd, INFd);
  auto featureSelection = std::make_shared<FeatureSelection<FiniteGradient>>(FiniteGradient{5, 0.01, 0.3, 0, 8.0}, 10, 1);

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

  for (size_t fId = 0; fId < fEnd; fId++) {
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
        kf->removeFeatures();
        kf = lf;
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
  if (thread.joinable()) {
    thread.join();
  }
  evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
}
