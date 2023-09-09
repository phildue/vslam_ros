#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "descriptor_matching/overlays.h"
#include "keypoint_selection/keypoint_selection.h"
#include "motion_model/ConstantVelocityModel.h"
#include "vslam/core.h"
#include "vslam/direct.h"
#include "vslam/evaluation.h"
#include "vslam/utils.h"
using namespace vslam;

int main(int argc, char **argv) {
  const std::string filename = argv[0];
  const std::string experimentId = filename.substr(filename.find_last_of("/") + 1);
  const std::string sequenceId = argc > 1 ? argv[1] : "rgbd_dataset_freiburg1_desk";
  auto dl = std::make_unique<evaluation::tum::DataLoader>("/mnt/dataset/tum_rgbd/", sequenceId);

  Trajectory::ShPtr trajGt = evaluation::tum::loadTrajectory(dl->pathGt());
  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  log::initialize(outPath, true, TEST_RESOURCE "/log.conf");

  auto rgbOdometry = std::make_shared<AlignmentRgb>(AlignmentRgb::defaultParameters());
  auto rgbPoseGraph = std::make_shared<AlignmentRgbPoseGraph>(4);
  auto motionModel = std::make_shared<ConstantVelocityModel>(0.03, INFd, INFd);
  auto featureSelector = [](Frame::ShPtr f) { return keypoint::selectPyramidGradient(f, 10, 0.01, 0.3, 0, 8.0f, false, 1); };

  log::config("AlignmentRgbPoseGraph")->show = false;
  log::config("AlignmentRgbPoseGraph")->delay = 1;
  log::config("Frame")->show = true;
  log::config("Frame")->delay = 1;
  log::config("Features")->show = true;
  log::config("Features")->delay = 1;

  auto config = el::Loggers::getLogger("default")->configurations();
  config->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  el::Loggers::reconfigureLogger("default", *config);

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  Trajectory::ShPtr trajOptimized = std::make_shared<Trajectory>();

  const size_t fEnd = 200;  // dl->timestamps().size();
  Frame::ShPtr kf = dl->loadFrame(0);
  kf->computePyramid(rgbOdometry->nLevels());
  kf->computeDerivatives();
  kf->computePcl();
  featureSelector(kf);
  log::append("Features", overlay::Features(kf, 20));

  motionModel->update(kf->pose(), kf->t());

  Frame::ShPtr lf = kf;
  traj->append(kf->t(), kf->pose().inverse());

  std::map<size_t, std::map<size_t, Pose>> poseGraph;
  std::vector<Frame::ShPtr> keyframes;
  keyframes.push_back(kf);
  poseGraph[kf->id()] = {};
  bool newKeyFrame = false;
  for (size_t fId = 0; fId < fEnd; fId++) {
    try {

      Frame::ShPtr f = dl->loadFrame(fId);
      f->computePyramid(rgbOdometry->nLevels());
      f->computeDerivatives();
      f->computePcl();

      f->pose() = motionModel->predict(f->t());
      log::append("Frame", overlay::frames({kf, lf, f}));
      rgbPoseGraph->align({kf, f}, {kf});

      print("{}/{}: {} m, {:.3f}°\n", fId, fEnd, f->pose().translation().transpose(), f->pose().totalRotationDegrees());
      kf = f;
      featureSelector(kf);

      log::append("Features", overlay::Features(f, 20));
      motionModel->update(f->pose(), f->t());
      traj->append(dl->timestamps()[fId], f->pose().inverse());
    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);

  // evaluation::tum::writeTrajectory(*trajOptimized, trajectoryAlgoPath);
  // evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
}
