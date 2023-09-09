#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "descriptor_matching/overlays.h"
#include "motion_model/ConstantVelocityModel.h"
#include "vslam/core.h"
#include "vslam/direct.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
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
  log::initialize(outPath, true);
  log::configure(TEST_RESOURCE "/log.conf");

  auto rgbOdometry = std::make_shared<AlignmentRgb>(AlignmentRgb::defaultParameters());
  auto rgbPoseGraph = std::make_shared<AlignmentRgbPoseGraph>(3);
  auto motionModel = std::make_shared<ConstantVelocityModel>(0.03, INFd, INFd);
  auto featureSelection = std::make_shared<FeatureSelection>(5, 0.01, 0.3, 0, 8.0, 20, 4);

  log::config("AlignmentRgbPoseGraph")->show = true;
  log::config("AlignmentRgbPoseGraph")->delay = 0;
  log::config("Frame")->show = true;
  log::config("Frame")->delay = 1;
  log::config("Features")->show = true;
  log::config("Features")->delay = 0;

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
  featureSelection->select(kf);
  log::append("Features", overlay::Features(kf, 20));

  motionModel->update(kf->pose(), kf->t());

  Frame::ShPtr lf = kf;
  traj->append(kf->t(), kf->pose().inverse());
  double entropyRef = 0.;

  std::map<size_t, std::map<size_t, Pose>> poseGraph;
  std::vector<Frame::ShPtr> keyframes;
  bool newKeyFrame = false;
  for (size_t fId = 1; fId < fEnd; fId++) {
    try {

      Frame::ShPtr f = dl->loadFrame(fId);
      f->computePyramid(rgbOdometry->nLevels());
      f->computeDerivatives();
      f->pose() = motionModel->predict(f->t());
      log::append("Frame", overlay::frames({kf, lf, f}));
      auto results = rgbOdometry->align(kf, f);
      f->pose() = results->pose;

      const double entropyRatio = std::log(f->pose().cov().determinant()) / entropyRef;
      print(
        "{}/{}: {} m, {:.3f}° err: {:.3f} #: {} |H|={:.3f}\n",
        fId,
        fEnd,
        f->pose().translation().transpose(),
        f->pose().totalRotationDegrees(),
        results->normalEquations[0].error,
        results->normalEquations[0].nConstraints,
        entropyRatio);

      if (entropyRatio < 0.95) {
        print("Keyframe selected.\n");
        newKeyFrame = true;
        kf = lf;
        kf->computeDerivatives();
        kf->computePcl();
        featureSelection->select(kf, false);

        keyframes.push_back(kf);
        poseGraph[kf->id()] = {};

        f->pose() = motionModel->predict(f->t());
        results = rgbOdometry->align(kf, f);
        f->pose() = results->pose;
        entropyRef = std::log(f->pose().cov().determinant());
      }

      /*
      for (size_t l = 3; l > 0; l--) {

        Feature2D::VecShPtr features(results->constraints[l].size());
        std::transform(
          std::execution::par_unseq,
          results->constraints[l].begin(),
          results->constraints[l].end(),
          features.begin(),
          [&](const AlignmentRgb::Constraint::ConstShPtr c) {
            const cv::Vec2f dIuv = f->dI(l).at<cv::Vec2f>(c->uv1(1), c->uv1(0));
            const Vec2d response{dIuv[0], dIuv[1]};
            return std::make_unique<Feature2D>(c->uv1.cast<double>(), f, l, response.norm());
          });
        f->addFeatures(features);
      }
      log::append("Features", overlay::Features(f, 20));
      */
      poseGraph.at(kf->id())[f->t()] = f->pose() * kf->pose().inverse();
      motionModel->update(f->pose(), f->t());
      traj->append(dl->timestamps()[fId], f->pose().inverse());
      lf = f;

      if (newKeyFrame && keyframes.size() >= 3) {
        newKeyFrame = false;
        rgbPoseGraph->align(keyframes, {keyframes.begin(), keyframes.end() - 1});
        for (const auto &kf : keyframes) {
          trajOptimized->append(kf->t(), kf->pose().inverse());
          for (const auto &[t, motion] : poseGraph.at(kf->id())) {
            trajOptimized->append(t, (motion * kf->pose()).inverse());
          }
        }
        evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
        evaluation::computeKPIs(dl->sequenceId(), experimentId, false);

        evaluation::tum::writeTrajectory(*trajOptimized, trajectoryAlgoPath);
        evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
      }

    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);

  // evaluation::tum::writeTrajectory(*trajOptimized, trajectoryAlgoPath);
  // evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
}
