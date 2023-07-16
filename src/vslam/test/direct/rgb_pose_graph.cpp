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
#if true
#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "descriptor_matching/overlays.h"
#include "vslam/core.h"
#include "vslam/direct.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/motion_model.h"
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

  Trajectory::ShPtr trajGt = evaluation::tum::loadTrajectory(dl->pathGt());
  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  const size_t nFrames = std::min(250UL, dl->nFrames());
  const size_t fNo0 = nFrames >= dl->nFrames() ? 0UL : random::U(0UL, dl->nFrames() - nFrames);

  log::initialize(outPath, true);
  log::configure(TEST_RESOURCE "/log/");
  auto alignerOdom = std::make_shared<AlignmentRgbd>(AlignmentRgbd::defaultParameters());
  auto alignerGraph = std::make_shared<AlignmentRgbPoseGraph>(1, 30);
  auto motionModel = std::make_shared<ConstantVelocityModel>(10.0, INFd, INFd);
  auto featureSelection = std::make_shared<FeatureSelection>(5, 0.01, 0.3, 0, 8.0, 20, 4);
  const int nKeyFramesOpt = 5;

  log::config("AlignmentRgbPoseGraph")->show = -1;
  log::config("Frame")->show = 1;
  log::config("Features")->show = 1;
  log::config("Keyframe")->show = 1;

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  Trajectory::ShPtr trajOptimized = std::make_shared<Trajectory>();

  Frame::ShPtr kf = dl->loadFrame(fNo0);
  kf->computePyramid(4);
  kf->computeDerivatives();
  kf->computePcl();
  featureSelection->select(kf, true);
  motionModel->update(kf->pose(), kf->t());

  Frame::ShPtr lf = nullptr;
  traj->append(kf->t(), kf->pose().inverse());
  double entropyRef = 0.;
  std::map<size_t, std::map<size_t, Pose>> poseGraph;
  std::vector<Frame::ShPtr> kfs;
  kfs.push_back(kf);
  poseGraph[kf->id()] = {};
  Point3D::VecShPtr points;
  for (size_t fId = fNo0 + 1; fId < fNo0 + nFrames; fId++) {
    try {

      Frame::ShPtr f = dl->loadFrame(fId);
      f->computePyramid(4);
      f->pose() = motionModel->predict(f->t());
      log::append("Frame", [&]() { return lf ? overlay::frames({kf, lf, f}) : overlay::frames({kf, f}); });
      auto r = alignerOdom->align(kf, f);
      f->pose() = r->pose;
      const double entropyRatio = std::log(f->pose().cov().determinant()) / entropyRef;

      print(
        "{}/{}: {} m, {:.3f}°, |H|={:.3f}, e={:.3f} \n",
        fId,
        fNo0 + nFrames,
        f->pose().translation().transpose(),
        f->pose().totalRotationDegrees(),
        entropyRatio,
        r->normalEquations[0].error);

      if (lf && (entropyRatio < 0.95 || r->normalEquations[0].nConstraints < 300)) {
        lf->computeDerivatives();
        lf->computePcl();

        for (const auto &ft : kf->features()) {
          const double scale = 1.0 / std::pow(2.0, ft->level());
          const Vec3d p3dw = kf->pose().SE3().inverse() * kf->p3d(ft->v(), ft->u());
          auto uv1 = lf->project(lf->pose().SE3() * p3dw);
          if (uv1.allFinite() && lf->withinImage(uv1, 7.0)) {
            const cv::Vec2f dIuv = lf->dI(ft->level()).at<cv::Vec2f>(uv1(1) * scale, uv1(0) * scale);
            const Vec2d response{dIuv[0], dIuv[1]};
            auto ft1 = std::make_shared<Feature2D>(uv1, lf, ft->level(), response.norm());
            if (ft->point()) {
              ft1->point() = ft->point();
              ft->point()->addFeature(ft1);
            } else {
              ft->point() = ft1->point() = std::make_shared<Point3D>(p3dw, Feature2D::VecShPtr{ft, ft1});
              points.push_back(ft->point());
            }
            lf->addFeature(ft1);
          }
        }
        print("Tracked points: {}/{}\n", lf->featuresWithPoints().size(), points.size());

        kf = lf;
        kfs.push_back(kf);
        poseGraph[kf->id()] = {};

        featureSelection->select(kf, false);
        log::append("Keyframe", overlay::Features(kf, 20));
        Frame::VecShPtr kfWindow;
        std::copy_if(kfs.begin(), kfs.end() - 1, std::back_inserter(kfWindow), [&](auto kf_) {
          auto fts = kf_->features();
          const size_t nObservationsInFrame = std::count_if(fts.begin(), fts.end(), [&](auto ft) {
            return ft->point() && ft->point()->features().back() == ft && kf->withinImage(kf->project(ft->point()->position()));
          });
          print("#Observations of {} in frame {}: {}\n", kf_->id(), kf->id(), nObservationsInFrame);
          return nObservationsInFrame > 150;
        });
        kfWindow.push_back(kf);
        print("#Frames in window: {}\n", kfWindow.size());
        if (kfWindow.size() > nKeyFramesOpt + 1) {
          alignerGraph->align(kfWindow, {kfWindow.begin(), kfWindow.end() - nKeyFramesOpt});
          for (const auto &kf : kfs) {
            trajOptimized->append(kf->t(), kf->pose().inverse());
            for (const auto &[t, motion] : poseGraph.at(kf->id())) {
              trajOptimized->append(t, (motion * kf->pose()).inverse());
            }
          }
        }

        r = alignerOdom->align(kf, f);
        f->pose() = r->pose;
        entropyRef = std::log(f->pose().cov().determinant());
        print(
          "{}/{}: New keyframe: {} m, {:.3f}°, H_0={:.3f}, e={:.3f} \n",
          fId,
          fNo0 + nFrames,
          f->pose().translation().transpose(),
          f->pose().totalRotationDegrees(),
          entropyRef,
          r->normalEquations[0].error);
      }
      poseGraph.at(kf->id())[f->t()] = f->pose() * kf->pose().inverse();
      motionModel->update(f->pose(), f->t());
      traj->append(dl->timestamps()[fId], f->pose().inverse());
      lf = f;

    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }
  for (const auto &_kf : kfs) {
    _kf->removeFeatures();
  }
  evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);

  evaluation::tum::writeTrajectory(*trajOptimized, trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
}
#endif