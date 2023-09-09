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
  const std::vector<std::string> sequences = evaluation::tum::sequencesTraining();
  const std::string sequenceId = argc > 1 ? argv[1] : sequences[random::U(0, sequences.size() - 1)];
  auto dl = std::make_unique<evaluation::tum::DataLoader>("/mnt/dataset/tum_rgbd/", sequenceId);

  Trajectory::ShPtr trajGt = evaluation::tum::loadTrajectory(dl->pathGt());
  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  const size_t nFrames = 100;
  const size_t fNo0 = nFrames >= dl->nFrames() ? 0UL : random::U(0UL, dl->nFrames() - nFrames);

  log::initialize(outPath, true);
  log::configure(TEST_RESOURCE "/log/");
  auto rgbPoseGraph = std::make_shared<AlignmentRgbPoseGraph>(1, 50);
  auto featureSelection = std::make_shared<FeatureSelection>(10, 0.01, 0.3, 0, 8.0f, 20.0, 1);

  log::config("AlignmentRgbPoseGraph")->show = 0;
  log::config("Frame")->show = 1;
  log::config("Features")->show = 0;

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();

  std::vector<Frame::ShPtr> frames;
  for (size_t fId = fNo0; fId < fNo0 + nFrames; fId++) {
    try {

      Frame::ShPtr f = dl->loadFrame(fId);
      f->pose() = *trajGt->poseAt(f->t());
      f->computePyramid(4);
      f->computeDerivatives();
      f->computePcl();
      log::append("Frame", [&]() { return overlay::frames({f}); });

      if (fId % 10 == 0) {
        if (!frames.empty()) {
          auto lf = *frames.rbegin();
          Feature2D::VecShPtr features;
          Point3D::VecShPtr points;
          for (auto ft : lf->features()) {
            const Vec3d p3dw = lf->pose().inverse().SE3() * lf->p3d(ft->v(), ft->u());
            const Vec2d uv1 = f->project(f->pose().SE3() * p3dw);
            if (f->withinImage(uv1, 7.0)) {
              const double scale = 1.0 / std::pow(2.0, ft->level());
              const cv::Vec2f dIuv = f->dI(ft->level()).at<cv::Vec2f>(uv1(1) * scale, uv1(0) * scale);
              const Vec2d response{dIuv[0], dIuv[1]};
              auto ft1 = std::make_shared<Feature2D>(uv1, f, ft->level(), response.norm());
              features.push_back(ft1);
              if (ft->point()) {
                ft->point()->addFeature(ft1);
              } else {
                auto p = std::make_shared<Point3D>(p3dw, Feature2D::VecShPtr{ft, ft1});
                ft1->point() = p;
                ft->point() = p;
                points.push_back(p);
              }
            }
          }
          f->addFeatures(features);
          LOG(INFO) << format("Keeping: {} features.", features.size());
          LOG(INFO) << format("Created: {} points.", points.size());
        }
        featureSelection->select(f, false);
        frames.push_back(f);

        if (frames.size() > 3) {
          rgbPoseGraph->align(frames, {frames.begin(), frames.end() - 1});
        }
        traj->append(f->t(), f->pose().inverse());
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
