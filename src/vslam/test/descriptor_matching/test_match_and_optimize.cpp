

#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "vslam/bundle_adjustment.h"
#include "vslam/core.h"
#include "vslam/descriptor_matching.h"
#include "vslam/direct_icp.h"
#include "vslam/evaluation.h"
#include "vslam/motion_model.h"
#include "vslam/utils.h"

using namespace vslam;

/*
Attempt: 
Improve trajectory by performing Pose/Point optimization
over key frames using descriptor based matching between keyframes and bundle adjustment.

Result:
- Matches look descent
- Bundle Adjustment reduces reprojection error reasonably
- Optimized trajectory is worse than non optimized

Explanation:
- Still quite some matches are not correct but suffer from aparture problem (e.g screen)
- Matches are relatively spares compared to direct icp
- ?
*/

TEST(DescriptorMatchingTest, DISABLED_EvaluateOnTum)
{
  const std::string experimentId = "test_cpp_descriptor_matching";
  auto dl = std::make_unique<evaluation::tum::DataLoader>(
    "/mnt/dataset/tum_rgbd/", "rgbd_dataset_freiburg2_desk");

  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  log::initialize(outPath, true);
  log::config("default")->delay = 1;

  auto directIcp = std::make_shared<DirectIcp>(DirectIcp::defaultParameters());
  auto motionModel =
    std::make_shared<ConstantVelocityModel>(ConstantVelocityModel::defaultParameters());
  auto descriptorMatching = std::make_shared<FeatureTracking>();
  auto bundleAdjustment = std::make_shared<BundleAdjustment>(50, 30);
  auto config = el::Loggers::getLogger("bundle_adjustment")->configurations();
  config->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  el::Loggers::reconfigureLogger("bundle_adjustment", *config);
  config = el::Loggers::getLogger("tracking")->configurations();
  config->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  el::Loggers::reconfigureLogger("tracking", *config);
  log::config("Frame")->delay = 1;
  log::config("Track")->delay = 1;
  log::config("BAAfter")->delay = 1;

  Trajectory::ShPtr trajectory = std::make_shared<Trajectory>();
  Trajectory::ShPtr trajectoryOptimized = std::make_shared<Trajectory>();
  const size_t fEnd = 300 + 1;  //dl->timestamps().size();
  Pose motion;
  cv::Mat img0 = dl->loadIntensity(0), depth0 = dl->loadDepth(0);
  Frame::ShPtr kf = std::make_shared<Frame>(img0, depth0, dl->cam(), dl->timestamps()[0]);
  kf->computePyramid(directIcp->nLevels());
  kf->computeDerivatives();
  kf->computePcl();
  trajectory->append(dl->timestamps()[0], kf->pose().inverse());
  motionModel->update(kf->pose(), kf->t());
  std::vector<Frame::ShPtr> kfs{kf};
  descriptorMatching->track(kf, {});

  std::map<uint64_t, Frame::VecShPtr> childFrames;
  Pose pose;  //copy of kf pose which is not optimized
  for (size_t fId = 0; fId < fEnd; fId++) {
    try {
      /*Front end computes relative motion from last key frame to current frame.
        Between all keyframes, descriptors are matched to have 2D-3D correspondences for bundle adjustment.
      */
      const cv::Mat img = dl->loadIntensity(fId);
      const cv::Mat depth = dl->loadDepth(fId);
      Frame::ShPtr f = std::make_shared<Frame>(img, depth, dl->cam(), dl->timestamps()[fId]);
      f->computePyramid(directIcp->nLevels());

      log::append("Frame", visualizeFramePair(kf, f));
      TIMED_SCOPE(timer, "computeFrame");
      motion = directIcp->computeEgomotion(kf, f, Pose());
      motionModel->update(f->pose(), f->t());
      f->pose() = motion * pose;
      trajectory->append(f->t(), f->pose().inverse());

      if (motion.translation().norm() > 0.1 || motion.totalRotationDegrees() > 1.0) {
        kf = f;
        kf->computeDerivatives();
        kf->computePcl();
        pose = kf->pose();

        descriptorMatching->track(kf, kfs);
        kfs.push_back(kf);
        if (kfs.size() >= 3) {
          log::append("Track", overlay::CorrespondingPoints({kfs.end() - 3, kfs.end()}));
        }
      } else {
        f->pose() = motion;
        childFrames[kf->id()].push_back(f);
      }
      print(
        "{}/{}: {} m, {:.3f}°\n", fId, fEnd, f->pose().translation().transpose(),
        f->pose().totalRotationDegrees());

      /* Every n frame perform optimization over a window of frames.
         Compute KPIs for "frontend only" and optimized trajectory.
         Optimized trajectory should be better. 
      */
      if (kfs.size() % 10 == 0) {
        log::append("BABefore", overlay::FeatureDisplacement({kfs[0], kfs[kfs.size() - 1]}));
        auto results = bundleAdjustment->optimize({kfs.begin(), kfs.end()}, {kfs[0]});
        print("Reprojection Error: {} --> {}\n", results->errorBefore, results->errorAfter);
        for (auto _kf : kfs) {
          if (results->poses.find(_kf->id()) == results->poses.end()) continue;
          print(
            "Updating: {} by {}\n", _kf->id(),
            (results->poses.at(_kf->id()) * _kf->pose().inverse()).translation().transpose());
          for (auto cf : childFrames[_kf->id()]) {
            cf->pose() = cf->pose() * results->poses.at(_kf->id());
            trajectoryOptimized->append(cf->t(), cf->pose().inverse());
          }
          _kf->pose() = results->poses.at(_kf->id());
          trajectoryOptimized->append(_kf->t(), _kf->pose().inverse());
        }
        for (auto p : kf->featuresWithPoints()) {
          p->point()->position() = results->positions.at(p->point()->id());
        }
        log::append("BAAfter", overlay::FeatureDisplacement({kfs[0], kfs[kfs.size() - 1]}));

        /*Start new track*/
        //kfs = {kf};
        //childFrames = {{kf->id(), childFrames.at(kf->id())}};
      }
      if (fId % 100 == 0) {
        try {
          evaluation::tum::writeTrajectory(*trajectory, trajectoryAlgoPath);
          evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
          evaluation::tum::writeTrajectory(*trajectoryOptimized, trajectoryAlgoPath);
          evaluation::computeKPIs(dl->sequenceId(), experimentId, false);

        } catch (const std::runtime_error & e) {
          print("{}\n", e.what());
        }
      }

    } catch (const std::runtime_error & e) {
      std::cerr << e.what() << std::endl;
    }
  }
}
