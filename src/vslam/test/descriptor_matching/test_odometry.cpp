
/*
Attempt:
Compute odometry by descriptor based matching and computing Fundamental Matrix.
Scale by GT (could be replaced by triangulating points and scale to depth map).
This was mainly to see if this simple VO suffers from the same problem as described
in test_match_and_optimize.

Result:
- Matches don't look too bad but VO is not great

Explanation:
- Similar as test_match_and_optimize? the descriptor based stuff does not seem to work well on this
data
*/
#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "BundleAdjustment.h"
#include "FeatureTracking.h"
#include "FeatureTrackingOcv.h"
#include "Matcher.h"
#include "overlays.h"
#include "vslam/core.h"
#include "vslam/evaluation.h"
#include "vslam/utils.h"

using namespace vslam;

TEST(DescriptorMatchingTest, DISABLED_EvaluateOnTum) {
  const std::string experimentId = "test_c++";
  auto dl = std::make_unique<evaluation::tum::DataLoader>("/mnt/dataset/tum_rgbd/", "rgbd_dataset_freiburg2_desk");

  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  const int tRmse = 200;
  std::thread thread;
  log::initialize(outPath, true);

  auto descriptorMatching = std::make_shared<FeatureTrackingOcv>();
  auto config = el::Loggers::getLogger("tracking")->configurations();
  config->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  el::Loggers::reconfigureLogger("tracking", *config);
  log::config("Correspondences")->show = 1;

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  Trajectory::ShPtr trajGt = evaluation::tum::loadTrajectory(dl->pathGt());
  const size_t fEnd = dl->timestamps().size();
  Pose motion;
  Pose pose;
  cv::Mat img0 = dl->loadIntensity(0), depth0 = dl->loadDepth(0);
  Frame::ShPtr kf = std::make_shared<Frame>(img0, depth0, dl->cam(), dl->timestamps()[0]);
  traj->append(dl->timestamps()[0], pose);
  for (size_t fId = 1; fId < fEnd; fId += 1) {
    try {
      print("{}/{}: {} m, {:.3f}°\n", fId, fEnd, pose.translation().transpose(), pose.totalRotationDegrees());

      const cv::Mat img = dl->loadIntensity(fId);
      const cv::Mat depth = dl->loadDepth(fId);
      Frame::ShPtr f = std::make_shared<Frame>(img, depth, dl->cam(), dl->timestamps()[fId]);

      {
        TIMED_SCOPE(timer, "computeFrame");
        motion = descriptorMatching->computeEgomotion(kf, f);
        // TODO(me) simple workaround for scaling
        motion.translation() *= trajGt->motionBetween(kf->t(), f->t())->translation().norm() / motion.translation().norm();
      }
      pose = motion * pose;
      traj->append(dl->timestamps()[fId], pose.inverse());
      kf = f;

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
