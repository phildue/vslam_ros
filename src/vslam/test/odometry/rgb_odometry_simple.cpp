

#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "vslam/core.h"
#include "vslam/evaluation.h"
#include "vslam/odometry.h"
#include "vslam/utils.h"

using namespace vslam;

/*This test aims to test the simple API which mostly uses opencv datatypes and only does the 2 frame alignment*/

int main(int argc, char **argv) {
  const std::string filename = argv[0];
  const std::string experimentId = filename.substr(filename.find_last_of("/") + 1);
  const std::string sequenceId = argc > 1 ? argv[1] : "rgbd_dataset_freiburg1_desk";
  auto dl = std::make_unique<evaluation::tum::DataLoader>("/mnt/dataset/tum_rgbd/", sequenceId);

  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  const int tRmse = 200;
  std::thread thread;

  auto alignment = std::make_shared<AlignmentRgb>(AlignmentRgb::defaultParameters());
  log::initialize(outPath, true);
  log::configure(TEST_RESOURCE "/log/");
  log::config("Features")->show = 1;
  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = dl->timestamps().size();
  Pose motion;
  Pose pose;
  cv::Mat img0 = dl->loadIntensity(0), depth0 = dl->loadDepth(0);
  traj->append(dl->timestamps()[0], pose);
  for (size_t fId = 0; fId < fEnd; fId++) {
    try {
      print("{}/{}: {} m, {:.3f}°\n", fId, fEnd, pose.translation().transpose(), pose.totalRotationDegrees());

      const cv::Mat img = dl->loadIntensity(fId);
      const cv::Mat depth = dl->loadDepth(fId);

      cv::imshow("Frame", colorizedRgbd(img, depth));
      cv::waitKey(1);
      {
        TIMED_SCOPE(timer, "computeFrame");
        motion = alignment->align(dl->cam(), img0, depth0, img, motion.SE3());
      }
      pose = motion * pose;
      traj->append(dl->timestamps()[fId], pose.inverse());
      img0 = img;
      depth0 = depth;

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
