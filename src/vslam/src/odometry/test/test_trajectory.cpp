//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include <core/core.h>
#include <utils/utils.h>
#include <opencv2/highgui.hpp>
#include "odometry/odometry.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(TrajectoryTest, Create)
{
  auto poseGraph =
    std::make_shared<Trajectory>(utils::loadTrajectory(TEST_RESOURCE "/trajectory.txt"));

}
TEST(TrajectoryTest, Interpolate)
{
  SE3d pose0, pose2;
  pose2.translation().x() += 2.0;
  std::map<Timestamp, SE3d> poses;
  poses[0] = pose0;
  poses[2] = pose2;
  auto poseGraph = std::make_shared<Trajectory>(poses);
  auto pose1 = poseGraph->poseAt(1);
  EXPECT_NEAR(pose1->pose().translation().x(), 1.0, 1e-7);
  std::cout << pose1->pose().log().transpose() << std::endl;

}
