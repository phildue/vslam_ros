// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>

#include "PoseGraph.h"
using namespace testing;
using namespace vslam;

TEST(PoseGraphTest, OptimizeSynthetic) {
  auto poseGraph = std::make_shared<PoseGraph>();

  Pose pose01;
  pose01.cov() = Mat6d::Identity();
  pose01.SE3().translation().x() = 0.1;
  Pose pose12;
  pose12.cov() = Mat6d::Identity();
  pose12.SE3().translation().x() = 0.1;
  Pose pose02;
  pose02.cov() = Mat6d::Identity() * 0.5;
  pose02.SE3().translation().x() = 0.21;
  poseGraph->addMeasurement(0, 1, pose01);
  poseGraph->addMeasurement(1, 2, pose12);
  poseGraph->addMeasurement(0, 2, pose02);
  poseGraph->optimize();
  EXPECT_NEAR(poseGraph->poses().at(2).translation().x(), 0.21, 0.005);
}
