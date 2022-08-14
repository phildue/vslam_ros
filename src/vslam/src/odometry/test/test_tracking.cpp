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

#include <core/core.h>
#include <gtest/gtest.h>
#include <utils/utils.h>

#include <opencv2/highgui.hpp>

#include "odometry/odometry.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(TrackingTest, Track)
{
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.png") / 5000.0;
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.png");

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f0 = std::make_shared<FrameRgbd>(img, depth, cam, 3, 0);
  auto f1 = std::make_shared<FrameRgbd>(img, depth, cam, 3, 0);
  std::vector<FrameRgbd::ShPtr> frames;
  frames.push_back(f0);
  auto tracking = std::make_shared<FeatureTracking>();
  tracking->extractFeatures(f0);
  tracking->extractFeatures(f1);

  auto featuresCandidate = tracking->selectCandidates(f1, frames);
  EXPECT_EQ(f0->features().size(), featuresCandidate.size());

  auto points = tracking->match(f1, featuresCandidate);
  EXPECT_EQ(points.size(), featuresCandidate.size());
}
