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
#include "core/Camera.h"
#include "core/Frame.h"
#include "core/Point3D.h"
#include <gtest/gtest.h>

using namespace testing;
using namespace vslam;

TEST(FrameTest, BadDimensions) {
  cv::Mat img(480, 640, CV_8UC1);

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  EXPECT_ANY_THROW(std::make_shared<Frame>(img, cam)) << "Should throw because image dimensions don't match with camera parameters";

  cv::Mat depth(520, 640, CV_32FC1);
  cv::Mat img1(480, 640, CV_8UC1);

  EXPECT_ANY_THROW(std::make_shared<Frame>(img1, depth, cam)) << "Should throw because image dimensions don't match depth dimensions";
}
TEST(FrameTest, BadAccessDepth) {
  cv::Mat img(480, 640, CV_8UC1);
  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  auto f = std::make_shared<Frame>(img, cam);

  EXPECT_EQ(-1, f->depth(10, 10));
}

TEST(FrameTest, GoodAccessDepth) {
  cv::Mat img(480, 640, CV_8UC1);
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(20));

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  auto f = std::make_shared<Frame>(img, depth, cam);

  EXPECT_EQ(20, f->depth(10, 10));
}

TEST(FrameTest, AccessPcl) {
  cv::Mat img(480, 640, CV_8UC1);
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(20));

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  auto f = std::make_shared<Frame>(img, depth, cam);

  EXPECT_ANY_THROW(f->pcl());

  f->computePcl();
  EXPECT_FALSE(f->pcl().empty());
}

TEST(FrameTest, Pyramid) {
  cv::Mat img(480, 640, CV_8UC1);
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(20));

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  auto f = std::make_shared<Frame>(img, cam);

  EXPECT_ANY_THROW(f->intensity(2));
  f->computePyramid(3);
  f->intensity(2);
}

TEST(FrameTest, BadAccessDerivative) {
  cv::Mat img(480, 640, CV_8UC1);
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(20));

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  auto f = std::make_shared<Frame>(img, cam);

  EXPECT_ANY_THROW(f->dI());
}
TEST(FrameTest, GoodAccessDerivative) {
  cv::Mat img(480, 640, CV_8UC1);
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(20));

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  auto f = std::make_shared<Frame>(img, cam);
  f->computeDerivatives();
  f->dI();
}
TEST(FrameTest, Pcl) {
  cv::Mat img(480, 640, CV_8UC1);
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(20));

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  auto f = std::make_shared<Frame>(img, depth, cam);
  f->computePyramid(3);
  f->computePcl();
  auto pclFull = f->pcl();
  ASSERT_NEAR((pclFull[100 * 640 + 100] - f->p3d(100, 100)).norm(), 0, 0.001);
  for (size_t i = 0; i < f->nLevels(); i++) {
    auto pcl = f->pcl(i, true);
    EXPECT_FALSE(pcl.empty());

    for (const auto &p : pcl) {
      auto uv = f->project(p, i);
      EXPECT_GE(uv.x(), 0);
      EXPECT_GE(uv.y(), 0);
      EXPECT_LE(uv.x(), f->width(i));
      EXPECT_LE(uv.y(), f->height(i));
      const double z = f->depth(uv.y(), uv.x(), i);
      EXPECT_TRUE(std::isfinite(z) && z > 0);
      EXPECT_NEAR(p.z(), z, 0.001);
      EXPECT_NEAR((f->reconstruct(uv, z, i) - p).norm(), 0.0, 0.001);

      EXPECT_NEAR((f->p3d(std::round(uv.y()), std::round(uv.x()), i) - p).norm(), 0.0, 0.001) << "Failed for uv=" << uv.transpose();
      EXPECT_NEAR((f->world2image(f->image2world(uv, z, i), i) - uv).norm(), 0.0, 0.001);
    }
  }
}
TEST(FrameTest, AddDeleteFeatures) {
  cv::Mat img(480, 640, CV_8UC1);
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(20));

  auto cam = std::make_shared<vslam::Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
  auto f0 = std::make_shared<Frame>(img, depth, cam);
  auto f1 = std::make_shared<Frame>(img, depth, cam);

  for (int i = 0; i < 6; i++) {
    auto ft = std::make_shared<Feature2D>(Vec2d::Random(), f0);
    f0->addFeature(ft);
  }
  for (int i = 0; i < 3; i++) {
    auto ft = std::make_shared<Feature2D>(Vec2d::Random(), f1);
    f1->addFeature(ft);
  }
  for (int i = 0; i < 3; i++) {
    auto ft0 = f0->features()[i];
    auto ft1 = f1->features()[i];

    ft0->point() = std::make_shared<Point3D>(Vec3d::Random(), Feature2D::VecShPtr({ft0, ft1}));
    ft1->point() = ft0->point();
  }
  EXPECT_EQ(f0->features().size(), 6);
  EXPECT_EQ(f1->featuresWithPoints().size(), 3);
  f0->removeFeatures();
  EXPECT_EQ(f0->features().size(), 0);
  EXPECT_EQ(f1->featuresWithPoints().size(), 0);
}