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

#include "core/Camera.h"
using namespace testing;
using namespace vslam;

class CameraTest : public Test {
public:
  Eigen::MatrixX3d _points3D;
  const int _nPoints = 8;
  const double precision = 1e-7;

  CameraTest() { setup3D(); }

  void setup3D() {
    _points3D.resize(_nPoints, Eigen::NoChange);

    _points3D.row(0) = Eigen::Vector3d(-0.5, -0.5, 10);
    _points3D.row(1) = Eigen::Vector3d(0.5, -0.5, 10);
    _points3D.row(2) = Eigen::Vector3d(0.5, -0.5, 10);
    _points3D.row(3) = Eigen::Vector3d(-0.5, 0.5, 10);
    _points3D.row(4) = Eigen::Vector3d(-0.5, -0.5, 10.5);
    _points3D.row(5) = Eigen::Vector3d(0.5, -0.5, 10.5);
    _points3D.row(6) = Eigen::Vector3d(0.5, -0.5, 10.5);
    _points3D.row(7) = Eigen::Vector3d(-0.5, 0.5, 10.5);
  }
};
TEST_F(CameraTest, Constructor) {
  const double f = 10;
  const double cx = 320;
  const double cy = 240;
  auto camera = std::make_shared<Camera>(f, f, cx, cy, 640, 480);
  EXPECT_EQ(camera->fx(), f);
  EXPECT_EQ(camera->cx(), cx);
}
TEST_F(CameraTest, ForwardBackwardProjection) {
  const double f = 10;
  const double cx = 320;
  const double cy = 240;
  auto camera = std::make_shared<Camera>(f, f, cx, cy, 640, 480);

  for (int i = 0; i < _points3D.rows(); i++) {
    const auto &p3d = _points3D.row(i);
    auto pImage = camera->project(p3d);

    EXPECT_NEAR(pImage.x(), (f * p3d.x() + cx * p3d.z()) / p3d.z(), precision) << "Projection failed for point: [" << i << "] -> " << p3d;
    EXPECT_NEAR(pImage.y(), (f * p3d.y() + cy * p3d.z()) / p3d.z(), precision) << "Projection failed for point: [" << i << "] -> " << p3d;

    Eigen::Vector3d pRay = camera->reconstruct(pImage);

    EXPECT_NEAR(pRay.x(), (pImage.x() - cx) / f, precision) << "Projection failed for point: [" << i << "] -> " << p3d;
    EXPECT_NEAR(pRay.y(), (pImage.y() - cy) / f, precision) << "Projection failed for point: [" << i << "] -> " << p3d;

    Eigen::Vector3d p3dBack = camera->reconstruct(pImage, p3d.z());

    EXPECT_NEAR(p3d.x(), p3dBack.x(), precision) << "Back-projection failed for point: [" << i << "] -> " << pImage << " | " << p3d;
    EXPECT_NEAR(p3d.y(), p3dBack.y(), precision) << "Back-projection failed for point: [" << i << "] -> " << pImage << " | " << p3d;
    EXPECT_NEAR(p3d.z(), p3dBack.z(), precision) << "Back-projection failed for point: [" << i << "] -> " << pImage << " | " << p3d;
  }
}

TEST_F(CameraTest, ProjectingInvalid) {
  const double f = 10;
  const double cx = 320;
  const double cy = 240;
  auto camera = std::make_shared<Camera>(f, f, cx, cy, 640, 480);
  {
    const auto &p3d = Eigen::Vector3d(1, 1, 0);
    auto pImage = camera->project(p3d);
    EXPECT_TRUE(std::isnan(pImage.x()));
    EXPECT_TRUE(std::isnan(pImage.y()));
  }
  {
    const auto &p3d = Eigen::Vector3d(1, 1, -1);
    auto pImage = camera->project(p3d);
    EXPECT_TRUE(std::isnan(pImage.x()));
    EXPECT_TRUE(std::isnan(pImage.y()));
  }
}

TEST_F(CameraTest, Reprojection) {
  const double f = 1;
  const double cx = 320;
  const double cy = 240;
  auto camera0 = std::make_shared<Camera>(f, f, cx, cy, 640, 480);
  auto camera1 = std::make_shared<Camera>(f, f, cx, cy, 640, 480);
  SE3d poseCam1;
  poseCam1.rotX(0.15);
  for (int i = 0; i < _points3D.rows(); i++) {
    const auto &p3d = _points3D.row(i);
    auto pImage = camera0->project(p3d);
    auto pImage1 = camera1->project(poseCam1.inverse() * camera0->reconstruct(pImage));

    EXPECT_NEAR(pImage1.x(), pImage.x() - poseCam1.translation().x(), precision) << "Projection failed for point: [" << i << "] -> " << p3d;
  }
}
