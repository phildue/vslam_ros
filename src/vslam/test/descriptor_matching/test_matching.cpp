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

#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "vslam/core.h"
#include "vslam/descriptor_matching.h"
#include "vslam/evaluation.h"
#include "vslam/utils.h"
using namespace testing;
using namespace vslam;
using namespace vslam::evaluation;

TEST(TrackingTest, SelectVisible)
{
  log::config("default")->show = TEST_VISUALIZE;
  cv::Mat depth = tum::loadDepth(TEST_RESOURCE "/depth.png");
  cv::Mat img = tum::loadIntensity(TEST_RESOURCE "/rgb.png");

  auto f0 = std::make_shared<Frame>(img, depth, tum::Camera(), 0);
  auto f1 = std::make_shared<Frame>(img, depth, tum::Camera(), 1);
  std::vector<Frame::ShPtr> frames;
  frames.push_back(f0);
  auto tracking = std::make_shared<FeatureTracking>();
  f0->addFeatures(tracking->extractFeatures(f0));
  f1->addFeatures(tracking->extractFeatures(f1));

  auto featuresCandidate = tracking->selectCandidates(f1, frames);
  EXPECT_EQ(f0->features().size(), featuresCandidate.size());
}

TEST(TrackingTest, GridSubsampling)
{
  log::config("default")->show = TEST_VISUALIZE;
  cv::Mat depth = tum::loadDepth(TEST_RESOURCE "/depth.png");
  cv::Mat img = tum::loadIntensity(TEST_RESOURCE "/rgb.png");

  auto f0 = std::make_shared<Frame>(img, depth, tum::Camera(), 0);

  cv::Ptr<cv::DescriptorExtractor> detector = cv::ORB::create();
  std::vector<cv::KeyPoint> kpts;
  detector->detect(f0->intensity(), kpts);
  FeatureTracking::createFeatures(kpts, f0);
  log::append("BeforeGridSubsampling", overlay::Features({f0}, f0->width() * 200));
  auto kptsFiltered = FeatureTracking::gridSubsampling(kpts, f0, 30);
  f0->removeFeatures();
  FeatureTracking::createFeatures(kptsFiltered, f0);
  log::append("AfterGridSubsampling", overlay::Features({f0}, 30));
}

TEST(TrackingTest, FeatureConversion)
{
  log::config("default")->show = TEST_VISUALIZE;
  cv::Mat depth = tum::loadDepth(TEST_RESOURCE "/depth.png");
  cv::Mat img = tum::loadIntensity(TEST_RESOURCE "/rgb.png");

  auto f0 = std::make_shared<Frame>(img, depth, tum::Camera(), 0);

  cv::Ptr<cv::DescriptorExtractor> detector = cv::ORB::create();
  std::vector<cv::KeyPoint> kpts;
  cv::Mat desc;
  detector->detectAndCompute(f0->intensity(), cv::Mat(), kpts, desc);
  auto features = FeatureTracking::createFeatures(kpts, desc, DescriptorType::ORB);

  for (size_t i = 0U; i < kpts.size(); i++) {
    EXPECT_NEAR(kpts[i].pt.x, features[i]->position().x(), 0.001);
    EXPECT_NEAR(kpts[i].pt.y, features[i]->position().y(), 0.001);
    EXPECT_NEAR(kpts[i].response, features[i]->response(), 0.001);
    EXPECT_NEAR(kpts[i].octave, features[i]->level(), 0.001);
    MatXd descriptor;
    cv::cv2eigen(desc.row(i), descriptor);
    EXPECT_NEAR((descriptor - features[i]->descriptor()).norm(), 0.0, 0.001);
  }

  auto descBack = FeatureTracking::createDescriptorMatrix(
    std::vector<Feature2D::ConstShPtr>(features.begin(), features.end()), desc.type());

  cv::Mat diff;
  cv::subtract(desc, descBack, diff);
  EXPECT_NEAR(cv::norm(diff), 0, 0.001);
}

TEST(TrackingTest, DISABLED_MatcherWithCombinedError)
{
  log::config("default")->show = TEST_VISUALIZE;
  const size_t nFeatures = 5;
  auto f0 = std::make_shared<Frame>(
    tum::loadIntensity(TEST_RESOURCE "/1311868164.363181.png"),
    tum::loadDepth(TEST_RESOURCE "/1311868164.338541.png"), tum::Camera(), 1311868164363181000U);

  auto f1 = std::make_shared<Frame>(
    tum::loadIntensity(TEST_RESOURCE "/1311868165.499179.png"),
    tum::loadDepth(TEST_RESOURCE "/1311868165.409979.png"), tum::Camera(), 1311868165499179000U);

  auto trajectoryGt = tum::loadTrajectory(TEST_RESOURCE "/trajectory.txt");
  f0->pose() = *trajectoryGt->poseAt(f0->t());
  f1->pose() = *trajectoryGt->poseAt(f1->t());

  auto tracking = std::make_shared<FeatureTracking>();
  f0->addFeatures(tracking->extractFeatures(f0, true));
  f1->addFeatures(tracking->extractFeatures(f1));

  MatXd reprojectionError = vslam::Matcher::computeDistanceMat(
    Frame::ConstShPtr(f0)->features(), Frame::ConstShPtr(f1)->features(),
    [](auto ft1, auto ft2) { return vslam::Matcher::reprojectionError(ft1, ft2); });

  EXPECT_NE(reprojectionError.norm(), 0.0);

  MatXd descriptorDistance = vslam::Matcher::computeDistanceMat(
    Frame::ConstShPtr(f0)->features(), Frame::ConstShPtr(f1)->features(),
    [](auto ft1, auto ft2) { return vslam::Matcher::descriptorHamming(ft1, ft2); });

  EXPECT_NE(descriptorDistance.norm(), 0.0);

  const VecXd reprojectionErrorMin = reprojectionError.rowwise().minCoeff();
  const VecXd descriptorDistanceMin = descriptorDistance.rowwise().minCoeff();
  for (size_t i = 0; i < f0->features().size(); i++) {
    reprojectionError.row(i) /= std::max<double>(reprojectionErrorMin(i), 1);
    descriptorDistance.row(i) /= std::max<double>(descriptorDistanceMin(i), 1);
  }
  MatXd combinedDistance = reprojectionError + descriptorDistance;
  const VecXd combinedDistanceMin = combinedDistance.rowwise().minCoeff();
  for (size_t i = 0; i < f0->features().size(); i++) {
    combinedDistance.row(i) /= std::max<double>(combinedDistanceMin(i), 1);
  }
  std::vector<Feature2D::ConstShPtr> featureSubset;
  for (size_t i = 0; i < nFeatures; i++) {
    const int idx0 = static_cast<int>(random::U(0, f0->features().size() - 1));
    featureSubset.push_back(f0->features()[idx0]);
    std::cout << "Reprojection Error Min:" << reprojectionError.row(idx0).minCoeff()
              << " Reprojection Error Max:" << reprojectionError.row(idx0).maxCoeff() << std::endl;

    log::append(
      "FeatureCandidatesReprojection",
      overlay::MatchCandidates(
        f0, f1, reprojectionError / reprojectionError.row(idx0).minCoeff(), 1.5, idx0));

    std::cout << "Descriptor Distance Min:" << descriptorDistance.row(idx0).minCoeff()
              << " Descriptor Distance Max:" << descriptorDistance.row(idx0).maxCoeff()
              << std::endl;

    log::append(
      "FeatureCandidatesDescriptorDistance",
      overlay::MatchCandidates(
        f0, f1, descriptorDistance / descriptorDistance.row(idx0).minCoeff(), 1.5, idx0));

    std::cout << "Combined Error Min:" << combinedDistance.row(idx0).minCoeff()
              << " Combined Error Max:" << combinedDistance.row(idx0).maxCoeff() << std::endl;

    log::append(
      "FeatureCandidatesCombined", overlay::MatchCandidates(f0, f1, combinedDistance, 1.5, idx0));
  }
  auto matcher = std::make_shared<vslam::Matcher>(vslam::Matcher::reprojectionHamming, 4.0, 0.8);
  const std::vector<vslam::Matcher::Match> matches =
    matcher->match(featureSubset, Frame::ConstShPtr(f1)->features());
  EXPECT_EQ(matches.size(), 3);
}
