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

#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <set>

#include "FeatureTrackingOcv.h"
#include "overlays.h"
#include "utils/log.h"
#define LOG_TRACKING(level) CLOG(level, "tracking")

namespace vslam
{
std::vector<cv::DMatch> FeatureTrackingOcv::removeNonDistinct(
  const std::vector<std::vector<cv::DMatch>> & matchesKnn) const
{
  std::vector<cv::DMatch> matchesDistinct;

  for (size_t i = 0; i < matchesKnn.size(); i++) {
    const auto & m = matchesKnn[i][0];
    if (m.distance < _nextBestMatchRatio * matchesKnn[i][1].distance) {
      matchesDistinct.push_back(m);
    }
  }
  LOG_TRACKING(DEBUG) << "Kept [" << matchesDistinct.size() << "/" << matchesKnn.size()
                      << "] distinct matches.";
  return matchesDistinct;
}
std::vector<cv::DMatch> FeatureTrackingOcv::removeModelOutliers(
  const std::vector<cv::DMatch> & matches, const Feature2D::VecConstShPtr & featuresCur,
  const Feature2D::VecConstShPtr & featuresRef) const
{
  std::vector<cv::Point> pRef;
  std::vector<cv::Point> pTarget;
  for (const auto & m : matches) {
    auto fRef = featuresRef[m.queryIdx];
    pRef.push_back(cv::Point(fRef->position().x(), fRef->position().y()));
    auto fCur = featuresCur[m.trainIdx];
    pTarget.push_back(cv::Point(fCur->position().x(), fCur->position().y()));
  }
  cv::Mat matchMask, K;
  cv::eigen2cv(featuresRef[0]->frame()->camera()->K(), K);
  auto E = cv::findEssentialMat(pRef, pTarget, K, cv::FM_RANSAC, 0.99, 3, matchMask);

  std::vector<cv::DMatch> matchesInliers;
  matchesInliers.reserve(matches.size());
  for (size_t i = 0U; i < matches.size(); i++) {
    if (matchMask.at<uchar>(i, 0) > 0U) {
      matchesInliers.push_back(matches[i]);
    }
  }
  LOG_TRACKING(DEBUG) << "Kept [" << matchesInliers.size() << "/" << matches.size() << "] inliers.";
  return matchesInliers;
}
std::vector<Point3D::ShPtr> FeatureTrackingOcv::track(
  Frame::ShPtr frameCur, const Frame::VecShPtr & framesRef) const
{
  cv::Mat descCur;
  std::vector<cv::KeyPoint> kptsCur;
  extractFeatures(frameCur, framesRef.empty(), kptsCur, descCur);
  if (framesRef.empty()) {
    LOG_TRACKING(WARNING) << "No reference frames passed.";
    FeatureTracking::createFeatures(kptsCur, descCur, DescriptorType::ORB, frameCur);
    return std::vector<Point3D::ShPtr>();
  }
  auto featuresRef = selectCandidates(frameCur, framesRef);
  auto featuresCur = FeatureTracking::createFeatures(kptsCur, descCur, DescriptorType::ORB);
  for (auto ft : featuresCur) {
    ft->frame() = frameCur;
  }
  auto descRef = FeatureTracking::createDescriptorMatrix(
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()), descCur.type());

  std::vector<std::vector<cv::DMatch>> matchesKnn;
  cv::Ptr<cv::DescriptorMatcher> matcherOcv =
    cv::DescriptorMatcher::create(_descriptorMatcherType.c_str());
  /*cv::Mat mask = MatcherOcv::computeMaskReprojection(
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()),
    std::vector<Feature2D::ConstShPtr>(featuresCur.begin(), featuresCur.end()),
    _maxReprojectionError);*/
  matcherOcv->knnMatch(descRef, descCur, matchesKnn, 2);
  LOG_TRACKING(DEBUG) << "Found [" << matchesKnn.size() << "] matches.";
  //TODO: first check what happened to the features that already have points and should be visible

  auto matches = removeNonDistinct(matchesKnn);

  matches = removeModelOutliers(
    matches, {featuresCur.begin(), featuresCur.end()}, {featuresRef.begin(), featuresRef.end()});

  size_t matchedPoints = 0;
  for (const auto & m : matches) {
    if (
      0 < m.queryIdx || static_cast<size_t>(m.queryIdx) < featuresRef.size() || 0 < m.trainIdx ||
      static_cast<size_t>(m.trainIdx) < featuresCur.size()) {
      continue;
    }
    auto fRef = featuresRef[m.queryIdx];
    auto fCur = featuresCur[m.trainIdx];
    auto z = frameCur->depth().at<float>(fCur->position().y(), fCur->position().x());
    if (fCur->point()) {
      LOG_TRACKING(DEBUG) << "Feature [" << fRef->id() << "] cannot be matched with Feature ["
                          << fCur->id() << "] because it was already matched to Point ["
                          << fCur->point()->id() << "]";
    } else if (fRef->point()) {
      if (!frameCur->observationOf(fRef->point()->id())) {
        fCur->point() = fRef->point();
        fRef->point()->addFeature(fCur);
        //LOG_TRACKING(DEBUG) << "Feature [" << fCur->id() << "] was matched to point: ["
        //                    << fRef->point()->id() << "]";
        matchedPoints++;
      } else {
        LOG_TRACKING(DEBUG) << "Point: [" << fRef->point()->id() << "] was already matched on ["
                            << frameCur->id() << "] with lower distance. Skipping..";
      }
    } else if (z > 0) {
      std::vector<Feature2D::ShPtr> features = {fRef, fCur};
      const Vec3d p3d = frameCur->image2world(fCur->position(), z);
      fCur->point() = std::make_shared<Point3D>(p3d, features);
      fRef->point() = fCur->point();
      //LOG_TRACKING(DEBUG) << "New Point between [" << fRef->id() << "] and [" << fCur->id()
      //                    << "]: [" << fCur->point()->id() << "] with distance: [" << m.distance
      //                    << "]";
    }
  }

  // Apply the grid only on the unmatched features.

  featuresCur = FeatureTracking::gridSubsampling(featuresCur, frameCur, _gridCellSize);
  frameCur->addFeatures(featuresCur);
  std::vector<Point3D::ShPtr> points;
  points.reserve(featuresCur.size());
  std::for_each(featuresCur.begin(), featuresCur.end(), [&](auto ft) {
    if (ft->point() && ft->point()->features().size() == 2) {
      points.push_back(ft->point());
    }
  });

  LOG_TRACKING(DEBUG) << "Frame [" << frameCur->id() << "]: #New Points: [" << points.size()
                      << "] #Matched Points: [" << matchedPoints << "] #Unmatched Features: ["
                      << frameCur->features().size() - frameCur->featuresWithPoints().size() << "]";

  return points;
  //LOG_IMG("Tracking") << std::make_shared<FeaturePlot>(frameCur, _gridCellSize);
}

Pose FeatureTrackingOcv::computeEgomotion(Frame::ShPtr f0, Frame::ShPtr f1)
{
  cv::Mat desc0, desc1;
  std::vector<cv::KeyPoint> kpts0, kpts1;

  extractFeatures(f0, true, kpts0, desc0);
  extractFeatures(f1, false, kpts1, desc1);

  std::vector<std::vector<cv::DMatch>> matchesKnn;
  cv::Ptr<cv::DescriptorMatcher> matcherOcv =
    cv::DescriptorMatcher::create(_descriptorMatcherType.c_str());
  matcherOcv->knnMatch(desc0, desc1, matchesKnn, 2);
  LOG_TRACKING(DEBUG) << "Found [" << matchesKnn.size() << "] matches.";
  auto matches = FeatureTrackingOcv::removeNonDistinct(matchesKnn);

  std::vector<cv::Point> pts0, pts1;
  for (const auto & m : matches) {
    pts0.push_back(kpts0[m.queryIdx].pt);
    pts1.push_back(kpts1[m.trainIdx].pt);
  }
  cv::Mat matchMask, K, R, t;
  cv::eigen2cv(f0->camera()->K(), K);
  cv::Mat E = cv::findEssentialMat(pts0, pts1, K, cv::FM_RANSAC, 0.99, 3, matchMask);
  cv::recoverPose(E, pts0, pts1, K, R, t, matchMask);
  log::append("Correspondences", overlay::Matches(f0, f1, pts0, pts1, matchMask));
  //TODO triangulate, scale with depth
  Mat3d R_;
  Vec3d t_;
  cv::cv2eigen(R, R_);
  cv::cv2eigen(t, t_);

  return Pose(SE3d(R_, t_).inverse());
}

std::vector<Vec3d> FeatureTrackingOcv::triangulateDlt(
  const std::vector<cv::Point> & pts0, const std::vector<cv::Point> & pts1, const Mat4d & P1,
  const Mat4d & P2) const
{
  std::vector<Vec3d> p3d(pts0.size());
  for (size_t i = 0; i < p3d.size(); i++) {
    Mat4d A = Mat4d::Zero();  // (n views, 4)
    A.row(0) = pts0[i].x * P1.row(2) - P1.row(0);
    A.row(1) = pts0[i].y * P1.row(2) - P1.row(1);
    A.row(2) = pts1[i].x * P2.row(2) - P2.row(0);
    A.row(3) = pts1[i].y * P2.row(2) - P2.row(1);
    //TODO(me)
    //s, vh = np.linalg.svd(A, full_matrices = True)
    // p3d[:, i] = (vh[-1] / vh[-1, -1])
  }
  return p3d;
}

void FeatureTrackingOcv::extractFeatures(
  Frame::ShPtr frame, bool applyGrid, std::vector<cv::KeyPoint> & kpts, cv::Mat & desc) const
{
  cv::Mat mask = cv::Mat(frame->height(), frame->width(), CV_8UC1);

  for (int v = 0; v < mask.rows; v++) {
    for (int u = 0; u < mask.cols; u++) {
      const float d = frame->depth().at<float>(v, u);
      if (std::isfinite(d) && d > 0.1f) {
        mask.at<std::uint8_t>(v, u) = 1U;
      }
    }
  }
  cv::Ptr<cv::DescriptorExtractor> detector = cv::FastFeatureDetector::create();

  detector->detect(frame->intensity(), kpts, mask);
  LOG_TRACKING(DEBUG) << "Frame [" << frame->id() << "]: Detected Keypoints: " << kpts.size();

  if (applyGrid) {
    kpts = FeatureTracking::gridSubsampling(kpts, frame, _gridCellSize);
  }

  LOG_TRACKING(DEBUG) << "Frame [" << frame->id()
                      << "]: Kept keypoints after grid: " << kpts.size();
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
  extractor->compute(frame->intensity(), kpts, desc);
  LOG_TRACKING(DEBUG) << "Frame [" << frame->id() << "]: Computed descriptors: " << desc.rows << "x"
                      << desc.cols;
}

}  // namespace vslam
