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
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <set>

#include "FeatureTracking.h"
#include "features/overlays.h"
#include "overlays.h"
#include "utils/log.h"

#define LOG_NAME "tracking"
#define MLOG(level) CLOG(level, LOG_NAME)

namespace vslam {
std::vector<cv::KeyPoint>
FeatureTracking::gridSubsampling(const std::vector<cv::KeyPoint> &keypoints, Frame::ConstShPtr frame, double cellSize) {
  const size_t nRows = static_cast<size_t>(static_cast<float>(frame->height(0)) / static_cast<float>(cellSize));
  const size_t nCols = static_cast<size_t>(static_cast<float>(frame->width(0)) / static_cast<float>(cellSize));

  /* Create grid for subsampling where each cell contains index of keypoint with the highest response
   *  or the total amount of keypoints if empty */
  std::vector<size_t> grid(nRows * nCols, keypoints.size());
  for (size_t idx = 0U; idx < keypoints.size(); idx++) {
    const auto &kp = keypoints[idx];
    const size_t r = static_cast<size_t>(static_cast<float>(kp.pt.y) / static_cast<float>(cellSize));
    const size_t c = static_cast<size_t>(static_cast<float>(kp.pt.x) / static_cast<float>(cellSize));
    if (grid[r * nCols + c] >= keypoints.size() || kp.response > keypoints[grid[r * nCols + c]].response) {
      grid[r * nCols + c] = idx;
    }
  }

  std::vector<cv::KeyPoint> kptsGrid;
  kptsGrid.reserve(nRows * nCols);
  std::for_each(grid.begin(), grid.end(), [&](auto idx) {
    if (idx < keypoints.size()) {
      kptsGrid.push_back(keypoints[idx]);
    }
  });
  return kptsGrid;
}

std::vector<Feature2D::ShPtr> FeatureTracking::gridSubsampling(
  const std::vector<Feature2D::ShPtr> &features, Frame::ConstShPtr frame, double cellSize, const Eigen::MatrixXi &mask) {
  const size_t nRows = static_cast<size_t>(static_cast<float>(frame->height(0)) / static_cast<float>(cellSize));
  const size_t nCols = static_cast<size_t>(static_cast<float>(frame->width(0)) / static_cast<float>(cellSize));

  /* Create grid for subsampling where each cell contains index of keypoint with the highest response
   *  or the total amount of keypoints if empty */
  auto grid = std::vector<size_t>(nRows * nCols, features.size());
  for (size_t idx = 0U; idx < features.size(); idx++) {
    const auto &ft = features[idx];
    const size_t r = static_cast<size_t>(static_cast<float>(ft->position().y()) / static_cast<float>(cellSize));
    const size_t c = static_cast<size_t>(static_cast<float>(ft->position().x()) / static_cast<float>(cellSize));
    if (mask.size() > 0 && mask(r, c) <= 0) {
      continue;
    } else if (grid[r * nCols + c] >= features.size() || (ft->response() > features[grid[r * nCols + c]]->response())) {
      grid[r * nCols + c] = idx;
    }
  }

  std::vector<Feature2D::ShPtr> ftsGrid;
  ftsGrid.reserve(nRows * nCols);
  std::for_each(grid.begin(), grid.end(), [&](auto idx) {
    if (idx < features.size()) {
      ftsGrid.push_back(features[idx]);
    }
  });
  return ftsGrid;
}

std::vector<Matcher::Match> FeatureTracking::removeModelOutliers(
  const std::vector<Matcher::Match> &matches,
  const std::vector<Feature2D::ConstShPtr> &featuresRef,
  const std::vector<Feature2D::ConstShPtr> &featuresCur) const {
  std::vector<cv::Point> pRef;
  std::vector<cv::Point> pTarget;
  for (const auto &m : matches) {
    auto fRef = featuresRef[m.idxRef];
    pRef.push_back(cv::Point(fRef->position().x(), fRef->position().y()));
    auto fCur = featuresCur[m.idxCur];
    pTarget.push_back(cv::Point(fCur->position().x(), fCur->position().y()));
  }

  cv::Mat matchMask, K;
  cv::eigen2cv(featuresRef[0]->frame()->camera()->K(), K);
  auto E = cv::findEssentialMat(pRef, pTarget, K, cv::FM_RANSAC, 0.99, 3, matchMask);

  std::vector<Matcher::Match> matchesInliers;
  matchesInliers.reserve(matches.size());
  for (size_t i = 0U; i < matches.size(); i++) {
    if (matchMask.at<uchar>(i, 0) > 0U) {
      matchesInliers.push_back(matches[i]);
    } else {
      // LOG_TRACKING(DEBUG) << "Dropped: [" << featuresRef[matches[i].queryIdx]->id() << "] to ["
      //                     << featuresCur[matches[i].trainIdx]->id() << "] as model outlier.";
    }
  }
  MLOG(DEBUG) << "Kept [" << matchesInliers.size() << "/" << matches.size() << "] inliers.";
  return matchesInliers;
}

FeatureTracking::FeatureTracking(Matcher::ConstShPtr matcher) :
    _matcher(matcher) {
  log::create(LOG_NAME);
}

std::vector<Point3D::ShPtr> FeatureTracking::track(Frame::ShPtr frameCur, const std::vector<Frame::ShPtr> &framesRef) const {
  std::vector<Point3D::ShPtr> points;
  if (framesRef.empty()) {
    MLOG(WARNING) << "No reference frames given. Assuming initial frame.";
    auto featuresCur = extractFeatures(frameCur, true);
    frameCur->addFeatures(featuresCur);
  } else {
    auto featuresCur = extractFeatures(frameCur, false);
    auto featuresRef = selectCandidates(frameCur, framesRef);
    points = match(featuresRef, featuresCur);

    Feature2D::VecShPtr unmatchedFeatures;
    MatXi mask = MatXi::Ones(frameCur->height(0) / _gridCellSize, frameCur->width(0) / _gridCellSize);
    Point3D::VecShPtr strongPoints;
    for (const auto &ft : featuresCur) {
      if (ft->point()) {
        mask((int)(ft->position().y() / _gridCellSize), (int)(ft->position().x() / _gridCellSize)) = 0;
        frameCur->addFeature(ft);
        if (ft->point()->features().size() >= framesRef.size() + 1) {
          strongPoints.push_back(ft->point());
        }
      } else {
        unmatchedFeatures.push_back(ft);
      }
    }
    frameCur->addFeatures(gridSubsampling(unmatchedFeatures, Frame::ConstShPtr(frameCur), _gridCellSize, mask));

    Point3D::VecShPtr lostPoints, matchedPoints;
    for (auto ft : featuresRef) {
      if (ft->point() && !frameCur->observationOf(ft->point()->id())) {
        lostPoints.push_back(ft->point());
        MLOG(DEBUG) << format(
          "Lost point: {} even though it should be visible at: {}",
          ft->point()->id(),
          frameCur->world2image(ft->point()->position()).transpose());
      } else if (ft->point()) {
        matchedPoints.push_back(ft->point());
      }
    }
    MLOG(DEBUG) << format(
      "Lost {} out of {} points. We have {} strong points with >{} observations.",
      lostPoints.size(),
      lostPoints.size() + matchedPoints.size(),
      strongPoints.size(),
      framesRef.size() + 1);
  }
  log::append(LOG_NAME, overlay::Features(frameCur, _gridCellSize));

  return points;
}

Pose FeatureTracking::computeEgomotion(Frame::ShPtr f0, Frame::ShPtr f1) {
  auto ft0 = f0->features().empty() ? extractFeatures(f0) : f0->features();
  auto ft1 = f1->features().empty() ? extractFeatures(f1) : f1->features();
  std::vector<Matcher::Match> matches = _matcher->match({ft0.begin(), ft0.end()}, {ft1.begin(), ft1.end()});
  MLOG(INFO) << format("Matches found: {}", matches.size());
  return estimatePose(matches, {ft0.begin(), ft0.end()}, {ft1.begin(), ft1.end()});
}

Feature2D::VecShPtr FeatureTracking::extractFeatures(Frame::ShPtr frame, bool applyGrid, size_t nMax) const {
  cv::Mat mask = cv::Mat(frame->height(), frame->width(), CV_8UC1);

  for (int v = 0; v < mask.rows; v++) {
    for (int u = 0; u < mask.cols; u++) {
      const float d = frame->depth().at<float>(v, u);
      if (std::isfinite(d) && d > 0.1f) {
        mask.at<std::uint8_t>(v, u) = 1U;
      }
    }
  }
  cv::Ptr<cv::DescriptorExtractor> detector = cv::GFTTDetector::create();
  std::vector<cv::KeyPoint> kpts;

  detector->detect(frame->intensity(), kpts, mask);
  MLOG(DEBUG) << "Detected Keypoints: " << kpts.size();

  if (applyGrid) {
    kpts = gridSubsampling(kpts, frame, _gridCellSize);
  }

  MLOG(DEBUG) << "Remaining keypoints after grid: " << kpts.size();
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
  cv::Mat desc;
  extractor->compute(frame->intensity(), kpts, desc);
  MLOG(DEBUG) << "Computed descriptors: " << desc.rows << "x" << desc.cols;

  std::vector<Feature2D::ShPtr> features;
  features.reserve(desc.rows);
  MatXd descriptor;
  cv::cv2eigen(desc, descriptor);
  for (int i = 0; i < desc.rows; i++) {
    const auto &kp = kpts[i];
    features.push_back(std::make_shared<Feature2D>(
      Vec2d(kp.pt.x, kp.pt.y), frame, kp.octave, kp.response, Descriptor(descriptor.row(i), DescriptorType::ORB)));
  }
  std::sort(features.begin(), features.end(), [&](auto ft0, auto ft1) { return ft0->response() < ft1->response(); });
  return std::vector<Feature2D::ShPtr>(features.begin(), features.begin() + std::min(features.size(), nMax));
}

std::vector<Point3D::ShPtr> FeatureTracking::match(Frame::ShPtr frameCur, const std::vector<Feature2D::ShPtr> &featuresRef) const {
  return match(featuresRef, frameCur->features());
}

std::vector<Point3D::ShPtr>
FeatureTracking::match(const std::vector<Feature2D::ShPtr> &featuresRef, const std::vector<Feature2D::ShPtr> &featuresCur) const {
  if (featuresRef.empty()) {
    MLOG(WARNING) << "No reference features given for matching.";
    return std::vector<Point3D::ShPtr>();
  }
  if (featuresCur.empty()) {
    MLOG(WARNING) << "No target features given for matching.";
    return std::vector<Point3D::ShPtr>();
  }
  std::vector<Matcher::Match> matches = _matcher->match(
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()),
    std::vector<Feature2D::ConstShPtr>(featuresCur.begin(), featuresCur.end()));

  matches = removeModelOutliers(
    matches,
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()),
    std::vector<Feature2D::ConstShPtr>(featuresCur.begin(), featuresCur.end()));

  std::vector<Point3D::ShPtr> points;
  points.reserve(matches.size());
  for (const auto &m : matches) {
    auto fRef = featuresRef[m.idxRef];
    auto fCur = featuresCur[m.idxCur];
    auto frameCur = fCur->frame();
    auto z = frameCur->depth().at<float>(fCur->position().y(), fCur->position().x());
    if (fCur->point()) {
      MLOG(DEBUG) << "Feature [" << fRef->id() << "] cannot be matched with Feature [" << fCur->id()
                  << "] because it was already matched to Point [" << fCur->point()->id() << "]";
    } else if (fRef->point()) {
      if (!frameCur->observationOf(fRef->point()->id())) {
        fCur->point() = fRef->point();
        fRef->point()->addFeature(fCur);
        MLOG(DEBUG) << "Feature [" << fCur->id() << "] was matched to point: [" << fRef->point()->id() << "]";

      } else {
        MLOG(DEBUG) << "Point: [" << fRef->point()->id() << "] was already matched on [" << frameCur->id()
                    << "] with lower distance. Skipping..";
      }
    } else if (z > 0) {
      std::vector<Feature2D::ShPtr> features = {fRef, fCur};
      const Vec3d p3d = frameCur->image2world(fCur->position(), z);
      fCur->point() = std::make_shared<Point3D>(p3d, features);
      fRef->point() = fCur->point();
      points.push_back(fCur->point());
      MLOG(DEBUG) << "New Point between [" << fRef->id() << "] and [" << fCur->id() << "]: [" << fCur->point()->id() << "] with distance: ["
                  << m.distance << "]";
    }
  }

  MLOG(DEBUG) << "#New Points: " << points.size();

  return points;
}

std::vector<Feature2D::ShPtr>
FeatureTracking::selectCandidates(Frame::ConstShPtr frameCur, const std::vector<Frame::ShPtr> &framesRef) const {
  /* Prefer features from newer frames as they should have closer appearance*/
  std::vector<Frame::ShPtr> framesSorted(framesRef.begin(), framesRef.end());
  std::sort(framesSorted.begin(), framesSorted.end(), [](auto f0, auto f1) { return f0->t() > f1->t(); });

  std::set<std::uint64_t> pointIds;
  std::vector<Feature2D::ShPtr> candidates;
  if (!framesRef.empty()) {
    candidates.reserve(framesRef.size() * framesRef.at(0)->features().size());
  }
  for (auto &f : framesSorted) {
    if (f->id() == frameCur->id()) {
      continue;
    }
    for (auto ft : f->features()) {
      if (!ft->point()) {
        candidates.push_back(ft);
      } else if (
        pointIds.find(ft->point()->id()) == pointIds.end() && frameCur->withinImage(frameCur->world2image(ft->point()->position()))) {
        candidates.push_back(ft);
        pointIds.insert(ft->point()->id());
      }
    }
  }
  MLOG(DEBUG) << "#Candidate Features: " << candidates.size() << " #Visible Points: " << pointIds.size()
              << " #Unmatched Features: " << candidates.size() - pointIds.size();

  return candidates;
}
std::vector<Feature2D::ShPtr> FeatureTracking::createFeatures(const std::vector<cv::KeyPoint> &keypoints, Frame::ShPtr frame) {
  std::vector<Feature2D::ShPtr> features;
  features.reserve(keypoints.size());
  for (const auto &kp : keypoints) {
    features.push_back(std::make_shared<Feature2D>(Vec2d(kp.pt.x, kp.pt.y), frame, kp.octave, kp.response));
  }
  if (frame) {
    frame->addFeatures(features);
  }
  return features;
}

std::vector<Feature2D::ShPtr> FeatureTracking::createFeatures(
  const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &desc, const DescriptorType &descType, Frame::ShPtr frame) {
  std::vector<Feature2D::ShPtr> features;
  features.reserve(keypoints.size());
  MatXd descriptor;
  cv::cv2eigen(desc, descriptor);
  for (int i = 0; i < desc.rows; i++) {
    const auto &kp = keypoints[i];
    features.push_back(
      std::make_shared<Feature2D>(Vec2d(kp.pt.x, kp.pt.y), frame, kp.octave, kp.response, Descriptor(descriptor.row(i), descType)));
  }
  if (frame) {
    frame->addFeatures(features);
  }
  return features;
}

cv::Mat FeatureTracking::createDescriptorMatrix(const std::vector<Feature2D::ConstShPtr> &features, int dtype) {
  // TODO is this fast enough?
  MatXd descriptors(features.size(), features[0]->descriptor().rows());
  for (size_t i = 0U; i < features.size(); i++) {
    descriptors.row(i) = features[i]->descriptor();
  }
  cv::Mat descriptorsOcv;
  cv::eigen2cv(descriptors, descriptorsOcv);
  descriptorsOcv.convertTo(descriptorsOcv, dtype);
  return descriptorsOcv;
}

Pose FeatureTracking::estimatePose(
  const std::vector<Matcher::Match> &matches, const Feature2D::VecConstShPtr &fts0, const Feature2D::VecConstShPtr &fts1) const {
  std::vector<cv::Point> pRef;
  std::vector<cv::Point> pTarget;
  for (const auto &m : matches) {
    auto fRef = fts0[m.idxRef];
    pRef.push_back(cv::Point(fRef->position().x(), fRef->position().y()));
    auto fCur = fts1[m.idxCur];
    pTarget.push_back(cv::Point(fCur->position().x(), fCur->position().y()));
  }

  cv::Mat matchMask;
  cv::Mat K, R, t;
  cv::eigen2cv(fts0[0]->frame()->camera()->K(), K);
  auto E = cv::findEssentialMat(pRef, pTarget, K, cv::FM_RANSAC, 0.99, 3, matchMask);
  cv::recoverPose(E, pRef, pTarget, K, R, t, matchMask);
  Mat3d R_;
  Vec3d t_;
  cv::cv2eigen(R, R_);
  cv::cv2eigen(t, t_);

  return Pose(SE3d(R_, t_));
}

}  // namespace vslam