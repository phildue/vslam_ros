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

#ifndef VSLAM_FEATURE_TRACKING_OCV_H__
#define VSLAM_FEATURE_TRACKING_OCV_H__
#include "FeatureTracking.h"
namespace vslam
{
class FeatureTrackingOcv : public FeatureTracking
{
public:
  typedef std::shared_ptr<FeatureTrackingOcv> ShPtr;
  typedef std::unique_ptr<FeatureTrackingOcv> UnPtr;
  typedef std::shared_ptr<const FeatureTrackingOcv> ConstShPtr;
  typedef std::unique_ptr<const FeatureTrackingOcv> ConstUnPtr;

  FeatureTrackingOcv() : FeatureTracking(nullptr){};

  std::vector<Point3D::ShPtr> track(
    Frame::ShPtr frameCur, const Frame::VecShPtr & framesRef) const override;

  std::vector<cv::DMatch> removeNonDistinct(
    const std::vector<std::vector<cv::DMatch>> & matchesKnn) const;

  std::vector<cv::DMatch> removeModelOutliers(
    const std::vector<cv::DMatch> & matches, const Feature2D::VecConstShPtr & featuresCur,
    const Feature2D::VecConstShPtr & featuresRef) const;
  Pose computeEgomotion(Frame::ShPtr f0, Frame::ShPtr f1);

private:
  const double _nextBestMatchRatio = 0.8;
  const double _maxReprojectionError = 50.0;
  const size_t _gridCellSize = 15;
  const std::string _descriptorMatcherType = "BruteForce-Hamming";
  void extractFeatures(
    Frame::ShPtr frame, bool applyGrid, std::vector<cv::KeyPoint> & kpts, cv::Mat & desc) const;

  std::vector<Vec3d> triangulateDlt(
    const std::vector<cv::Point> & pts0, const std::vector<cv::Point> & pts1, const Mat4d & P1,
    const Mat4d & P2) const;
};
}  // namespace vslam

#endif  //VSLAM_FEATURE_TRACKING_OCV_H__
