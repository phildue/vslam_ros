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

#ifndef VSLAM_OVERLAY_FEATURES_H__
#define VSLAM_OVERLAY_FEATURES_H__
#include "core/Frame.h"
#include "core/Point3D.h"
#include "utils/utils.h"
#include <map>
#include <opencv2/highgui/highgui.hpp>
namespace vslam::overlay {
class Features {
public:
  Features(Frame::ConstShPtr frame, double cellSize, bool annotate = false) :
      _frame(frame),
      _gridCellSize(cellSize),
      _annotate(annotate) {}
  cv::Mat draw() const;
  cv::Mat operator()() const { return draw(); }

private:
  const Frame::ConstShPtr _frame;
  const double _gridCellSize;
  const bool _annotate;
};

class MatchCandidates {
public:
  MatchCandidates(Frame::ConstShPtr f0, Frame::ConstShPtr f1, const MatXd &mask, double maxMask, int idx) :
      _f0(f0),
      _f1(f1),
      _mask(mask),
      _maxMask(maxMask),
      _idx(idx) {}
  cv::Mat draw() const;
  cv::Mat operator()() const { return draw(); }

private:
  void drawFeature(cv::Mat &mat, Feature2D::ConstShPtr ft, const std::string &annotation = "", double radius = 5) const;
  const Frame::ConstShPtr _f0, _f1;
  const MatXd _mask;
  const double _maxMask;
  const int _idx;
};

class CorrespondingPoints {
public:
  CorrespondingPoints(const std::vector<Frame::ConstShPtr> &frames) :
      _frames(frames) {}

  cv::Mat draw() const;
  cv::Mat operator()() const { return draw(); }

private:
  std::vector<Frame::ConstShPtr> _frames;
};

class Matches {
public:
  Matches(
    Frame::ConstShPtr f0,
    Frame::ConstShPtr f1,
    const std::vector<cv::Point> &pts0,
    const std::vector<cv::Point> &pts1,
    const cv::Mat &mask) :
      _f0(f0),
      _f1(f1),
      _pts0(pts0),
      _pts1(pts1),
      _mask(mask) {}

  cv::Mat draw() const;
  cv::Mat operator()() const { return draw(); }

private:
  Frame::ConstShPtr _f0, _f1;
  std::vector<cv::Point> _pts0, _pts1;
  cv::Mat _mask;
};

class FeatureDisplacement {
public:
  FeatureDisplacement(
    const Frame::VecConstShPtr &frames, const Point3D::VecConstShPtr &points, unsigned int maxWidth = 2560, unsigned int nRows = 2);
  FeatureDisplacement(const Frame::VecConstShPtr &frames, unsigned int maxWidth = 2560, unsigned int nRows = 2);

  cv::Mat draw() const;
  cv::Mat operator()() const { return draw(); }

private:
  Frame::VecConstShPtr _frames;
  Point3D::VecConstShPtr _points;
  const unsigned int _maxWidth;
  const unsigned int _nRows;
  static std::map<uint64_t, cv::Scalar> _colorMap;
};

}  // namespace vslam::overlay
#endif  // VSLAM_OVERLAY_FEATURES_H__