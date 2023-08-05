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
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "core/Point3D.h"
#include "overlays.h"
#include "utils/visuals.h"
namespace vslam::overlay {
std::map<uint64_t, cv::Scalar> FeatureDisplacement::_colorMap = {};

cv::Mat Features::draw() const {
  cv::Mat mat;
  cv::cvtColor(_frame->intensity(), mat, cv::COLOR_GRAY2BGR);
  cv::putText(
    mat, format("{} | {} |", _frame->id(), _frame->t()), cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

  if (_gridCellSize > 1) {
    for (size_t r = 0; r < _frame->height(0); r += _gridCellSize) {
      cv::line(mat, cv::Point(0, r), cv::Point(_frame->width(0), r), cv::Scalar(128, 128, 128));
    }
    for (size_t c = 0; c < _frame->width(0); c += _gridCellSize) {
      cv::line(mat, cv::Point(c, 0), cv::Point(c, _frame->height(0)), cv::Scalar(128, 128, 128));
    }
  }

  // std::stringstream ss;
  // ss << format("#{} *{}", _frame->features().size(), _frame->featuresWithPoints().size());
  // cv::putText(mat, ss.str(), cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
  for (auto ft : _frame->features()) {
    cv::Point center(ft->position().x(), ft->position().y());
    const double radius = 1 * std::pow(2.0, ft->level());
    if (ft->point()) {
      if (_annotate) {
        std::stringstream ss;
        ss << ft->point()->id();
        cv::putText(mat, ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
      }

      cv::circle(mat, center, radius, cv::Scalar(255, 0, 0), 2);
    } else {
      if (_annotate) {
        std::stringstream ss;
        ss << ft->id();
        cv::putText(mat, ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
      }
      cv::rectangle(mat, cv::Rect(center - cv::Point(radius, radius), center + cv::Point(radius, radius)), cv::Scalar(0, 0, 255), 2);
    }
  }
  return mat;
}

cv::Mat MatchCandidates::draw() const {
  cv::Mat mat0, mat1;
  cv::cvtColor(_f0->intensity(), mat0, cv::COLOR_GRAY2BGR);
  auto ft = _f0->features()[_idx];
  drawFeature(mat0, ft, std::to_string(ft->id()));
  cv::cvtColor(_f1->intensity(), mat1, cv::COLOR_GRAY2BGR);
  for (size_t j = 0U; j < _f1->features().size(); j++) {
    if (_mask(_idx, j) < _maxMask) {
      drawFeature(mat1, _f1->features()[j], std::to_string(_mask(_idx, j)));
    }
  }
  cv::Mat mat;
  cv::hconcat(std::vector<cv::Mat>({mat0, mat1}), mat);
  return mat;
}
void MatchCandidates::drawFeature(cv::Mat &mat, Feature2D::ConstShPtr ft, const std::string &annotation, double radius) const {
  cv::Point center(ft->position().x(), ft->position().y());
  cv::putText(mat, annotation, center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

  cv::rectangle(mat, cv::Rect(center - cv::Point(radius, radius), center + cv::Point(radius, radius)), cv::Scalar(0, 0, 255), 2);
}

cv::Mat CorrespondingPoints::draw() const {
  std::vector<cv::Mat> mats;
  for (const auto &f : _frames) {
    cv::Mat mat;
    cv::cvtColor(f->intensity(), mat, cv::COLOR_GRAY2BGR);
    cv::rectangle(mat, cv::Point(0, 0), cv::Point(20, 20), cv::Scalar(0, 0, 0), -1);
    cv::putText(mat, std::to_string(f->id()), cv::Point(5, 12), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255));
    mats.push_back(mat);
  }

  std::set<uint64_t> points;
  for (size_t i = 0U; i < _frames.size(); i++) {
    for (auto ftRef : _frames[i]->featuresWithPoints()) {
      auto p = ftRef->point();
      if (points.find(p->id()) != points.end()) {
        continue;
      }
      points.insert(p->id());
      cv::Scalar color((double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255);
      for (size_t j = 0U; j < _frames.size(); j++) {
        auto ft = _frames[j]->observationOf(p->id());
        if (ft) {
          cv::Point center(ft->position().x(), ft->position().y());
          const double radius = 2;
          cv::circle(mats[j], center, radius, color, 2);
          if (_legend) {
            std::stringstream ss;
            ss << ft->point()->id();
            cv::putText(mats[j], ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255));
          }
        }
      }
    }
  }
  return arrangeInGrid(mats, _rows, _cols, _h, _w);
}

FeatureDisplacement::FeatureDisplacement(
  const Frame::VecConstShPtr &frames, const Point3D::VecConstShPtr &points, unsigned int maxWidth, unsigned int nRows) :
    _frames(frames),
    _points(points),
    _maxWidth(maxWidth),
    _nRows(nRows) {
  std::sort(_frames.begin(), _frames.end(), [](auto f0, auto f1) { return f0->t() < f1->t(); });
}
FeatureDisplacement::FeatureDisplacement(const Frame::VecConstShPtr &frames, unsigned int maxWidth, unsigned int nRows) :
    _frames(frames),
    _maxWidth(maxWidth),
    _nRows(nRows) {
  std::sort(_frames.begin(), _frames.end(), [](auto f0, auto f1) { return f0->t() < f1->t(); });

  for (auto f : frames) {
    for (auto ft : f->featuresWithPoints()) {
      if (std::find(_points.begin(), _points.end(), ft->point()) == _points.end()) {
        _points.push_back(ft->point());
      }
    }
  }
}

cv::Mat Matches::draw() const {
  cv::Mat mat0, mat1, mat;
  cv::cvtColor(_f0->intensity(), mat0, cv::COLOR_GRAY2BGR);
  cv::cvtColor(_f1->intensity(), mat1, cv::COLOR_GRAY2BGR);
  const int radius = 7;
  for (size_t i = 0; i < _pts0.size(); i++) {
    if (_mask.at<int>(i) == 0)
      continue;
    const cv::Scalar color(
      (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255);
    cv::circle(mat0, _pts0[i], radius, color, 1);
    cv::circle(mat1, _pts1[i], radius, color, 1);
  }
  cv::hconcat(mat0, mat1, mat);
  return mat;
}
cv::Mat FeatureDisplacement::draw() const {
  std::vector<cv::Mat> mats;
  for (const auto &f : _frames) {
    cv::Mat mat;
    cv::cvtColor(f->intensity(), mat, cv::COLOR_GRAY2BGR);
    cv::rectangle(mat, cv::Point(0, 0), cv::Point(40, 20), cv::Scalar(0, 0, 0), -1);

    mats.push_back(mat);
  }
  const double radius = 5;
  std::map<uint64_t, double> reprojectionError;
  std::map<uint64_t, int> nPoints;

  for (auto p : _points) {
    auto colorIt = _colorMap.find(p->id());
    if (colorIt == _colorMap.end()) {
      _colorMap[p->id()] =
        cv::Scalar((double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255);
    }
    const cv::Scalar &color = _colorMap[p->id()];
    for (size_t i = 0U; i < _frames.size(); i++) {
      auto f = _frames[i];
      auto pInF = f->observationOf(p->id());

      if (pInF) {
        auto mat = mats[i];

        const Vec2d reprojection = f->world2image(p->position());
        if (reprojection.allFinite()) {
          const double error = (pInF->position() - reprojection).norm();
          cv::circle(mat, cv::Point(reprojection.x(), reprojection.y()), radius, color, error > 10 ? 3 : 1);
          cv::rectangle(
            mat,
            cv::Point(pInF->position().x() - radius, pInF->position().y() - radius),
            cv::Point(pInF->position().x() + radius, pInF->position().y() + radius),
            color,
            error > 10 ? 3 : 1);
          reprojectionError[f->id()] += error;
          nPoints[f->id()]++;
        }
      }
    }
  }
  const int nMatsPerRow = std::ceil(static_cast<double>(mats.size()) / static_cast<double>(_nRows));
  const double s = std::min(1.0, (static_cast<double>(_maxWidth) / static_cast<double>(nMatsPerRow)) / static_cast<double>(mats[0].cols));
  int i = 0U;
  const Timestamp t0 = (*_frames.begin())->t();
  for (auto f : _frames) {
    auto &mat = mats[i++];
    cv::putText(
      mat,
      fmt::format(
        "#{0:d}, {1:.3f}, #Points: {2:d}, err: {3:.3f}",
        f->id(),
        static_cast<double>(f->t() - t0) / static_cast<double>(1e9),
        nPoints[f->id()],
        reprojectionError[f->id()]),
      cv::Point(5, 12),
      cv::FONT_HERSHEY_COMPLEX,
      0.5,
      cv::Scalar(255, 255, 255));
    cv::resize(mat, mat, cv::Size(0, 0), s, s, cv::INTER_AREA);
  }
  std::vector<cv::Mat> rows;
  for (unsigned int r = 0; r < _nRows; r++) {
    std::vector<cv::Mat> matsInRow;
    for (int c = 0; c < nMatsPerRow; c++) {
      if (r * nMatsPerRow + c >= mats.size()) {
        matsInRow.push_back(cv::Mat(mats[0].size(), mats[0].type(), cv::Scalar(0, 0, 0)));
      } else {
        matsInRow.push_back(mats[r * nMatsPerRow + c]);
      }
    }
    cv::Mat row;
    cv::hconcat(matsInRow, row);
    rows.push_back(row);
  }
  cv::Mat mat;
  cv::vconcat(rows, mat);
  return mat;
}
}  // namespace vslam::overlay