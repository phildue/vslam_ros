#include "overlays.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "core/Point3D.h"
#include "utils/visuals.h"
namespace vslam::overlay {

cv::Mat Features::draw() const {
  cv::Mat mat = visualizeFrame(_frame);

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
    const double scale = std::pow(2, ft->level());
    cv::Point center(ft->position().x() * scale, ft->position().y() * scale);
    if (ft->point()) {
      if (_annotate) {
        std::stringstream ss;
        ss << ft->point()->id();
        cv::putText(mat, ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
      }

      cv::circle(mat, center, _radius, cv::Scalar(255, 0, 0), 2);
    } else {
      if (_annotate) {
        std::stringstream ss;
        ss << ft->id();
        cv::putText(mat, ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
      }
      cv::rectangle(mat, cv::Rect(center - cv::Point(_radius, _radius), center + cv::Point(_radius, _radius)), cv::Scalar(0, 0, 255), 2);
    }
  }
  return mat;
}

cv::Mat ReprojectedFeatures::draw() const {
  cv::Mat mat = visualizeFrame(_f1);

  if (_gridCellSize > 1) {
    for (size_t r = 0; r < _f1->height(0); r += _gridCellSize) {
      cv::line(mat, cv::Point(0, r), cv::Point(_f1->width(0), r), cv::Scalar(128, 128, 128));
    }
    for (size_t c = 0; c < _f1->width(0); c += _gridCellSize) {
      cv::line(mat, cv::Point(c, 0), cv::Point(c, _f1->height(0)), cv::Scalar(128, 128, 128));
    }
  }

  // std::stringstream ss;
  // ss << format("#{} *{}", _frame->features().size(), _frame->featuresWithPoints().size());
  // cv::putText(mat, ss.str(), cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
  SE3d relativePose = _relativePose ? _relativePose->SE3() : _f1->pose().SE3() * _f0->pose().SE3().inverse();
  for (auto ft : _f0->features()) {
    const Vec2d uv1 = _f1->project(relativePose * ft->frame()->p3d(ft->v(), ft->u()));
    if (!uv1.allFinite() || !_f1->withinImage(uv1, 1.0)) {
      continue;
    }
    cv::Point center(uv1.x(), uv1.y());
    if (ft->point()) {
      if (_annotate) {
        std::stringstream ss;
        ss << ft->point()->id();
        cv::putText(mat, ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
      }

      cv::circle(mat, center, _radius, cv::Scalar(255, 0, 0), 2);
    } else {
      if (_annotate) {
        std::stringstream ss;
        ss << ft->id();
        cv::putText(mat, ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
      }
      cv::rectangle(mat, cv::Rect(center - cv::Point(_radius, _radius), center + cv::Point(_radius, _radius)), cv::Scalar(0, 0, 255), 2);
    }
  }
  return mat;
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

}  // namespace vslam::overlay