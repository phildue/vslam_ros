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

}  // namespace vslam::overlay