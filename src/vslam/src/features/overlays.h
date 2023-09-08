#pragma once
#include "core/Frame.h"
#include "core/types.h"
namespace vslam::overlay {

class Features {
public:
  Features(Frame::ConstShPtr frame, double radius = 1, double cellSize = 0, bool annotate = false) :
      _frame(frame),
      _radius(radius),
      _gridCellSize(cellSize),
      _annotate(annotate) {}
  cv::Mat draw() const;
  cv::Mat operator()() const { return draw(); }

private:
  const Frame::ConstShPtr _frame;
  const double _radius;
  const double _gridCellSize;
  const bool _annotate;
};

class ReprojectedFeatures {
public:
  ReprojectedFeatures(
    Frame::ConstShPtr f0,
    Frame::ConstShPtr f1,
    double radius = 1,
    double cellSize = 0,
    Pose::ConstShPtr relativePose = nullptr,
    bool annotate = false) :
      _f0(f0),
      _f1(f1),
      _radius(radius),
      _gridCellSize(cellSize),
      _relativePose(relativePose),
      _annotate(annotate) {}
  cv::Mat draw() const;
  cv::Mat operator()() const { return draw(); }

private:
  const Frame::ConstShPtr _f0, _f1;
  const double _radius;
  const double _gridCellSize;
  const Pose::ConstShPtr _relativePose;
  const bool _annotate;
};

}  // namespace vslam::overlay
