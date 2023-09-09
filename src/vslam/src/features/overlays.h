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

class CorrespondingPoints {
public:
  CorrespondingPoints(const std::vector<Frame::ConstShPtr> &frames, int rows, int cols, int h = 240, int w = 320, bool legend = false) :
      _frames(frames),
      _rows(rows),
      _cols(cols),
      _h(h),
      _w(w),
      _legend(legend) {}

  cv::Mat draw() const;
  cv::Mat operator()() const { return draw(); }

private:
  const std::vector<Frame::ConstShPtr> _frames;
  const int _rows, _cols, _h, _w;
  const bool _legend;
};

}  // namespace vslam::overlay
