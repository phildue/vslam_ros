#pragma once

#include <map>
#include <vector>

#include "core/Frame.h"
#include "core/types.h"
namespace vslam::overlay {
class DepthCorrespondence {
public:
  DepthCorrespondence(
    const std::vector<Vec2d> &uv0,
    Frame::ConstShPtr f0,
    Frame::ConstShPtr f1,
    const SE3d &pose,
    const std::vector<double> &z0,
    int level,
    int patchSize,
    bool drawEpipolarLine = true);
  DepthCorrespondence(
    const std::vector<Feature2D::ConstShPtr> &uv0,
    Frame::ConstShPtr f0,
    Frame::ConstShPtr f1,
    const SE3d &pose,
    const std::vector<double> &z0,
    int level,
    int patchSize,
    bool drawEpipolarLine = true);

  cv::Mat operator()() const;

  void drawCorrespondence(cv::Mat &img0, cv::Mat &img1, const Vec2d &uv0, double z, const cv::Scalar &color) const;
  void drawEpipolarLine(cv::Mat &img1, const Vec2d &uv0, double z, const cv::Scalar &color) const;

  cv::Mat drawPatches(const Vec2d &uv0, double z) const;
  void drawResidual(cv::Mat &r, const Vec2d &uv0, double z, int i) const;

  MatXd interpolatePatch1For(const Vec2d &uv0, double z) const;

  MatXd extractPatch0(const Vec2d &uv0) const;

  cv::Mat computeLossMap(const Vec2d &uv0) const;

  double interpolate(const cv::Mat &intensity, const Vec2d &uv) const;

private:
  const bool _drawEpipolarLine;
  const double _zd = 0.25;
  std::vector<Vec2d> _uv0;
  const Frame::ConstShPtr _f0, _f1;
  const SE3d _pose01;
  const int _level, _radius, _patchSize;
  const std::vector<double> _z0;
  static std::map<size_t, cv::Scalar> colorMap;
};
}  // namespace vslam::overlay