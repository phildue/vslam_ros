#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/random.h"
#include "overlay.h"
#include "utils/log.h"
#include "utils/visuals.h"

namespace vslam::overlay {
std::map<size_t, cv::Scalar> DepthCorrespondence::colorMap = {};
DepthCorrespondence::DepthCorrespondence(
  const std::vector<Vec2d> &uv0,
  Frame::ConstShPtr f0,
  Frame::ConstShPtr f1,
  const SE3d &pose,
  const std::vector<double> &z0,
  int level,
  int patchSize,
  bool drawLine) :
    _drawEpipolarLine(drawLine),
    _uv0(uv0),
    _f0(f0),
    _f1(f1),
    _pose01(pose),
    _level(level),
    _radius(std::floor(patchSize / 2.0)),
    _patchSize(patchSize),
    _z0(z0) {}

DepthCorrespondence::DepthCorrespondence(
  const std::vector<Feature2D::ConstShPtr> &uv0,
  Frame::ConstShPtr f0,
  Frame::ConstShPtr f1,
  const SE3d &pose,
  const std::vector<double> &z0,
  int level,
  int patchSize,
  bool drawLine) :
    _drawEpipolarLine(drawLine),
    _uv0(uv0.size()),
    _f0(f0),
    _f1(f1),
    _pose01(pose),
    _level(level),
    _radius(std::floor(patchSize / 2.0)),
    _patchSize(patchSize),
    _z0(z0) {
  std::transform(uv0.begin(), uv0.end(), _uv0.begin(), [](auto ft) { return ft->position(); });
}

cv::Mat DepthCorrespondence::operator()() const {
  cv::Mat img0 = visualizeFrame(_f0, _level);
  cv::Mat img1 = visualizeFrame(_f1, _level);
  cv::Mat residual(img0.rows, img0.cols, CV_8UC1, cv::Scalar(0));
  std::vector<cv::Mat> patches;
  for (size_t i = 0; i < _uv0.size(); i++) {
    if (DepthCorrespondence::colorMap.find(i) == DepthCorrespondence::colorMap.end()) {
      DepthCorrespondence::colorMap[i] =
        cv::Scalar((double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255);
    }
    const double s = 1.0 / std::pow(2.0, _level);
    const Vec3d p0t = _pose01 * _f0->reconstruct(_uv0.at(i) * s, _z0.at(i), _level);
    const Vec2d uv1 = _f1->project(p0t, _level);

    if (p0t.z() <= 0 || !_f1->withinImage(uv1, 7.0, _level)) {
      continue;
    }
    const auto &color = DepthCorrespondence::colorMap[i];
    if (_drawEpipolarLine) {
      drawEpipolarLine(img1, _uv0[i] * s, _z0.at(i), color);
    }
    drawCorrespondence(img0, img1, _uv0[i] * s, _z0.at(i), color);
    // patches.push_back(drawPatches(_uv0[i]*s, _z0.at(i)));
    drawResidual(residual, _uv0[i] * s, _z0.at(i), i);
  }
  cv::resize(img0, img0, cv::Size(640, 480));
  cv::resize(img1, img1, cv::Size(640, 480));
  const double rnorm = cv::sum(residual)[0];
  cv::resize(residual, residual, cv::Size(640, 480));
  cv::cvtColor(residual, residual, cv::COLOR_GRAY2BGR);
  cv::putText(residual, format("|r|={:.3f}", rnorm), cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
  cv::Mat patch;
  // cv::vconcat(patches, patch);
  // cv::imshow("patches", patch);

  cv::Mat mat;
  cv::hconcat(std::vector<cv::Mat>({img0, img1, residual}), mat);

  return mat;
}

void DepthCorrespondence::drawCorrespondence(cv::Mat &img0, cv::Mat &img1, const Vec2d &uv0, double z, const cv::Scalar &color) const {
  cv::rectangle(img0, cv::Point(uv0(0) - _radius, uv0(1) - _radius), cv::Point(uv0(0) + _radius, uv0(1) + _radius), color, 1);
  const Vec3d p0t = _pose01 * _f0->reconstruct(uv0, z, _level);
  const Vec2d uv1 = _f1->project(p0t, _level);
  // LOG(INFO) << format(
  //   "uv0: {} z0: {:.3f} --> uv1: {} z1 {:.3f} pose1: {}", uv0.transpose(), z, uv1.transpose(), p0t.z(), _pose.log().transpose());
  if (uv1.allFinite()) {
    cv::rectangle(img1, cv::Point(uv1(0) - _radius, uv1(1) - _radius), cv::Point(uv1(0) + _radius, uv1(1) + _radius), color, 1);
  }
}

void DepthCorrespondence::drawEpipolarLine(cv::Mat &img1, const Vec2d &uv0, double z, const cv::Scalar &color) const {
  const Vec2d uv10 = _f1->project(_pose01 * _f0->reconstruct(uv0, z - _zd, _level), _level);

  const Vec2d uv11 = _f1->project(_pose01 * _f0->reconstruct(uv0, z + _zd, _level), _level);
  if (uv10.allFinite() && uv11.allFinite()) {
    cv::line(img1, cv::Point(uv10(0), uv10(1)), cv::Point(uv11(0), uv11(1)), color);
  }
}

cv::Mat DepthCorrespondence::drawPatches(const Vec2d &uv0, double z) const {
  cv::Mat img0 = _f0->intensity(_level);
  cv::Mat img1 = _f1->intensity(_level);
  cv::Mat patch0(_patchSize, _patchSize, CV_8UC1), patch1(_patchSize, _patchSize, CV_8UC1), residual(_patchSize, _patchSize, CV_8UC1);
  for (int r = 0; r < _patchSize; r++) {
    for (int c = 0; c < _patchSize; c++) {
      const int u0 = uv0.x() - std::floor(_patchSize / 2.0) + c;
      const int v0 = uv0.y() - std::floor(_patchSize / 2.0) + r;
      const Vec2d uv1 = _f1->project(_pose01 * _f0->reconstruct(Vec2d(u0, v0), z, _level), _level);

      if (!uv1.allFinite())
        continue;

      patch0.at<uint8_t>(r, c) = img0.at<uint8_t>(v0, u0);
      patch1.at<uint8_t>(r, c) = static_cast<uint8_t>(interpolate(img1, uv1));
      residual.at<uint8_t>(r, c) = std::abs(patch0.at<uint8_t>(r, c) - patch1.at<uint8_t>(r, c));
    }
  }
  const double r = cv::norm(residual);
  cv::resize(patch0, patch0, cv::Size(300, 300));
  cv::resize(patch1, patch1, cv::Size(300, 300));
  cv::resize(residual, residual, cv::Size(300, 300));
  cv::putText(residual, format("r={:.2f}", r), cv::Point(40, 40), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

  cv::Mat mat;
  cv::hconcat(std::vector<cv::Mat>({patch0, patch1, residual}), mat);
  return mat;
}

void DepthCorrespondence::drawResidual(cv::Mat &r, const Vec2d &uv0, double z0, int i) const {
  cv::Mat img0 = _f0->intensity(_level);
  cv::Mat img1 = _f1->intensity(_level);
  const Vec3d p0t = _pose01 * _f0->reconstruct(uv0, z0, _level);
  const Vec2d uv1 = _f1->project(p0t, _level);

  if (p0t.z() <= 0 || !_f1->withinImage(uv1, 7.0, _level)) {
    r.at<uint8_t>(uv0(1), uv0(0)) = 255;
  } else {
    r.at<uint8_t>(uv0(1), uv0(0)) =
      static_cast<uint8_t>(std::abs(img0.at<uint8_t>(uv0(1), uv0(0)) - static_cast<uint8_t>(interpolate(img1, uv1))));
  }

  LOG_IF(false, DEBUG) << "me  : uv0: " << uv0.transpose() << " z0: " << z0 << " i0: " << (double)img0.at<uint8_t>(uv0(1), uv0(0))
                       << " --> uv1: " << uv1.transpose() << " z1: " << p0t.z() << " i1: " << (interpolate(img1, uv1))
                       << " r=" << r.at<uint8_t>(uv0(1), uv0(0)) << " pose1: " << _pose01.log().transpose();
}

MatXd DepthCorrespondence::interpolatePatch1For(const Vec2d &uv0, double z) const {
  MatXd patch;
  for (int r = 0; r < _patchSize; r++) {
    for (int c = 0; c < _patchSize; c++) {
      const int u0 = uv0.x() - std::floor(_patchSize / 2.0) + c;
      const int v0 = uv0.y() - std::floor(_patchSize / 2.0) + r;
      // const Vec2d uv1 =
      //   _f1->project(_pose * _f0->reconstruct(Vec2d(u0, v0), 1.0 / (*_z0), _level), _level);
      // patch(r, c) = static_cast<T>(interpolate(_f1->intensity(_level), uv1));
    }
  }
  return patch;
}

MatXd DepthCorrespondence::extractPatch0(const Vec2d &uv0) const {
  MatXd patch;
  for (int r = 0; r < _patchSize; r++) {
    for (int c = 0; c < _patchSize; c++) {
      const int u0 = uv0.x() - std::floor(_patchSize / 2.0) + c;
      const int v0 = uv0.y() - std::floor(_patchSize / 2.0) + r;
      patch(r, c) = _f0->intensity(_level).at<uint8_t>(v0, u0);
    }
  }
  return patch;
}

cv::Mat DepthCorrespondence::computeLossMap(const Vec2d &uv0) const {
  MatXd lossMap;
  MatXd patch0 = extractPatch0(uv0);
  for (int r = 0; r < _patchSize; r++) {
    for (int c = 0; c < _patchSize; c++) {
      const int u0 = uv0.x() - std::floor(_patchSize / 2.0) + c;
      const int v0 = uv0.y() - std::floor(_patchSize / 2.0) + r;
      // const MatXd patch1 =
      // interpolatePatch1For(Vec2d(u0, v0));
      // lossMap(r, c) = (patch0.cast<double>() - patch1.cast<double>()).norm();
    }
  }
  lossMap /= lossMap.maxCoeff();
  lossMap *= 255.0;
  cv::Mat lossMap_;
  // cv::eigen2cv(lossMap.cast<uint8_t>(), lossMap_);
  // cv::resize(lossMap_, lossMap_, cv::Size(100, 100));
  return lossMap_;
}

double DepthCorrespondence::interpolate(const cv::Mat &intensity, const Vec2d &uv) const {
  const double u = uv(0);
  const double v = uv(1);
  const double u0 = std::floor(u);
  const double u1 = std::ceil(u);
  const double v0 = std::floor(v);
  const double v1 = std::ceil(v);
  const double w_u1 = u - u0;
  const double w_u0 = 1.0 - w_u1;
  const double w_v1 = v - v0;
  const double w_v0 = 1.0 - w_v1;
  const double iz00 = intensity.at<uint8_t>(v0, u0);
  const double iz01 = intensity.at<uint8_t>(v0, u1);
  const double iz10 = intensity.at<uint8_t>(v1, u0);
  const double iz11 = intensity.at<uint8_t>(v1, u1);

  const double w00 = w_v0 * w_u0;
  const double w01 = w_v0 * w_u1;
  const double w10 = w_v1 * w_u0;
  const double w11 = w_v1 * w_u1;

  double iw = w00 * iz00 + w01 * iz01 + w10 * iz10 + w11 * iz11;
  iw /= w00 + w01 + w10 + w11;
  return iw;
}

}  // namespace vslam::overlay