#pragma once
#include "core/types.h"
namespace vslam {

template <typename NumType> float interpolate(const cv::Mat &intensity, const Vec2f &uv) {
  auto sample = [&](int v, int u) -> float { return (float)(intensity.at<NumType>(v, u)); };
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
  const float iz00 = sample(v0, u0);
  const float iz01 = sample(v0, u1);
  const float iz10 = sample(v1, u0);
  const float iz11 = sample(v1, u1);

  const double w00 = w_v0 * w_u0;
  const double w01 = w_v0 * w_u1;
  const double w10 = w_v1 * w_u0;
  const double w11 = w_v1 * w_u1;

  float izw = w00 * iz00 + w01 * iz01 + w10 * iz10 + w11 * iz11;
  izw /= w00 + w01 + w10 + w11;
  return izw;
}

template <typename NumType1, typename NumType2> Vec2f interpolate(const cv::Mat &intensity, const cv::Mat &depth, const Vec2f &uv) {
  auto sample = [&](int v, int u) -> Vec2f {
    const double z = depth.at<NumType2>(v, u);
    return Vec2f(intensity.at<NumType1>(v, u), std::isfinite(z) ? z : 0);
  };
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
  const Vec2f iz00 = sample(v0, u0);
  const Vec2f iz01 = sample(v0, u1);
  const Vec2f iz10 = sample(v1, u0);
  const Vec2f iz11 = sample(v1, u1);

  const double w00 = iz00(1) > 0 ? w_v0 * w_u0 : 0;
  const double w01 = iz01(1) > 0 ? w_v0 * w_u1 : 0;
  const double w10 = iz10(1) > 0 ? w_v1 * w_u0 : 0;
  const double w11 = iz11(1) > 0 ? w_v1 * w_u1 : 0;

  Vec2f izw = w00 * iz00 + w01 * iz01 + w10 * iz10 + w11 * iz11;
  izw /= w00 + w01 + w10 + w11;
  return izw;
}
}  // namespace vslam