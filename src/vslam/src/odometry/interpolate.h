#pragma once
#include "core/types.h"
namespace vslam {

template <typename Sample, typename T> Vec<T, -1> interpolate(const Vec<T, 2> &uv, Sample sample) {
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
  const Vec<T, -1> iz00 = sample(v0, u0);
  const Vec<T, -1> iz01 = sample(v0, u1);
  const Vec<T, -1> iz10 = sample(v1, u0);
  const Vec<T, -1> iz11 = sample(v1, u1);

  const double w00 = iz00.allFinite() ? w_v0 * w_u0 : 0;
  const double w01 = iz00.allFinite() ? w_v0 * w_u1 : 0;
  const double w10 = iz00.allFinite() ? w_v1 * w_u0 : 0;
  const double w11 = iz00.allFinite() ? w_v1 * w_u1 : 0;

  Vec<T, -1> izw = w00 * iz00 + w01 * iz01 + w10 * iz10 + w11 * iz11;
  izw /= w00 + w01 + w10 + w11;
  return izw;
}

}  // namespace vslam