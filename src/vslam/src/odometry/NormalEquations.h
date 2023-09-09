#pragma once
#include "core/types.h"
namespace vslam {

struct NormalEquations {
  Mat6f A = Mat6f::Zero();
  Vec6f b = Vec6f::Zero();
  float error = 0.0f;
  int nConstraints = 0;
  void operator+=(const NormalEquations &that);
};
NormalEquations operator+(const NormalEquations &ne0, const NormalEquations &ne1);
}  // namespace vslam
