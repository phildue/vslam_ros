#pragma once
#include "core/types.h"
namespace vslam {

struct NormalEquations {
  Mat6f A;
  Vec6f b;
  float error;
  int nConstraints;
  NormalEquations operator+(const NormalEquations &that) const {
    return NormalEquations({A + that.A, b + that.b, error + that.error, nConstraints + that.nConstraints});
  }
  void operator+=(const NormalEquations &that) {
    A += that.A;
    b += that.b;
    error += that.error;
    nConstraints += that.nConstraints;
  }
};
}  // namespace vslam
