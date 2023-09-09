#include "NormalEquations.h"
namespace vslam {

void NormalEquations::operator+=(const NormalEquations &that) {
  A.noalias() += that.A;
  b.noalias() += that.b;
  error += that.error;
  nConstraints += that.nConstraints;
}
NormalEquations operator+(const NormalEquations &ne0, const NormalEquations &ne1) {
  return {ne0.A + ne1.A, ne0.b + ne1.b, ne0.error + ne1.error, ne0.nConstraints + ne1.nConstraints};
}
}  // namespace vslam
