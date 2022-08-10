#include "Loss.h"
#include "core/core.h"
#include "utils/utils.h"

namespace pd::vslam::least_squares
{


double TukeyLoss::compute(double r) const
{
  // If the residuals falls within the 95% the loss is quadratic
  if (std::abs(r) < TukeyLoss::C) {
    const double r_c = r / TukeyLoss::C;

    return C2_6 * (1.0 - std::pow(1.0 - r_c * r_c, 3));

  } else {
    // Outliers are disregarded
    return C2_6;
  }
}

double TukeyLoss::computeDerivative(double r) const
{
  // If the residuals falls within the 95% the loss is quadratic
  if (std::abs(r) < TukeyLoss::C) {
    const double r_c = r / TukeyLoss::C;

    return r * std::pow(1.0 - r_c * r_c, 2);

  } else {
    // Outliers are disregarded
    return 0.0;
  }
}

double TukeyLoss::computeWeight(double r) const
{

  // If the residuals falls within the 95% the loss is quadratic
  if (std::abs(r) < TukeyLoss::C) {
    const double r_c = r / TukeyLoss::C;

    return std::pow(1.0 - r_c * r_c, 2);

  } else {
    // Outliers are disregarded
    return 0.0;
  }
}

double HuberLoss::computeWeight(double r) const
{
  if (std::abs(r) < _c) {
    return 1.0;
  } else {
    return (_c * r > 0.0 ? 1.0 : -1.0) / r;
  }
}
//dl/dr
double HuberLoss::computeDerivative(double r) const
{
  if (std::abs(r) < _c) {
    return r;
  } else {
    return _c * r > 0.0 ? 1.0 : -1.0;
  }
}
//l(r)
double HuberLoss::compute(double r) const
{
  if (std::abs(r) < _c) {
    return 0.5 * r * r;
  } else {
    return _c * std::abs(r) - 0.5 * r * r;
  }
}

double LossTDistribution::computeWeight(double r) const
{

  return (_v + 1.0) / (_v + r * r);
}
double LossTDistribution::computeDerivative(double UNUSED(r)) const
{
  return 0.0;               //TODO
}
double LossTDistribution::compute(double UNUSED(r)) const
{
  return 0.0;               //TODO

}


}
