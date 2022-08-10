#include <utils/utils.h>
#include "Scaler.h"
namespace pd::vslam::least_squares
{

Scaler::Scale MedianScaler::compute(const VecXd & r) const
{
  /*
  std::vector<double> rs();
  rs.reserve(r.rows());
  for (int i = 0; i < r.rows(); i++)
  {
      algorithm::insertionSort(rs,r(i));
  }*/

  auto median = algorithm::median(r, false);
  auto std = std::sqrt((r.array() - median).array().abs().sum() / (r.rows() - 1));
  LOG_PLT("MedianScaler") << std::make_shared<vis::Histogram>(r, "ErrorDistribution", 30);
  return {median, std};
}


Scaler::Scale MeanScaler::compute(const VecXd & r) const
{
  if (r.rows() == 0) {
    SOLVER(WARNING) << "Empty residual.";
    return {0.0, 1.0};
  }
  auto mean = r.mean();
  auto std = std::sqrt((r.array() - mean).array().abs().sum() / (r.rows() - 1));
  LOG_PLT("MedianScaler") << std::make_shared<vis::Histogram>(r, "ErrorDistribution", 30);
  return {mean, std};

}


Scaler::Scale ScalerTDistribution::compute(const VecXd & r) const
{
  double stepSize = std::numeric_limits<double>::max();
  size_t iter = 0;
  double sigma = 1.0;
  for (;
    iter < _maxIterations && stepSize > _minStepSize;
    iter++)
  {
    double sum = 0.0;
    for (int i = 0; i < r.rows(); i++) {
      sum += r(i) * r(i) * (_v + 1) / (_v + std::pow(r(i) / sigma, 2));
    }
    const double sigma_i = std::sqrt(sum / (double)r.rows());
    stepSize = std::abs(sigma - sigma_i);
    SOLVER(DEBUG) << "Sigma_i: " << iter << " with: " << sigma_i << " stepSize: " << stepSize;

    sigma = sigma_i;
  }
  SOLVER(DEBUG) << "Converged after: " << iter << " with: " << sigma;
  return {0.0, sigma};
}


} // namespace pd::vslam::solver
