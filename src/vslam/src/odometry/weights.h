#pragma once
#include "core/types.h"
namespace vslam {

template <typename Constraint> class TDistribution {
public:
  typedef std::shared_ptr<TDistribution<Constraint>> ShPtr;
  TDistribution(double dof, double precision = 1e-3, int maxIterations = 50) :
      _dof(dof),
      _precision(precision),
      _maxIterations(maxIterations) {}

  void computeWeights(const std::vector<std::shared_ptr<Constraint>> &features) {
    VecXf weights = VecXf::Ones(features.size());
    std::vector<float> rrT(features.size());
    for (size_t n = 0; n < features.size(); n++) {
      rrT[n] = features[n]->residual * features[n]->residual;
    }

    for (int i = 0; i < _maxIterations; i++) {
      std::vector<float> wrrT(features.size());
      for (size_t n = 0; n < features.size(); n++) {
        wrrT[n] = weights(n) * rrT[n];
      }
      float sum = std::accumulate(rrT.begin(), rrT.end(), 0.f);

      const float scale_i = 1.f / (sum / features.size());

      const double diff = std::abs(_scale - scale_i);
      _scale = scale_i;
      for (size_t n = 0; n < features.size(); n++) {
        weights(n) = computeWeight(features[n]->residual);
        features[n]->weight = weights(n) * _scale;
      }

      if (diff < _precision) {
        break;
      }
    }
  }
  double computeWeight(float r) const { return (_dof + 1.0) / (_dof + r * _scale * r); }
  const float &scale() const { return _scale; };

private:
  const double _dof, _precision;
  const int _maxIterations;
  float _scale;
};

template <typename Constraint> class TDistributionBivariate {
public:
  typedef std::shared_ptr<TDistributionBivariate<Constraint>> ShPtr;
  TDistributionBivariate(double dof, double precision = 1e-3, int maxIterations = 50) :
      _dof(dof),
      _precision(precision),
      _maxIterations(maxIterations) {}
  void computeWeights(const std::vector<std::shared_ptr<Constraint>> &features) {
    VecXf weights = VecXf::Ones(features.size());
    std::vector<Mat2f> rrT(features.size());
    for (size_t n = 0; n < features.size(); n++) {
      rrT[n] = features[n]->residual * features[n]->residual.transpose();
    }

    for (int i = 0; i < _maxIterations; i++) {
      std::vector<Mat2f> wrrT(features.size());
      for (size_t n = 0; n < features.size(); n++) {
        wrrT[n] = weights(n) * rrT[n];
      }
      Mat2f sum = std::accumulate(rrT.begin(), rrT.end(), Mat2f::Zero().eval());

      const Mat2f scale_i = (sum / features.size()).inverse();

      const double diff = (_scale - scale_i).norm();
      _scale = scale_i;
      for (size_t n = 0; n < features.size(); n++) {
        weights(n) = computeWeight(features[n]->residual);
        features[n]->weight = weights(n) * _scale;
      }

      if (diff < _precision) {
        break;
      }
    }
  }
  double computeWeight(const Vec2f &r) const { return (_dof + 2.0) / (_dof + r.transpose() * _scale * r); }
  const Mat2f &scale() const { return _scale; };

private:
  const double _dof, _precision;
  const int _maxIterations;
  Mat2f _scale;
};

}  // namespace vslam