#ifndef VSLAM_SOLVER_SCALER
#define VSLAM_SOLVER_SCALER
#include <core/core.h>

namespace pd::vslam::least_squares {


  class Scaler {
public:
    typedef std::shared_ptr < Scaler > ShPtr;
    typedef std::unique_ptr < Scaler > UnPtr;
    typedef std::shared_ptr < const Scaler > ConstShPtr;
    typedef std::unique_ptr < const Scaler > ConstUnPtr;

    struct Scale
    {
      double offset;
      double scale;
    };

    virtual Scale compute(const VecXd & UNUSED(r)) const {return {0.0, 1.0};}

  };

  class MedianScaler: public Scaler {
public:
    Scale compute(const VecXd & r) const override;

private:
    double _median = 0.0;
    double _std = 1.0;
  };
  class MeanScaler: public Scaler {
public:
    Scale compute(const VecXd & r) const override;

private:
    double _mean = 0.0;
    double _std = 1.0;
  };

  class ScalerTDistribution: public Scaler {
public:
    ScalerTDistribution(double v = 5.0, uint64_t maxIterations = 30, double minStepSize = 1e-5)
      : _v(v), _maxIterations(maxIterations), _minStepSize(minStepSize) {
    }

    Scale compute(const VecXd & r) const override;

private:
    const double _v = 5.0;     //Experimentally, Robust Odometry Estimation From Rgbd Cameras
    double _sigma = 1.0;
    double _sigma2 = 1.0;
    uint64_t _maxIterations = 100;
    double _minStepSize = 1e-5;
  };
} // namespace pd::vslam::least_squares

#endif
