#ifndef VSLAM_SOLVER_PRIOR
#define VSLAM_SOLVER_PRIOR
#include <core/core.h>
#include "GaussNewton.h"
namespace pd::vslam::least_squares {
  class Prior {
public:
    typedef std::shared_ptr < Prior > ShPtr;
    typedef std::unique_ptr < Prior > UnPtr;
    typedef std::shared_ptr < const Prior > ConstShPtr;
    typedef std::unique_ptr < const Prior > ConstUnPtr;

    virtual void apply(typename NormalEquations::ShPtr ne, const Eigen::VectorXd & x) const = 0;

  };

} // namespace pd::vslam::least_squares

#endif
