#ifndef VSLAM_SOLVER_H__
#define VSLAM_SOLVER_H__

#include "NormalEquations.h"
namespace pd::vslam::least_squares {

  class Problem {
public:
    typedef std::shared_ptr < Problem > ShPtr;
    typedef std::unique_ptr < Problem > UnPtr;
    typedef std::shared_ptr < const Problem > ConstShPtr;
    typedef std::unique_ptr < const Problem > ConstUnPtr;

    size_t nParameters() const {return _nParameters;}
    Problem(size_t nParameters) : _nParameters(nParameters) {
    }

    virtual void updateX(const Eigen::VectorXd & dx) = 0;
    virtual void setX(const Eigen::VectorXd & x) = 0;
    virtual Eigen::VectorXd x() const = 0;
    virtual NormalEquations::ConstShPtr computeNormalEquations() = 0;

private:
    size_t _nParameters;
  };


  class Solver {
public:
    typedef std::shared_ptr < Solver > ShPtr;
    typedef std::unique_ptr < Solver > UnPtr;
    typedef std::shared_ptr < const Solver > ConstShPtr;
    typedef std::unique_ptr < const Solver > ConstUnPtr;
    struct Results
    {
      typedef std::shared_ptr < Results > ShPtr;
      typedef std::unique_ptr < Results > UnPtr;
      typedef std::shared_ptr < const Results > ConstShPtr;
      typedef std::unique_ptr < const Results > ConstUnPtr;

      Eigen::VectorXd chi2, stepSize;
      Eigen::MatrixXd x;
      std::vector < NormalEquations::ConstShPtr > normalEquations;
      size_t iteration;
    };

    virtual Results::ConstUnPtr solve(std::shared_ptr < Problem > problem) = 0;
  };
}
#endif
