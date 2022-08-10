#ifndef VSLAM_LEAST_SQUARES_NORMAL_EQUATIONS_H__
#define VSLAM_LEAST_SQUARES_NORMAL_EQUATIONS_H__
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "core/core.h"

namespace pd::vslam::least_squares {

  class NormalEquations {
public:
    typedef std::shared_ptr < NormalEquations > ShPtr;
    typedef std::unique_ptr < NormalEquations > UnPtr;
    typedef std::shared_ptr < const NormalEquations > ConstShPtr;
    typedef std::unique_ptr < const NormalEquations > ConstUnPtr;

    NormalEquations(size_t nParameters);
    NormalEquations(const std::vector < NormalEquations > &normalEquations);
    NormalEquations(const std::vector < NormalEquations::ConstShPtr > &normalEquations);
    NormalEquations(
      const Eigen::MatrixXd & J, const Eigen::VectorXd & r,
      const Eigen::VectorXd & w);

    void addConstraint(const Eigen::VectorXd & J, double r, double w);
    void combine(const NormalEquations & that);

    std::string toString() const;

    Eigen::MatrixXd & A() {return _A;}
    Eigen::VectorXd & b() {return _b;}
    double & chi2() {return _chi2;}
    size_t & nConstraints() {return _nConstraints;}

    const Eigen::MatrixXd & A()  const {return _A;}
    const Eigen::VectorXd & b()  const {return _b;}
    const double & chi2()  const {return _chi2;}
    const size_t & nConstraints() const {return _nConstraints;}

private:
    Eigen::MatrixXd _A;
    Eigen::VectorXd _b;
    double _chi2;
    size_t _nConstraints;

  };

}
#endif
