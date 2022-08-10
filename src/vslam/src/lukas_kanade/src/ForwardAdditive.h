#ifndef VSLAM_LUKAS_KANADE_H__
#define VSLAM_LUKAS_KANADE_H__
#include <memory>

#include "core/core.h"
#include "least_squares/least_squares.h"

#include "Warp.h"
namespace pd::vslam::lukas_kanade {

  class ForwardAdditive: public least_squares::Problem {
public:
    ForwardAdditive(
      const Image & templ, const MatXd & dX, const MatXd & dY, const Image & image,
      std::shared_ptr < Warp > w0,
      least_squares::Loss::ShPtr = std::make_shared < least_squares::QuadraticLoss > (),
      double minGradient = 0,
      std::shared_ptr < const least_squares::Prior > prior = nullptr);
    const std::shared_ptr < const Warp > warp();

    void updateX(const Eigen::VectorXd & dx) override;
    void setX(const Eigen::VectorXd & x) override {_w->setX(x);}

    Eigen::VectorXd x() const override {return _w->x();}
    least_squares::NormalEquations::ConstShPtr computeNormalEquations() override;

protected:
    const Image _T;
    const Image _Iref;
    Eigen::MatrixXd _dIdx;
    Eigen::MatrixXd _dIdy;
    const std::shared_ptr < Warp > _w;
    const std::shared_ptr < least_squares::Loss > _loss;
    const double _minGradient;
    const std::shared_ptr < const least_squares::Prior > _prior;

    std::vector < Eigen::Vector2i > _interestPoints;
  };

}
#endif
