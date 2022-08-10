//
// Created by phil on 07.08.21.
//

#ifndef VSLAM_VISUALS_H
#define VSLAM_VISUALS_H

#include <memory>

#include <opencv2/core/mat.hpp>
#include <matplotlibcpp.h>

#include "core/core.h"

namespace pd::vslam {
  class Frame;
  class FrameRGBD;

  namespace vis {
    namespace plt = matplotlibcpp;

    cv::Mat drawAsImage(const Eigen::MatrixXd & mat);
    //static void logJacobianImage(int iteration, const Eigen::MatrixXd& jacobian);

    cv::Mat drawMat(const Image & mat);

    class Drawable {
public:
      typedef std::unique_ptr < Drawable > Ptr;
      typedef std::unique_ptr < const Drawable > ConstPtr;
      typedef std::shared_ptr < Drawable > ShPtr;
      typedef std::shared_ptr < const Drawable > ConstShPtr;

      virtual cv::Mat draw() const = 0;
    };
    template < typename T >
    class DrawableMat: public Drawable {
public:
      DrawableMat(const Eigen::Matrix < T, -1, -1 > &mat) : _mat(mat) {
      }
      cv::Mat draw() const override {return drawAsImage(_mat.template cast < double > ());}

private:
      const Eigen::Matrix < T, -1, -1 > _mat;
    };

    class Plot {
public:
      typedef std::unique_ptr < Plot > Ptr;
      typedef std::unique_ptr < const Plot > ConstPtr;
      typedef std::shared_ptr < Plot > ShPtr;
      typedef std::shared_ptr < const Plot > ConstShPtr;


      virtual void plot() const = 0;
      virtual std::string csv() const = 0;
    };

    class Histogram: public Plot
    {
public:
      Histogram(
        const Eigen::VectorXd & h, const std::string & title = "Histogram",
        int nBins = 10) : _h(h), _title(title), _nBins(nBins) {
      };
      const Eigen::VectorXd _h;
      const std::string _title;
      const int _nBins;
      void plot() const override;
      std::string csv() const override {return "";}
    };


    class PlotLevenbergMarquardt: public Plot {

public:
      PlotLevenbergMarquardt(
        int nIterations, const Eigen::VectorXd & chi2,
        const Eigen::VectorXd & dChi2, const Eigen::VectorXd & chi2pred,
        const Eigen::VectorXd & lambda, const Eigen::VectorXd & stepSize,
        const Eigen::VectorXd & rho)
        : _chi2(chi2),
        _chi2pred(chi2pred),
        _lambda(lambda),
        _stepSize(stepSize),
        _nIterations(nIterations),
        _dChi2(dChi2),
        _rho(rho)
      {
      }

      void plot() const override;
      std::string csv() const override;

private:
      const Eigen::VectorXd _chi2;
      const Eigen::VectorXd _chi2pred;
      const Eigen::VectorXd _lambda;
      const Eigen::VectorXd _stepSize;
      const int _nIterations;
      const Eigen::VectorXd _dChi2;
      const Eigen::VectorXd _rho;


    };

    class PlotGaussNewton: public Plot {

public:
      PlotGaussNewton(
        int nIterations, const Eigen::VectorXd & chi2,
        const Eigen::VectorXd & stepSize)
        : _chi2(chi2),
        _stepSize(stepSize),
        _nIterations(nIterations)
      {
      }

      void plot() const override;
      std::string csv() const override;

private:
      const Eigen::VectorXd _chi2;
      const Eigen::VectorXd _stepSize;
      const int _nIterations;

    };

  }
}

#endif //VSLAM_LOG_H
