#ifndef VSLAM_DIRECT_ICP_H__
#define VSLAM_DIRECT_ICP_H__
#include <map>
#include <memory>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "NormalEquations.h"
#include "core/Camera.h"
#include "core/Frame.h"
#include "core/Pose.h"
#include "core/types.h"
#include "weights.h"
namespace vslam {
class AlignmentRgbd {
public:
  typedef std::shared_ptr<AlignmentRgbd> ShPtr;

  struct Constraint {
    typedef std::shared_ptr<Constraint> ShPtr;
    typedef std::shared_ptr<Constraint> ConstShPtr;
    typedef std::vector<ShPtr> VecShPtr;
    typedef std::vector<ConstShPtr> VecConstShPtr;

    size_t fId;
    Vec2f uv0, uv1;
    Vec3f p0;
    Vec2f iz0;
    Vec6f JZJw;
    Matf<2, 6> J;
    Mat2f weight;
    Vec2f residual;
    bool valid;
  };

  struct Results {
    typedef std::unique_ptr<Results> UnPtr;
    Pose pose;
    std::vector<Constraint::VecConstShPtr> constraints;
    std::vector<Mat2d> scale;
    std::vector<int> iteration;
    std::vector<NormalEquations> normalEquations;
  };

  static std::map<std::string, double> defaultParameters();

  AlignmentRgbd(const std::map<std::string, double> params);
  AlignmentRgbd(int nLevels = 4, double maxIterations = 100, double minParameterUpdate = 1e-4, double maxErrorIncrease = 1.1);

  Pose align(
    Camera::ConstShPtr cam,
    const cv::Mat &intensity0,
    const cv::Mat &depth0,
    const cv::Mat &intensity1,
    const cv::Mat &depth1,
    const SE3d &guess,
    const Mat6d &guessCovariance = Mat6d::Identity() * INFd);

  Results::UnPtr align(Frame::ConstShPtr frame0, Frame::ShPtr frame1);
  Results::UnPtr align(Frame::VecConstShPtr frame0, Frame::ShPtr frame1);

  int nLevels() { return _nLevels; }

private:
  const TDistributionBivariate<Constraint>::ShPtr _weightFunction;
  const int _nLevels;
  const double _maxIterations, _minParameterUpdate, _maxErrorIncrease;

  int _level, _iteration;

  Constraint::VecShPtr setupConstraints(const Frame::VecConstShPtr &framesRef, const std::vector<SE3f> &motion) const;

  Constraint::VecShPtr
  computeResidualsAndJacobian(const Constraint::VecShPtr &constraints, Frame::ConstShPtr f1, const std::vector<SE3f> &motion) const;

  NormalEquations computeNormalEquations(const std::vector<Constraint::ShPtr> &constraints) const;

  NormalEquations computeNormalEquations(const Pose &prior, const SE3f &pose);

  Matf<2, 6> computeJacobianWarp(const Vec3f &p, Camera::ConstShPtr cam) const;

  Vec6f computeJacobianSE3z(const Vec3f &p) const;
};

}  // namespace vslam
#endif
