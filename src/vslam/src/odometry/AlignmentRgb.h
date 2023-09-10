#pragma once
#include <map>
#include <memory>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "core/Camera.h"
#include "core/Frame.h"
#include "core/Pose.h"
#include "core/macros.h"
#include "core/types.h"

#include "NormalEquations.h"
#include "weights.h"
namespace vslam::odometry {
class AlignmentRgb {
public:
  typedef std::shared_ptr<AlignmentRgb> ShPtr;

  struct Constraint {
    TYPEDEF_PTR(Constraint)

    size_t fId;
    Vec2f uv0, uv1;
    Vec3f p0;
    float i0;
    Vec6f J;
    float weight;
    float residual;
    bool valid;
  };

  struct Results {
    TYPEDEF_PTR(Results)

    Pose pose;
    std::vector<Constraint::VecConstShPtr> constraints;
    std::vector<double> scale;
    std::vector<int> iteration;
    std::vector<NormalEquations> normalEquations;
  };

  static std::map<std::string, double> defaultParameters();

  AlignmentRgb(const std::map<std::string, double> params);
  AlignmentRgb(int nLevels = 4, int maxIterations = 100, double minParameterUpdate = 1e-4, double maxErrorIncrease = 1.1);

  Pose align(
    Camera::ConstShPtr cam,
    const cv::Mat &intensity0,
    const cv::Mat &depth0,
    const cv::Mat &intensity1,
    const SE3d &guess,
    const Mat6d &guessCovariance = Mat6d::Identity() * NANd);

  Pose align(
    Camera::ConstShPtr cam,
    const cv::Mat &intensity0,
    const cv::Mat &depth0,
    const cv::Mat &intensity1,
    const cv::Mat &depth1,
    const SE3d &guess,
    const Mat6d &guessCovariance = Mat6d::Identity() * NANd);

  Results::UnPtr align(Frame::ConstShPtr frame0, Frame::ConstShPtr frame1, const Pose &prior) const;
  Results::UnPtr align(const Feature2D::VecConstShPtr &features, Frame::ConstShPtr frame1, const Pose &prior) const;

  int nLevels() { return _nLevels; }

private:
  const TDistribution<Constraint>::ShPtr _weightFunction;
  const int _nLevels;
  const double _maxIterations, _minParameterUpdate, _maxErrorIncrease;

  Constraint::VecShPtr setupConstraints(const Frame::VecConstShPtr &framesRef, const Feature2D::VecConstShPtr &features, int level) const;

  Constraint::VecShPtr computeResidualsAndJacobian(
    const Constraint::VecShPtr &constraints, Frame::ConstShPtr f1, const std::vector<SE3f> &motion, int level) const;

  NormalEquations computeNormalEquations(const std::vector<Constraint::ShPtr> &constraints, int level) const;

  NormalEquations computeNormalEquations(const Pose &prior, const SE3f &pose) const;
};

}  // namespace vslam::odometry
