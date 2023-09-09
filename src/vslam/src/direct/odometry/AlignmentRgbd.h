#ifndef VSLAM_DIRECT_ICP_H__
#define VSLAM_DIRECT_ICP_H__
#include <map>
#include <memory>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "core/Camera.h"
#include "core/Frame.h"
#include "core/Point3D.h"
#include "core/Pose.h"
#include "core/macros.h"
#include "core/types.h"
#include "direct/NormalEquations.h"
#include "direct/weights.h"
namespace vslam {
class AlignmentRgbd {
public:
  TYPEDEF_PTR(AlignmentRgbd)
  struct Constraint {
    TYPEDEF_PTR(Constraint)

    size_t fId;
    Vec2f uv0, uv1;
    Vec3f p0;
    Vec2f iz0;
    Matf<1, 6> JZJw;
    Matf<2, 6> J;
    Mat2f weight;
    Vec2f residual;
    bool valid;
  };

  struct Results {
    TYPEDEF_PTR(Results)
    Pose pose;
    std::vector<Constraint::VecConstShPtr> constraints;
    std::vector<Mat2d> scale;
    std::vector<int> iteration, levels;
    std::vector<NormalEquations> normalEquations;
  };

  static std::map<std::string, double> defaultParameters();

  AlignmentRgbd(const std::map<std::string, double> params);
  AlignmentRgbd(int nLevels = 4, double maxIterations = 100, double minParameterUpdate = 1e-4, double maxErrorIncrease = 1.1);
  AlignmentRgbd(const std::vector<int> &levels, double maxIterations, double minParameterUpdate, double maxErrorIncrease);

  Pose align(
    Camera::ConstShPtr cam,
    const cv::Mat &intensity0,
    const cv::Mat &depth0,
    const cv::Mat &intensity1,
    const cv::Mat &depth1,
    const SE3d &guess,
    const Mat6d &guessCovariance = Mat6d::Identity() * INFd);

  Results::UnPtr align(Frame::ConstShPtr frame0, Frame::ConstShPtr frame1);
  Results::UnPtr align(Frame::ConstShPtr frame0, Frame::ConstShPtr frame1, const Pose &prior);
  Results::UnPtr align(Feature2D::VecConstShPtr features, Frame::ConstShPtr frame1);
  Results::UnPtr align(Feature2D::VecConstShPtr features, Frame::ConstShPtr frame1, const Pose &prior);

  int nLevels() { return *std::max_element(_levels.begin(), _levels.end()) + 1; }

private:
  const TDistributionBivariate<Constraint>::ShPtr _weightFunction;
  std::vector<int> _levels;
  const double _maxIterations, _minParameterUpdate, _maxErrorIncrease;

  int _level, _iteration;

  Constraint::VecShPtr setupConstraints(const Frame::VecConstShPtr &framesRef, const Feature2D::VecConstShPtr &features) const;

  Constraint::VecShPtr
  computeResidualsAndJacobian(const Constraint::VecShPtr &constraints, Frame::ConstShPtr f1, const std::vector<SE3f> &motion) const;

  NormalEquations computeNormalEquations(const std::vector<Constraint::ShPtr> &constraints) const;

  NormalEquations computeNormalEquations(const Pose &prior, const SE3f &pose);
};

}  // namespace vslam
#endif
