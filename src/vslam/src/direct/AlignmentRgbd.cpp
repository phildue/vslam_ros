

#include <execution>
#include <numeric>

#include "AlignmentRgbd.h"
#include "core/Feature2D.h"
#include "core/random.h"
#include "features/FeatureSelection.h"

#include "utils/log.h"

#include "interpolate.h"
#define PERFORMANCE_RGBD_ALIGNMENT false
#define LOG_NAME "direct_odometry"
#define MLOG(level) CLOG(level, LOG_NAME)

namespace vslam {
std::map<std::string, double> AlignmentRgbd::defaultParameters() {
  return {{"nLevels", 4.0}, {"maxIterations", 100}, {"minParameterUpdate", 1e-4}, {"maxErrorIncrease", 1.5}};
}

AlignmentRgbd::AlignmentRgbd(const std::map<std::string, double> params) :
    AlignmentRgbd(params.at("nLevels"), params.at("maxIterations"), params.at("minParameterUpdate"), params.at("maxErrorIncrease")) {}

AlignmentRgbd::AlignmentRgbd(int nLevels, double maxIterations, double minParameterUpdate, double maxErrorIncrease) :
    _weightFunction(std::make_shared<TDistributionBivariate<Constraint>>(5.0, 1e-3, 10)),
    _nLevels(nLevels),
    _maxIterations(maxIterations),
    _minParameterUpdate(minParameterUpdate),
    _maxErrorIncrease(maxErrorIncrease) {
  log::create(LOG_NAME);
}

Pose AlignmentRgbd::align(
  Camera::ConstShPtr cam,
  const cv::Mat &intensity0,
  const cv::Mat &depth0,
  const cv::Mat &intensity1,
  const cv::Mat &depth1,
  const SE3d &guess,
  const Mat6d &guessCovariance) {
  auto f0 = std::make_shared<Frame>(intensity0, depth0, cam);
  f0->computePyramid(_nLevels);
  f0->computeDerivatives();
  f0->computePcl();
  auto featureSelection = std::make_shared<FeatureSelection>(5, 0.01, 0.3, 0, 8.0, 20, 4);

  featureSelection->select(f0, true);
  auto f1 = std::make_shared<Frame>(intensity1, depth1, cam);
  f1->computePyramid(_nLevels);
  f1->pose() = Pose(guess, guessCovariance);
  Pose pose = align(f0, f1)->pose;
  f0->removeFeatures();
  return pose;
}
AlignmentRgbd::Results::UnPtr AlignmentRgbd::align(Frame::ConstShPtr frame0, Frame::ShPtr frame1) {
  return align(Frame::VecConstShPtr({frame0}), frame1);
}
AlignmentRgbd::Results::UnPtr AlignmentRgbd::align(Frame::VecConstShPtr framesRef, Frame::ShPtr frame1) {
  TIMED_SCOPE(timer, "align");

  const Pose &prior = frame1->pose();
  SE3f pose = prior.SE3().cast<float>();
  Mat6f covariance;
  std::vector<SE3f> motions(framesRef.size());
  std::vector<Constraint::VecShPtr> constraints(_nLevels);
  std::vector<NormalEquations> normalEquations(_nLevels);
  std::vector<int> iterations(_nLevels);
  std::vector<Mat2d> scale(_nLevels);
  for (_level = _nLevels - 1; _level >= 0; _level--) {
    TIMED_SCOPE_IF(timerLevel, format("computeLevel{}", _level), PERFORMANCE_RGBD_ALIGNMENT);

    auto constraintsAll = setupConstraints(framesRef, motions);
    // constraintsAll = keypoint::subsampling::uniform(
    //   constraintsAll, frame1->height(_level), frame1->width(_level), frame1->size(_level), [](auto c) { return c->uv0; });

    std::string reason = "Max iterations exceeded";
    double error = INFd;
    Vec6f dx = Vec6f::Zero();
    for (_iteration = 0; _iteration < _maxIterations; _iteration++) {
      iterations[_level] = _iteration;
      TIMED_SCOPE_IF(timerIter, format("computeIteration{}", _level), PERFORMANCE_RGBD_ALIGNMENT);
      std::transform(framesRef.begin(), framesRef.end(), motions.begin(), [&](const Frame::ConstShPtr &f) {
        return SE3f(pose * f->pose().SE3().inverse().cast<float>());
      });

      constraints[_level] = computeResidualsAndJacobian(constraintsAll, frame1, motions);

      if (constraints[_level].size() < 6) {
        reason = format("Not enough constraints: {}", constraints[_level].size());
        MLOG(WARNING) << reason;
        pose = SE3f();
        break;
      }
      {
        TIMED_SCOPE_IF(timer3, format("computeWeights{}", _level), PERFORMANCE_RGBD_ALIGNMENT);
        _weightFunction->computeWeights(constraints[_level]);
        scale[_level] = _weightFunction->scale().cast<double>();
      }

      normalEquations[_level] = computeNormalEquations(constraints[_level]);

      if (prior.cov().allFinite()) {
        normalEquations[_level] += computeNormalEquations(prior, pose);
      }

      if (normalEquations[_level].error / error > _maxErrorIncrease) {
        reason = format("Error increased: {:.2f}/{:.2f}", normalEquations[_level].error, error);
        pose = SE3f::exp(dx) * pose;
        break;
      }
      error = normalEquations[_level].error;

      dx = normalEquations[_level].A.ldlt().solve(normalEquations[_level].b);

      pose = SE3f::exp(-dx) * pose;
      covariance = normalEquations[_level].A.inverse();

      if (dx.norm() < _minParameterUpdate) {
        reason = format("Minimum step size reached: {:.5f}/{:.5f}", dx.norm(), _minParameterUpdate);
        break;
      }
      MLOG(DEBUG) << format("level: {}, iteration: {}, error: {}, #: {},", _level, _iteration, error, constraints[_level].size());
    }
    MLOG(DEBUG) << format(
      "Done: {}, level: {}, iteration: {}, error: {}, #: {},", reason, _level, _iteration, error, constraints[_level].size());
  }
  return std::make_unique<Results>(
    Results{Pose(pose.cast<double>(), covariance.cast<double>()), constraints, scale, iterations, normalEquations});
}

AlignmentRgbd::Constraint::VecShPtr
AlignmentRgbd::setupConstraints(const Frame::VecConstShPtr &frames, const std::vector<SE3f> &motion) const {
  TIMED_SCOPE_IF(timer2, format("setupConstraints{}", _level), PERFORMANCE_RGBD_ALIGNMENT);

  // TODO we could also first stack all features and then do one transform..
  std::vector<Constraint::ShPtr> constraintsAll;
  for (size_t fId = 0; fId < frames.size(); fId++) {
    const auto &frame = frames[fId];
    const cv::Mat &intensity = frame->intensity(_level);
    const cv::Mat &dI = frame->dI(_level);
    const cv::Mat &dZ = frame->dZ(_level);
    const double scale = 1.0 / std::pow(2.0, _level);
    auto features = frame->features();

    std::vector<Constraint::ShPtr> constraints(features.size());
    std::transform(std::execution::par_unseq, features.begin(), features.end(), constraints.begin(), [&](const auto &ft) {
      const Vec2d uv = ft->position() * scale;
      const float i = intensity.at<uint8_t>(uv(1), uv(0));
      const cv::Vec2f dIuv = dI.at<cv::Vec2f>(uv(1), uv(0));
      const cv::Vec2f dZuv = dZ.at<cv::Vec2f>(uv(1), uv(0));

      auto c = std::make_unique<Constraint>();
      c->fId = fId;
      c->uv0 = uv.cast<float>();
      c->p0 = frame->p3d(uv(1), uv(0), _level).cast<float>();
      c->iz0 = Vec2f(i, c->p0.z());

      Mat<float, 2, 6> Jw = computeJacobianWarp(motion[fId] * c->p0, frame->camera(_level));
      c->J.row(0) = dIuv[0] * Jw.row(0) + dIuv[1] * Jw.row(1);
      c->JZJw = dZuv[0] * Jw.row(0) + dZuv[1] * Jw.row(1);
      return c;
    });
    constraintsAll.insert(constraintsAll.end(), constraints.begin(), constraints.end());
  }
  // LOG(INFO) << format("Created: {} constraints from {} frames.", constraintsAll.size(), frames.size());

  return constraintsAll;
}
Matf<2, 6> AlignmentRgbd::computeJacobianWarp(const Vec3f &p, Camera::ConstShPtr cam) const {
  const double &x = p.x();
  const double &y = p.y();
  const double z_inv = 1. / p.z();
  const double z_inv_2 = z_inv * z_inv;

  Matf<2, 6> J;
  J(0, 0) = z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -x * z_inv_2;
  J(0, 3) = y * J(0, 2);
  J(0, 4) = 1.0 - x * J(0, 2);
  J(0, 5) = -y * z_inv;
  J.row(0) *= cam->fx();
  J(1, 0) = 0.0;
  J(1, 1) = z_inv;
  J(1, 2) = -y * z_inv_2;
  J(1, 3) = -1.0 + y * J(1, 2);
  J(1, 4) = -J(1, 3);
  J(1, 5) = x * z_inv;
  J.row(1) *= cam->fy();

  return J;
}

AlignmentRgbd::Constraint::VecShPtr AlignmentRgbd::computeResidualsAndJacobian(
  const AlignmentRgbd::Constraint::VecShPtr &constraints, Frame::ConstShPtr f1, const std::vector<SE3f> &motion) const {
  TIMED_SCOPE_IF(timer1, format("computeResidualAndJacobian{}", _level), PERFORMANCE_RGBD_ALIGNMENT);

  /*Cache some constants for faster loop*/
  const Camera::ConstShPtr cam = f1->camera(_level);

  const Mat3f K = cam->K().cast<float>();
  std::vector<Mat3f> R(motion.size());
  std::transform(motion.begin(), motion.end(), R.begin(), [](auto m) { return m.rotationMatrix(); });
  std::vector<Vec3f> t(motion.size());
  std::transform(motion.begin(), motion.end(), t.begin(), [](auto m) { return m.translation(); });

  std::vector<SE3f> motionInv(motion.size());
  std::transform(motion.begin(), motion.end(), motionInv.begin(), [](auto m) { return m.inverse(); });
  std::vector<Mat3f> Rinv(motion.size());
  std::transform(motionInv.begin(), motionInv.end(), Rinv.begin(), [](auto m) { return m.rotationMatrix(); });
  std::vector<Vec3f> tinv(motion.size());
  std::transform(motionInv.begin(), motionInv.end(), tinv.begin(), [](auto m) { return m.translation(); });
  const Mat3f Kinv = cam->Kinv().cast<float>();

  const cv::Mat &I1 = f1->I(_level);
  const cv::Mat &Z1 = f1->Z(_level);
  const float h = f1->height(_level);
  const float w = f1->width(_level);
  const int bh = std::max<int>(1, (int)(0.01f * h));
  const int bw = std::max<int>(1, (int)(0.01f * w));

  auto withinImage = [&](const Vec2f &uv) -> bool { return (bw < uv(0) && uv(0) < w - bw && bh < uv(1) && uv(1) < h - bh); };

  std::for_each(std::execution::par_unseq, constraints.begin(), constraints.end(), [&](auto c) {
    const size_t &i = c->fId;
    const Vec3f p0t = K * ((R[i] * (c->p0)) + t[i]);
    const Vec2f uv0t = Vec2f(p0t(0), p0t(1)) / p0t(2);
    c->uv1 = uv0t;
    const Vec2f iz1w = withinImage(uv0t) ? interpolate<uint8_t, float>(I1, Z1, uv0t) : Vec2f::Constant(NANf);

    const Vec3f p1t = ((Rinv[i] * (iz1w(1) * (Kinv * Vec3f(uv0t(0), uv0t(1), 1.0)))) + tinv[i]);

    c->residual = Vec2f(iz1w(0), p1t.z()) - c->iz0;

    c->J.row(1) = c->JZJw - computeJacobianSE3z(p1t);

    c->valid = p0t.z() > 0 && std::isfinite(iz1w.norm()) && std::isfinite(c->residual.norm()) && std::isfinite(c->J.norm());
  });
  std::vector<Constraint::ShPtr> constraintsValid;
  std::copy_if(constraints.begin(), constraints.end(), std::back_inserter(constraintsValid), [](auto c) { return c->valid; });
  return constraintsValid;
}

NormalEquations AlignmentRgbd::computeNormalEquations(const std::vector<AlignmentRgbd::Constraint::ShPtr> &constraints) const {
  TIMED_SCOPE_IF(timer2, format("computeNormalEquations{}", _level), PERFORMANCE_RGBD_ALIGNMENT);
  NormalEquations ne = std::transform_reduce(
    std::execution::par_unseq,
    constraints.begin(),
    constraints.end(),
    NormalEquations({Mat6f::Zero(), Vec6f::Zero(), 0.0, 0}),
    std::plus<NormalEquations>{},
    [](auto c) {
      return NormalEquations(
        {c->J.transpose() * c->weight * c->J,
         c->J.transpose() * c->weight * c->residual,
         c->residual.transpose() * c->weight * c->residual,
         1});
    });

  return ne;
}

NormalEquations AlignmentRgbd::computeNormalEquations(const Pose &prior, const SE3f &pose) {
  const Mat6f priorInformation = prior.cov().inverse().cast<float>();
  const Vec6f priorError = priorInformation * ((pose * prior.SE3().inverse().cast<float>()).log());
  return {priorInformation, priorError, priorError.norm(), 1};
}

Vec6f AlignmentRgbd::computeJacobianSE3z(const Vec3f &p) const {
  Vec6f J;
  J(0) = 0.0;
  J(1) = 0.0;
  J(2) = 1.0;
  J(3) = p(1);
  J(4) = -p(0);
  J(5) = 0.0;

  return J;
}

}  // namespace vslam