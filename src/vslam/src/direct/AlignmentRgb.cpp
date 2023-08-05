

#include <execution>
#include <functional>
#include <numeric>

#include "AlignmentRgb.h"
#include "core/Feature2D.h"
#include "core/random.h"
#include "features/FeatureSelection.h"
#include "interpolate.h"
#include "jacobians.h"
#include "utils/log.h"
#define PERFORMANCE_RGBD_ALIGNMENT false
#define LOG_NAME "direct_odometry"
#define MLOG(level) CLOG(level, LOG_NAME)

namespace vslam {
std::map<std::string, double> AlignmentRgb::defaultParameters() {
  return {{"nLevels", 4.0}, {"maxIterations", 100}, {"minParameterUpdate", 1e-4}, {"maxErrorIncrease", 1.5}};
}

AlignmentRgb::AlignmentRgb(const std::map<std::string, double> params) :
    AlignmentRgb(params.at("nLevels"), params.at("maxIterations"), params.at("minParameterUpdate"), params.at("maxErrorIncrease")) {}

AlignmentRgb::AlignmentRgb(int nLevels, double maxIterations, double minParameterUpdate, double maxErrorIncrease) :
    _weightFunction(std::make_shared<TDistribution<Constraint>>(5.0, 1e-3, 10)),
    _nLevels(nLevels),
    _maxIterations(maxIterations),
    _minParameterUpdate(minParameterUpdate),
    _maxErrorIncrease(maxErrorIncrease) {
  log::create(LOG_NAME);
}

Pose AlignmentRgb::align(
  Camera::ConstShPtr cam,
  const cv::Mat &intensity0,
  const cv::Mat &depth0,
  const cv::Mat &intensity1,
  const SE3d &guess,
  const Mat6d &guessCovariance) {
  auto f0 = std::make_shared<Frame>(intensity0, depth0, cam);
  f0->computePyramid(_nLevels);
  f0->computeDerivatives();
  f0->computePcl();
  auto featureSelection = std::make_shared<FeatureSelection<FiniteGradient>>(FiniteGradient{5, 0.01, 0.3, 0, 8.0}, 10, 4);
  featureSelection->select(f0);
  auto f1 = std::make_shared<Frame>(intensity1, cam);
  f1->computePyramid(_nLevels);
  f1->pose() = Pose(guess, guessCovariance);
  Pose pose = align(f0, f1)->pose;
  f0->removeFeatures();
  return pose;
}

Pose AlignmentRgb::align(
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
  auto featureSelection = std::make_shared<FeatureSelection<FiniteGradient>>(FiniteGradient{5, 0.01, 0.3, 0, 8.0}, 10, 4);

  featureSelection->select(f0);
  auto f1 = std::make_shared<Frame>(intensity1, depth1, cam);
  f1->computePyramid(_nLevels);
  f1->pose() = Pose(guess, guessCovariance);
  Pose pose = align(f0, f1)->pose;
  f0->removeFeatures();
  return pose;
}
AlignmentRgb::Results::UnPtr AlignmentRgb::align(Frame::ConstShPtr frame0, Frame::ConstShPtr frame1) {
  return align(frame0->features(), frame1);
}
AlignmentRgb::Results::UnPtr AlignmentRgb::align(const Feature2D::VecConstShPtr &features, Frame::ConstShPtr frame1) {
  TIMED_SCOPE(timer, "align");

  const Pose &prior = frame1->pose();
  SE3f pose = prior.SE3().cast<float>();
  Mat6f covariance;
  Frame::VecConstShPtr framesRef;
  std::transform(features.begin(), features.end(), std::back_inserter(framesRef), [](auto ft) { return ft->frame(); });

  std::vector<SE3f> motions(framesRef.size());
  std::vector<Constraint::VecShPtr> constraints(_nLevels);
  std::vector<NormalEquations> normalEquations(_nLevels);
  std::vector<int> iterations(_nLevels);
  std::vector<double> scale(_nLevels);
  for (_level = _nLevels - 1; _level >= 0; _level--) {
    TIMED_SCOPE_IF(timerLevel, format("computeLevel{}", _level), PERFORMANCE_RGBD_ALIGNMENT);
    auto constraintsAll = setupConstraints(framesRef, features);

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
        scale[_level] = _weightFunction->scale();
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

AlignmentRgb::Constraint::VecShPtr
AlignmentRgb::setupConstraints(const Frame::VecConstShPtr &framesRef, const Feature2D::VecConstShPtr &features) const {
  TIMED_SCOPE_IF(timer2, format("setupConstraints{}", _level), PERFORMANCE_RGBD_ALIGNMENT);

  // TODO we could also first stack all features and then do one transform..

  std::vector<Constraint::ShPtr> constraints(features.size());
  std::transform(std::execution::par_unseq, features.begin(), features.end(), constraints.begin(), [&](const auto &ft) {
    const Frame::ConstShPtr frame = ft->frame();
    const cv::Mat &intensity = frame->intensity(_level);
    const cv::Mat &dI = frame->dI(_level);
    const double scale = 1.0 / std::pow(2.0, _level);

    const Vec2d uv = ft->position() * scale;
    const float i = intensity.at<uint8_t>(uv(1), uv(0));
    const cv::Vec2f dIuv = dI.at<cv::Vec2f>(uv(1), uv(0));

    auto c = std::make_unique<Constraint>();
    for (c->fId = 0; c->fId < framesRef.size(); c->fId++) {
      if (framesRef[c->fId]->id() == frame->id()) {
        break;
      }
    }
    c->uv0 = uv.cast<float>();
    c->p0 = frame->p3d(uv(1), uv(0), _level).cast<float>();
    c->i0 = i;

    /*TODO
    - dont we have to recompute this each iteration? NO we always linearize and differentiate around the identity warp
    */
    const Mat<float, 1, 2> JI(dIuv[0], dIuv[1]);

    c->J = JI * jacobian::project_p<float>(c->p0, frame->camera(_level)->fx(), frame->camera(_level)->fy()) *
           jacobian::transform_se3(SE3f(), c->p0);
    return c;
  });

  return constraints;
}

AlignmentRgb::Constraint::VecShPtr AlignmentRgb::computeResidualsAndJacobian(
  const AlignmentRgb::Constraint::VecShPtr &constraints, Frame::ConstShPtr f1, const std::vector<SE3f> &motion) const {
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

  const cv::Mat &I1 = f1->I(_level);
  const float h = f1->height(_level);
  const float w = f1->width(_level);
  const int bh = std::max<int>(1, (int)(0.01f * h));
  const int bw = std::max<int>(1, (int)(0.01f * w));

  auto withinImage = [&](const Vec2f &uv) -> bool { return (bw < uv(0) && uv(0) < w - bw && bh < uv(1) && uv(1) < h - bh); };

  std::function<float(const Vec2f &)> sample = [&](const Vec2f &uv) -> float {
    return interpolate(uv, [&](int v, int u) { return Vec<float, 1>::Constant(I1.at<uint8_t>(v, u)); })(0);
  };
  if (f1->hasDepth()) {
    sample = [&](const Vec2f &uv) -> float {
      return interpolate(uv, [&](int v, int u) {
        const float i1 = I1.at<uint8_t>(v, u);
        const float z1 = f1->depth(_level).at<float>(v, u);
        return z1 > 0 ? Vec2f(i1, z1) : Vec2f::Constant(NANf);
      })(0);
    };
  }

  std::for_each(std::execution::par_unseq, constraints.begin(), constraints.end(), [&](auto c) {
    const size_t &i = c->fId;
    const Vec3f p0t = K * ((R[i] * (c->p0)) + t[i]);
    const Vec2f uv0t = Vec2f(p0t(0), p0t(1)) / p0t(2);
    c->uv1 = uv0t;
    const float i1w = withinImage(uv0t) ? sample(uv0t) : NANf;

    c->residual = i1w - c->i0;

    c->valid = p0t.z() > 0 && std::isfinite(i1w) && std::isfinite(c->residual) && std::isfinite(c->J.norm());
  });
  std::vector<Constraint::ShPtr> constraintsValid;
  std::copy_if(constraints.begin(), constraints.end(), std::back_inserter(constraintsValid), [](auto c) { return c->valid; });
  return constraintsValid;
}

NormalEquations AlignmentRgb::computeNormalEquations(const std::vector<AlignmentRgb::Constraint::ShPtr> &constraints) const {
  TIMED_SCOPE_IF(timer2, format("computeNormalEquations{}", _level), PERFORMANCE_RGBD_ALIGNMENT);
  NormalEquations ne = std::transform_reduce(
    std::execution::par_unseq,
    constraints.begin(),
    constraints.end(),
    NormalEquations({Mat6f::Zero(), Vec6f::Zero(), 0.0, 0}),
    std::plus<NormalEquations>{},
    [](auto c) {
      return NormalEquations(
        {c->J * c->weight * c->J.transpose(), c->J * c->weight * c->residual, c->residual * c->weight * c->residual, 1});
    });

  return ne;
}

NormalEquations AlignmentRgb::computeNormalEquations(const Pose &prior, const SE3f &pose) {
  const Mat6f priorInformation = prior.cov().inverse().cast<float>();
  const Vec6f priorError = priorInformation * ((pose * prior.SE3().inverse().cast<float>()).log());
  return {priorInformation, priorError, priorError.norm(), 1};
}

}  // namespace vslam