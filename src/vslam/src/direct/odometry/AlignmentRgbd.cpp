
#include <execution>
#include <numeric>

#include "AlignmentRgbd.h"
#include "core/Feature2D.h"
#include "core/random.h"
#include "features/FeatureSelection.h"

#include "utils/log.h"

#include "direct/interpolate.h"
#include "direct/jacobians.h"
#define PERFORMANCE_RGBD_ALIGNMENT false
#define LOG_NAME "direct_odometry"
#define MLOG(level) CLOG(level, LOG_NAME)
#define exec_policy std::execution::par_unseq
namespace vslam {
std::map<std::string, double> AlignmentRgbd::defaultParameters() {
  return {{"nLevels", 4.0}, {"maxIterations", 100}, {"minParameterUpdate", 1e-4}, {"maxErrorIncrease", 1.5}};
}

AlignmentRgbd::AlignmentRgbd(const std::map<std::string, double> params) :
    AlignmentRgbd(params.at("nLevels"), params.at("maxIterations"), params.at("minParameterUpdate"), params.at("maxErrorIncrease")) {}

AlignmentRgbd::AlignmentRgbd(int nLevels, double maxIterations, double minParameterUpdate, double maxErrorIncrease) :
    _weightFunction(std::make_shared<TDistributionBivariate<Constraint>>(5.0, 1e-3, 10)),
    _maxIterations(maxIterations),
    _minParameterUpdate(minParameterUpdate),
    _maxErrorIncrease(maxErrorIncrease) {
  log::create(LOG_NAME);
  for (int i = nLevels - 1; i >= 0; i--) {
    _levels.push_back(i);
  }
}
AlignmentRgbd::AlignmentRgbd(const std::vector<int> &levels, double maxIterations, double minParameterUpdate, double maxErrorIncrease) :
    _weightFunction(std::make_shared<TDistributionBivariate<Constraint>>(5.0, 1e-3, 10)),
    _levels(levels),
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
  f0->computePyramid(*std::max_element(_levels.begin(), _levels.end()) + 1);
  f0->computeDerivatives();
  f0->computePcl();
  auto featureSelection = std::make_shared<FeatureSelection<FiniteGradient>>(FiniteGradient{5, 0.01, 0.3, 0, 8.0}, 10, 4);

  featureSelection->select(f0);
  auto f1 = std::make_shared<Frame>(intensity1, depth1, cam);
  f1->computePyramid(*std::max_element(_levels.begin(), _levels.end()) + 1);
  f1->pose() = Pose(guess, guessCovariance);
  Pose pose = align(f0, f1)->pose;
  f0->removeFeatures();
  return pose;
}
AlignmentRgbd::Results::UnPtr AlignmentRgbd::align(Frame::ConstShPtr frame0, Frame::ConstShPtr frame1) {
  return align(frame0->features(), frame1);
}
AlignmentRgbd::Results::UnPtr AlignmentRgbd::align(Frame::ConstShPtr frame0, Frame::ConstShPtr frame1, const Pose &prior) {
  return align(frame0->features(), frame1, prior);
}
AlignmentRgbd::Results::UnPtr AlignmentRgbd::align(Feature2D::VecConstShPtr features, Frame::ConstShPtr frame1) {
  // TODO require prior explicitly
  return align(features, frame1, frame1->pose());
}
AlignmentRgbd::Results::UnPtr AlignmentRgbd::align(Feature2D::VecConstShPtr features, Frame::ConstShPtr frame1, const Pose &prior) {

  TIMED_SCOPE(timer, "align");

  SE3f pose = prior.SE3().cast<float>();
  Mat6f covariance;
  Frame::VecConstShPtr framesRef;
  std::transform(features.begin(), features.end(), std::back_inserter(framesRef), [](auto ft) { return ft->frame(); });
  std::vector<SE3f> motions(framesRef.size());
  std::vector<Constraint::VecShPtr> constraints(nLevels());
  std::vector<NormalEquations> normalEquations(nLevels());
  std::vector<int> iterations(nLevels());
  std::vector<Mat2d> scale(nLevels());
  for (const auto &l : _levels) {
    _level = l;
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
        reason = format("Not enough constraints: {} at {} iteration {}", constraints[_level].size(), _level, _iteration);
        MLOG(WARNING) << reason;
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

      if (!dx.allFinite()) {
        reason = format("NaN during optimization");
        break;
      }

      pose = SE3f::exp(-dx) * pose;
      covariance = normalEquations[_level].A.inverse();

      if (dx.norm() < _minParameterUpdate) {
        reason = format("Minimum step size reached: {:.5f}/{:.5f}", dx.norm(), _minParameterUpdate);
        break;
      }
      MLOG(DEBUG) << format(
        "level: {}, i: {}, #: {}, err: {}, cov: {}",
        _level,
        _iteration,
        constraints[_level].size(),
        error,
        covariance.diagonal().transpose().block(0, 0, 1, 3).norm());
    }
    MLOG(INFO) << format(
      "Done: {}, level: {}, i: {}, #: {}, err: {}, cov: {},",
      reason,
      _level,
      _iteration,
      constraints[_level].size(),
      error,
      covariance.diagonal().transpose().block(0, 0, 1, 3).norm());
  }
  auto r = std::make_unique<Results>(Results{
    Pose(pose.cast<double>(), covariance.cast<double>()),
    std::vector<Constraint::VecConstShPtr>(constraints.size()),
    scale,
    iterations,
    _levels,
    normalEquations});
  std::transform(constraints.begin(), constraints.end(), r->constraints.begin(), [](auto c) {
    return Constraint::VecConstShPtr{c.begin(), c.end()};
  });
  return r;
}

AlignmentRgbd::Constraint::VecShPtr
AlignmentRgbd::setupConstraints(const Frame::VecConstShPtr &framesRef, const Feature2D::VecConstShPtr &features) const {
  TIMED_SCOPE_IF(timer2, format("setupConstraints{}", _level), PERFORMANCE_RGBD_ALIGNMENT);

  std::vector<Constraint::ShPtr> constraints(features.size());
  std::transform(exec_policy, features.begin(), features.end(), constraints.begin(), [&](const auto &ft) {
    const Vec2d uv = ft->position() / std::pow(2.0, _level);
    auto frame = ft->frame();

    auto c = std::make_unique<Constraint>();
    for (c->fId = 0; c->fId < framesRef.size(); c->fId++) {
      if (framesRef[c->fId]->id() == frame->id()) {
        break;
      }
    }
    c->uv0 = Vec2d(ft->position()).cast<float>();
    c->p0 = Vec3d(frame->p3d(uv(1), uv(0), _level)).cast<float>();
    c->iz0 = Vec2f(frame->intensity(uv(1), uv(0), _level) / 255., c->p0.z());

    // Dont we have to recompute this each iteration? NO we always linearize and differentiate around the identity warp
    const Mat<float, 1, 2> JI(frame->dIx(uv(1), uv(0), _level), frame->dIy(uv(1), uv(0), _level));
    const Mat<float, 1, 2> JZ(frame->dZx(uv(1), uv(0), _level), frame->dZy(uv(1), uv(0), _level));

    c->J.row(0) = 1. / 255. * JI * jacobian::project_p<float>(c->p0, frame->camera(_level)->fx(), frame->camera(_level)->fy()) *
                  jacobian::transform_se3(SE3f(), c->p0);
    c->JZJw = JZ * jacobian::project_p<float>(c->p0, frame->camera(_level)->fx(), frame->camera(_level)->fy()) *
              jacobian::transform_se3(SE3f(), c->p0);
    return c;
  });

  return constraints;
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

  std::for_each(exec_policy, constraints.begin(), constraints.end(), [&](auto c) {
    const size_t &i = c->fId;
    const Vec3f p0t = K * ((R[i] * (c->p0)) + t[i]);
    const Vec2f uv0t = Vec2f(p0t(0), p0t(1)) / p0t(2);
    c->uv1 = uv0t;
    const Vec2f iz1w = withinImage(uv0t) ? interpolate(
                                             uv0t,
                                             [&](int v, int u) {
                                               const float i1 = I1.at<uint8_t>(v, u);
                                               const float z1 = Z1.at<float>(v, u);
                                               return z1 > 0 ? Vec2f(i1, z1) : Vec2f::Constant(NANf);
                                             })
                                         : Vec2f::Constant(NANf);

    const Vec3f p1t = ((Rinv[i] * (iz1w(1) * (Kinv * Vec3f(uv0t(0), uv0t(1), 1.0)))) + tinv[i]);

    c->residual = Vec2f(iz1w(0) / 255.0, p1t.z()) - c->iz0;

    c->J.row(1) = c->JZJw - jacobian::transform_se3(motion[i], c->p0).row(2);

    c->valid = p0t.z() > 0 && std::isfinite(iz1w.norm()) && std::isfinite(c->residual.norm()) && std::isfinite(c->J.norm());
  });
  std::vector<Constraint::ShPtr> constraintsValid;
  std::copy_if(constraints.begin(), constraints.end(), std::back_inserter(constraintsValid), [](auto c) { return c->valid; });
  return constraintsValid;
}

NormalEquations AlignmentRgbd::computeNormalEquations(const std::vector<AlignmentRgbd::Constraint::ShPtr> &constraints) const {
  TIMED_SCOPE_IF(timer2, format("computeNormalEquations{}", _level), PERFORMANCE_RGBD_ALIGNMENT);
  // TODO why does this give inconsistent results when run in parallel?
  NormalEquations ne = std::transform_reduce(
    std::execution::seq, constraints.begin(), constraints.end(), NormalEquations{}, std::plus<NormalEquations>{}, [](auto c) {
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

}  // namespace vslam