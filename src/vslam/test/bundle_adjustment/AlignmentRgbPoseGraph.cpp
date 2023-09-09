#include "AlignmentRgbPoseGraph.h"

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include <opencv2/highgui.hpp>
#include <sophus/ceres_manifold.hpp>
#include <thread>

#include "SE3Parameterization.h"
#include "core/Point3D.h"
#include "core/macros.h"
#include "core/types.h"
#include "interpolate.h"
#include "jacobians.h"
#include "overlay.h"
#include "utils/log.h"
#define PERFORMANCE_RGB_ALIGNMENT true
#define LOG_NAME "direct_pose_graph"
#define MLOG(level) CLOG(level, LOG_NAME)
namespace vslam {

namespace autodiff {
class PhotometricReprojectionError {
public:
  PhotometricReprojectionError(
    int patchSize, const cv::Mat &img0, const cv::Mat &img1, const Mat3d &K, const Vec2d &uv0, double z, size_t idx, double scale) :
      _patchSize(patchSize),
      _grid1(img1.ptr<uint8_t>(0), 0, img1.rows, 0, img1.cols),
      _img1(_grid1),
      _K{K},
      _Kinv{K.inverse()},
      _uv0(uv0(0), uv0(1), 1.0),
      _z0(z),
      _idx(idx),
      _scale(scale) {
    _i0 = MatXui8::Zero(patchSize, patchSize);
    for (int r = 0; r < patchSize; r++) {
      for (int c = 0; c < patchSize; c++) {
        const int u0 = _uv0.x() - std::floor(patchSize / 2.0) + c;
        const int v0 = _uv0.y() - std::floor(patchSize / 2.0) + r;
        _i0(r, c) = img0.at<uint8_t>(v0, u0);
        // Zero mean
        //_i0.array() -= -_i0.mean();
      }
    }
  }

  template <typename T> bool operator()(T const *const *parameters, T *residuals) const {
    Eigen::Map<Sophus::SE3<T> const> const pose0(parameters[0]);
    Eigen::Map<Sophus::SE3<T> const> const pose1(parameters[1]);
    const double patchSize_2 = std::floor(_patchSize / 2.0);
    Mat<T, -1, -1> i1 = Mat<T, -1, -1>::Zero(_patchSize, _patchSize);
    for (int r = 0; r < _patchSize; r++) {
      for (int c = 0; c < _patchSize; c++) {
        const T u0 = _uv0.x() - (T)patchSize_2 + (T)c;
        const T v0 = _uv0.y() - (T)patchSize_2 + (T)r;
        const Mat<T, 3, 1> uv0(u0, v0, (T)1.0);
        const Mat<T, 3, 1> p0t = (pose1 * pose0.inverse()) * (_Kinv * _z0 * uv0);
        if (p0t.z() <= 0) {
          residuals[r * _patchSize + c] = (T)(255.0 / _scale);
        } else {
          const Mat<T, 3, 1> uv1 = _K * p0t / p0t.z();
          _img1.Evaluate(uv1(1), uv1(0), &i1(r, c));
          residuals[r * _patchSize + c] = (T)(((T)i1(r, c) - (T)_i0(r, c)) / _scale);
        }
      }
    }
    return true;
  }

  static ceres::CostFunction *Create(
    int patchSize, const cv::Mat &img0, const cv::Mat &img1, const Mat3d &K, const Vec2d &uv0, double z, size_t idx, double scale = 1.0) {
    auto cost = new ceres::DynamicAutoDiffCostFunction<PhotometricReprojectionError>(
      new PhotometricReprojectionError(patchSize, img0, img1, K, uv0, z, idx, scale));
    cost->SetNumResiduals(patchSize * patchSize);
    cost->AddParameterBlock(SE3d::num_parameters);
    cost->AddParameterBlock(SE3d::num_parameters);

    return cost;
  }

private:
  int _patchSize;
  ceres::Grid2D<uint8_t, 1> _grid1;
  ceres::BiCubicInterpolator<ceres::Grid2D<uint8_t, 1>> _img1;
  MatXui8 _i0;
  const Mat3d _K, _Kinv;
  const Vec3d _uv0;
  const double _z0;
  const size_t _idx;
  const double _scale;
};
}  // namespace autodiff
namespace analytical::forward {
class PhotometricReprojectionError : public ceres::SizedCostFunction<1, SE3d::num_parameters, SE3d::num_parameters> {
public:
  virtual ~PhotometricReprojectionError() {}
  PhotometricReprojectionError(Feature2D::ConstShPtr ft, Frame::ConstShPtr f1, int level) :
      _ft(ft),
      _f0(ft->frame()),
      _f1(f1),
      _level(level) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

    Eigen::Map<Sophus::SE3d const> const pose0(parameters[0]);
    Eigen::Map<Sophus::SE3d const> const pose1(parameters[1]);
    const Vec3d p0 = _f0->p3d(_ft->v(), _ft->u());
    const Vec3d pw = pose0.inverse() * p0;
    const Vec3d p1 = pose1 * pw;
    const Vec2d uv1 = _f1->project(p1, _level);
    const Matd<1, -1> i1dIxdIy = interpolate(Vec2d(uv1(0), uv1(1)), [&](int v, int u) {
                                   const double i1 = _f1->intensity(_level).at<uint8_t>(v, u);
                                   const cv::Vec2f dI1 = _f1->dI(_level).at<cv::Vec2f>(v, u);
                                   return _f1->withinImage(Vec2d(u, v), 1.0, _level) ? Vec3d(i1, dI1[0], dI1[1]) : Vec3d(NANd, NANd, NANd);
                                 }).transpose();
    if (i1dIxdIy.allFinite()) {

      if (residuals != NULL) {
        residuals[0] =
          (i1dIxdIy(0) - (double)_f0->intensity(_level).at<uint8_t>(_ft->v() / std::pow(2.0, _level), _ft->u() / std::pow(2.0, _level)));
      }
      if (jacobians != NULL) {
        const Matd<1, 2> JI1(i1dIxdIy(1), i1dIxdIy(2));
        if (jacobians[0] != NULL) {
          Eigen::Map<Matd<1, 7>> J(jacobians[0]);
          J.segment<6>(0) = JI1 * jacobian::project_p(p1, _f1->camera(_level)->fx(), _f1->camera(_level)->fy()) *
                            jacobian::transform_p(SE3d(pose1)) * jacobian::inverseTransform_se3(SE3d(pose0), p0);
          J(6) = 0.0;
        }
        if (jacobians[1] != NULL) {
          Eigen::Map<Matd<1, 7>> J(jacobians[1]);
          // We have to do this workaround using 7-dim error see "SE3Parameterization.cpp"
          J.segment<6>(0) =
            JI1 * jacobian::project_p(p1, _f1->camera(_level)->fx(), _f1->camera(_level)->fy()) * jacobian::transform_se3(SE3d(pose1), pw);
          J(6) = 0.0;
        }
      }
    } else {
      if (residuals != NULL) {
        residuals[0] = 255.0;
      }
      if (jacobians != NULL) {
        if (jacobians[0] != NULL) {
          Eigen::Map<Matd<1, 7>> J(jacobians[0]);
          J.setZero();
        }
        if (jacobians[1] != NULL) {
          Eigen::Map<Matd<1, 7>> J(jacobians[1]);
          J.setZero();
        }
      }
    }
    return true;
  }

private:
  Feature2D::ConstShPtr _ft;
  Frame::ConstShPtr _f0, _f1;
  int _level;
};
};  // namespace analytical::forward
namespace analytical::inverse {
class PhotometricReprojectionError : public ceres::SizedCostFunction<1, SE3d::num_parameters, SE3d::num_parameters> {
public:
  virtual ~PhotometricReprojectionError() {}
  PhotometricReprojectionError(Feature2D::ConstShPtr ft, Frame::ConstShPtr f1, int level) :
      _ft(ft),
      _f0(ft->frame()),
      _f1(f1),
      _level(level) {
    const Vec3d p0 = _f0->p3d(_ft->v(), _ft->u());

    const cv::Vec2f dI0 = _f0->dI(_level).at<cv::Vec2f>(_ft->v() / std::pow(2.0, _level), _ft->u() / std::pow(2.0, _level));
    const Matd<1, 2> JI0(dI0[0], dI0[1]);
    _J0 = JI0 * jacobian::project_p(p0, _f1->camera(_level)->fx(), _f1->camera(_level)->fy()) * jacobian::transform_p(SE3d()) *
          jacobian::inverseTransform_se3(SE3d(), p0);
    _J1 = JI0 * jacobian::project_p(p0, _f1->camera(_level)->fx(), _f1->camera(_level)->fy()) * jacobian::transform_se3(SE3d(), p0);
  }

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

    Eigen::Map<Sophus::SE3d const> const pose0(parameters[0]);
    Eigen::Map<Sophus::SE3d const> const pose1(parameters[1]);
    const Vec3d p0 = _f0->p3d(_ft->v(), _ft->u());
    const Vec3d pw = pose0.inverse() * p0;
    const Vec3d p1 = pose1 * pw;
    const Vec2d uv1 = _f1->project(p1, _level);
    const Mat<double, -1, 1> i1 = interpolate(uv1, [&](int v, int u) {
      return _f1->withinImage(Vec2d(u, v), 1.0, _level) ? Vec<double, 1>(_f1->intensity(_level).at<uint8_t>(v, u)) : Vec<double, 1>(NANd);
    });
    if (i1.allFinite()) {

      if (residuals != NULL) {
        residuals[0] =
          (i1(0) - (double)_f0->intensity(_level).at<uint8_t>(_ft->v() / std::pow(2.0, _level), _ft->u() / std::pow(2.0, _level)));
      }
      if (jacobians != NULL) {

        if (jacobians[0] != NULL) {
          Eigen::Map<Matd<1, 7>> J(jacobians[0]);
          J.segment<6>(0) = _J0;
          J(6) = 0.0;
        }
        if (jacobians[1] != NULL) {
          Eigen::Map<Matd<1, 7>> J(jacobians[1]);
          // We have to do this workaround using 7-dim error see "SE3Parameterization.cpp"
          J.segment<6>(0) = _J1;
          J(6) = 0.0;
        }
      }
    } else {
      if (residuals != NULL) {
        residuals[0] = 255.0;
      }
      if (jacobians != NULL) {
        if (jacobians[0] != NULL) {
          Eigen::Map<Matd<1, 7>> J(jacobians[0]);
          J.setZero();
        }
        if (jacobians[1] != NULL) {
          Eigen::Map<Matd<1, 7>> J(jacobians[1]);
          J.setZero();
        }
      }
    }
    return true;
  }

private:
  Feature2D::ConstShPtr _ft;
  Frame::ConstShPtr _f0, _f1;
  int _level;
  Matd<1, 6> _J0, _J1;
};
};  // namespace analytical::inverse

class OverlayCallback : public ceres::IterationCallback {
public:
  OverlayCallback(
    const std::vector<Vec2d> &uv0,
    Frame::ConstShPtr f0,
    Frame::ConstShPtr f1,
    const SE3d *pose0,
    const SE3d *pose1,
    const std::vector<double> &z0,
    int level) :
      _uv0(uv0),
      _f0(f0),
      _f1(f1),
      _pose0(pose0),
      _pose1(pose1),
      _level(level),
      _z0(z0) {}
  OverlayCallback(
    const std::vector<Feature2D::ConstShPtr> &uv0,
    Frame::ConstShPtr f0,
    Frame::ConstShPtr f1,
    const SE3d *pose0,
    const SE3d *pose1,
    const std::vector<double> &z0,
    int level) :
      _uv0(uv0.size()),
      _f0(f0),
      _f1(f1),
      _pose0(pose0),
      _pose1(pose1),
      _level(level),
      _z0(z0) {
    std::transform(uv0.begin(), uv0.end(), _uv0.begin(), [](auto ft) { return ft->position(); });
  }
  virtual ~OverlayCallback() {}
  ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
    // print("t0= {}\n", _pose0->log().transpose());
    // print("t1= {}\n", _pose1->log().transpose());
    // print("t01= {}\n", ((*_pose1) * (*_pose0).inverse()).log().transpose());
    if (summary.iteration == 0) {
      MLOG(DEBUG) << format("iter | cost     | cost_change | |gradient| | |step|   | tr_ratio | tr_radius | successful");
    }
    MLOG(DEBUG) << format(
      "{:03d}  | {:3.2e} | {:3.2e}    | {:3.2e}   | {:3.2e} | {:3.2e} | {:3.2e} | {}",
      summary.iteration,
      summary.cost,
      summary.cost_change,
      summary.gradient_max_norm,
      summary.step_norm,
      summary.relative_decrease,
      summary.trust_region_radius,
      summary.step_is_successful ? "yes" : "no ");
    log::append("AlignmentRgbPoseGraph", overlay::DepthCorrespondence(_uv0, _f0, _f1, (*_pose1) * ((*_pose0).inverse()), _z0, _level, 1));
    return ceres::SOLVER_CONTINUE;
  }

private:
  std::vector<Vec2d> _uv0;
  const Frame::ConstShPtr _f0, _f1;
  const SE3d *_pose0;
  const SE3d *_pose1;

  const int _level;
  const std::vector<double> _z0;
};
AlignmentRgbPoseGraph::AlignmentRgbPoseGraph(int nLevels, int maxIterations, double tukeyA) :
    _nLevels(nLevels),
    _maxIterations(maxIterations),
    _tukeyA(tukeyA) {
  log::create(LOG_NAME);
}

void AlignmentRgbPoseGraph::align(Frame::VecShPtr frames, Frame::VecShPtr framesFixed) {
  for (int level = _nLevels - 1; level >= 0; level--) {
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.update_state_every_iteration = true;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = _maxIterations;
    options.check_gradients = false;

    for (auto fIt0 = frames.begin(); fIt0 != frames.end(); ++fIt0) {
      for (auto fIt1 = fIt0 + 1; fIt1 != frames.end(); ++fIt1) {
        auto f0 = *fIt0;
        auto f1 = *fIt1;
        if (
          std::find(framesFixed.begin(), framesFixed.end(), f0) != framesFixed.end() &&
          std::find(framesFixed.begin(), framesFixed.end(), f1) != framesFixed.end())
          continue;

        auto features = f0->features();

        if (features.empty()) {
          MLOG(INFO) << format("Skipping frame {} as it did not have any features.", f0->id());
          continue;
        }
        MLOG(INFO) << format("Aligning frame {} to {} at level {} with {} keypoints.", f0->id(), f1->id(), level, features.size());

        std::vector<double> z(features.size());
        std::transform(features.begin(), features.end(), z.begin(), [&](auto ft) { return f0->depth().at<float>(ft->v(), ft->u()); });

        log::append(
          "AlignmentRgbPoseGraph",
          overlay::DepthCorrespondence(
            Feature2D::VecConstShPtr{features.begin(), features.end()}, f0, f1, (f1->pose() * f0->pose().inverse()).SE3(), z, level, 1));
#ifdef AUTODIFF
        problem.AddParameterBlock(f0->pose().SE3().data(), SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());

        problem.AddParameterBlock(f1->pose().SE3().data(), SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());
#else
        problem.AddParameterBlock(f0->pose().SE3().data(), SE3d::num_parameters, new SE3Parameterization());
        problem.AddParameterBlock(f1->pose().SE3().data(), SE3d::num_parameters, new SE3Parameterization());
#endif
        auto loss = new ceres::TukeyLoss(_tukeyA);

        for (size_t i = 0; i < features.size(); i++) {
#ifdef AUTODIFF
          problem.AddResidualBlock(
            autodiff::PhotometricReprojectionError::Create(
              1,
              f0->intensity(level),
              f1->intensity(level),
              f0->camera(level)->K(),
              features[i]->position() / std::pow(2.0, level),
              z.at(i),
              features[i]->id(),
              1.0),
            loss,
            f0->pose().SE3().data(),
            f1->pose().SE3().data());

#else
          problem.AddResidualBlock(
            new analytical::forward::PhotometricReprojectionError(features[i], f1, level),
            loss,
            f0->pose().SE3().data(),
            f1->pose().SE3().data());
#endif
        }
        // if (options.callbacks.empty()) {
        options.callbacks.push_back(new OverlayCallback(
          Feature2D::VecConstShPtr{features.begin(), features.end()}, f0, f1, &f0->pose().SE3(), &f1->pose().SE3(), z, level));
        //}
      }
    }
    for (auto f : framesFixed) {
      MLOG(INFO) << format("Fixing frame {}", f->id());
      problem.SetParameterBlockConstant(f->pose().SE3().data());
    }

    ceres::Solver::Summary summary;
    TIMED_SCOPE(timer, "solve");
    ceres::Solve(options, &problem, &summary);
    for (auto cb : options.callbacks) {
      delete cb;
    }
    MLOG(INFO) << summary.BriefReport();
    MLOG(DEBUG) << summary.FullReport();
  }
}

}  // namespace vslam