// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <sophus/ceres_manifold.hpp>

#include "BundleAdjustment.h"
#include "core/Point3D.h"
#include "utils/log.h"

#define LOG_NAME "bundle_adjustment"
#define MLOG(level) CLOG(level, LOG_NAME)

namespace vslam
{
class ReprojectionErrorManifold
{
public:
  ReprojectionErrorManifold(const Eigen::Vector2d & observation, const Eigen::Matrix3d & K)
  : _obs{observation}, _K{K}
  {
  }

  template <typename T>
  bool operator()(const T * const poseData, const T * const pointData, T * residuals) const
  {
    Eigen::Map<Sophus::SE3<T> const> const pose(poseData);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> pWcs(pointData);

    auto pCcs = pose * pWcs;

    if (pCcs[2] > 0.1) {
      auto pIcs = _K * pCcs;

      residuals[0] = pIcs.x() / pIcs.z() - T(_obs.x());
      residuals[1] = pIcs.y() / pIcs.z() - T(_obs.y());
    } else {
      residuals[0] = T(0.0);
      residuals[1] = T(0.0);
    }
    return true;
  }
  static ceres::CostFunction * Create(
    const Eigen::Vector2d & observation, const Eigen::Matrix3d & K)
  {
    return new ceres::AutoDiffCostFunction<
      ReprojectionErrorManifold, 2, Sophus::SE3d::num_parameters, 3>(
      new ReprojectionErrorManifold(observation, K));
  }

private:
  const Eigen::Vector2d _obs;
  const Eigen::Matrix3d _K;
};
BundleAdjustment::BundleAdjustment(size_t maxIterations, double tukeyScale)
: _maxIterations(maxIterations), _tukeyScale(tukeyScale)
{
  log::create(LOG_NAME);
}

BundleAdjustment::Results::ConstUnPtr BundleAdjustment::optimize(
  const Frame::VecConstShPtr & frames, const Frame::VecConstShPtr & fixedFrames) const
{
  Results::UnPtr results = std::make_unique<Results>();
  ceres::Problem problem;
  if (frames.empty()) {
    MLOG(WARNING) << "No Frames given to optimize.";
    return results;
  }
  for (const auto & f : frames) {
    results->poses[f->id()] = f->pose();
    problem.AddParameterBlock(
      results->poses[f->id()].pose().data(), SE3d::num_parameters,
      new Sophus::Manifold<Sophus::SE3>());

    for (const auto & ft : f->featuresWithPoints()) {
      auto pointId = ft->point()->id();
      auto pit = results->positions.find(pointId);
      if (pit == results->positions.end()) {
        results->positions[pointId] = ft->point()->position();
      }
      problem.AddResidualBlock(
        ReprojectionErrorManifold::Create(ft->position(), f->camera()->K()),
        new ceres::TukeyLoss(_tukeyScale), results->poses.at(f->id()).pose().data(),
        results->positions.at(pointId).data());
    }
  }
  if (fixedFrames.empty()) {
    MLOG(WARNING) << "No fixed frames given. Fixing first frame in list: [" << frames.at(0)->id()
                  << "]";
    problem.SetParameterBlockConstant(results->poses[frames.at(0)->id()].pose().data());
  } else {
    for (const auto & f : fixedFrames) {
      problem.AddParameterBlock(
        results->poses.at(f->id()).pose().data(), SE3d::num_parameters,
        new Sophus::Manifold<Sophus::SE3>());
      problem.SetParameterBlockConstant(results->poses.at(f->id()).pose().data());
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = _maxIterations;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  results->errorBefore = summary.initial_cost;
  results->errorAfter = summary.final_cost;

  MLOG(DEBUG) << summary.FullReport();
  MLOG(DEBUG) << "BA: Reduced reprojection error from [" << results->errorBefore << "] to ["
              << results->errorAfter << "]";

  return results;
}
}  // namespace vslam
