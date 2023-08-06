#include "PoseGraph.h"
#include "utils/log.h"
#include <ceres/ceres.h>
#include <sophus/ceres_manifold.hpp>
namespace vslam {

class OdometryError {
public:
  OdometryError(const Pose &pose01) :
      _measurement01(pose01.SE3()),
      _information(pose01.cov().inverse()) {}

  template <typename T> bool operator()(T const *pose0d, T const *pose1d, T *residualsd) const {
    Eigen::Map<Sophus::SE3<T> const> const pose0(pose0d);
    Eigen::Map<Sophus::SE3<T> const> const pose1(pose1d);
    Eigen::Map<Mat<T, 6, 1>> residuals(residualsd);

    Sophus::SE3<T> state01 = pose1 * pose0.inverse();
    residuals = _information * (_measurement01 * state01.inverse()).log();

    return true;
  }

  static ceres::CostFunction *Create(const Pose &pose01) {
    auto cost = new ceres::AutoDiffCostFunction<OdometryError, 6, SE3d::num_parameters, SE3d::num_parameters>(new OdometryError(pose01));

    return cost;
  }

private:
  const SE3d _measurement01;
  const Mat6d _information;
};

PoseGraph::PoseGraph() { log::create(LOG_NAME); }

void PoseGraph::addMeasurement(size_t frameId0, size_t frameId1, const Pose &pose01) {
  if (_nodes.find(frameId0) == _nodes.end()) {
    _nodes[frameId0] = SE3d();
    CLOG(INFO, LOG_NAME) << format("Adding node {} at identity", frameId0);
  }
  if (_nodes.find(frameId1) == _nodes.end()) {
    _nodes[frameId1] = pose01.SE3() * _nodes[frameId0];
    CLOG(INFO, LOG_NAME) << format("Adding node {} at {}", frameId1, _nodes[frameId1].log());
  }
  _edges.push_back(std::make_shared<Constraint>(frameId0, frameId1, pose01));
}
void PoseGraph::optimize() {
  ceres::Problem problem;
  ceres::Solver::Options options;
  options.update_state_every_iteration = true;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 50;
  options.check_gradients = false;
  for (auto &[id, node] : _nodes) {
    problem.AddParameterBlock(node.data(), SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());
  }
  problem.SetParameterBlockConstant(_nodes.begin()->second.data());

  for (auto &e : _edges) {
    problem.AddResidualBlock(OdometryError::Create(e->pose), nullptr, _nodes.at(e->from).data(), _nodes.at(e->to).data());
  }
  ceres::Solver::Summary summary;
  TIMED_SCOPE(timer, "solve");
  ceres::Solve(options, &problem, &summary);
  for (auto cb : options.callbacks) {
    delete cb;
  }
  CLOG(INFO, LOG_NAME) << summary.BriefReport();
  CLOG(DEBUG, LOG_NAME) << summary.FullReport();
}

}  // namespace vslam
