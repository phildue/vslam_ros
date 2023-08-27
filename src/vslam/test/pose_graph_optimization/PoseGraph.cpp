#include "PoseGraph.h"
#include "utils/log.h"
#include <ceres/ceres.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/ceres_manifold.hpp>
namespace vslam {

class OdometryError {
public:
  OdometryError(const Pose &pose01) :
      _measurement01(pose01.SE3()),
      _weight(pose01.cov().inverse()) {}

  template <typename T> bool operator()(T const *pose0d, T const *pose1d, T *residualsd) const {
    Eigen::Map<Sophus::SE3<T> const> const pose0(pose0d);
    Eigen::Map<Sophus::SE3<T> const> const pose1(pose1d);
    Eigen::Map<Mat<T, 6, 1>> residuals(residualsd);

    Sophus::SE3<T> state01 = pose1 * pose0.inverse();
    residuals = _weight * (_measurement01 * state01.inverse()).log();

    return true;
  }

  static ceres::CostFunction *Create(const Pose &pose01) {
    auto cost = new ceres::AutoDiffCostFunction<OdometryError, 6, SE3d::num_parameters, SE3d::num_parameters>(new OdometryError(pose01));

    return cost;
  }

private:
  const SE3d _measurement01;
  const Mat6d _weight;
};

struct Overlay {

  const std::map<size_t, SE3d> &poses;
  const PoseGraph::Constraint::VecShPtr &edges;

public:
  cv::Mat operator()() const { return draw(); }
  cv::Mat draw() const {
    double minX, minY, maxX, maxY;
    for (const auto &[id, pose] : poses) {
      if (pose.translation().x() > maxX) {
        maxX = pose.translation().x();
      }
      if (pose.translation().y() > maxY) {
        maxY = pose.translation().y();
      }
      if (pose.translation().x() < minX) {
        minX = pose.translation().x();
      }
      if (pose.translation().y() < minY) {
        minY = pose.translation().y();
      }
    }
    cv::Mat mat(960, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
    const double s = 1. / 0.001;
    const double oy = 480;
    const double ox = 640;
    for (const auto &[id, pose] : poses) {
      cv::circle(mat, cv::Point(pose.translation().x() * s + ox, pose.translation().y() * s + oy), 2, cv::Scalar(255, 0, 0), 2);
    }
    for (const auto &edge : edges) {
      SE3d pose0 = poses.at(edge->from);
      SE3d pose1 = poses.at(edge->to);

      Vec3d d = pose1.translation() - pose0.translation();
      Vec3d c = pose0.translation() + d / 2.;
      cv::line(
        mat,
        cv::Point(pose0.translation().x() * s + ox, pose0.translation().y() * s + oy),
        cv::Point(pose1.translation().x() * s + ox, pose1.translation().y() * s + oy),
        cv::Scalar(0, 255, 0));
      cv::putText(
        mat,
        format("{:.2f}", (edge->pose.translation().norm())),
        cv::Point(c.x() * s + ox, c.y() * s + oy),
        cv::FONT_HERSHEY_COMPLEX,
        0.5,
        cv::Scalar(255, 255, 255));
    }
    return mat;
  }
};

class Callback : public ceres::IterationCallback {
public:
  virtual ~Callback() {}
  ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
    if (summary.iteration == 0) {
      CLOG(DEBUG, PoseGraph::LOG_NAME) << format(
        "iter | cost     | cost_change | |gradient| | |step|   | tr_ratio | tr_radius | successful");
    }
    CLOG(DEBUG, PoseGraph::LOG_NAME) << format(
      "{:03d}  | {:3.2e} | {:3.2e}    | {:3.2e}   | {:3.2e} | {:3.2e} | {:3.2e} | {}",
      summary.iteration,
      summary.cost,
      summary.cost_change,
      summary.gradient_max_norm,
      summary.step_norm,
      summary.relative_decrease,
      summary.trust_region_radius,
      summary.step_is_successful ? "yes" : "no ");
    return ceres::SOLVER_CONTINUE;
  }

private:
};
PoseGraph::PoseGraph() { log::create(LOG_NAME); }

bool PoseGraph::hasMeasurement(size_t frameId0, size_t frameId1) {
  return std::find_if(_edges.begin(), _edges.end(), [&](auto c) { return (c->from == frameId0 && c->to == frameId1); }) != _edges.end();
}
void PoseGraph::addMeasurement(size_t frameId0, size_t frameId1, const Pose &pose01) {
  if (_nodes.find(frameId0) == _nodes.end()) {
    _nodes[frameId0] = SE3d();
  }
  if (_nodes.find(frameId1) == _nodes.end()) {
    _nodes[frameId1] = pose01.SE3() * _nodes[frameId0];
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
  options.callbacks.push_back(new Callback());
  auto loss = nullptr;  // ceres::TukeyLoss(0.1);
  for (auto &e : _edges) {
    problem.AddResidualBlock(OdometryError::Create(e->pose), loss, _nodes.at(e->from).data(), _nodes.at(e->to).data());
  }
  ceres::Solver::Summary summary;
  TIMED_SCOPE(timer, "solve");
  log::append("GraphBefore", Overlay{_nodes, _edges});
  ceres::Solve(options, &problem, &summary);
  for (auto cb : options.callbacks) {
    delete cb;
  }
  log::append("GraphAfter", Overlay{_nodes, _edges});

  CLOG(INFO, LOG_NAME) << summary.BriefReport();
  CLOG(DEBUG, LOG_NAME) << summary.FullReport();
}

}  // namespace vslam
