#pragma once
#include "core/Pose.h"
#include "core/macros.h"
#include "core/types.h"
#include <map>
namespace vslam {
class PoseGraph {
public:
  TYPEDEF_PTR(PoseGraph)
  PoseGraph();
  void addMeasurement(size_t frameId0, size_t frameId1, const Pose &pose01);
  void optimize();
  const std::map<size_t, SE3d> &poses() const { return _nodes; }
  static constexpr const char LOG_NAME[] = "pose_graph_optimization";

private:
  struct Constraint {
    TYPEDEF_PTR(Constraint)
    size_t from, to;
    Pose pose;
  };
  std::map<size_t, SE3d> _nodes;
  Constraint::VecShPtr _edges;
};
}  // namespace vslam
