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
  bool hasMeasurement(Timestamp t0, Timestamp t1);
  void addMeasurement(Timestamp t0, Timestamp t1, const Pose &pose01);
  void optimize();
  const std::map<Timestamp, SE3d> &poses() const { return _nodes; }
  static constexpr const char LOG_NAME[] = "pose_graph_optimization";
  struct Constraint {
    TYPEDEF_PTR(Constraint)
    Timestamp t0, t1;
    Pose pose;
  };

private:
  std::map<Timestamp, SE3d> _nodes;
  Constraint::VecShPtr _edges;
};
}  // namespace vslam
