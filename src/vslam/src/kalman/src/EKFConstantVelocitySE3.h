#ifndef VSLAM_KALMAN_FILTER_SE3_H__
#define VSLAM_KALMAN_FILTER_SE3_H__

#include "core/core.h"
namespace pd::vslam::kalman {

  class EKFConstantVelocitySE3 {
public:
    typedef std::shared_ptr < EKFConstantVelocitySE3 > ShPtr;
    typedef std::unique_ptr < EKFConstantVelocitySE3 > UnPtr;
    typedef std::shared_ptr < const EKFConstantVelocitySE3 > ConstPtr;

    struct State
    {
      typedef std::shared_ptr < State > ShPtr;
      typedef std::unique_ptr < State > UnPtr;
      typedef std::shared_ptr < const State > ConstPtr;

      Vec6d pose;
      Vec6d velocity;
      Matd < 6, 6 > covPose;
      Matd < 6, 6 > covVel;
    };

    EKFConstantVelocitySE3(
      const Matd < 12, 12 > &covarianceProcess,
      Timestamp t0 = std::numeric_limits < uint64_t > ::max());

    State::UnPtr predict(Timestamp t) const;

    void update(const Vec6d & motion, const Matd < 6, 6 > & covMotion, Timestamp t);

private:
    void predict(
      Timestamp t, Vec6d & pose, Vec6d & vel, Matd < 12, 12 > & P, Matd < 12,
      12 > & Jfx) const;

    Matd < 12, 12 > computeJacobianProcess(const SE3d & pose) const;
    Matd < 6, 12 > computeJacobianMeasurement(Timestamp t) const;

    Timestamp _t;
    Vec6d _pose;
    Vec6d _velocity;
    Matd < 12, 12 > _covState;
    Matd < 12, 12 > _covProcess;
  };
}
#endif //VSLAM_KALMAN_H__
