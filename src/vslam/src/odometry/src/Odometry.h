#ifndef VSLAM_ODOMETRY
#define VSLAM_ODOMETRY

#include <core/core.h>
#include <least_squares/least_squares.h>
#include "SE3Alignment.h"
#include "Map.h"
#include "IterativeClosestPoint.h"
#include "RgbdAlignmentOpenCv.h"
#include "IterativeClosestPointOcv.h"

namespace pd::vslam {
  class Odometry {
public:
    typedef std::shared_ptr < Odometry > ShPtr;
    typedef std::unique_ptr < Odometry > UnPtr;
    typedef std::shared_ptr < const Odometry > ConstShPtr;
    typedef std::unique_ptr < const Odometry > ConstUnPtr;

    virtual void update(FrameRgbd::ConstShPtr frame) = 0;

    virtual PoseWithCovariance::ConstShPtr pose() const = 0;
    virtual PoseWithCovariance::ConstShPtr speed() const = 0;

    static ShPtr make();

  };


  class OdometryRgbd: public Odometry {
public:
    typedef std::shared_ptr < OdometryRgbd > ShPtr;
    typedef std::unique_ptr < OdometryRgbd > UnPtr;
    typedef std::shared_ptr < const OdometryRgbd > ConstShPtr;
    typedef std::unique_ptr < const OdometryRgbd > ConstUnPtr;

    OdometryRgbd(
      double minGradient,
      least_squares::Solver::ShPtr solver,
      least_squares::Loss::ShPtr loss,
      Map::ConstShPtr map);

    void update(FrameRgbd::ConstShPtr frame) override;

    PoseWithCovariance::ConstShPtr pose() const override {return _pose;}
    PoseWithCovariance::ConstShPtr speed() const override {return _speed;}

protected:
    const SE3Alignment::ConstShPtr _aligner;
    const Map::ConstShPtr _map;
    const bool _includeKeyFrame, _trackKeyFrame;
    PoseWithCovariance::ConstShPtr _speed;
    PoseWithCovariance::ConstShPtr _pose;

  };
  class OdometryIcp: public Odometry {
public:
    typedef std::shared_ptr < OdometryIcp > ShPtr;
    typedef std::unique_ptr < OdometryIcp > UnPtr;
    typedef std::shared_ptr < const OdometryIcp > ConstShPtr;
    typedef std::unique_ptr < const OdometryIcp > ConstUnPtr;

    OdometryIcp(int level, int maxIterations, Map::ConstShPtr map);

    void update(FrameRgbd::ConstShPtr frame) override;

    PoseWithCovariance::ConstShPtr pose() const override {return _pose;}
    PoseWithCovariance::ConstShPtr speed() const override {return _speed;}

protected:
    const IterativeClosestPoint::ConstShPtr _aligner;
    PoseWithCovariance::ConstShPtr _speed;
    PoseWithCovariance::ConstShPtr _pose;
    const Map::ConstShPtr _map;

  };

}
#endif// VSLAM_ODOMETRY
