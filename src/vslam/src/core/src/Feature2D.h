//
// Created by phil on 30.06.21.
//

#ifndef VSLAM_FEATURE2D_H
#define VSLAM_FEATURE2D_H

#include <memory>
#include <Eigen/Dense>
#include "types.h"
namespace pd::vslam {

  class Point3D;
  class FrameRgb;

  class Feature2D {
public:
    typedef std::shared_ptr < Feature2D > ShPtr;
    typedef std::shared_ptr < const Feature2D > ConstShPtr;
    typedef std::unique_ptr < Feature2D > UnPtr;
    typedef std::unique_ptr < const Feature2D > ConstUnPtr;

    Feature2D(
      const Vec2d & position, std::shared_ptr < FrameRgb > frame, size_t level = 0U,
      double response = 0.0, const VecXd & descriptor = VecXd::Zero(
        10), std::shared_ptr < Point3D > p3d = nullptr);

    std::shared_ptr < const Point3D > point() const {return _point;}
    std::shared_ptr < Point3D > &point() {
      return _point;
    }
    const Vec2d & position() const {return _position;}
    std::shared_ptr < FrameRgb > frame() {
      return _frame;
    }
    std::shared_ptr < const FrameRgb > frame() const {return _frame;}
    const VecXd & descriptor() const {return _descriptor;}
    size_t level() const {return _level;}
    double response() const {return _response;}
    const std::uint64_t & id() const {return _id;}

private:
    const Vec2d _position;
    const std::shared_ptr < FrameRgb > _frame;
    const size_t _level;
    const double _response;
    const VecXd _descriptor;
    std::shared_ptr < Point3D > _point;
    const std::uint64_t _id;
    static std::uint64_t _idCtr;

  };

}


#endif //VSLAM_FEATURE2D_H
