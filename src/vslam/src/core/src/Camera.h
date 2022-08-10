//
// Created by phil on 30.06.21.
//

#ifndef VSLAM_CAMERA_H
#define VSLAM_CAMERA_H

#include <Eigen/Dense>
#include <memory>

#include "types.h"
namespace pd::vslam {


  class Camera {
public:
    using ConstShPtr = std::shared_ptr < const Camera >;
    using ShPtr = std::shared_ptr < Camera >;
    using Ptr = std::unique_ptr < Camera >;
    typedef std::vector < ConstShPtr > ConstShPtrVec;

    Camera(double f, double cx, double cy);
    Camera(double fx, double fy, double cx, double cy);

    Eigen::Vector2d camera2image(const Eigen::Vector3d & pCamera) const;
    Eigen::Vector3d image2camera(const Eigen::Vector2d & pImage, double depth = 1.0) const;
    Eigen::Vector3d image2ray(const Eigen::Vector2d & pImage) const;
    void resize(double s);

    const double & focalLength() const {return _K(0, 0);}
    const double & fx() const {return _K(0, 0);}
    const double & fy() const {return _K(1, 1);}

    Eigen::Vector2d principalPoint() const {return {_K(0, 2), _K(1, 2)};}
    const Eigen::Matrix3d & K() const {return _K;}
    const Eigen::Matrix3d & Kinv() const {return _Kinv;}
    ShPtr static resize(ConstShPtr cam, double s);

private:
    Eigen::Matrix < double, 3, 3 > _K;       //< Intrinsic camera matrix
    Eigen::Matrix < double, 3, 3 > _Kinv;       //< Intrinsic camera matrix inverted
  };
}

#endif //VSLAM_CAMERA_H
