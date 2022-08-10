//
// Created by phil on 30.06.21.
//

#include "Camera.h"
namespace pd::vslam
{


Eigen::Vector2d Camera::camera2image(const Eigen::Vector3d & pWorld) const
{
  if (pWorld.z() <= 0) {
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  }
  Eigen::Vector3d pProj = _K * pWorld;
  return {pProj.x() / pProj.z(), pProj.y() / pProj.z()};
}

Eigen::Vector3d Camera::image2camera(const Eigen::Vector2d & pImage, double depth) const
{
  return _Kinv * (Eigen::Vector3d({pImage.x(), pImage.y(), 1}) * depth);
}
Eigen::Vector3d Camera::image2ray(const Eigen::Vector2d & pImage) const
{
  return _Kinv * Eigen::Vector3d({pImage.x(), pImage.y(), 1});
}

Camera::Camera(double f, double cx, double cy)
: Camera(f, f, cx, cy) {}

Camera::Camera(double fx, double fy, double cx, double cy)
{
  _K << fx, 0, cx,
    0, fy, cy,
    0, 0, 1;
  _Kinv = _K.inverse();
}
void Camera::resize(double s)
{
  _K *= s;
  _Kinv = _K.inverse();
}
Camera::ShPtr Camera::resize(Camera::ConstShPtr cam, double s)
{
  return std::make_shared<Camera>(
    cam->fx() * s, cam->fy() * s,
    cam->principalPoint().x() * s, cam->principalPoint().y() * s);
}


}
