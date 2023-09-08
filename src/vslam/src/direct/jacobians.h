#pragma once
#include "core/Camera.h"
#include "core/types.h"
namespace vslam::jacobian {

template <typename T> Mat<T, 3, 3> hat(const Vec<T, 3> &p) {
  /*Converts a vector to its skew symmetric matrix*/
  Mat<T, 3, 3> M = Mat<T, 3, 3>::Zero();
  M(0, 0) = 0.0;
  M(0, 1) = -p.z();
  M(0, 2) = p.y();

  M(1, 0) = p.z();
  M(1, 1) = 0.0;
  M(1, 2) = -p.x();

  M(2, 0) = -p.y();
  M(2, 1) = p.x();
  M(2, 2) = 0.0;
  return M;
}

template <typename T> Mat<T, 2, 3> project_p(const Vec<T, 3> &p, T fx, T fy) {
  /*A tutorial on SE3 parameterization: Eq.: A.6*/

  Mat<T, 2, 3> J = Mat<T, 2, 3>::Zero();
  J << fx / p.z(), 0, -fx * p.x() / (p.z() * p.z()), 0, fy / p.z(), -fy * p.y() / (p.z() * p.z());
  return J;
}

template <typename T> Mat<T, 3, 6> transform_se3(const SE3<T> &D, const Vec<T, 3> &p) {
  /*A tutorial on SE3 parameterization: Eq.: 10.23*/
  Mat<T, 3, 6> J = Mat<T, 3, 6>::Zero();

  J.block(0, 0, 3, 3).setIdentity();
  J.block(0, 3, 3, 3) = -hat(D * p);

  return J;
}

template <typename T> Mat<T, 2, 6> transformAndProject_se3(const Vec<T, 3> &p, T fx, T fy) {
  /*A tutorial on SE3 parameterization: Eq.: A.8*/
  const T &x = p.x();
  const T &y = p.y();
  const T z_inv = 1. / p.z();
  const T z_inv_2 = z_inv * z_inv;

  Mat<T, 2, 6> J = Mat<T, 2, 6>::Zero();
  J(0, 0) = z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -x * z_inv_2;
  J(0, 3) = y * J(0, 2);
  J(0, 4) = 1.0 - x * J(0, 2);
  J(0, 5) = -y * z_inv;
  J.row(0) *= fx;
  J(1, 0) = 0.0;
  J(1, 1) = z_inv;
  J(1, 2) = -y * z_inv_2;
  J(1, 3) = -1.0 + y * J(1, 2);
  J(1, 4) = -J(1, 3);
  J(1, 5) = x * z_inv;
  J.row(1) *= fy;
  return J;
}

template <typename T> Mat<T, 3, 6> inverseTransform_se3(const SE3<T> &D, const Vec<T, 3> &p) {
  /*
  A tutorial on SE3 parameterization: Eq.: 10.25
  */
  Mat<T, 3, 6> J = Mat<T, 3, 6>::Zero();

  const Mat3d R = D.rotationMatrix();

  J.block(0, 0, 3, 3) = -R.transpose();
  J.col(3) = R.row(1) * p.z() - R.row(2) * p.y();
  J.col(4) = R.row(2) * p.x() - R.row(0) * p.z();
  J.col(5) = R.row(0) * p.y() - R.row(1) * p.x();
  return J;
}
template <typename T> Mat<T, 3, 3> transform_p(const SE3<T> &D) {
  /*
  A tutorial on SE3 parameterization: Eq.: 10.15
  */
  return D.rotationMatrix();
}
}  // namespace vslam::jacobian