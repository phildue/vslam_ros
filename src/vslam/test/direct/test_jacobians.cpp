#include <gtest/gtest.h>

using namespace testing;
#include "direct/jacobians.h"
#include "vslam/core.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/utils.h"

using namespace vslam;
using namespace vslam::evaluation;
TEST(Jacobians, JacobianReprojection) {

  /*
  Check if the jacobians make sense.

  When we have two frames we can compute the update for the pose of frame0 or of frame 1.

  While the jacobian is different (frame 0 transforms the point with its inverse pose, frame 1 with its pose),

  The final relative should be the same as its the optimal update with the same residual.
  */

  auto f0 = std::make_shared<Frame>(
    tum::loadIntensity(TEST_RESOURCE "/1311868164.363181.png"),
    tum::loadDepth(TEST_RESOURCE "/1311868164.338541.png"),
    tum::Camera(),
    1311868164363181000U);

  f0->computePyramid(4);
  f0->computeDerivatives();
  f0->computePcl();

  auto f1 = std::make_shared<Frame>(
    tum::loadIntensity(TEST_RESOURCE "/1311868165.499179.png"),
    tum::loadDepth(TEST_RESOURCE "/1311868165.409979.png"),
    tum::Camera(),
    1311868165499179000U);
  f1->computePyramid(4);
  f1->computeDerivatives();
  f1->computePcl();
  auto featureSelection = std::make_shared<FeatureSelection<FiniteGradient>>(FiniteGradient{5, 0.01, 0.3, 0, 8.0}, 10, 4);
  const int l = 3;
  const double s = 1.0 / std::pow(2.0, l);
  featureSelection->select(f0);
  SE3d T0, T1;
  Vec6d dx0, dx1;
  {
    Mat6d H = Mat6d::Zero();
    Vec6d Jr = Vec6d::Zero();
    for (const auto &ft : f0->features(l)) {
      const Vec3d p0 = f0->p3d(ft->v(), ft->u());
      const Vec3d pw = f0->pose().SE3().inverse() * p0;

      const Vec3d p1 = f1->pose().SE3() * pw;
      const Vec2d uv1 = f1->project(p1, l);
      const double i1 = f1->intensity(l).at<uint8_t>(uv1(1), uv1(0));
      const double r = f0->intensity(l).at<uint8_t>(ft->v() * s, ft->u() * s) - i1;
      const cv::Vec2f dIuv = f0->dI(l).at<cv::Vec2f>(ft->v() * s, ft->u() * s);
      const Matd<1, 2> JI(dIuv[0], dIuv[1]);
      const Matd<2, 3> Jp = jacobian::project_p(p1, f1->camera(l)->fx(), f1->camera(l)->fy());
      const Matd<3, 6> Jt = jacobian::transform_se3(f1->pose().SE3(), pw);
      const Matd<1, 6> J = JI * Jp * Jt;

      LOG_IF(ft->id() % 100 == 0, INFO) << format("{}:\n{}", ft->id(), Jt);

      ASSERT_TRUE(J.allFinite()) << "J= " << J << "\nJI=" << JI << "\nJp= " << Jp << "\nJt= " << Jt;

      H.noalias() += J.transpose() * J;
      Jr.noalias() += J.transpose() * r;
    }
    dx0 = H.ldlt().solve(Jr);
    // print("H={}\n", H);
    // print("Jr={}\n", Jr.transpose());
    print("dx0={}\n", dx0.transpose());
    T0 = SE3d::exp(dx0) * f1->pose().SE3() * f0->pose().SE3().inverse();
  }
  {
    Mat6d H = Mat6d::Zero();
    Vec6d Jr = Vec6d::Zero();
    for (const auto &ft : f0->features(l)) {
      const Vec3d p0 = f0->p3d(ft->v(), ft->u());
      const Vec3d pw = f0->pose().SE3().inverse() * p0;
      const Vec3d p1 = f1->pose().SE3() * pw;
      const Vec2d uv1 = f1->project(p1, l);
      const double i1 = f1->intensity(l).at<uint8_t>(uv1(1), uv1(0));
      const double r = f0->intensity(l).at<uint8_t>(ft->v() * s, ft->u() * s) - i1;
      const cv::Vec2f dIuv = f0->dI(l).at<cv::Vec2f>(ft->v() * s, ft->u() * s);
      const Matd<1, 2> JI(dIuv[0], dIuv[1]);
      const Matd<2, 3> Jp = jacobian::project_p(p1, f1->camera(l)->fx(), f1->camera(l)->fy());
      const Matd<3, 3> Jt = jacobian::transform_p(f1->pose().SE3());
      const Matd<3, 6> Jinvt = jacobian::inverseTransform_se3(f0->pose().SE3(), p0);
      const Matd<1, 6> J = JI * Jp * Jt * Jinvt;
      LOG_IF(ft->id() % 100 == 0, INFO) << format("{}:\n{}", ft->id(), Jinvt);

      ASSERT_TRUE(J.allFinite()) << "J= " << J << " \nJI= " << JI << "\nJp= " << Jp << "\nJt= " << Jt << "\nJinvt= " << Jinvt;

      H.noalias() += J.transpose() * J;
      Jr.noalias() += J.transpose() * r;
    }

    dx1 = H.ldlt().solve(Jr);
    // print("H={}\n", H);
    // print("Jr={}\n", Jr.transpose());
    print("dx1={}\n", dx1.transpose());

    T1 = f1->pose().SE3() * (SE3d::exp(dx1) * f0->pose().SE3()).inverse();
  }
  Vec6d diff = (T1 * T0.inverse()).log();
  print("diff = {}\n", diff.transpose());

  EXPECT_NEAR(diff.norm(), 0, 1e-7) << "T0: " << T0.log().transpose() << "\nT1: " << T1.log().transpose();
}