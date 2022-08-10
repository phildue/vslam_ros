#include <utils/utils.h>
#include "IterativeClosestPointOcv.h"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include <opencv2/rgbd.hpp>
#define LOG_ODOM(level) CLOG(level, "odometry")
namespace pd::vslam
{


PoseWithCovariance::UnPtr IterativeClosestPointOcv::align(
  FrameRgbd::ConstShPtr from,
  FrameRgbd::ConstShPtr to) const
{

  cv::Mat camMat, srcImage, srcDepth, dstImage, dstDepth, guess;
  cv::eigen2cv(from->camera()->K(), camMat);
  cv::eigen2cv(
    algorithm::computeRelativeTransform(
      from->pose().pose(),
      to->pose().pose()).matrix(), guess);

  LOG_ODOM(DEBUG) << "K=" << camMat;
  cv::rgbd::ICPOdometry estimator(camMat,
    cv::rgbd::Odometry::DEFAULT_MIN_DEPTH(), cv::rgbd::Odometry::DEFAULT_MAX_DEPTH(),
    cv::rgbd::Odometry::DEFAULT_MAX_DEPTH_DIFF(),
    cv::rgbd::Odometry::DEFAULT_MAX_POINTS_PART(), std::vector<int>({30, 30, 30}));

  cv::eigen2cv(from->intensity(), srcImage);
  cv::eigen2cv(from->depth(), srcDepth);
  cv::eigen2cv(to->intensity(), dstImage);
  cv::eigen2cv(to->depth(), dstDepth);
  srcImage.convertTo(srcImage, CV_8UC1);
  srcDepth.convertTo(srcDepth, CV_32FC1);
  dstImage.convertTo(dstImage, CV_8UC1);
  dstDepth.convertTo(dstDepth, CV_32FC1);
  auto odomFrameFrom = cv::rgbd::OdometryFrame::create(srcImage, srcDepth);
  auto odomFrameTo = cv::rgbd::OdometryFrame::create(dstImage, dstDepth);

  cv::Mat RtCv;
  MatXd Rt;
  bool success = estimator.compute(odomFrameFrom, odomFrameTo, RtCv, guess);
  cv::cv2eigen(RtCv, Rt);

  if (success) {
    LOG_ODOM(DEBUG) << "Successfully aligned: Rt=" << Rt;
    return std::make_unique<PoseWithCovariance>(
      SE3d(Rt) * from->pose().pose(), MatXd::Identity(
        6,
        6) );
  } else {
    LOG_ODOM(DEBUG) << "Alignment failed.";
    return std::make_unique<PoseWithCovariance>(from->pose() );
  }
}


}
