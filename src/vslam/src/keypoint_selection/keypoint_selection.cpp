#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/macros.h"
#include "core/random.h"
#include "keypoint_selection.h"
#include "utils/log.h"
#include "utils/visuals.h"
namespace vslam::keypoint
{
std::vector<Vec2d> selectManual(Frame::ConstShPtr f, int patchSize)
{
  const int radius = std::floor(patchSize / 2.0);
  std::vector<cv::Scalar> colors(
    500, cv::Scalar(
           (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255,
           (double)std::rand() / RAND_MAX * 255));
  cv::Mat img_ = visualizeFrame(f);
  cv::namedWindow("FeatureSelection");

  struct KeyPointCollector
  {
    cv::Mat depth;
    std::vector<Vec2d> keypoints;
  };
  KeyPointCollector cc{f->depth(), {}};
  cv::setMouseCallback(
    "FeatureSelection",
    [](int evt, int x, int y, int UNUSED(flags), void * param) {
      if (evt == cv::EVENT_LBUTTONDOWN) {
        KeyPointCollector * c = (KeyPointCollector *)param;
        cv::Mat depth = c->depth;
        if (std::isfinite(depth.at<float>(y, x)) && depth.at<float>(y, x) > 0) {
          c->keypoints.push_back(Vec2d(x, y));
        } else {
          print("No valid depth! Did not select");
        }
      }
    },
    (void *)&cc);
  while (cv::waitKey(10) != 27 /*ESC*/) {
    cv::imshow("FeatureSelection", img_);
    for (size_t i = 0; i < cc.keypoints.size(); i++) {
      const auto & kp = cc.keypoints[i];
      cv::rectangle(
        img_, cv::Point(kp.x() - radius, kp.y() - radius),
        cv::Point(kp.x() + radius, kp.y() + radius), colors[i], 1);
    }
  }
  LOG(INFO) << format("Selected keypoints: {}", cc.keypoints.size());
  return cc.keypoints;
}

}  // namespace vslam::keypoint