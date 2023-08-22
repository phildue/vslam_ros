#ifndef VSLAM_VISUALS_H__
#define VSLAM_VISUALS_H__

#include "core/Frame.h"
#include "core/types.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>

namespace vslam {
cv::Mat colorizedDepth(const cv::Mat &depth, double zMax = -1);
cv::Mat blend(const cv::Mat &mat1, const cv::Mat &mat2, double weight1);
cv::Mat colorizedRgbd(const cv::Mat &intensity, const cv::Mat &depth, double zMax = -1);
cv::Mat visualizeFrame(Frame::ConstShPtr frame, int level = 0);
cv::Mat visualizeFramePair(Frame::ConstShPtr f0, Frame::ConstShPtr f1);
namespace overlay {
cv::Mat frames(Frame::VecConstShPtr frames);

template <class... Drawables> class Hstack {
private:
  std::tuple<Drawables...> _drawables;

public:
  Hstack(Drawables &&...drawables) :
      _drawables(std::forward<Drawables>(drawables)...) {}

  cv::Mat operator()() const {
    std::vector<cv::Mat> mats;
    std::apply([&](const auto &...drawables) { (mats.push_back(drawables.draw()), ...); }, _drawables);
    cv::Mat mat;
    cv::hconcat(mats, mat);
    return mat;
  }
};
}  // namespace overlay
}  // namespace vslam
#endif  // VSLAM_VISUALS_H__
