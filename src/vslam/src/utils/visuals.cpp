#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "visuals.h"
namespace vslam {
cv::Mat colorizedDepth(const cv::Mat &depth, double zMax) {
  double Min, zMax_ = zMax;
  if (zMax < 0) {
    cv::minMaxLoc(depth, &Min, &zMax_);
  }

  cv::Mat depthNorm, depthBgr;
  depth.convertTo(depthBgr, CV_8UC3, 255.0 / zMax_);
  cv::applyColorMap(depthBgr, depthBgr, cv::COLORMAP_JET);
  return depthBgr;
}

cv::Mat blend(const cv::Mat &mat1, const cv::Mat &mat2, double weight1) {
  cv::Mat joint;
  cv::Mat weightsI(cv::Size(mat1.cols, mat1.rows), CV_32FC1, weight1);
  cv::Mat weightsZ(cv::Size(mat1.cols, mat1.rows), CV_32FC1, 1.0 - weight1);
  cv::blendLinear(mat1, mat2, weightsI, weightsZ, joint);
  return joint;
}

cv::Mat colorizedRgbd(const cv::Mat &intensity, const cv::Mat &depth, double zMax) {
  cv::Mat depthColor = colorizedDepth(depth, zMax);
  cv::Mat intensityColor;
  cv::cvtColor(intensity, intensityColor, cv::COLOR_GRAY2BGR);
  return blend(intensityColor, depthColor, 0.7);
}

cv::Mat visualizeFrame(Frame::ConstShPtr frame, int level) {
  cv::Mat img = colorizedRgbd(frame->I(level), frame->depth(level));
  cv::putText(img, format("{} | {}", frame->id(), frame->t()), cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
  return img;
}

cv::Mat visualizeFramePair(Frame::ConstShPtr f0, Frame::ConstShPtr f1) {
  cv::Mat overlay;
  cv::hconcat(std::vector<cv::Mat>({visualizeFrame(f0), visualizeFrame(f1)}), overlay);
  return overlay;
}
namespace overlay {
cv::Mat frames(Frame::VecConstShPtr frames) {
  cv::Mat overlay;
  std::vector<cv::Mat> mats(frames.size());
  std::transform(frames.begin(), frames.end(), mats.begin(), [](auto f) { return visualizeFrame(f); });
  cv::hconcat(mats, overlay);
  return overlay;
}

cv::Mat frames(Frame::VecConstShPtr frames, int rows, int cols, int h, int w) {
  cv::Mat overlay;
  std::vector<cv::Mat> mats(frames.size());
  std::transform(frames.begin(), frames.end(), mats.begin(), [](auto f) { return visualizeFrame(f); });
  return arrangeInGrid(mats, rows, cols, h, w);
}

cv::Mat arrangeInGrid(const std::vector<cv::Mat> &mats_, int nRows, int nCols, int h, int w) {
  auto mats = mats_;
  for (auto &m : mats) {
    cv::resize(m, m, cv::Size(w, h));
  }
  std::vector<std::vector<cv::Mat>> grid(nRows, std::vector<cv::Mat>(nCols, cv::Mat(h, w, CV_8UC3, cv::Scalar(0, 0, 0))));
  std::vector<cv::Mat> rows(nRows);
  for (int i = 0; i < nRows; i++) {
    for (int j = 0; j < nCols; j++) {
      const size_t idx = i * nCols + j;
      if (idx < mats.size()) {
        grid[i][j] = mats[idx];
      }
    }
    cv::hconcat(grid[i], rows[i]);
  }
  cv::Mat mat;
  cv::vconcat(rows, mat);
  return mat;
}

}  // namespace overlay

}  // namespace vslam