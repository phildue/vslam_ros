#ifndef VSLAM_VISUALS_H__
#define VSLAM_VISUALS_H__

#include "core/Frame.h"
#include "core/types.h"
namespace vslam {
cv::Mat colorizedDepth(const cv::Mat &depth, double zMax = -1);
cv::Mat blend(const cv::Mat &mat1, const cv::Mat &mat2, double weight1);
cv::Mat colorizedRgbd(const cv::Mat &intensity, const cv::Mat &depth, double zMax = -1);
cv::Mat visualizeFrame(Frame::ConstShPtr frame, int level = 0);
cv::Mat visualizeFramePair(Frame::ConstShPtr f0, Frame::ConstShPtr f1);
namespace overlay {
cv::Mat frames(Frame::VecConstShPtr frames);
cv::Mat frames(Frame::VecConstShPtr frames, int nRows, int nCols, int h = 240, int w = 320);

cv::Mat arrangeInGrid(const std::vector<cv::Mat> &mats_, int nRows, int nCols, int h = 240, int w = 320);

}  // namespace overlay
}  // namespace vslam
#endif  // VSLAM_VISUALS_H__
