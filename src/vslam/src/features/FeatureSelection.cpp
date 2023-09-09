#include "FeatureSelection.h"
#include "core/Feature2D.h"
#include "core/macros.h"
#include "keypoint_selection.h"
#include "overlays.h"
#include "utils/log.h"
#include <functional>
#define LOG_NAME "features"
#define MLOG(level) CLOG(level, LOG_NAME)
namespace vslam {

FeatureSelection::FeatureSelection(
  float intensityGradientMin, float depthGradientMin, float depthGradientMax, float depthMin, float depthMax, float gridSize, int nLevels) :
    _intensityGradientMin(intensityGradientMin),
    _depthGradientMin(depthGradientMin),
    _depthGradientMax(depthGradientMax),
    _depthMin(depthMin),
    _depthMax(depthMax),
    _gridSize(gridSize),
    _nLevels(nLevels) {
  log::create(LOG_NAME);
}

void FeatureSelection::select(Frame::ShPtr f, bool override) const {
  for (int level = _nLevels - 1; level >= 0; level--) {
    const double scale = std::pow(2.0, level);
    const cv::Mat &depth = f->depth(level);
    const cv::Mat &dI = f->dI(level);
    const cv::Mat &dZ = f->dZ(level);

    /*We want features at all levels, so we create an occupancy grid of existing features at that level*/
    const std::vector<bool> occupied =
      override
        ? std::vector<bool>(f->size(), false)
        : keypoint::createOccupancyGrid(f->features(level), f->height(), f->width(), _gridSize, [](auto ft) { return ft->position(); });

    const int nOccupied = std::count(occupied.begin(), occupied.end(), true);
    const int nCols = f->width() / _gridSize;
    const int nRows = occupied.size() / (float)nCols;
    MLOG(INFO) << format(
      "Present features: {} occupying {} in {}x{} cells. {}",
      f->features().size(),
      nOccupied,
      nRows,
      nCols,
      override ? "Overriding..." : "Preserving..");

    /* Propose features in empty cells that meet feature criterion */
    std::vector<Vec2d> kps = keypoint::select(f, level, [&](const Vec2d &uv) {
      const float z = depth.at<float>(uv(1), uv(0));
      const cv::Vec2f dIuv = dI.at<cv::Vec2f>(uv(1), uv(0));
      const cv::Vec2f dZuv = dZ.at<cv::Vec2f>(uv(1), uv(0));
      const size_t r = static_cast<size_t>(uv(1) * scale / _gridSize);
      const size_t c = static_cast<size_t>(uv(0) * scale / _gridSize);
      return !occupied[r * nCols + c] && (std::isfinite(z) && std::isfinite(dZuv[0]) && std::isfinite(dZuv[1]) && _depthMin < z &&
                                          z < _depthMax && std::abs(dZuv[0]) < _depthGradientMax && std::abs(dZuv[1]) < _depthGradientMax &&
                                          (std::abs(dIuv[0]) > _intensityGradientMin || std::abs(dIuv[1]) > _intensityGradientMin ||
                                           std::abs(dZuv[0]) > _depthGradientMin || std::abs(dZuv[1]) > _depthGradientMin));
    });
    MLOG(INFO) << format("Newly selected features: {}", kps.size());

    std::vector<Feature2D::ShPtr> features(kps.size());
    std::transform(std::execution::par_unseq, kps.begin(), kps.end(), features.begin(), [&](const auto &kp) {
      const cv::Vec2f dIuv = dI.at<cv::Vec2f>(kp(1) / scale, kp(0) / scale);
      const Vec2d response{dIuv[0], dIuv[1]};
      return std::make_unique<Feature2D>(kp, f, level, response.norm());
    });

    /*For each cell use best entry.*/
    features = keypoint::subsampling::grid(
      features,
      f->height(),
      f->width(),
      _gridSize,
      [&](auto kp0, auto kp1) { return kp0->response() > kp1->response(); },
      [](auto ft) { return ft->position(); });

    MLOG(INFO) << format(
      "Newly selected features: {}. Subsampling in {}x{} cells. Remaining: {}",
      kps.size(),
      f->height() / _gridSize,
      f->width() / _gridSize,
      features.size());

    f->addFeatures(features);
    MLOG(INFO) << format("Total features in frame: {}", f->features().size());
    log::append("Features", overlay::Features(f, _gridSize));
  }
}

}  // namespace vslam
