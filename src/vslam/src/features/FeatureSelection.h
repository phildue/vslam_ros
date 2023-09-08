#pragma once
#include "core/Feature2D.h"
#include "core/Frame.h"
#include "core/macros.h"
#include "keypoint_selection.h"
#include "overlays.h"
#include "utils/log.h"
namespace vslam {

struct FiniteGradient {
  const float _intensityGradientMin, _depthGradientMin, _depthGradientMax, _depthMin, _depthMax;

  bool operator()(const Vec2d &uv, Frame::ConstShPtr f, int level) const {
    const float z = f->depth(level).at<float>(uv(1), uv(0));
    const cv::Vec2f dIuv = f->dI(level).at<cv::Vec2f>(uv(1), uv(0));
    const cv::Vec2f dZuv = f->dZ(level).at<cv::Vec2f>(uv(1), uv(0));
    return (
      std::isfinite(z) && std::isfinite(dZuv[0]) && std::isfinite(dZuv[1]) && _depthMin < z && z < _depthMax &&
      std::abs(dZuv[0]) < _depthGradientMax && std::abs(dZuv[1]) < _depthGradientMax &&
      (std::abs(dIuv[0]) > _intensityGradientMin || std::abs(dIuv[1]) > _intensityGradientMin || std::abs(dZuv[0]) > _depthGradientMin ||
       std::abs(dZuv[1]) > _depthGradientMin));
  }
};

class FeatureMask {

  std::vector<std::vector<std::vector<bool>>> _grid;
  const float _gridSize;

public:
  FeatureMask(Frame::ConstShPtr f, float gridSize) :
      _gridSize(gridSize) {
    for (size_t level = 0; level < f->nLevels(); level++) {
      _grid.push_back(keypoint::createOccupancyGrid(f->features(level), f->height(level), f->width(level), gridSize, [](auto ft) {
        return ft->position() / std::pow(2.0, ft->level());
      }));
      const int nOccupied = std::accumulate(_grid[level].begin(), _grid[level].end(), 0, [](auto s, auto row) {
        s += std::count(row.begin(), row.end(), true);
        return s;
      });
      const int nCols = f->width() / _gridSize;
      const int nRows = _grid[level].size() / (float)nCols;
      CLOG(INFO, "features") << format(
        "Present features: {} occupying {} in {}x{} cells at level {}.", f->features().size(), nOccupied, nRows, nCols, level);
    }
  }
  bool operator()(const Vec2d &uv, Frame::ConstShPtr UNUSED(f), int level) const {
    return _grid[level][uv(1) / _gridSize][uv(0) / _gridSize];
  }
};

template <typename Criterion> class FeatureSelection {
public:
  typedef std::shared_ptr<FeatureSelection<Criterion>> ShPtr;
  typedef std::unique_ptr<FeatureSelection<Criterion>> UnPtr;

  static constexpr char LOG_NAME[] = "features";

  struct NoMask {
    bool operator()(const Vec2d &UNUSED(uv), Frame::ConstShPtr UNUSED(f), int UNUSED(level)) const { return false; }
  };

  FeatureSelection(const Criterion &criterion, float gridSize, int nLevels = 4) :
      _criterion(criterion),
      _gridSize(gridSize),
      _nLevels(nLevels) {
    log::create(LOG_NAME);
  }

  template <typename Mask = NoMask> void select(Frame::ShPtr f, const Mask &masked = NoMask()) const {
    Feature2D::VecShPtr featuresAll;
    for (int level = _nLevels - 1; level >= 0; level--) {
      const double scale = std::pow(2.0, level);

      /* Propose features in empty cells that meet feature criterion */
      std::vector<Vec2d> kps = keypoint::select(
        f, level, [&](const Vec2d &uv) { return !masked(uv, Frame::ConstShPtr(f), level) && _criterion(uv, Frame::ConstShPtr(f), level); });
      CLOG(INFO, LOG_NAME) << format("Newly selected features: {}", kps.size());

      Feature2D::VecShPtr features(kps.size());
      std::transform(std::execution::par_unseq, kps.begin(), kps.end(), features.begin(), [&](const auto &kp) {
        const cv::Vec2f dIuv = f->dI(level).at<cv::Vec2f>(kp(1) / scale, kp(0) / scale);
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
      CLOG(INFO, LOG_NAME) << format(
        "Subsampling in {}x{} cells. Remaining: {}", f->height() / _gridSize, f->width() / _gridSize, featuresAll.size());

      f->addFeatures(features);
    }

    CLOG(INFO, LOG_NAME) << format("Total features in frame: {}", f->features().size());
    log::append("Features", overlay::Features(f, 1, _gridSize));
  }

  const float &gridSize() const { return _gridSize; }

private:
  const Criterion _criterion;
  const float _gridSize;
  const int _nLevels;
};

}  // namespace vslam
