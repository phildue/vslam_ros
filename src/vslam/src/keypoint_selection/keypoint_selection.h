#pragma once
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>

#include "core/Frame.h"
#include "core/types.h"
namespace vslam::keypoint
{
std::vector<Vec2d> selectManual(Frame::ConstShPtr f, int patchSize = 7);

template <typename Criterion>
std::vector<Vec2d> select(Frame::ConstShPtr f, int level, Criterion criterion)
{
  std::vector<int> vs(f->height(level));
  for (size_t v = 0; v < f->height(level); v++) {
    vs[v] = v;
  }

  std::vector<Vec2d> keypoints = std::transform_reduce(
    std::execution::par_unseq, vs.begin(), vs.end(), std::vector<Vec2d>(),
    [](auto v0, auto v1) {
      v0.insert(v0.end(), v1.begin(), v1.end());
      return v0;
    },
    [&](int v) {
      std::vector<Vec2d> kps;
      kps.reserve(f->width(level));
      for (size_t u = 0; u < f->width(level); u++) {
        if (criterion(Vec2d(u, v))) {
          kps.push_back(Vec2d(u, v));
        }
      }
      return kps;
    });
  return keypoints;
}
namespace subsampling
{
template <typename KeyPoint, typename Position>
std::vector<KeyPoint> uniform(
  const std::vector<KeyPoint> & keypoints, int height, int width, int nPoints, Position getPosition)
{
  const size_t nNeeded = std::max<size_t>(20, nPoints);
  std::vector<bool> mask(height * width, false);
  std::vector<KeyPoint> subset;
  subset.reserve(keypoints.size());
  if (nNeeded < keypoints.size()) {
    while (subset.size() < nNeeded) {
      auto kp = keypoints[random::U(0, keypoints.size() - 1)];
      const auto & pos = getPosition(kp);
      const size_t idx = pos(1) * width + pos(0);
      if (!mask[idx]) {
        subset.push_back(kp);
        mask[idx] = true;
      }
    }
    return subset;
  }
  return keypoints;
}
template <typename KeyPoint, typename Score, typename Position>
std::vector<KeyPoint> grid(
  const std::vector<KeyPoint> & keypoints, int height, int width, float cellSize, Score betterScore,
  Position getPosition)
{
  const size_t nRows = static_cast<size_t>(static_cast<float>(height) / cellSize);
  const size_t nCols = static_cast<size_t>(static_cast<float>(width) / cellSize);

  /* Create grid for subsampling where each cell contains index of keypoint with the highest response
  *  or the total amount of keypoints if empty */
  std::vector<size_t> grid(nRows * nCols, keypoints.size());
  for (size_t idx = 0U; idx < keypoints.size(); idx++) {
    const auto & kp = keypoints[idx];
    const auto & pos = getPosition(kp);
    const size_t r = static_cast<size_t>(pos.y() / cellSize);
    const size_t c = static_cast<size_t>(pos.x() / cellSize);
    if (
      grid[r * nCols + c] >= keypoints.size() || betterScore(kp, keypoints[grid[r * nCols + c]])) {
      grid[r * nCols + c] = idx;
    }
  }

  std::vector<KeyPoint> kptsGrid;
  kptsGrid.reserve(nRows * nCols);
  std::for_each(grid.begin(), grid.end(), [&](auto idx) {
    if (idx < keypoints.size()) {
      kptsGrid.push_back(keypoints[idx]);
    }
  });
  return kptsGrid;
}
template <typename KeyPoint, typename Score, typename Position>
std::vector<KeyPoint> grid(
  const std::vector<KeyPoint> & keypoints, int height, int width, int cellSize, Score betterScore)
{
  return grid(keypoints, height, width, cellSize, betterScore, [](auto kp) { return kp; });
}

}  // namespace subsampling

}  // namespace vslam::keypoint