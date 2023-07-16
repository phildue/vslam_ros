#pragma once
#include "core/Frame.h"
namespace vslam {
class FeatureSelection {
public:
  typedef std::shared_ptr<FeatureSelection> ShPtr;
  typedef std::unique_ptr<FeatureSelection> UnPtr;
  FeatureSelection(
    float intensityGradientMin = 5,
    float depthGradientMin = 0.01,
    float depthGradientMax = 0.3,
    float depthMin = 0,
    float depthMax = 8.0,
    float gridSize = 20.0f,
    int nLevels = 4);

  void select(Frame::ShPtr f, bool override = true) const;

private:
  const float _intensityGradientMin, _depthGradientMin, _depthGradientMax, _depthMin, _depthMax, _gridSize = 1.0f;
  const int _nLevels;
};

}  // namespace vslam
