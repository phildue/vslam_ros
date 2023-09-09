// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "DifferentialEntropy.h"
#include "utils/log.h"
namespace vslam::keyframe_selection {

Frame::ShPtr DifferentialEntropy::select(Frame::ShPtr currentFrame) {
  const double entropy = std::log(currentFrame->pose().cov().determinant());
  if (!_lastFrame) {
    _entropyRef = entropy;
    return nullptr;
  }
  if (entropy / _entropyRef < _minEntropyRatio) {
    auto kf = _lastFrame;
    _lastFrame = nullptr;
    return kf;
  }
  return nullptr;
}
void DifferentialEntropy::update(Frame::ShPtr currentFrame) {
  const double entropy = std::log(currentFrame->pose().cov().determinant());

  if (_newKeyFrame || !std::isfinite(_entropyRef)) {
    _entropyRef = entropy;
    _newKeyFrame = false;
  } else if (entropy / _entropyRef < _minEntropyRatio) {
    _entropyRef = std::numeric_limits<double>::quiet_NaN();
    _newKeyFrame = true;
    _keyFrame = _lastFrame;
  }
  CLOG(DEBUG, LOG_NAME) << format(
    "Entropy ratio: [{:.3f}/{:.3f}] [{:.3f}/{:.3f}] {}",
    entropy / _entropyRef,
    _minEntropyRatio,
    entropy,
    _entropyRef,
    _newKeyFrame ? "New KeyFrame" : "");
  _lastFrame = currentFrame;
}
bool DifferentialEntropy::newKeyFrame() const { return _newKeyFrame; }

Frame::ShPtr DifferentialEntropy::keyFrame() const { return _keyFrame; }

}  // namespace vslam::keyframe_selection
