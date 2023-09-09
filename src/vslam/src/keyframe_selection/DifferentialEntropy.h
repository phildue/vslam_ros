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

#ifndef VSLAM_KEY_FRAME_SELECTION
#define VSLAM_KEY_FRAME_SELECTION

#include "core/Frame.h"
#include "core/macros.h"
namespace vslam::keyframe_selection {
class DifferentialEntropy {
public:
  TYPEDEF_PTR(DifferentialEntropy)

  DifferentialEntropy(double minEntropyRatio = 0.9) :
      _minEntropyRatio(minEntropyRatio),
      _lastFrame(nullptr),
      _entropyRef(std::numeric_limits<double>::quiet_NaN()),
      _newKeyFrame(false) {}
  void update(Frame::ShPtr currentFrame);
  bool newKeyFrame() const;
  Frame::ShPtr keyFrame() const;
  Frame::ShPtr select(Frame::ShPtr currentFrame);

private:
  const double _minEntropyRatio;
  Frame::ShPtr _lastFrame, _keyFrame;
  double _entropyRef;
  bool _newKeyFrame;
  static constexpr const char LOG_NAME[] = "keyframe_selection";
};
}  // namespace vslam::keyframe_selection
#endif  // VSLAM_KEY_FRAME_SELECTION
