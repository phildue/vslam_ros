#ifndef VSLAM_KEY_FRAME_SELECTION
#define VSLAM_KEY_FRAME_SELECTION

#include "core/core.h"
namespace pd::vslam {
  class KeyFrameSelection {
public:
    typedef std::shared_ptr < KeyFrameSelection > ShPtr;
    typedef std::unique_ptr < KeyFrameSelection > UnPtr;
    typedef std::shared_ptr < const KeyFrameSelection > ConstShPtr;
    typedef std::unique_ptr < const KeyFrameSelection > ConstUnPtr;

    virtual void update(FrameRgbd::ConstShPtr frame) = 0;
    virtual bool isKeyFrame() const = 0;

    static ShPtr make();

  };
  class KeyFrameSelectionIdx: public KeyFrameSelection {
public:
    typedef std::shared_ptr < KeyFrameSelectionIdx > ShPtr;
    typedef std::unique_ptr < KeyFrameSelectionIdx > UnPtr;
    typedef std::shared_ptr < const KeyFrameSelectionIdx > ConstShPtr;
    typedef std::unique_ptr < const KeyFrameSelectionIdx > ConstUnPtr;

    KeyFrameSelectionIdx(uint64_t period = 2) : KeyFrameSelection(), _period(period), _ctr(0U) {
    }
    void update(FrameRgbd::ConstShPtr UNUSED(frame)) override {_ctr++;}
    bool isKeyFrame() const override {return _ctr == 0 || _ctr % _period == 0;}

private:
    const uint64_t _period;
    uint64_t _ctr;
  };
}
#endif// VSLAM_RGBD_ODOMETRY
