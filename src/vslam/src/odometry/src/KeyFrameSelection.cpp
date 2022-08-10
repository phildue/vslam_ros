#include "KeyFrameSelection.h"
namespace pd::vslam
{

KeyFrameSelection::ShPtr KeyFrameSelection::make()
{
  return std::make_shared<KeyFrameSelectionIdx>();
}
}
